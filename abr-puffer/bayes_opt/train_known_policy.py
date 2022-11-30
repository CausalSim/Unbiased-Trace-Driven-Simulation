import os
import time
from typing import List, Dict, Union, Tuple
import numpy as np
from tqdm import tqdm
from multiprocessing import Queue, Process
from queue import Empty
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kernels

from env.bola import BolaAgent
from env.bba import BBA
from env.abr import ABRSimEnv, db_ssim, ssim_db
from env.csim_mlp import MLP
from utils import make_folders, save_args, set_omp_thrs
from argparse import ArgumentParser

parser = ArgumentParser(description='Puffer RL parameters')

parser.add_argument('--seed', type=int, default=10, help='random seed (default: 42)')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers (default: 8)')
parser.add_argument('--rebuf_penalty', type=float, default=100, help='Rebuffering penalty (default: 100)')

parser.add_argument('--sim_mode', type=str, choices=['causalsim', 'expertsim'], help='Simulation mode', required=True)
parser.add_argument('--policy', type=str, choices=['bba', 'bola1', 'bola2'], help='Which policy to use', required=True)
parser.add_argument('--csim_buf_path', type=str, help='Buffer CausalSim Model', required=True)
parser.add_argument('--csim_dt_path', type=str, help='Download Time CausalSim Model', required=True)
parser.add_argument('--dir', type=str, help='Puffer trace path', required=True)


config = parser.parse_args()


def eval_policy(num_traces: int, agent_config: Dict[str, Union[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    traces = np.random.RandomState(config.seed).choice(num_traces, size=num_traces, replace=False)
    q_exp = Queue(config.num_workers * 8)
    worker_procs = []
    for i in range(config.num_workers):
        worker_procs.append(Process(target=run_worker, args=(i, traces[traces % config.num_workers == i],
                                                             q_exp, agent_config)))
    for i in range(config.num_workers):
        worker_procs[i].start()

    # Set up reward stats
    # 1) Reward 2) Chunk size 3) SSIM 4) Rebuffer 5) Smoothness penalty 6) Download time 7) Buffer size
    reward_stats = [None for _ in range(num_traces)]
    done_stats = 0

    with tqdm(total=num_traces, leave=False) as pbar:
        while done_stats < num_traces:
            try:
                index, stats = q_exp.get(block=False)
                reward_stats[index] = stats
                done_stats += 1
                pbar.update()
            except Empty:
                time.sleep(1 / 100.)

    for worker in worker_procs:
        worker.join()

    return np.array(reward_stats, dtype=object), traces


def run_worker(worker_id: int, trace_list: List[int], queue_exp: Queue,
               agent_config: Dict[str, Union[str, np.ndarray]]):
    assert np.all(np.array(trace_list) % config.num_workers == worker_id)
    assert len(trace_list) == len(np.unique(trace_list))
    assert np.min(trace_list) >= 0
    set_omp_thrs()

    # Setting up environment
    env = ABRSimEnv(env_config={
        "sim_mode": config.sim_mode,
        "seed": config.seed,
        "trace_path": config.dir + '/gp_cooked/',
        "csim_buf_model": config.csim_buf_path,
        "csim_dt_model": config.csim_dt_path,
        "id": worker_id,
        "num_w": config.num_workers,
        'rebuf_p': config.rebuf_penalty,
    })
    env.seed(config.seed + worker_id)
    assert np.max((np.array(trace_list) - worker_id) // config.num_workers) < len(env.all_traces)

    # Shorthands
    act_len = env.CHOICES

    # Known agent
    if agent_config['name'] == 'bba':
        assert 'params' in agent_config and agent_config['params'].shape == (2,)
        known_agent = BBA(act_len, reservoir=agent_config['params'][0], cushion=agent_config['params'][1])
    elif agent_config['name'] == 'bola1':
        assert 'params' in agent_config and agent_config['params'].shape == (2,)
        known_agent = BolaAgent(1, act_len, reservoir=agent_config['params'][0], cushion=agent_config['params'][1])
    elif agent_config['name'] == 'bola2':
        assert 'params' in agent_config and agent_config['params'].shape == (2,)
        known_agent = BolaAgent(2, act_len, reservoir=agent_config['params'][0], cushion=agent_config['params'][1])
    else:
        raise ValueError('No such policy is known!')

    for epoch in range(len(trace_list)):
        # Restart the environment
        transformed_trace_index = (trace_list[epoch]-worker_id) // config.num_workers
        obs = env.reset(trace_choice=transformed_trace_index)
        done = False
        rew_index = 0
        # Set up reward stats
        # 1) Reward 2) Chunk size 3) SSIM 4) Rebuffer 5) Smoothness penalty 6) Download time 7) Buffer size
        reward_stats = np.zeros((env.curr_trace_chunks+1, 7))

        while not done:
            # Get action
            act = known_agent.sample_action(obs)

            reward_stats[rew_index, 6] = env.buffer_size

            # Apply action
            next_obs, rew, done, info = env.step(act)

            reward_stats[rew_index, 0] = rew
            reward_stats[rew_index, 1] = info['chunk_size']
            reward_stats[rew_index, 2] = info['ssim']
            reward_stats[rew_index, 3] = info['stall_time']
            reward_stats[rew_index, 4] = info['ssim_change']
            reward_stats[rew_index, 5] = info['download_time']
            rew_index += 1

            obs = next_obs

        reward_stats[rew_index, 0] = np.nan
        reward_stats[rew_index, 1] = np.nan
        reward_stats[rew_index, 2] = np.nan
        reward_stats[rew_index, 3] = np.nan
        reward_stats[rew_index, 4] = np.nan
        reward_stats[rew_index, 5] = np.nan
        reward_stats[rew_index, 6] = env.buffer_size

        queue_exp.put((trace_list[epoch], reward_stats))


def run_experiment():
    set_omp_thrs()
    output_folder = f'{config.dir.rstrip("/")}/tests/gp_{config.policy}_{config.sim_mode}/'
    make_folders(output_folder)
    os.makedirs(output_folder + '/data_s/')
    save_args(config, output_folder)

    # Setting up environment
    env = ABRSimEnv(env_config={
        "sim_mode": config.sim_mode,
        "seed": config.seed,
        "trace_path": config.dir + '/gp_cooked/',
        "csim_buf_model": config.csim_buf_path,
        "csim_dt_model": config.csim_dt_path,
    })
    num_traces = len(env.all_traces)
    del env

    run_stats = []
    next_params = []

    kappa = 10
    num_samples = 100
    starting_samples = 15

    params_all = np.array([(x / 10, y / 10) for x in range(150) for y in range(150) if x + y <= 150])

    for i in np.random.choice(len(params_all), starting_samples, replace=False):
        next_params.append(params_all[i])

    # Original params for BBA, BOLA1, BOLA2
    next_params = [
        np.array([3, 10.5]),
        np.array([3, 8.7]),
        np.array([3, 9.12]),
    ] + next_params

    print(f'Have {len(next_params)} points to try first!!!')

    while len(run_stats) < num_samples:
        while len(next_params) > 0:
            # folder_run = output_folder + f'/data_s/{config.policy}_{next_params[0][0]}_{next_params[0][1]}/'
            # os.makedirs(folder_run)
            rew_stats, traces = eval_policy(num_traces, {'name': config.policy, 'params': next_params[0]})
            if config.sim_mode == 'expertsim':
                # ExpertSim simulations might have fewer chunks than original trajectory
                rew_stats = [arr[np.any(arr != 0, axis=-1)] for arr in rew_stats]
            # np.save(f'{folder_run}/trace_ind.npy', traces)
            # np.save(f'{folder_run}/rewards_test.npy', rew_stats)
            run_stats.append([*extract_fitness(next_params[0], rew_stats)])
            next_params = next_params[1:]
            stat_np = np.array(run_stats)
            np.save(f'{output_folder}/run_stats.npy', stat_np)

        sigma, x_s, y_pred, y_s = train_gp_and_infer(params_all, run_stats)
        x_new = params_all[np.argmax(y_pred[:, 0] + kappa * sigma)] + np.random.normal(0, 0.1, size=(2,))
        x_new = np.maximum([0, 0], x_new)
        next_params.append(x_new)

        print(f'Best point so far: {x_s[np.argmax(y_s)]} -> {np.max(y_s)}, next point is {x_new} with predicted '
              f'reward {y_pred[np.argmax(y_pred[:, 0] + kappa * sigma)]} and '
              f'uncertainty {sigma[np.argmax(y_pred[:, 0] + kappa * sigma)]}')

    _, x_s, y_pred, y_s = train_gp_and_infer(params_all, run_stats)
    arg_best_pred = np.argmax(y_pred)
    x_hat = params_all[arg_best_pred]

    print(f'Best point so far: {x_s[np.argmax(y_s)]} -> {np.max(y_s)}, final suggestion is {x_hat} with predicted '
          f'reward {y_pred[arg_best_pred]}')


def train_gp_and_infer(params_all, run_stats):
    stat_np = np.array(run_stats)
    x_s = stat_np[:, :2]
    y_s = ssim_db(stat_np[:, [3]]) - config.rebuf_penalty * stat_np[:, [4]]
    model = GaussianProcessRegressor(kernel=kernels.Matern(), alpha=1e-4)
    model.fit(x_s, y_s)
    y_pred, sigma = model.predict(params_all, return_std=True)
    return sigma, x_s, y_pred, y_s


def extract_fitness(params: np.ndarray, rew_stats: List[np.ndarray]) -> Tuple[float, float, float, float, float]:
    wt_weights = np.array([(arr.shape[0] - 1) * 2.002 + arr[1:-1, 3].sum() - arr[-1, 6] for arr in rew_stats])
    ssim_arr = [arr[:-1][arr[:-1, 2] <= 50, 2] for arr in rew_stats]
    wt_weights_ssim = [warr for warr, arr in zip(wt_weights, ssim_arr) if len(arr) > 0]
    ssim_arr = [arr for arr in ssim_arr if len(arr) > 0]
    ssim_score = np.average([arr.mean() for arr in ssim_arr], weights=wt_weights_ssim)
    ssim_db_score = np.average([db_ssim(arr).mean() for arr in ssim_arr], weights=wt_weights_ssim)
    stall_score = np.sum([arr[1:-1, 3].sum() for arr in rew_stats]) / wt_weights.sum()
    return params[0], params[1], ssim_score, ssim_db_score, stall_score


if __name__ == '__main__':
    run_experiment()
