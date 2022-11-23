import os
from typing import List
import numpy as np
from tqdm import tqdm, trange
from policies import Agent
import argparse

from env.abr import ABRSimEnv
from policies import get_all_policies


def run_trajectories(env: ABRSimEnv, policy_agent: Agent, save_path: str) -> np.ndarray:
    # shorthand
    num_traces = len(env.all_traces)
    len_vid = env.total_num_chunks
    size_obs = env.obs_high.shape[0]

    # trajectory to return, each step has obs{of size_obs}, action{of size 1}, reward{of size 1}
    traj = np.empty((num_traces, len_vid, size_obs + 1 + 1))

    for trace_index in trange(num_traces):
        # Choose specific trace and start from the initial point in the trace
        obs = env.reset(trace_choice=trace_index)

        for epi_step in range(len_vid):
            # choose action through policy
            act = policy_agent.take_action(obs)

            # take action
            next_obs, rew, done, info = env.step(act)
            if np.abs((next_obs[9] - obs[-6+act])/next_obs[9] * 100) > 1e-8:
                import pdb
                pdb.set_trace()
                assert False
            if np.abs((next_obs[4] - obs[-12+act])/next_obs[4] * 100) > 1e-8:
                assert False

            # save action
            traj[trace_index][epi_step][:size_obs] = obs
            traj[trace_index][epi_step][size_obs] = act
            traj[trace_index][epi_step][size_obs+1] = rew

            # episode should not finish before video length
            assert not done or epi_step == len_vid-1

            # next state
            obs = next_obs

    np.save(save_path, traj)

    return traj


def run_expert_cf(traj: np.ndarray, cf_path: str, policies: List[Agent], mpc_lookback: int):
    if os.path.exists(cf_path):
        print(f'CF for {cf_path} already exists, continuing')
        raise OSError
    chunk_sizes = traj[0, :, 2*mpc_lookback+3:2*mpc_lookback+3+ABRSimEnv.CHOICES]
    cf_traj = np.empty((len(policies), traj.shape[0], traj.shape[1], 3))
    one_matched = False

    for p, policy_agent in enumerate(tqdm(policies, leave=False)):
        for i in tqdm(range(traj.shape[0]), leave=False):
            obs = np.array(traj[i, 0, :-14])
            for j in range(traj.shape[1]-1):
                cf_traj[p, i, j, 0] = obs[2*mpc_lookback]
                cf_traj[p, i, j, 1] = obs[2*mpc_lookback-1]
                cf_traj[p, i, j, 2] = obs[2*mpc_lookback+2]
                act = policy_agent.take_action(obs)

                dtime_orig = traj[i, j+1, 2*mpc_lookback-1]
                dtime = dtime_orig / chunk_sizes[j, int(traj[i, j, -2])] * chunk_sizes[j, act]
                obs[:mpc_lookback] = traj[i, j+1, :mpc_lookback]
                obs[mpc_lookback:2*mpc_lookback-1] = obs[mpc_lookback+1:2*mpc_lookback]
                obs[2*mpc_lookback-1] = dtime
                obs[2*mpc_lookback] = min(max(obs[2*mpc_lookback] - dtime, 0) + ABRSimEnv.CHUNK_LENGTH,
                                          ABRSimEnv.MAX_BUFF_S)
                obs[2*mpc_lookback + 1] -= 1
                obs[2 * mpc_lookback + 2] = act
                obs[2 * mpc_lookback + 2:] = np.array(traj[i, j+1, 2 * mpc_lookback + 2:-14])
            cf_traj[p, i, -1, 0] = obs[2 * mpc_lookback]
            cf_traj[p, i, -1, 1] = obs[2 * mpc_lookback - 1]
            cf_traj[p, i, -1, 2] = obs[2 * mpc_lookback + 2]
            # Must do this to keep seeds consistent
            policy_agent.take_action(obs)

        cf = cf_traj[p, :, :]
        orig = traj[:, :, [2 * mpc_lookback, 2 * mpc_lookback-1, 2 * mpc_lookback + 2]]
        one_matched = one_matched or np.allclose(cf, orig)

    assert one_matched

    np.save(cf_path, cf_traj)


def main():
    parser = argparse.ArgumentParser(description='parameters')

    # -- Basic --
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--eps', type=float, default=1e-6, help='epsilon (default: 1e-6)')
    parser.add_argument('--trace_sim_count', type=int, default=5000, help='Number of generated traces (default: 5000)')
    parser.add_argument('--bba_reservoir', type=float, default=5, help='BBA - Reservoir (default: 5)')
    parser.add_argument('--bba_cushion', type=float, default=10, help='BBA - Cushion (default: 10)')
    parser.add_argument('--mpc_lookback', type=int, default=5, help='MPC - Throughput lookback (default: 5)')
    parser.add_argument('--mpc_lookahead', type=int, default=5, help='MPC - Throughput lookahead (default: 5)')
    parser.add_argument('--dir', type=str, required=True, help='Output folder')

    config = parser.parse_args()

    # Create output folder
    os.makedirs(config.dir, exist_ok=True)

    # set up environments for workers
    print('Setting up environment..')
    env = ABRSimEnv(mpc_lookahead=config.mpc_lookahead, mpc_lookback=config.mpc_lookback, seed=config.seed,
                    trace_folder=config.dir, num_traces=config.trace_sim_count)

    for pol, name, path in zip(*get_all_policies(config)):
        print(f'Starting {name}..')
        traj = run_trajectories(env, pol, f'{config.dir}/{path}')
        run_expert_cf(traj, f'{config.dir}/cf_{path}', get_all_policies(config)[0], config.mpc_lookback)

    print('DONE')


if __name__ == '__main__':
    main()
