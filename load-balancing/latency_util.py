import numpy as np
from tqdm import tqdm
import torch

from create_dataset import (
    PowerofKPolicy,
    OptimalPolicy,
    RandomPolicy,
    ShortestQueuePolicy,
    JobScheduler,
    ProcessTimeManager,
    TrackerPolicy,
)


def collect_traces_sim_traj_fact(
    job_sizes,
    inter_arrs,
    ptimes_obs,
    actions_obs,
    feature_extractor,
    action_factor_net,
    r,
    pt_mean,
    pt_std,
    test_pol_idx,
    p_change=0,
):
    ns = 8
    seed = 43
    load_target = 0.6
    # p_change should be 0 if '_0' and 0.5 if '_50'
    assert job_sizes.ndim == 2
    assert job_sizes.shape == inter_arrs.shape
    assert ptimes_obs.shape == actions_obs.shape
    assert ptimes_obs.shape[:] == job_sizes.shape

    no_traj, length = job_sizes.shape

    # Load info arrays
    time_jobs = np.cumsum(inter_arrs, axis=-1)
    actions = np.empty((1, no_traj, length), dtype=int)
    proc_times = np.empty((1, no_traj, length), dtype=float)
    latencies = np.empty((1, no_traj, length), dtype=float)
    feature = np.empty((no_traj, length), dtype=float)
    max_action = 7
    env = JobScheduler(ns)
    pt_mgr = ProcessTimeManager(ns, seed, p_change)
    # Load policies
    pols = [
        RandomPolicy(seed, ns),
        ShortestQueuePolicy(seed, ns),
        PowerofKPolicy(seed, ns, 2),
        PowerofKPolicy(seed, ns, 3),
        PowerofKPolicy(seed, ns, 4),
        PowerofKPolicy(seed, ns, 5),
        OptimalPolicy(seed, ns),
        TrackerPolicy(seed, ns, 0.995),
    ]
    for i_pol, policy in enumerate(pols[test_pol_idx : test_pol_idx + 1]):
        # Reset environment
        # Load rate manager
        pt_mgr = ProcessTimeManager(ns, seed, p_change)
        # Register rate manager for optimal policy
        policy.register(pt_manager=pt_mgr)
        for index in tqdm(range(no_traj)):
            obs = env.reset()
            for i in range(length):
                # Choose server
                act = policy.act(obs)
                # Calculate processing time
                pt_o = ptimes_obs[index, i]
                orig_action = int(actions_obs[index, i])
                # get one-hot encoder
                orig_action_oh = np.zeros(max_action + 1)
                orig_action_oh[orig_action] = 1

                pt_o_white = (pt_o - pt_mean) / pt_std

                action = int(act)
                # get one-hot encoder for action
                action_oh = np.zeros(max_action + 1)
                action_oh[action] = 1

                input_numpy = np.array(
                    [pt_o_white] + list(orig_action_oh)
                )  # Model only accepts normalized inputs
                input_numpy = np.expand_dims(input_numpy, axis=0)
                input_tensor = torch.as_tensor(
                    input_numpy, dtype=torch.float32, device=torch.device("cpu")
                )
                with torch.no_grad():
                    feature_tensor = feature_extractor(input_tensor)
                input_numpy = np.array(action_oh)
                input_numpy = np.expand_dims(input_numpy, axis=0)
                input_tensor = torch.as_tensor(
                    input_numpy, dtype=torch.float32, device=torch.device("cpu")
                )
                action_factor = action_factor_net(input_tensor)

                cf_processing_time_white = torch.mul(feature_tensor, action_factor)
                cf_processing_time_white = torch.matmul(
                    cf_processing_time_white, torch.ones([r, 1], dtype=torch.float32)
                )

                cf_processing_time_white = (
                    cf_processing_time_white.cpu().detach().numpy()[0][0]
                )
                ptime = (cf_processing_time_white * pt_std) + pt_mean

                # ########## Use the counterfactual model to predict ptime
                # Submit processing time for tracker policy
                policy.submit(act, ptime)
                # Receive latency and queue sizes
                latency, obs = env.step(ptime, act, inter_arrs[index, i])
                # Save info
                proc_times[i_pol, index, i] = ptime
                feature[index, i] = feature_tensor.detach().cpu().numpy()[0]
                actions[i_pol, index, i] = act
                latencies[i_pol, index, i] = latency

    return feature, actions, proc_times, latencies


def collect_traces_direct_traj(
    job_sizes,
    inter_arrs,
    ptimes_obs,
    buffer_predictor,
    pt_mean,
    pt_std,
    test_pol_idx,
    p_change=0,
):
    ns = 8
    seed = 43
    load_target = 0.6
    # p_change should be 0 if '_0' and 0.5 if '_50'
    assert job_sizes.ndim == 2
    assert job_sizes.shape == inter_arrs.shape
    assert ptimes_obs.shape[:] == job_sizes.shape

    no_traj, length = job_sizes.shape

    # Load policies
    pols = [
        RandomPolicy(seed, ns),
        ShortestQueuePolicy(seed, ns),
        PowerofKPolicy(seed, ns, 2),
        PowerofKPolicy(seed, ns, 3),
        PowerofKPolicy(seed, ns, 4),
        PowerofKPolicy(seed, ns, 5),
        OptimalPolicy(seed, ns),
        TrackerPolicy(seed, ns, 0.995),
    ]
    if test_pol_idx is not None:
        pols = pols[test_pol_idx : test_pol_idx + 1]
        p_out = 1
    else:
        p_out = len(pols)
    # Load info arrays
    time_jobs = np.cumsum(inter_arrs, axis=-1)
    actions = np.empty((p_out, no_traj, length), dtype=int)
    proc_times = np.empty((p_out, no_traj, length), dtype=float)
    latencies = np.empty((p_out, no_traj, length), dtype=float)
    feature = np.empty((no_traj, length), dtype=float)
    max_action = 7
    env = JobScheduler(ns)
    pt_mgr = ProcessTimeManager(ns, seed, p_change)
    for i_pol, policy in enumerate(pols):
        # Reset environment
        # Load rate manager
        pt_mgr = ProcessTimeManager(ns, seed, p_change)
        # Register rate manager for optimal policy
        policy.register(pt_manager=pt_mgr)
        for index in tqdm(range(no_traj)):
            obs = env.reset()
            for i in range(length):
                # Choose server
                act = policy.act(obs)
                # Calculate processing time
                pt_o = ptimes_obs[index, i]
                pt_o_white = (pt_o - pt_mean) / pt_std

                action = int(act)
                # get one-hot encoder for action
                input_numpy = np.zeros(max_action + 2)
                input_numpy[action + 1] = 1
                input_numpy[0] = pt_o_white
                input_numpy = np.array([input_numpy])
                input_tensor = torch.as_tensor(
                    input_numpy,
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                )
                with torch.no_grad():
                    cf_processing_time_white = buffer_predictor(input_tensor)
                cf_processing_time_white = cf_processing_time_white.cpu().numpy()[0][0]
                ptime = (cf_processing_time_white * pt_std) + pt_mean

                ########### Use the counterfactual model to predict ptime
                # Submit processing time for tracker policy
                policy.submit(act, ptime)
                # Receive latency and queue sizes
                latency, obs = env.step(ptime, act, inter_arrs[index, i])
                # Save info
                proc_times[i_pol, index, i] = ptime
                actions[i_pol, index, i] = act
                latencies[i_pol, index, i] = latency

    return actions, proc_times, latencies
