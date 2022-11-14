import heapq
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt


class JobScheduler(object):
    def __init__(self, num_servers):
        self.ns = num_servers
        self.queue_lens = np.zeros(self.ns, dtype=int)
        self.queue_times = np.zeros(self.ns, dtype=float)
        self.events = []
        self.t_sim = 0

    def reset(self):
        self.queue_lens = np.zeros(self.ns, dtype=int)
        self.queue_times = np.zeros(self.ns, dtype=float)
        self.events.clear()
        self.t_sim = 0
        return self.queue_lens

    def _sim(self, ia_time):
        # Simulate for ia_time msecs
        heapq.heappush(self.events, (self.t_sim + ia_time, -1))
        while self.events[0][1] != -1:
            key, item = heapq.heappop(self.events)
            self.queue_lens[item] -= 1
        key, item = heapq.heappop(self.events)
        self.t_sim = key
        assert item == -1
        assert np.min(self.queue_lens) >= 0
        self.queue_times = (self.queue_times - ia_time).clip(min=0)
        return self.queue_lens

    def _schedule_job(self, proc_time, server):
        # Add new job
        self.queue_lens[server] += 1
        self.queue_times[server] += proc_time
        heapq.heappush(self.events, (self.t_sim + self.queue_times[server], server))
        return self.queue_times[server]

    def step(self, proc_time, server, ia_time):
        latency = self._schedule_job(proc_time, server)
        return latency, self._sim(ia_time)


class ProcessTimeManager(object):
    def __init__(self, num_servers, seed, p_change_per_hour):
        self.ns = num_servers
        self.rng = np.random.RandomState(seed)
        self.gamma = -np.log(1 - p_change_per_hour) / 3600 / 1000
        self.slow_down_rates = np.power(5, self.rng.random(self.ns) * 2 - 1)

    def _sim(self, ia_time):
        p_stay = np.exp(-self.gamma * ia_time)
        change_arr = self.rng.choice([0, 1], self.ns, p=[p_stay, 1 - p_stay])
        self.slow_down_rates = (
            change_arr * np.power(5, self.rng.random(self.ns) * 2 - 1)
            + (1 - change_arr) * self.slow_down_rates
        )

    def _get_proc_time(self, job_size, server):
        return self.slow_down_rates[server] * job_size

    def step(self, job_size, server, ia_time):
        proc_time = self._get_proc_time(job_size, server)
        self._sim(ia_time)
        return proc_time


class Policy(object):
    def __init__(self, seed, act_len):
        self.n = act_len
        self.rng = np.random.RandomState(seed)

    def register(self, **kwargs):
        self.__dict__.update(kwargs)

    def act(self, obs_np):
        return 0

    def submit(self, server, proc_time):
        pass


class RandomPolicy(Policy):
    def __init__(self, seed, act_len):
        super(RandomPolicy, self).__init__(seed, act_len)

    def act(self, obs_np):
        return self.rng.choice(self.n)


class ShortestQueuePolicy(Policy):
    def __init__(self, seed, act_len):
        super(ShortestQueuePolicy, self).__init__(seed, act_len)

    def act(self, obs_np):
        return np.argmin(obs_np)


class PowerofKPolicy(Policy):
    def __init__(self, seed, act_len, k):
        super(PowerofKPolicy, self).__init__(seed, act_len)
        self.k = k

    def act(self, obs_np):
        invalid_mask = np.ones(self.n, dtype=bool)
        invalid_mask[self.rng.choice(self.n, self.k, replace=True)] = False
        return np.ma.array(obs_np, mask=invalid_mask).argmin()


class OptimalPolicy(Policy):
    def __init__(self, seed, act_len):
        super(OptimalPolicy, self).__init__(seed, act_len)

    def act(self, obs_np):
        return np.argmin((obs_np + 1) * self.pt_manager.slow_down_rates)


class TrackerPolicy(Policy):
    def __init__(self, seed, act_len, alpha):
        super(TrackerPolicy, self).__init__(seed, act_len)
        self.alpha = alpha
        self.q_means = np.zeros(self.n, dtype=float)

    def act(self, obs_np):
        return np.argmin((obs_np + 1) * self.q_means)

    def submit(self, server, proc_time):
        self.q_means[server] = (
            self.alpha * self.q_means[server] + (1 - self.alpha) * proc_time
        )


def collect_traces_real(job_sizes, inter_arrs, actions_rnd, p_change=0):
    ns = 8
    seed = 43
    load_target = 0.6
    # p_change should be 0 if '_0' and 0.5 if '_50'
    assert job_sizes.ndim == 2
    assert job_sizes.shape == inter_arrs.shape
    no_trajs, length = job_sizes.shape
    # Load info arrays
    time_jobs = np.cumsum(inter_arrs, axis=-1)
    actions = np.empty((16, no_trajs, length), dtype=int)
    proc_times = np.empty((16, no_trajs, length), dtype=float)
    latencies = np.empty((16, no_trajs, length), dtype=float)
    env = JobScheduler(ns)
    pt_mgr = ProcessTimeManager(ns, seed, p_change)
    # job_sizes /= (job_sizes.mean()/inter_arrs.mean()*pt_mgr.slow_down_rates.mean()/ns/load_target)
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
    ] + list(range(8))
    for i_pol, policy in enumerate(pols):

        # Reset environment

        # Load rate manager
        pt_mgr = ProcessTimeManager(ns, seed, p_change)
        # Register rate manager for optimal policy
        if i_pol < 8:
            policy.register(pt_manager=pt_mgr)
        for index in tqdm(range(no_trajs)):
            obs = env.reset()
            for i in range(length):
                # Choose server
                if i_pol >= 8:
                    act = actions_rnd[i_pol, index, i]
                else:
                    act = policy.act(obs)
                # Calculate processing time
                ptime = pt_mgr.slow_down_rates[act] * job_sizes[index, i]

                ########### Use the counterfactual model to predict ptime
                # Submit processing time for tracker policy
                if i_pol < 8:
                    policy.submit(act, ptime)
                # Receive latency and queue sizes
                latency, obs = env.step(ptime, act, inter_arrs[index, i])
                # Save info
                proc_times[i_pol, index, i] = ptime
                actions[i_pol, index, i] = act
                latencies[i_pol, index, i] = latency

    return actions, proc_times, latencies


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

                ########### Use the counterfactual model to predict ptime
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
    pols = pols[test_pol_idx : test_pol_idx + 1]
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
                input_tensor = torch.as_tensor(
                    [input_numpy], dtype=torch.float32, device=torch.device("cpu")
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
