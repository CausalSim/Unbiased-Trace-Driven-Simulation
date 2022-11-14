import heapq
import os
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Auto experiment launch')
parser.add_argument('--dir', type=str, required=True, help='Load Balance dataset directory')

args = parser.parse_args()


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
        self.slow_down_rates = change_arr * np.power(5, self.rng.random(self.ns) * 2 - 1) + (
                    1 - change_arr) * self.slow_down_rates

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
        self.q_means[server] = self.alpha * self.q_means[server] + (1 - self.alpha) * proc_time


def collect_traces(job_sizes, inter_arrs, load_target, seed, p_change=0.05, ns=8, include_obs=False):
    assert job_sizes.ndim == 1
    assert job_sizes.shape == inter_arrs.shape
    length = job_sizes.shape[0]

    # Load info arrays
    time_jobs = np.cumsum(inter_arrs, axis=-1)
    actions = np.empty((8, length), dtype=int)
    proc_times = np.empty((8, length), dtype=float)
    latencies = np.empty((8, length), dtype=float)
    obs_s = None
    if include_obs:
        obs_s = np.empty((8, length + 1, ns), dtype=float)
    env = JobScheduler(ns)
    pt_mgr = ProcessTimeManager(ns, seed, p_change)
    job_sizes /= (job_sizes.mean() / inter_arrs.mean() * pt_mgr.slow_down_rates.mean() / ns / load_target)
    # Load policies
    pols = [RandomPolicy(seed, ns), ShortestQueuePolicy(seed, ns), PowerofKPolicy(seed, ns, 2),
            PowerofKPolicy(seed, ns, 3), PowerofKPolicy(seed, ns, 4), PowerofKPolicy(seed, ns, 5),
            OptimalPolicy(seed, ns), TrackerPolicy(seed, ns, 0.995)]
    for i_pol, policy in enumerate(pols):
        # Reset environment
        obs = env.reset()
        if include_obs:
            obs_s[i_pol, 0] = obs
        # Load rate manager
        pt_mgr = ProcessTimeManager(ns, seed, p_change)
        # Register rate manager for optimal policy
        policy.register(pt_manager=pt_mgr)
        for i in range(length):
            # Choose server
            act = policy.act(obs)
            # Calculate processing time
            ptime = pt_mgr.step(job_sizes[i], act, inter_arrs[i])
            # Submit processing time for tracker policy
            policy.submit(act, ptime)
            # Receive latency and queue sizes
            latency, obs = env.step(ptime, act, inter_arrs[i])
            # Save info
            proc_times[i_pol, i] = ptime
            actions[i_pol, i] = act
            latencies[i_pol, i] = latency
            if include_obs:
                obs_s[i_pol, i + 1] = obs

    if include_obs:
        return job_sizes, inter_arrs, time_jobs, actions, proc_times, latencies, obs_s
    else:
        return job_sizes, inter_arrs, time_jobs, actions, proc_times, latencies


def non_iid_workload(seed, n_jobs, num_states, p_change):
    rng = np.random.RandomState(seed)
    size_states = np.logspace(1, 2.5, num_states)
    std_ratio_states = np.linspace(0, 0.5, num_states)
    job_sizes = np.empty(n_jobs, dtype=float)
    mean_state = rng.choice(size_states)
    std_state = rng.choice(std_ratio_states) * mean_state
    for i in range(n_jobs):
        job_sizes[i] = rng.normal(mean_state, std_state)
        if rng.random() < p_change:
            mean_state = rng.choice(size_states)
            std_state = rng.choice(std_ratio_states) * mean_state
    job_sizes = job_sizes.clip(3, 1000)
    ia_times = rng.exponential(scale=50, size=n_jobs)
    return job_sizes, ia_times


def main():
    os.makedirs(f"{args.dir}/", exist_ok=True)
    job_size_arr, ia_time_arr = non_iid_workload(seed=42, n_jobs=10000000, num_states=50, p_change=1 / 12000)
    job_size_arr, ia_time_arr, time_job_arr, act_arr, ptime_arr, lat_arr = collect_traces(job_sizes=job_size_arr,
                                                                                          inter_arrs=ia_time_arr,
                                                                                          load_target=0.6, seed=43,
                                                                                          p_change=0, ns=8)
    dict_exp = {
        'job_size': job_size_arr,
        'ia_time': ia_time_arr,
        'time_jobs': time_job_arr,
        'actions': act_arr,
        'proc_times': ptime_arr,
        'latencies': lat_arr
    }
    with open(f"{args.dir}/non_iid_0_big.pkl", 'wb') as fandle:
        pickle.dump(dict_exp, fandle)


if __name__ == '__main__':
    main()
