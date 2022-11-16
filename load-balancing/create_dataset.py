import heapq
import os
import pickle
import numpy as np
import argparse
from tqdm import tqdm


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


class RandomKServersPolicy(Policy):
    def __init__(self, seed, act_len, k, servers):
        super(RandomKServersPolicy, self).__init__(seed, act_len)
        self.servers = servers
        self.k = k
        assert len(self.servers) == k

    def act(self, obs_np):
        return self.rng.choice(self.servers)


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


def collect_traces(
    job_sizes, inter_arrs, seed, p_change=0, ns=8, load_target=0.6
):
    assert job_sizes.ndim == 2
    assert job_sizes.shape == inter_arrs.shape

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
    job_sizes /= (
        job_sizes.mean()
        / inter_arrs.mean()
        * pt_mgr.slow_down_rates.mean()
        / ns
        / load_target
    )
    # Load policies
    k = 2
    pols = [
        RandomPolicy(seed, ns),
        ShortestQueuePolicy(seed, ns),
        PowerofKPolicy(seed, ns, 2),
        PowerofKPolicy(seed, ns, 3),
        PowerofKPolicy(seed, ns, 4),
        PowerofKPolicy(seed, ns, 5),
        OptimalPolicy(seed, ns),
        TrackerPolicy(seed, ns, 0.995),
        RandomKServersPolicy(seed, ns, k, servers=[3, 4]),
        RandomKServersPolicy(seed, ns, k, servers=[4, 5]),
        RandomKServersPolicy(seed, ns, k, servers=[1, 2]),
        RandomKServersPolicy(seed, ns, k, servers=[0, 3]),
        RandomKServersPolicy(seed, ns, k, servers=[3, 6]),
        RandomKServersPolicy(seed, ns, k, servers=[0, 1]),
        RandomKServersPolicy(seed, ns, k, servers=[5, 7]),
        RandomKServersPolicy(seed, ns, k, servers=[1, 7]),
    ]
    for i_pol, policy in tqdm(enumerate(pols), total=len(pols)):

        # Reset environment
        obs = env.reset()
        # Load rate manager
        pt_mgr = ProcessTimeManager(ns, seed, p_change)
        # Register rate manager for optimal policy
        policy.register(pt_manager=pt_mgr)
        for index in tqdm(range(no_trajs), leave=False):
            # Reset environment
            obs = env.reset()
            for i in range(length):
                # Choose server
                act = policy.act(obs)
                # Calculate processing time
                ptime = pt_mgr.slow_down_rates[act] * job_sizes[index, i]
                # Submit processing time for tracker policy
                policy.submit(act, ptime)
                # Receive latency and queue sizes
                latency, obs = env.step(ptime, act, inter_arrs[index, i])
                # Save info
                proc_times[i_pol, index, i] = ptime
                actions[i_pol, index, i] = act
                latencies[i_pol, index, i] = latency

    return job_sizes, inter_arrs, time_jobs, actions, proc_times, latencies


def non_iid_workload(seed, traj_length, no_traj, num_states, p_change):
    n_jobs = no_traj * traj_length
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
    ia_times = rng.exponential(scale=50, size=(no_traj, traj_length))
    ia_times[:, 0] = 0
    job_sizes_traj = job_sizes.reshape([no_traj, traj_length])
    return job_sizes_traj, ia_times


def main():
    parser = argparse.ArgumentParser(description="Auto experiment launch")
    parser.add_argument(
        "--dir", type=str, required=True, help="Load Balance dataset directory"
    )

    args = parser.parse_args()

    os.makedirs(f"{args.dir}/", exist_ok=True)
    job_size_arr, ia_time_arr = non_iid_workload(
        seed=42, no_traj=5000, traj_length=1000, num_states=50, p_change=1 / 12000
    )
    (
        job_size_arr,
        ia_time_arr,
        time_job_arr,
        act_arr,
        ptime_arr,
        lat_arr,
    ) = collect_traces(
        job_sizes=job_size_arr,
        inter_arrs=ia_time_arr,
        load_target=0.6,
        seed=43,
        p_change=0,
        ns=8,
    )
    dict_exp = {
        "job_size": job_size_arr,
        "ia_time": ia_time_arr,
        "time_jobs": time_job_arr,
        "actions": act_arr,
        "proc_times": ptime_arr,
        "latencies": lat_arr,
    }
    with open(f"{args.dir}/non_iid_0_big.pkl", "wb") as fandle:
        pickle.dump(dict_exp, fandle)


if __name__ == "__main__":

    main()
