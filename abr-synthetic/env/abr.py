import os
import numpy as np
from collections import deque
from .trace_loader import load_traces
from tqdm import tqdm


class ABRSimEnv(object):
    MAX_BUFF_S = 10.0
    CHUNK_LENGTH = 4.0
    CHOICES = 6
    # mapping between action and bitrate level
    BITRATE_MAP = [0.3, 0.75, 1.2, 1.85, 2.85, 4.3]
    REBUF_PENALTY = 4.3

    def __init__(self, mpc_lookahead: int, mpc_lookback: int, seed: int, trace_folder: str, num_traces: int):
        # observation and action space
        self.mpc_lookahead = mpc_lookahead
        self.mpc_lookback = mpc_lookback
        self.delay_list = None
        self.chunk_idx = None
        self.buffer_size = None
        self.past_action = None
        self.past_chunk_throughputs = None
        self.past_chunk_download_times = None
        self.np_random = None
        self.obs_high = None
        self.obs_low = None
        self.observation_space = None
        self.action_space = None
        self.setup_space()
        # set up seed
        self.seed(seed)
        # load all video chunk sizes
        self.chunk_sizes = np.load(os.path.dirname(__file__) + '/video_sizes.npy').T
        # assert number of chunks for different bitrates are all the same
        self.total_num_chunks = len(self.chunk_sizes)
        # load all trace files
        self.all_traces, self.all_rtts = load_traces(trace_folder=trace_folder, seed=seed,
                                                     length_trace=self.total_num_chunks,
                                                     num_traces=num_traces)
        # how many past throughput to report
        self.past_chunk_len = max(mpc_lookahead, mpc_lookback)

        print('Precomputing all download times...')
        self.all_delays = np.array([thr_slow_start(trace, self.chunk_sizes, rtt)
                                    for trace, rtt in tqdm(zip(self.all_traces, self.all_rtts), total=num_traces)])
        print('Finished')

    def observe(self):
        if self.chunk_idx < self.total_num_chunks:
            valid_chunk_idx = self.chunk_idx
        else:
            valid_chunk_idx = 0

        if self.past_action is not None:
            valid_past_action = self.past_action
        else:
            valid_past_action = 0

        # network throughput of past chunk, past chunk download time,
        # current buffer, number of chunks left and the last bitrate choice
        obs_arr = [self.past_chunk_throughputs[-i] for i in range(self.mpc_lookback, 0, -1)]
        obs_arr.extend([self.past_chunk_download_times[-i] for i in range(self.mpc_lookback, 0, -1)])
        obs_arr.extend([self.buffer_size, self.total_num_chunks - self.chunk_idx, valid_past_action])

        # current chunk size of different bitrates
        for chunk_idx_add in range(valid_chunk_idx, self.mpc_lookahead+valid_chunk_idx):
            obs_arr.extend(self.chunk_sizes[chunk_idx_add % self.total_num_chunks, i] for i in range(6))

        for i in range(6):
            obs_arr.append(self.chunk_sizes[valid_chunk_idx, i] / self.delay_list[valid_chunk_idx, i])
        for i in range(6):
            obs_arr.append(self.delay_list[valid_chunk_idx, i])

        obs_arr = np.array(obs_arr)
        assert np.all(obs_arr >= self.obs_low), obs_arr
        assert np.all(obs_arr <= self.obs_high), obs_arr

        return obs_arr

    def reset(self, trace_choice=None):
        assert trace_choice < len(self.all_traces)
        self.delay_list = self.all_delays[trace_choice]
        self.chunk_idx = 0
        self.buffer_size = 0.0  # initial download time not counted
        self.past_action = None
        self.past_chunk_throughputs = deque(maxlen=self.past_chunk_len)
        self.past_chunk_download_times = deque(maxlen=self.past_chunk_len)
        for _ in range(self.past_chunk_len):
            self.past_chunk_throughputs.append(0)
            self.past_chunk_download_times.append(0)

        return self.observe()

    def seed(self, seed):
        self.np_random = np.random.RandomState(seed)

    def setup_space(self):
        # Set up the observation and action space
        self.obs_low = np.array([0] * (3 + 2 * self.mpc_lookback + 6 * self.mpc_lookahead + 12))
        self.obs_high = np.array([100e6] * self.mpc_lookback + [5000] * self.mpc_lookback + [100, 500, 5] +
                                 [10e6] * (6*self.mpc_lookahead) + [100e6] * 6 + [5000] * 6)

    def step(self, action):
        # 0 <= action < num_bitrates
        assert 0 <= action < self.CHOICES

        # Note: sizes are in bytes, times are in seconds
        chunk_size = self.chunk_sizes[self.chunk_idx, action]

        # compute chunk download time based on trace
        delay = self.delay_list[self.chunk_idx, action]

        # compute buffer size
        rebuffer_time = max(delay - self.buffer_size, 0)

        # update video buffer
        self.buffer_size = max(self.buffer_size - delay, 0)
        self.buffer_size += self.CHUNK_LENGTH  # each chunk is 4 seconds of video

        # cap the buffer size
        self.buffer_size = min(self.buffer_size, self.MAX_BUFF_S)

        # bitrate change penalty
        if self.past_action is None:
            bitrate_change = 0
        else:
            bitrate_change = np.abs(self.BITRATE_MAP[action] - self.BITRATE_MAP[self.past_action])

        # linear reward
        # (https://dl.acm.org/citation.cfm?id=3098843 section 5.1, QoE metrics (1))
        reward = self.BITRATE_MAP[action] - self.REBUF_PENALTY * rebuffer_time - bitrate_change

        # store action for future bitrate change penalty
        self.past_action = action

        # update observed network bandwidth and duration
        self.past_chunk_throughputs.append(chunk_size / float(delay))
        self.past_chunk_download_times.append(delay)

        # advance video
        self.chunk_idx += 1
        done = (self.chunk_idx == self.total_num_chunks)

        return self.observe(), reward, done, \
            {'bitrate': self.BITRATE_MAP[action],
             'stall_time': rebuffer_time,
             'bitrate_change': bitrate_change}


def thr_slow_start(trace: np.ndarray, chunk_sizes: np.ndarray, rtt: float, thr_start: float = 2*1500) -> np.ndarray:
    delays = np.empty_like(chunk_sizes, dtype=float)
    # thr_start: bytes/second, Two packets, MTU = 1500 bytes
    thr_end = trace / 8.0 * 1e6  # bytes/second
    len_thr_exp_arr = np.ceil(np.log2(thr_end / thr_start)).astype(int)
    assert np.all(len_thr_exp_arr > 0)
    for i in range(delays.shape[0]):
        thr_arr = np.exp2(np.arange(len_thr_exp_arr[i]+1)) * thr_start
        thr_arr[-1] = thr_end[i]
        time_arr = np.ones(len_thr_exp_arr[i]) * rtt / 1000
        cumul_sum_thr = np.cumsum(thr_arr[:-1] * time_arr)
        for j, chunk in enumerate(chunk_sizes[i]):
            index_start = np.where(cumul_sum_thr > chunk)[0]
            index_start = len(thr_arr) - 1 if len(index_start) == 0 else index_start[0]
            time_first = 0 if index_start == 0 else rtt / 1000 * index_start
            size_first = 0 if index_start == 0 else cumul_sum_thr[index_start - 1]
            delays[i, j] = time_first + (chunk - size_first) / thr_arr[index_start]
    return delays
