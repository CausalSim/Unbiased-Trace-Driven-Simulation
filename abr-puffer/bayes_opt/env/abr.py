from typing import List, Tuple, Dict
import numpy as np
import torch
import datetime


MIN_SSIM = 0
MAX_SSIM = 60


def ssim_db(ssim: np.ndarray) -> np.ndarray:
    return np.where(ssim == 1, MAX_SSIM, np.clip(-10 * np.log10(1 - ssim), a_min=MIN_SSIM, a_max=MAX_SSIM))


def db_ssim(db: np.ndarray) -> np.ndarray:
    return 1 - np.power(10, -db / 10)


def load_traces(trace_path: str, wid: int, wnum: int) -> Tuple[List[np.ndarray], np.ndarray]:
    all_traces = []
    all_p_traces = []

    all_so_far = 0
    start_date = datetime.date(2020, 7, 27)
    end_date = datetime.date(2021, 6, 1)
    all_days = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]

    for today in all_days:
        date_string = "%d-%02d-%02d" % (today.year, today.month, today.day)
        trajs = np.load(f"{trace_path}/{date_string}_trc.npy", allow_pickle=True)
        len_all_traj = len(trajs)
        trajs = [traj for i, traj in enumerate(trajs) if (i+all_so_far-wid) % wnum == 0]
        all_traces.extend(trajs)
        all_p_traces.extend(len(traj) for traj in trajs)

        all_so_far += len_all_traj

    assert all_so_far == 235298, all_so_far
    assert len(all_traces) - 235298 // wnum in [0, 1]
    assert len(all_p_traces) == len(all_traces)

    assert all(n > 0 for n in all_p_traces)
    assert all((arr[:, 2] > 0).all() for arr in all_traces)
    assert all(np.all((arr[1:, 3] > arr[:-1, 3])) for arr in all_traces)
    assert all((arr[:, 4] > 0).all() for arr in all_traces)

    all_p_traces = np.ones_like(all_p_traces)
    all_p_traces = all_p_traces / all_p_traces.sum()

    return all_traces, np.array(all_p_traces)


class ABRSimEnv:
    MAX_BUFFER_S = 15
    CHUNK_LENGTH = 2.002
    REBUF_PENALTY = 100
    SSIM_PENALTY = 1
    CHOICES = 12

    def __init__(self, env_config: Dict):
        if 'id' not in env_config:
            env_config['id'] = 0
        if 'num_w' not in env_config:
            env_config['num_w'] = 1
        if 'rebuf_p' in env_config:
            self.REBUF_PENALTY = env_config['rebuf_p']
        self.mode = env_config['sim_mode']

        self.rebuf_dynamic = self.REBUF_PENALTY * np.log(10) / 10 / self.CHUNK_LENGTH * (1 - 0.9)

        self.trace = None
        # Currently, chunk_idx and curr_t_idx are equal, but they don't have to be
        self.curr_t_idx = None
        self.chunk_idx = None
        self.curr_trace_chunks = None
        self.chunk_time_left = None
        self.tot_ssim = None
        self.count_ssim = None

        self.buffer_size = None
        self.past_action = None
        self.past_ssim_db = None

        self.np_random = None
        self.obs_high = None
        self.obs_low = None
        self.observation_space = None
        self.action_space = None
        self.setup_space()
        self.last_seed = env_config['seed']
        self.seed(env_config['seed'])

        if self.mode == 'causalsim':
            self.buf_csim = torch.load(env_config['csim_buf_model'], map_location=torch.device('cpu')).cpu()
            self.buf_csim = torch.jit.script(self.buf_csim)
            self.dt_csim = torch.load(env_config['csim_dt_model'], map_location=torch.device('cpu')).cpu()
            self.dt_csim = torch.jit.script(self.dt_csim)
            self.cs_prm = {
                'in_mu': torch.as_tensor([
                    np.load(env_config['trace_path'] + '/buffer_mean.npy').item(),
                    np.load(env_config['trace_path'] + '/chosen_chunk_size_mean.npy').item(),
                    0,
                ], dtype=torch.float32),
                'in_std': torch.as_tensor([
                    np.load(env_config['trace_path'] + '/buffer_std.npy').item(),
                    np.load(env_config['trace_path'] + '/chosen_chunk_size_std.npy').item(),
                    1,
                ], dtype=torch.float32),
                'out_mu': torch.as_tensor([
                    np.load(env_config['trace_path'] + '/next_buffer_mean.npy').item(),
                    np.load(env_config['trace_path'] + '/download_time_mean.npy').item(),
                ], dtype=torch.float32),
                'out_std': torch.as_tensor([
                    np.load(env_config['trace_path'] + '/next_buffer_std.npy').item(),
                    np.load(env_config['trace_path'] + '/download_time_std.npy').item(),
                ], dtype=torch.float32),
            }

        self.all_traces, self.all_p_traces = load_traces(env_config['trace_path'], env_config['id'],
                                                         env_config['num_w'])
        # all_traces is a list of traces, not equal in length
        # each traces is a numpy array with the following:
        # BW hidden feature, DT hidden feature, factual throughput, time between actions, min rtt, bitrate choices,
        # ssim of choices in db
        assert all([trace.shape[1] == 5+2*self.CHOICES for trace in self.all_traces])

    def seed(self, seed: int = None):
        if seed:
            self.last_seed = seed
        self.np_random = np.random.RandomState(seed=self.last_seed)

    def setup_space(self):
        min_rver = [-1, -1]
        max_rver = [1.2, 1]

        self.obs_low = np.array(min_rver + [0] * (2 * self.CHOICES), dtype=np.float32)
        self.obs_high = np.array(max_rver + [155] * self.CHOICES + [1] * self.CHOICES, dtype=np.float32)

    def observe(self) -> np.ndarray:
        if self.chunk_idx < self.curr_trace_chunks:
            valid_chunk_idx = self.chunk_idx
        else:
            valid_chunk_idx = 0

        if self.past_action is not None:
            valid_past_action = self.past_action
        else:
            valid_past_action = 0

        # 1-4: Aux variables
        #       1) network throughputs of past chunks
        #       2) past chunks' download times
        #       3) past normalized ssims in db
        #       4) current buffer and last bitrate choice
        # 5-6: Set variables
        #       5) bitrates for current chunk
        #       6) NORM SSIMs in db for current chunk

        obs_arr = [
                self.buffer_size / self.MAX_BUFFER_S * 2 - 1,
                valid_past_action / self.CHOICES - 1
            ]

        obs_arr.extend(self.trace[valid_chunk_idx, 5:5+self.CHOICES, ] / 1e6 * 8)
        obs_arr.extend((self.trace[valid_chunk_idx, 5+self.CHOICES:5+2*self.CHOICES] - MIN_SSIM) / MAX_SSIM)

        obs_arr = np.array(obs_arr, dtype=np.float32)
        if np.any(obs_arr > self.obs_high) or np.any(obs_arr < self.obs_low):
            import pdb
            pdb.set_trace()

        obs_check_arr = np.nan_to_num(obs_arr)
        assert np.all(obs_check_arr >= self.obs_low), obs_arr
        assert np.all(obs_check_arr <= self.obs_high), obs_arr

        return obs_arr

    def reset(self, trace_choice: int) -> np.ndarray:
        assert trace_choice < len(self.all_traces), f"{trace_choice}, {len(self.all_traces)}"
        self.trace = self.all_traces[trace_choice]
        self.curr_t_idx = 0

        # For now we start from zero
        assert self.curr_t_idx == 0
        self.chunk_idx = 0
        self.curr_trace_chunks = self.trace.shape[0]
        self.chunk_time_left = self.trace[self.curr_t_idx, 3]

        self.buffer_size = 0.0  # initial download time not counted
        self.past_action = None
        self.past_ssim_db = None
        self.tot_ssim = 0
        self.count_ssim = 0

        return self.observe()

    def causal_sim(self, chunk_size: float) -> Tuple[float, float]:
        input_buf = torch.as_tensor([self.buffer_size, chunk_size, self.trace[self.curr_t_idx, 0]], dtype=torch.float32)
        input_dt = torch.as_tensor([self.buffer_size, chunk_size, self.trace[self.curr_t_idx, 1]], dtype=torch.float32)
        out_buf = self.buf_csim((input_buf - self.cs_prm['in_mu']) / self.cs_prm['in_std'])
        out_dt = self.dt_csim((input_dt - self.cs_prm['in_mu']) / self.cs_prm['in_std'])
        out_buf = out_buf * self.cs_prm['out_std'] + self.cs_prm['out_mu']
        out_dt = out_dt * self.cs_prm['out_std'] + self.cs_prm['out_mu']
        delay = max(out_dt[1].item(), 0.1)
        new_buffer = max(out_buf[0].item(), 0)
        new_buffer = min(new_buffer, self.MAX_BUFFER_S)
        return delay, new_buffer

    def expert_sim(self, chunk_size: float) -> Tuple[float, float, int, bool]:
        # keep experiencing the network trace
        # until the chunk is downloaded
        delay = 0
        over = False
        cti = self.curr_t_idx
        while chunk_size > 1e-8:  # floating number business

            fact_thr = self.trace[cti, 2]  # bytes/second
            chunk_time_used = min(self.chunk_time_left, chunk_size / fact_thr)

            chunk_size -= fact_thr * chunk_time_used
            self.chunk_time_left -= chunk_time_used
            delay += chunk_time_used

            if self.chunk_time_left == 0:
                cti += 1
                if cti == self.curr_trace_chunks:
                    cti = 0
                    over = True
                    self.chunk_time_left = self.trace[cti, 3]
                else:
                    self.chunk_time_left = self.trace[cti, 3]-self.trace[cti-1, 3]
                self.chunk_time_left = max(self.chunk_time_left, 0.1)

        new_buffer = min(max(self.buffer_size - delay, 0) + self.CHUNK_LENGTH, self.MAX_BUFFER_S)
        return delay, new_buffer, cti, over

    def step(self, action: int):
        assert 0 <= action < self.CHOICES

        # Note: sizes are in bytes, times are in seconds
        chunk_size = self.trace[self.chunk_idx, 5+action]
        ssim = self.trace[self.chunk_idx, 5+self.CHOICES+action]
        assert not np.isnan(chunk_size) and not np.isnan(ssim), f'\n{action}\n{self.observe()}'

        # compute chunk download time based on trace
        if self.mode == 'causalsim':
            delay, new_buffer = self.causal_sim(chunk_size=chunk_size)
            self.curr_t_idx += 1
            if self.curr_t_idx == self.curr_trace_chunks:
                self.curr_t_idx = 0
            done = self.chunk_idx == self.curr_trace_chunks - 1
        elif self.mode == 'expertsim':
            delay, new_buffer, self.curr_t_idx, over = self.expert_sim(chunk_size=chunk_size)
            done = self.chunk_idx == self.curr_trace_chunks - 1 or over
        else:
            raise ValueError(f'No simulation mode {self.mode}')

        assert delay >= 0
        assert new_buffer >= 0

        # compute buffer size
        rebuffer_time = max(delay - self.buffer_size, 0)
        self.buffer_size = new_buffer

        # bitrate change penalty
        if self.past_ssim_db is None:
            ssim_change = 0
        else:
            ssim_change = np.abs(ssim - self.past_ssim_db)

        reward = ssim - self.REBUF_PENALTY * rebuffer_time - self.SSIM_PENALTY * ssim_change

        # store action for future bitrate change penalty
        self.past_action = action
        self.past_ssim_db = ssim

        # advance video
        self.chunk_idx += 1
        obs = self.observe()

        return obs, reward, done, {
                   'chunk_size': chunk_size,
                   'ssim': ssim,
                   'stall_time': rebuffer_time,
                   'ssim_change': ssim_change,
                   'download_time': delay,
               }

    def render(self, mode="human"):
        pass
