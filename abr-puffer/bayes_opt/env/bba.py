import numpy as np

from .abr import ABRSimEnv


class BBA(object):
    MAX_BUF_S = ABRSimEnv.MAX_BUFFER_S

    def __init__(self, act_len: int, reservoir: float, cushion: float):
        self.act_n = act_len
        self.upper = reservoir + cushion
        self.lower = reservoir

    def sample_action(self, obs: np.ndarray) -> int:
        invalid_mask = np.logical_or(np.isnan(obs[-2 * self.act_n: -self.act_n]),
                                     np.isnan(obs[-self.act_n:]))
        size_arr_valid = np.ma.array(obs[-2 * self.act_n: -self.act_n], mask=invalid_mask)
        ssim_arr_valid = np.ma.array(obs[-self.act_n:], mask=invalid_mask)
        min_choice = size_arr_valid.argmin()
        max_choice = size_arr_valid.argmax()
        buffer = (obs[0] + 1) / 2 * self.MAX_BUF_S
        if buffer < self.lower:
            act = min_choice
        elif buffer >= self.upper:
            act = max_choice
        else:
            ratio = (buffer - self.lower) / float(self.upper - self.lower)
            min_chunk = size_arr_valid[min_choice]
            max_chunk = size_arr_valid[max_choice]
            bitrate = ratio * (max_chunk - min_chunk) + min_chunk
            mask = np.logical_or(invalid_mask, size_arr_valid > bitrate)
            act = np.ma.array(ssim_arr_valid, mask=mask).argmax()
        return act
