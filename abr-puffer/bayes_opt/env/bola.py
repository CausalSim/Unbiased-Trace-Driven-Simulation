from typing import Tuple

import numpy as np
from .abr import ABRSimEnv, MIN_SSIM, MAX_SSIM, db_ssim, ssim_db


class BolaAgent(object):
    size_ladder_bytes = [44319, 93355, 115601, 142904, 196884, 263965, 353752, 494902, 632193, 889893]
    ssim_index_ladder = [0.91050748, 0.94062527, 0.94806355, 0.95498943, 0.96214503, 0.96717277, 0.97273958, 0.97689813,
                         0.98004106, 0.98332605]
    MIN_BUF_S = 3
    MAX_BUF_S = ABRSimEnv.MAX_BUFFER_S

    def __init__(self, version: int, act_len: int, reservoir: float = None, cushion: float = None):
        super(BolaAgent, self).__init__()
        assert self.size_ladder_bytes[0] < self.size_ladder_bytes[1]
        assert self.ssim_index_ladder[0] < self.ssim_index_ladder[1]

        self.ssim_index_ladder = (ssim_db(np.array(self.ssim_index_ladder)) - MIN_SSIM) / MAX_SSIM

        assert self.MIN_BUF_S < self.MAX_BUF_S
        assert version in [1, 2]
        self.version = version
        self.act_n = act_len

        smallest = {'size': self.size_ladder_bytes[0],
                    'utility': self.utility(self.ssim_index_ladder[0])}
        second_smallest = {'size': self.size_ladder_bytes[1],
                           'utility': self.utility(self.ssim_index_ladder[1])}
        second_largest = {'size': self.size_ladder_bytes[-2],
                          'utility': self.utility(self.ssim_index_ladder[-2])}
        largest = {'size': self.size_ladder_bytes[-1],
                   'utility': self.utility(self.ssim_index_ladder[-1])}

        size_delta = self.size_ladder_bytes[1] - self.size_ladder_bytes[0]
        if version == 1:
            utility_high = largest['utility']
        else:
            utility_high = self.utility(1)

        size_utility_term = second_smallest['size'] * smallest['utility'] - \
            smallest['size'] * second_smallest['utility']

        gp_nominator = self.MAX_BUF_S * size_utility_term - utility_high * self.MIN_BUF_S * size_delta
        gp_denominator = ((self.MIN_BUF_S - self.MAX_BUF_S) * size_delta)
        if reservoir is not None or cushion is not None:
            assert reservoir is not None and cushion is not None
            int_first_pair = -size_utility_term/size_delta
            size_delta_last = self.size_ladder_bytes[-1] - self.size_ladder_bytes[-2]
            size_utility_term_last = second_largest['size'] * largest['utility'] - \
                largest['size'] * second_largest['utility']
            int_last_pair = size_utility_term_last / size_delta_last
            self.Vp = cushion / (int_first_pair - int_last_pair)
            self.gp = reservoir / self.Vp + int_first_pair
        else:
            self.gp = gp_nominator / gp_denominator
            self.Vp = self.MAX_BUF_S / (utility_high + self.gp)

    def utility(self, ssim_index: float or np.ndarray) -> float or np.ndarray:
        unnorm_db = ssim_index * MAX_SSIM + MIN_SSIM
        if self.version == 1:
            return unnorm_db
        else:
            return db_ssim(unnorm_db)

    def objective(self, utility: float or np.ndarray, size: float or np.ndarray, buffer: float) -> float or np.ndarray:
        return (self.Vp * (utility + self.gp) - buffer) / size

    def choose_max_objective(self, format_sizes: np.ndarray, format_ssims: np.ndarray,
                             buffer: float) -> Tuple[int, float]:
        objs = self.objective(self.utility(format_ssims), format_sizes, buffer)
        chosen_index = np.argmax(objs).item()
        return chosen_index, objs[chosen_index]

    def choose_max_scaled_utility(self, format_ssims: np.ndarray) -> int:
        chosen_index = np.argmax(self.utility(format_ssims)).item()
        return chosen_index

    def sample_action(self, obs: np.ndarray) -> int:
        buffer = (obs[0] + 1) / 2 * self.MAX_BUF_S
        valid_mask = np.logical_not(np.logical_or(np.isnan(obs[-2 * self.act_n: -self.act_n]),
                                                  np.isnan(obs[-self.act_n:])))
        size_arr_valid = obs[-2 * self.act_n: -self.act_n][valid_mask]
        ssim_arr_valid = obs[-self.act_n:][valid_mask]
        index_arr_valid = np.arange(self.act_n)[valid_mask]

        max_obj_index, max_obj = self.choose_max_objective(size_arr_valid, ssim_arr_valid, buffer)

        if self.version == 1 or max_obj >= 0:
            return index_arr_valid[max_obj_index]
        else:
            max_util_index = self.choose_max_scaled_utility(ssim_arr_valid)
            return index_arr_valid[max_util_index]
