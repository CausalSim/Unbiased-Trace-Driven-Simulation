from argparse import Namespace
import numpy as np
from abc import ABC, abstractmethod
from env.abr import ABRSimEnv
from typing import Tuple, List
from cpolicies.mpc import take_action_py


class Agent(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def take_action(self, obs_np: np.ndarray) -> int:
        pass


def get_all_policies(config: Namespace) -> Tuple[List[Agent], List[str], List[str]]:
    policies = [
        BBAAgent(act_len=ABRSimEnv.CHOICES, buf_pos=2 * config.mpc_lookback, bba_reservoir=config.bba_reservoir,
                 bba_cushion=config.bba_cushion),
        BBAAgentMIX(act_len=ABRSimEnv.CHOICES, buf_pos=2 * config.mpc_lookback,
                    bba_reservoir=config.bba_reservoir, bba_cushion=config.bba_cushion,
                    mult=1, ratio=0.5, seed=config.seed + 1),
        BBAAgentMIX(act_len=ABRSimEnv.CHOICES, buf_pos=2 * config.mpc_lookback,
                    bba_reservoir=config.bba_reservoir, bba_cushion=config.bba_cushion,
                    mult=2, ratio=0.5, seed=config.seed + 1),
        CMPCAgent(act_len=ABRSimEnv.CHOICES, mpc_lookahead=config.mpc_lookahead,
                  bitrate_map=ABRSimEnv.BITRATE_MAP, mpc_lookback=config.mpc_lookback, eps=config.eps,
                  rebuf_penalty=ABRSimEnv.REBUF_PENALTY),
        RNDAgent(act_len=ABRSimEnv.CHOICES, seed=config.seed + 1),
        BolaAgent(act_len=ABRSimEnv.CHOICES, buf_pos=2 * config.mpc_lookback,
                  bitrate_map=ABRSimEnv.BITRATE_MAP, chunk_length=ABRSimEnv.CHUNK_LENGTH,
                  max_buf_s=ABRSimEnv.MAX_BUFF_S),
        RateAgent(act_len=ABRSimEnv.CHOICES, eps=config.eps, mpc_lookback=config.mpc_lookback),
        OptimisticRateAgent(act_len=ABRSimEnv.CHOICES, eps=config.eps, mpc_lookback=config.mpc_lookback),
        PessimisticRateAgent(act_len=ABRSimEnv.CHOICES, eps=config.eps, mpc_lookback=config.mpc_lookback),
    ]
    policy_names = [
        'BBA',
        'BBAMIX-x1-50%',
        'BBAMIX-x2-50%',
        'MPC',
        'Random',
        'BOLA',
        'Rate Based',
        'Optimistic Rate Based',
        'Pessimistic Rate Based',
    ]
    policy_paths = [
        'bba_traj.npy',
        'bbamix_X1.0_RND50%_traj.npy',
        'bbamix_X2.0_RND50%_traj.npy',
        'mpc_traj.npy',
        'rnd_traj_0.npy',
        'bola_traj.npy',
        'rate_traj.npy',
        'opt_rate_traj.npy',
        'pess_rate_traj.npy',
    ]
    return policies, policy_names, policy_paths


class BBAAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()
        self.act_n = kwargs['act_len']
        self.upper = kwargs['bba_reservoir'] + kwargs['bba_cushion']
        self.lower = kwargs['bba_reservoir']
        self.buf_pos = kwargs['buf_pos']

    def take_action(self, obs_np: np.ndarray) -> int:
        buffer_size = obs_np[self.buf_pos]
        if buffer_size < self.lower:
            act = 0
        elif buffer_size >= self.upper:
            act = self.act_n - 1
        else:
            ratio = (buffer_size - self.lower) / float(self.upper - self.lower)
            min_chunk = np.min(obs_np[self.buf_pos+3:self.buf_pos+3+self.act_n])
            max_chunk = np.max(obs_np[self.buf_pos+3:self.buf_pos+3+self.act_n])
            bitrate = ratio * (max_chunk - min_chunk) + min_chunk
            act = max([i for i in range(self.act_n) if bitrate >= obs_np[self.buf_pos+3+i]])
        return act


class BBAAgentMIX(BBAAgent):
    def __init__(self, **kwargs):
        super(BBAAgentMIX, self).__init__(**kwargs)
        self.upper *= kwargs['mult']
        self.lower *= kwargs['mult']
        self.ratio_rnd = kwargs['ratio']
        assert 0 <= self.ratio_rnd < 1
        self.rng = np.random.RandomState(kwargs['seed'])

    def take_action(self, obs_np: np.ndarray) -> int:
        if self.rng.random() < self.ratio_rnd:
            return self.rng.choice(self.act_n)
        else:
            return BBAAgent.take_action(self, obs_np)


class RNDAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()
        self.act_n = kwargs['act_len']
        # random generator, to make runs deterministic, use different seed than env
        self.rng = np.random.RandomState(kwargs['seed'])

    def take_action(self, obs_np: np.ndarray) -> int:
        return self.rng.choice(self.act_n)


class CMPCAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()
        self.act_len = kwargs['act_len']
        self.mpc_lookahead = kwargs['mpc_lookahead']
        self.mpc_lookback = kwargs['mpc_lookback']
        self.eps = kwargs['eps']
        self.rebuf_penalty = kwargs['rebuf_penalty']
        self.vid_bit_rate = np.array(kwargs['bitrate_map'])

    def take_action(self, obs_np: np.ndarray) -> int:
        return take_action_py(obs_np, self.act_len, self.vid_bit_rate, self.rebuf_penalty, self.mpc_lookback,
                              self.mpc_lookahead, self.eps)


class RateAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()
        self.act_n = kwargs['act_len']
        self.mpc_lookback = kwargs['mpc_lookback']
        self.eps = kwargs['eps']

    def take_action(self, obs_np: np.ndarray) -> int:
        past_bandwidths = np.trim_zeros(obs_np[:self.mpc_lookback], 'f')
        if len(past_bandwidths) > 0:
            bandwidth = self.estimate_bandwidth(past_bandwidths)
        else:
            bandwidth = self.eps
        bit_rates = obs_np[2 * self.mpc_lookback+3:2 * self.mpc_lookback+3+self.act_n] / 4
        act = max([i for i in range(self.act_n) if bandwidth >= bit_rates[i]] + [0])
        return act

    @staticmethod
    def estimate_bandwidth(past_bws: np.ndarray) -> float:
        return 1 / (1 / np.array(past_bws)).mean()


class OptimisticRateAgent(RateAgent):
    def __init__(self, **kwargs):
        super(OptimisticRateAgent, self).__init__(**kwargs)

    @staticmethod
    def estimate_bandwidth(past_bws: np.ndarray) -> float:
        return np.max(past_bws)


class PessimisticRateAgent(RateAgent):
    def __init__(self, **kwargs):
        super(PessimisticRateAgent, self).__init__(**kwargs)

    @staticmethod
    def estimate_bandwidth(past_bws: np.ndarray) -> float:
        return np.min(past_bws)


class BolaAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()
        self.min_buf_s = 3
        self.max_buf_s = kwargs['max_buf_s']
        self.chunk_length = kwargs['chunk_length']
        self.buf_pos = kwargs['buf_pos']

        self.act_len = kwargs['act_len']
        self.size_ladder_bytes = np.array(kwargs['bitrate_map'])
        self.utility_ladder = self.utility(self.size_ladder_bytes)

        assert self.size_ladder_bytes[0] < self.size_ladder_bytes[1]
        assert self.utility_ladder[0] < self.utility_ladder[1]
        assert self.min_buf_s < self.max_buf_s

        smallest = {'size': self.size_ladder_bytes[0],
                    'utility': self.utility_ladder[0]}
        second_smallest = {'size': self.size_ladder_bytes[1],
                           'utility': self.utility_ladder[1]}
        largest = {'size': self.size_ladder_bytes[-1],
                   'utility': self.utility_ladder[-1]}

        size_delta = self.size_ladder_bytes[1] - self.size_ladder_bytes[0]
        utility_high = largest['utility']

        size_utility_term = second_smallest['size'] * smallest['utility'] - smallest['size'] * \
            second_smallest['utility']
        gp_nominator = self.max_buf_s * size_utility_term - utility_high * self.min_buf_s * size_delta
        gp_denominator = ((self.min_buf_s - self.max_buf_s) * size_delta)
        self.gp = gp_nominator / gp_denominator
        self.Vp = self.max_buf_s / self.chunk_length / (utility_high + self.gp)

    @staticmethod
    def utility(sizes: np.ndarray) -> np.ndarray:
        return np.log(sizes/sizes[0])

    def take_action(self, obs_np: np.ndarray) -> int:
        buffer_size = obs_np[self.buf_pos]
        buffer_in_chunks = buffer_size / self.chunk_length
        size_arr = obs_np[self.buf_pos + 3:self.buf_pos + 3 + self.act_len]
        objs = (self.Vp * (self.utility(size_arr) + self.gp) - buffer_in_chunks) / size_arr
        return np.argmax(objs).item()
