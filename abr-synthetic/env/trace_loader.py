from typing import Tuple, List
import numpy as np
from tqdm import trange
from scipy.optimize import fsolve


def load_traces(trace_folder: str, seed: int, length_trace: int, num_traces: int) -> Tuple[List[np.ndarray],
                                                                                           np.ndarray]:
    all_traces, all_rtts = load_sim_traces_process(length=length_trace, seed=seed, num_traces=num_traces)
    np.save(trace_folder + '/traces.npy', all_traces)
    np.save(trace_folder + '/rtts.npy', all_rtts)

    return all_traces, all_rtts


def load_sim_traces_process(length: int, num_traces: int, seed: int) -> Tuple[List[np.ndarray], np.ndarray]:
    rng = np.random.RandomState(seed)
    all_traces = []
    print('Creating traces')
    for i in trange(num_traces):
        p_transition = 1 - 1 / rng.randint(30, 100)
        var_coeff = rng.random() * 0.25 + 0.05
        low_thresh, high_thresh = uniform_thresh(4.5, 0.5, rng)
        all_bandwidth = np.empty(length)
        state = rng.random() * (high_thresh-low_thresh) + low_thresh
        for j in range(length):
            all_bandwidth[j] = np.clip(rng.normal(state, state * var_coeff), low_thresh, high_thresh)
            if rng.random() > p_transition:
                state = doubly_exponential(state, high_thresh, low_thresh, rng)
        all_traces.append(all_bandwidth)

    all_rtts = np.random.RandomState(seed).random(size=len(all_traces)) * 490 + 10
    print('Created!!!')

    return all_traces, all_rtts


def doubly_exponential(position: float, high: float, low: float, rng: np.random.RandomState) -> float:
    lamb = fsolve(lambda la: 1 - np.exp(-la * (high-position)) - np.exp(-la * (position-low)), np.array([0.5]))[0]
    rnd = rng.random()
    if rnd < 1 - np.exp(-lamb * (high-position)):
        return position - np.log(1-rnd)/lamb
    else:
        return position + np.log(rnd) / lamb


def uniform_thresh(high: float, low: float, rng: np.random.RandomState) -> Tuple[float, float]:
    low_thresh, high_thresh = 1, 1
    while (high_thresh-low_thresh) / (high_thresh+low_thresh) < 0.3:
        threshes = rng.random(size=2) * (high - low) + low
        low_thresh, high_thresh = np.min(threshes), np.max(threshes)
    return low_thresh, high_thresh
