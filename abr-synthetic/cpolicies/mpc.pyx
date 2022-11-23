# distutils: language = c++
import cython
from numpy.math cimport INFINITY
import numpy as np
cimport numpy as np
np.import_array()


def take_action_py(obs_np, act_n, vid_bit_rate, rebuf_penalty, mpc_lookback, mpc_lookahead, eps):
    next_chunks_len = min(mpc_lookahead-1, int(obs_np[2 * mpc_lookback + 1]))
    next_chunk_sizes = obs_np[3 + 2 * mpc_lookback:3 + 2 * mpc_lookback + act_n * next_chunks_len]
    past_bandwidths = np.trim_zeros(obs_np[:mpc_lookback], 'f')
    if len(past_bandwidths) > 0:
        harmonic_bandwidth = 1 / (1/past_bandwidths).mean()
    else:
        harmonic_bandwidth = eps
    future_bandwidth = harmonic_bandwidth
    return recursive_best_mpc_c(obs_np[2 * mpc_lookback], 0, next_chunks_len, int(obs_np[2 * mpc_lookback + 2]),
                                next_chunk_sizes / future_bandwidth, act_n, vid_bit_rate, rebuf_penalty)[1]


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef recursive_best_mpc_c(double curr_buffer, int position, int recursions_left, int last_quality,
                          double[:] download_times, int act_n, double[:] vid_bit_rate,
                          double rebuf_penalty):

    if recursions_left == 0:
        assert position * act_n == len(download_times)
        return 0, 0

    cdef double best_reward = -INFINITY
    cdef int best_act = -1
    cdef int chunk_quality = 0
    cdef double reward_act
    cdef double buffer_act
    cdef double download_time

    for chunk_quality in range(act_n):
        reward_act = 0
        buffer_act = curr_buffer
        # this is MB/MB/s --> seconds
        download_time = download_times[position * act_n + chunk_quality]
        if buffer_act < download_time:
            reward_act -= rebuf_penalty * (download_time - buffer_act)
            buffer_act = 0
        else:
            buffer_act -= download_time
        buffer_act += 4
        reward_act += vid_bit_rate[chunk_quality]
        reward_act -= abs(vid_bit_rate[chunk_quality] - vid_bit_rate[last_quality])
        reward_act += recursive_best_mpc_c(buffer_act, position+1, recursions_left-1, chunk_quality,
                                           download_times, act_n, vid_bit_rate, rebuf_penalty)[0]

        if best_reward < reward_act:
            best_reward = reward_act
            best_act = chunk_quality

    return best_reward, best_act
