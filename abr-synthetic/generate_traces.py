import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
import torch


def collect_traces_causalsim(
    pols,
    c_hat_observed,
    actions_observed,
    chunk_sizes,
    min_rtt,
    feature_extractor,
    buffer_predictor,
    action_factor_net,
    r,
    buf_mean,
    buf_std,
    next_buf_mean,
    next_buf_std,
    chat_mean,
    chat_std,
    size_mean,
    size_std,
    min_rtts_mean,
    min_rtts_std,
    down_time_mean,
    down_time_std,
):

    # Load policies
    length = c_hat_observed.shape[0]
    observation_array = np.zeros(
        [length + 1, 55]
    )  # 0-4 thpt, 5-9 download time, 10 buffer, 12 last action,
    observation_array[:, 13:43] = chunk_sizes
    observation_array[:, 11] = np.arange(1, length + 2)[::-1]
    observation_array[:, 43:] = np.nan
    cf_trajs = np.zeros([len(pols), length + 1, 3])
    feature = np.zeros([length])
    for i_pol, policy in enumerate(pols):
        observation_array[:, :11] = 0
        observation_array[:, 12] = 0
        for i in range(length):
            # Choose action
            act = policy.take_action(observation_array[i, :])

            ########### Use the counterfactual model to predict buffer level and download time
            chat = c_hat_observed[i]
            orig_chosen_size = observation_array[i, 13 + actions_observed[i]]
            chat_white = (chat - chat_mean) / chat_std
            selected_size = observation_array[i, 13 + act]
            selected_size_white = (selected_size - size_mean) / size_std
            orig_chosen_size_white = (orig_chosen_size - size_mean) / size_std
            buffer = observation_array[i, 10]
            buffer_white = (buffer - buf_mean) / buf_std
            min_rtt_white = (min_rtt - min_rtts_mean) / min_rtts_std

            input_numpy = np.array(
                [orig_chosen_size_white, min_rtt_white, chat_white]
            )  # Model only accepts normalized inputs
            input_numpy = np.expand_dims(input_numpy, axis=0)
            input_tensor = torch.as_tensor(
                input_numpy, dtype=torch.float32, device=torch.device("cpu")
            )
            with torch.no_grad():
                feature_tensor = feature_extractor(input_tensor)
            feature[i] = feature_tensor.cpu().numpy()[0][0]
            action_factor = action_factor_net(
                torch.tensor([selected_size_white], dtype=torch.float32)
            )
            predicted_thpt = torch.mul(feature_tensor, action_factor)
            predicted_thpt = torch.matmul(
                predicted_thpt, torch.ones([r, 1], dtype=torch.float32)
            )

            input_numpy = np.array([buffer_white, selected_size_white, min_rtt_white])
            input_numpy = np.expand_dims(input_numpy, axis=0)
            input_tensor = torch.as_tensor(
                input_numpy, dtype=torch.float32, device=torch.device("cpu")
            )
            input_tensor = torch.cat((input_tensor, predicted_thpt), dim=1)
            with torch.no_grad():
                prediction = buffer_predictor(input_tensor)
            next_buffer_white_tensor, down_time_white_tensor = (
                prediction[0, 0],
                prediction[0, 1],
            )
            next_buffer_white = next_buffer_white_tensor.cpu().numpy()
            down_time_white = down_time_white_tensor.cpu().numpy()
            download_time = (down_time_white * down_time_std) + down_time_mean
            next_buffer = (next_buffer_white * next_buf_std) + next_buf_mean

            # update observation_array
            observation_array[i + 1, 10] = next_buffer
            observation_array[i + 1, 12] = act
            observation_array[i + 1, 0:4] = observation_array[i, 1:5]
            observation_array[i + 1, 4] = selected_size / download_time
            observation_array[i + 1, 5:9] = observation_array[i, 6:10]
            observation_array[i + 1, 9] = download_time

            # Save results
            # NOTE: download time and next buffer are stored one index later, as is the case with original data
            cf_trajs[i_pol, i + 1, 0] = next_buffer
            cf_trajs[i_pol, i + 1, 1] = download_time
            cf_trajs[i_pol, i + 1, 2] = act
        act = policy.take_action(observation_array[i + 1, :])

    return (
        cf_trajs,
        feature,
    )


def collect_traces_slsim(
    pols,
    c_hat_observed,
    chunk_sizes,
    min_rtt,
    buffer_predictor,
    buf_mean,
    buf_std,
    next_buf_mean,
    next_buf_std,
    chat_mean,
    chat_std,
    size_mean,
    size_std,
    min_rtts_mean,
    min_rtts_std,
    down_time_mean,
    down_time_std,
):

    # Load policies
    length = c_hat_observed.shape[0]
    observation_array = np.zeros(
        [length + 1, 55]
    )  # 0-4 thpt, 5-9 download time, 10 buffer, 12 last action,
    observation_array[:, 13:43] = chunk_sizes
    observation_array[:, 11] = np.arange(1, length + 2)[::-1]
    observation_array[:, 43:] = np.nan
    cf_trajs = np.zeros([len(pols), length + 1, 3])
    for i_pol, policy in enumerate(pols):
        observation_array[:, :11] = 0
        observation_array[:, 12] = 0
        for i in range(length):
            # Choose action
            act = policy.take_action(observation_array[i, :])

            ########### Use the counterfactual model to predict buffer level and download time
            chat = c_hat_observed[i]
            chat_white = (chat - chat_mean) / chat_std
            selected_size = observation_array[i, 13 + act]
            selected_size_white = (selected_size - size_mean) / size_std
            buffer = observation_array[i, 10]
            buffer_white = (buffer - buf_mean) / buf_std
            min_rtt_white = (min_rtt - min_rtts_mean) / min_rtts_std
            input_numpy = np.array(
                [buffer_white, selected_size_white, min_rtt_white, chat_white]
            )
            input_numpy = np.expand_dims(input_numpy, axis=0)
            input_tensor = torch.as_tensor(
                input_numpy, dtype=torch.float32, device=torch.device("cpu")
            )
            with torch.no_grad():
                prediction = buffer_predictor(input_tensor)
            next_buffer_white_tensor, down_time_white_tensor = (
                prediction[0, 0],
                prediction[0, 1],
            )
            next_buffer_white = next_buffer_white_tensor.cpu().numpy()
            down_time_white = down_time_white_tensor.cpu().numpy()
            download_time = (down_time_white * down_time_std) + down_time_mean
            next_buffer = (next_buffer_white * next_buf_std) + next_buf_mean

            # update observation_array
            observation_array[i + 1, 10] = next_buffer
            observation_array[i + 1, 12] = act
            observation_array[i + 1, 0:4] = observation_array[i, 1:5]
            observation_array[i + 1, 4] = selected_size / download_time
            observation_array[i + 1, 5:9] = observation_array[i, 6:10]
            observation_array[i + 1, 9] = download_time

            # Save results
            # NOTE: download time and next buffer are stored one index later, as is the case with original data
            cf_trajs[i_pol, i + 1, 0] = next_buffer
            cf_trajs[i_pol, i + 1, 1] = download_time
            cf_trajs[i_pol, i + 1, 2] = act
        act = policy.take_action(observation_array[i + 1, :])

    return cf_trajs  
