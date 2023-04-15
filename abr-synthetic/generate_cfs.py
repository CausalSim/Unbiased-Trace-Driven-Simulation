from generate_traces import collect_traces_causalsim, collect_traces_slsim
import numpy as np
from policies import get_all_policies
import torch
from tqdm import tqdm


def generate_cfs(
    datapath,
    training_datapath,
    models_path,
    test_policy_idx,
    config,
    alg="causalsim",
    r=None,
):
    buffers = np.load(f"{training_datapath}/raw_train_buffers_synthetic.npy")
    next_buffers = np.load(f"{training_datapath}/raw_train_next_buffers_synthetic.npy")
    c_hats = np.load(f"{training_datapath}/raw_train_c_hats_synthetic.npy")
    chosen_chunk_sizes = np.load(
        f"{training_datapath}/raw_train_chosen_chunk_sizes_synthetic.npy"
    )
    min_rtts = np.load(f"{training_datapath}/raw_train_min_rtts_synthetic.npy")
    download_time = np.load(
        f"{training_datapath}/raw_train_download_time_synthetic.npy"
    )

    buffer_mean = np.mean(buffers)
    next_buffer_mean = np.mean(next_buffers)
    c_hat_mean = 0
    chosen_chunk_size_mean = np.mean(chosen_chunk_sizes)
    min_rtts_mean = np.mean(min_rtts)
    download_time_mean = np.mean(download_time)

    buffer_std = np.std(buffers)
    next_buffer_std = np.std(next_buffers)
    c_hat_std = np.std(c_hats)
    chosen_chunk_size_std = np.std(chosen_chunk_sizes)
    min_rtts_std = np.std(min_rtts)
    download_time_std = np.std(download_time)

    policies, _, _ = get_all_policies(config)
    if test_policy_idx is None:
        pols = policies
    else:
        pols = [policies[test_policy_idx]]

    train_trajectories = np.load(
        f"{training_datapath}/train_trajectories.npy", allow_pickle=True
    )
    features = np.zeros(
        [
            train_trajectories.shape[0],
            train_trajectories.shape[1],
            train_trajectories.shape[2] - 1,
        ]
    )
    cfs = np.zeros(
        [
            len(pols),
            train_trajectories.shape[1],
            train_trajectories.shape[2],
            3,
        ]
    )
    rtts = np.load(f"{datapath}/rtts.npy")
    trajs = train_trajectories[0, :]
    if alg == "causalsim":
        feature_extractor = torch.load(
            "%sbest_feature_extractor.pth" % models_path,
            map_location=torch.device("cpu"),
        )
        buffer_predictor = torch.load(
            "%sbest_buffer_predictor.pth" % models_path,
            map_location=torch.device("cpu"),
        )
        action_factor_net = torch.load(
            "%sbest_action_factor.pth" % models_path, map_location=torch.device("cpu")
        )

        for idx, traj in tqdm(enumerate(trajs)):
            cfs[0, idx, :], features[0, idx, :] = collect_traces_causalsim(
                pols,
                traj[1:, 4],
                traj[:-1, -2].astype(int),
                traj[:, 13 : 13 + 30],
                rtts[idx],
                feature_extractor,
                buffer_predictor,
                action_factor_net,
                r,
                buffer_mean,
                buffer_std,
                next_buffer_mean,
                next_buffer_std,
                c_hat_mean,
                c_hat_std,
                chosen_chunk_size_mean,
                chosen_chunk_size_std,
                min_rtts_mean,
                min_rtts_std,
                download_time_mean,
                download_time_std,
            )

        return cfs, features
    elif alg == "slsim":
        buffer_predictor = torch.load(
            "%sbest_buffer_predictor.pth" % models_path,
            map_location=torch.device("cpu"),
        )

        for idx, traj in tqdm(enumerate(trajs)):
            cfs[:, idx, :] = collect_traces_slsim(
                pols,
                traj[1:, 4],
                traj[:, 13 : 13 + 30],
                rtts[idx],
                buffer_predictor,
                buffer_mean,
                buffer_std,
                next_buffer_mean,
                next_buffer_std,
                c_hat_mean,
                c_hat_std,
                chosen_chunk_size_mean,
                chosen_chunk_size_std,
                min_rtts_mean,
                min_rtts_std,
                download_time_mean,
                download_time_std,
            )
        return cfs
