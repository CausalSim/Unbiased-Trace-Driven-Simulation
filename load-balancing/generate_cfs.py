from latency_util import *


def generate_cfs(
    dict_exp,
    training_datapath,
    models_path,
    test_policy_idx,
    alg="causalsim",
    N_test=5000,
    r=None,
):

    job_size = dict_exp["job_size"][:N_test, :]
    inter_arrs = dict_exp["ia_time"][:N_test, :]

    pts = np.load(f"{training_datapath}/raw_train_pt.npy")
    pt_mean = 0
    pt_std = np.std(pts)

    train_trajectories = np.load(
        f"{training_datapath}/train_trajectories.npy", allow_pickle=True
    )

    actions_obs = train_trajectories[:N_test, :, 0]
    ptimes_obs = train_trajectories[:N_test, :, 1]
    if alg == "causalsim":
        assert r is not None

        feature_extractor = torch.load(
            "%sbest_feature_extractor.pth" % models_path,
            map_location=torch.device("cpu"),
        )
        action_factor = torch.load(
            "%sbest_action_factor.pth" % models_path, map_location=torch.device("cpu")
        )

        features, actions, proc_times, latencies = collect_traces_sim_traj_fact(
            job_size,
            inter_arrs,
            ptimes_obs,
            actions_obs,
            feature_extractor,
            action_factor,
            r,
            pt_mean,
            pt_std,
            test_pol_idx=test_policy_idx,
            p_change=0,
        )
    elif alg == "slsim":
        buffer_predictor = torch.load(
            "%sbest_buffer_predictor.pth" % models_path,
            map_location=torch.device("cpu"),
        )
        actions, proc_times, latencies = collect_traces_direct_traj(
            job_size,
            inter_arrs,
            ptimes_obs,
            buffer_predictor,
            pt_mean,
            pt_std,
            test_pol_idx=test_policy_idx,
            p_change=0,
        )
        features = None
    else:
        raise ValueError(f"Unknown value {alg} for alg")

    cf = np.zeros([8, N_test, 1000, 3])
    cf[:, :, :, 0] = proc_times
    cf[:, :, :, 1] = latencies
    cf[:, :, :, 2] = actions

    return cf, features
