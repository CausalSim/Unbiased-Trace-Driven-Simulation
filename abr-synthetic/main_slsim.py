import numpy as np
import os
from slsim import train_slsim
import argparse

from generate_cfs import generate_cfs
import matplotlib.pyplot as plt
import torch

NO_POLICIES = 9
POLICIES = [
    "BBA",
    "BBAMIX-x1-50",
    "BBAMIX-x2-50",
    "MPC",
    "Random",
    "BOLA",
    "Rate Based",
    "Optimistic Rate Based",
    "Pessimistic Rate Based",
]
policy_paths_names = [
    "bba_traj",
    "bbamix_X1.0_RND50%_traj",
    "bbamix_X2.0_RND50%_traj",
    "mpc_traj",
    "rnd_traj_0",
    "bola_traj",
    "rate_traj",
    "opt_rate_traj",
    "pess_rate_traj",
]

torch.manual_seed(0)


def cdf(x, plot=True, *args, **kwargs):
    x = sorted(x)[:]
    x, y = sorted(x), np.arange(len(x)) / len(x)
    return plt.plot(x, 100 * y, *args, **kwargs) if plot else (x, y)


def get_mse(truth, estimate, policy_assignment):
    policies = np.unique(policy_assignment)
    MSE = []
    for p in policies:
        truth_p = truth[policy_assignment == p, :]
        estimate_p = estimate[policy_assignment == p, :]
        mse_p = np.square(estimate_p - truth_p).mean()
        MSE.append(mse_p)
    return MSE


def load_and_create_datasets(data_path, policy_out, dir_out, loss):
    # init policy out and list of training policies

    policy_out_idx = POLICIES.index(policy_out)
    policies = list(policy_paths_names)
    del policies[policy_out_idx]

    total_no_traj = 5000
    split = total_no_traj // len(policies)
    buffers = []
    next_buffers = []
    download_time = []
    chosen_chunk_sizes = []
    c_hats = []
    numbers = []
    min_rtts = []

    file_name = f"{policy_out}_{loss}"
    savepath = f"{dir_out}/train_data_{file_name}/"
    try:
        os.makedirs(savepath)
    except:
        pass

    rtts = np.load(f"{data_path}/rtts.npy")
    policy_assignment = np.zeros(total_no_traj)
    # loop over policies and load the data
    for i, policy in enumerate(policies):
        data = np.load(f"{data_path}/{policy}.npy")
        # select a fraction of the trajectories (not randomly, it's sliced in order)
        if i + 1 == len(policies):
            data = data[i * split : total_no_traj]
            policy_assignment[i * split : total_no_traj] = i
        else:
            data = data[i * split : (i + 1) * split]
            policy_assignment[i * split : (i + 1) * split] = i
        # concatenate all data
        if i == 0:
            data_all = np.array(data)
        else:
            data_all = np.concatenate([data_all, data])

        # loop over trajectories
        for idx, traj in enumerate(data):
            # c_hat of each step is observed in the next batch
            c_hats.extend(traj[1:, 4])
            # get download time
            download_time.extend(traj[1:, 9])
            # get the action index and the chosen chunk size
            action = traj[:-1, -2].astype(int)
            chunk_size = traj[np.arange(len(action)), 13 + action]
            chosen_chunk_sizes.extend(chunk_size)
            buffers.extend(traj[:-1, 10])
            min_rtts.extend(rtts[i * split + idx] * np.ones(action.shape[0]))
            next_buffers.extend(traj[1:, 10])
            # store which policy has been chosen
            numbers.extend([i for _ in traj[:-1, 0]])

    data_all = data_all.reshape(
        [1, total_no_traj, data_all.shape[1], data_all.shape[2]], order="C"
    )
    print(len(numbers), total_no_traj * 489)
    try:
        os.mkdir(f"{savepath}")
    except:
        pass
    np.save(f"{savepath}/train_trajectories.npy", data_all)
    np.save(f"{savepath}/policy_assignment.npy", policy_assignment)
    # use all points as training data
    train_indices = np.arange(len(numbers))
    buffers = np.array(buffers)
    next_buffers = np.array(next_buffers)
    chosen_chunk_sizes = np.array(chosen_chunk_sizes)
    numbers = np.array(numbers)
    c_hats = np.array(c_hats)
    min_rtts = np.array(min_rtts)
    download_time = np.array(download_time)

    train_buffers = buffers[train_indices]
    train_next_buffers = next_buffers[train_indices]
    train_c_hats = c_hats[train_indices]
    train_chosen_chunk_sizes = chosen_chunk_sizes[train_indices]
    train_numbers = numbers[train_indices]
    train_min_rtts = min_rtts[train_indices]
    download_time = download_time[train_indices]
    print(
        train_min_rtts.shape,
        train_next_buffers.shape,
        train_chosen_chunk_sizes.shape,
        download_time.shape,
    )

    np.save(f"{savepath}/raw_train_buffers_synthetic.npy", np.array(train_buffers))
    np.save(
        f"{savepath}/raw_train_next_buffers_synthetic.npy",
        np.array(train_next_buffers),
    )
    np.save(f"{savepath}/raw_train_c_hats_synthetic.npy", np.array(train_c_hats))
    np.save(
        f"{savepath}/raw_train_chosen_chunk_sizes_synthetic.npy",
        np.array(train_chosen_chunk_sizes),
    )
    np.save(f"{savepath}/raw_train_numbers_synthetic.npy", np.array(train_numbers))
    np.save(f"{savepath}/raw_train_min_rtts_synthetic.npy", np.array(train_min_rtts))
    np.save(
        f"{savepath}/raw_train_download_time_synthetic.npy", np.array(download_time)
    )

    train_buffers = train_buffers - np.mean(train_buffers)
    train_next_buffers = train_next_buffers - np.mean(train_next_buffers)
    # do not make c_hat mean zero, it affects the rank-2 structure
    train_c_hats = train_c_hats  # - np.mean(train_c_hats)

    train_chosen_chunk_sizes = train_chosen_chunk_sizes - np.mean(
        train_chosen_chunk_sizes
    )
    train_min_rtts = train_min_rtts - np.mean(train_min_rtts)
    download_time = download_time - np.mean(download_time)

    train_buffers = train_buffers / np.std(train_buffers)
    train_next_buffers = train_next_buffers / np.std(train_next_buffers)
    train_chosen_chunk_sizes = train_chosen_chunk_sizes / np.std(
        train_chosen_chunk_sizes
    )
    train_c_hats = train_c_hats / np.std(train_c_hats)
    train_min_rtts = train_min_rtts / np.std(train_min_rtts)
    download_time = download_time / np.std(download_time)

    train_inputs = np.concatenate(
        (
            np.expand_dims(train_buffers, axis=1),
            np.expand_dims(train_chosen_chunk_sizes, axis=1),
            np.expand_dims(train_min_rtts, axis=1),
            np.expand_dims(train_c_hats, axis=1),
        ),
        axis=1,
    )
    train_outputs = np.concatenate(
        (
            np.expand_dims(train_next_buffers, axis=1),
            np.expand_dims(download_time, axis=1),
            np.expand_dims(train_numbers, axis=1),
        ),
        axis=1,
    )
    print(train_inputs.shape, train_outputs.shape)
    np.save(f"{savepath}/white_train_inputs_synthetic.npy", train_inputs)
    np.save(f"{savepath}/white_train_outputs_synthetic.npy", train_outputs)
    assert len(train_inputs) == len(train_outputs)
    return savepath


def parse_args():
    parser = argparse.ArgumentParser(description="main parser")
    parser.add_argument("--dir", type=str, required=True, help="dataset directory")
    parser.add_argument(
        "--policy_out",
        type=str,
        default="Random",
        help=f"choose test policy from {POLICIES}",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse_loss",
        help="choose the loss from mse_loss, l1_loss, huber_loss",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--eps", type=float, default=1e-6, help="epsilon (default: 1e-6)"
    )
    parser.add_argument(
        "--trace_sim_count",
        type=int,
        default=5000,
        help="Number of generated traces (default: 5000)",
    )
    parser.add_argument(
        "--bba_reservoir", type=float, default=5, help="BBA - Reservoir (default: 5)"
    )
    parser.add_argument(
        "--bba_cushion", type=float, default=10, help="BBA - Cushion (default: 10)"
    )
    parser.add_argument(
        "--mpc_lookback",
        type=int,
        default=5,
        help="MPC - Throughput lookback (default: 5)",
    )
    parser.add_argument(
        "--mpc_lookahead",
        type=int,
        default=5,
        help="MPC - Throughput lookahead (default: 5)",
    )
    return parser.parse_args()


def main():
    # get parameters
    args = parse_args()

    policy_index = POLICIES.index(args.policy_out)
    # generate data wihtout the test policy
    print("GENERATE DATASETS .. ")
    generated_datapath = load_and_create_datasets(
        args.dir, args.policy_out, args.dir, args.loss
    )

    print("TRAIN SLSIM .. ")
    # train direct
    train_slsim(
        generated_datapath,
        models_path=f"{args.dir}/models/{args.policy_out}/{args.loss}",
        loss=args.loss,
    )

    print("GENERATE SLSIM COUNTERFACTUALS")
    alg = "slsim"
    cf_slsim = generate_cfs(
        args.dir,
        generated_datapath,
        models_path=f"{args.dir}/models/{args.policy_out}/{args.loss}/{alg}/",
        config=args,
        test_policy_idx=None,
        alg=alg,
    )
    policy = POLICIES[policy_index]
    policy_path = policy_paths_names[policy_index]
    np.save(f"cfs/cf_{args.policy_out}_{args.loss}.npy", cf_slsim)

    print("Plotting results")

    policy_assignment = np.load(f"{generated_datapath}/policy_assignment.npy")
    true_buffer = np.load(f"{args.dir}/{policy_path}.npy")[:, :, 10]

    MAPE_slsim = get_mse(
        true_buffer,
        cf_slsim[0, ..., 0],
        policy_assignment,
    )
    os.makedirs("figures", exist_ok=True)
    plt.figure()
    plt.title(f"buffer size predictions for test policy {policy}")
    cdf(MAPE_slsim, label="SLSim")
    plt.xlabel("buffer size prediction MSE")
    plt.ylabel("CDF(%)")
    plt.legend()
    plt.savefig(f"figures/buffer_size_MAPE_{args.policy_out}_{args.loss}.png")


if __name__ == "__main__":
    main()
