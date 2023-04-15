import numpy as np
import pickle
import os
from causalsim import train_causal_sim
from slsim import train_slsim
import argparse
from generate_cfs import generate_cfs
import matplotlib.pyplot as plt
import torch

NO_POLICIES = 16
POLICIES = [
    "random",
    "Shortest queue",
    "pow2",
    "pow3",
    "pow4",
    "pow5",
    "PseudoOptimal",
    "Tracker",
]


DATAPATH = "non_iid_0_big.pkl"
torch.manual_seed(0)


def cdf(x, plot=True, *args, **kwargs):
    x = sorted(x)[:]
    x, y = sorted(x), np.arange(len(x)) / len(x)
    return plt.plot(x, 100 * y, *args, **kwargs) if plot else (x, y)


def get_mape(truth, estimate, policy_assignment):
    policies = np.unique(policy_assignment)
    mapes = []
    for p in policies:
        truth_p = truth[policy_assignment == p, :]
        estimate_p = estimate[policy_assignment == p, :]
        mape_p = 100 * (np.abs(estimate_p - truth_p) / truth_p).mean()
        mapes.append(mape_p)
    return mapes


def load_and_create_datasets(dict_exp, policy_out, dir_out, loss):
    policy_out_idx = POLICIES.index(policy_out)

    actions = dict_exp["actions"]
    processing_times = dict_exp["proc_times"]
    no_traj, T = processing_times.shape[1], processing_times.shape[2]
    policies_range = np.arange(NO_POLICIES)
    policies_range = np.delete(policies_range, [policy_out_idx])
    numbers_policies = np.random.choice(policies_range, size=no_traj).astype(int)
    data_all = np.zeros([no_traj, T, 2])
    processing_times_list = []
    numbers = []
    actions_list = []
    for i, policy in enumerate(policies_range):
        actions_policy = actions[policy, numbers_policies == policy, :]
        pt_policy = processing_times[policy, numbers_policies == policy, :]
        data_all[numbers_policies == policy, :, 0] = actions_policy
        data_all[numbers_policies == policy, :, 1] = pt_policy
        processing_times_list.extend(pt_policy.flatten())
        actions_list.extend(actions_policy.flatten())
        numbers.extend([i for _ in range(actions_policy.flatten().shape[0])])
    file_name = f"{policy_out}_{loss}"
    savepath = f"{dir_out}/train_data_{file_name}/"
    try:
        os.makedirs(savepath)
    except:
        pass

    np.save(f"{savepath}/train_trajectories.npy", data_all)
    max_pol = max(numbers)
    train_indices = [i for i in range(len(numbers)) if numbers[i] <= max_pol]

    train_pt = np.array(processing_times_list)[train_indices]
    train_actions = np.array(actions_list)[train_indices].astype(int)
    train_numbers = np.array(numbers)[train_indices]

    np.save(f"{savepath}/raw_train_pt.npy", np.array(train_pt))
    np.save(f"{savepath}/raw_train_actions.npy", np.array(train_actions))
    np.save(f"{savepath}/raw_train_numbers.npy", np.array(train_numbers))

    np.save(f"{savepath}/policy_assignment.npy", np.array(numbers_policies))

    # train_pt = train_pt - np.mean(train_pt)
    train_pt = train_pt / np.std(train_pt)

    train_actions_onehot = np.zeros([train_actions.shape[0], int(actions.max()) + 1])

    train_actions_onehot[np.arange(train_actions.shape[0]), train_actions] = 1

    train_inputs = np.concatenate(
        (np.expand_dims(train_pt, axis=1), train_actions_onehot), axis=1
    )
    train_outputs = np.concatenate(
        (np.expand_dims(train_pt, axis=1), np.expand_dims(train_numbers, axis=1)),
        axis=1,
    )
    np.save(f"{savepath}/white_train_inputs.npy", train_inputs)
    np.save(f"{savepath}/white_train_outputs.npy", train_outputs)
    assert len(train_inputs) == len(train_outputs)
    return savepath


def parse_args():
    parser = argparse.ArgumentParser(description="main parser")
    parser.add_argument(
        "--dir", type=str, required=True, help="Load Balance dataset directory"
    )
    parser.add_argument(
        "--policy_out",
        type=str,
        default="random",
        help=f"choose test policy from {POLICIES}",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse_loss",
        help="choose the loss from mse_loss, l1_loss, huber_loss",
    )
    return parser.parse_args()


def main():
    # get parameters
    args = parse_args()
    with open(f"{args.dir}/{DATAPATH}", "rb") as fandle:
        dict_exp = pickle.load(fandle)

    policy_index = POLICIES.index(args.policy_out)
    # generate data wihtout the test policy
    print("GENERATE DATASETS .. ")
    generated_datapath = load_and_create_datasets(
        dict_exp, args.policy_out, args.dir, args.loss
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
    cf_slsim, features_slsim = generate_cfs(
        dict_exp,
        generated_datapath,
        models_path=f"{args.dir}/models/{args.policy_out}/{args.loss}/{alg}/",
        test_policy_idx=None,
        alg=alg,
        N_test=5000,
    )
    np.save(f"cfs/cf_{args.policy_out}_{args.loss}.npy", cf_slsim)
    print("Plotting results")
    policy_assignment = np.load(f"{generated_datapath}/policy_assignment.npy")
    truth_processing_time = dict_exp["proc_times"]

    MAPE_slsim = get_mape(
        truth_processing_time[policy_index, :, :],
        cf_slsim[0, ..., 0],
        policy_assignment,
    )
    os.makedirs("figures", exist_ok=True)

    plt.figure()
    plt.title("processing time")
    cdf(MAPE_slsim, label="Direct")
    plt.xlabel("Processing time MAPE")
    plt.ylabel("CDF(%)")
    plt.legend()
    plt.savefig(f"figures/processing_time_MAPE_{args.policy_out}.png")

    truth_latency = dict_exp["latencies"]

    MAPE_slsim = get_mape(
        truth_latency[policy_index, :, :],
        cf_slsim[0, ..., 1],
        policy_assignment,
    )

    plt.figure()
    plt.title("latency")
    cdf(MAPE_slsim, label="Direct")
    plt.xlabel("Latency MAPE")
    plt.ylabel("CDF(%)")
    plt.legend()
    plt.savefig(f"figures/latency_MAPE_{args.policy_out}.png")


if __name__ == "__main__":
    main()
