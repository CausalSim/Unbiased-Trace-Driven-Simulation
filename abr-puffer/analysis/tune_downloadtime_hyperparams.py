import numpy as np
import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="source directory")
args = parser.parse_args()
policy_names = ['bola_basic_v2', 'bola_basic_v1', 'puffer_ttp_cl', 'puffer_ttp_20190202', 'linear_bba']
buffer_based_names = ['bola_basic_v2', 'bola_basic_v1', 'linear_bba']
downloadtime_hyperparams = {policy: [] for policy in buffer_based_names}

C_list = ['0.05', '0.1', '0.5', '1.0', '5.0', '10.0', '15.0', '20.0', '25.0', '30.0', '40.0']

for left_out_policy in buffer_based_names:
    sim_data = {C: {policy: [] for policy in buffer_based_names} for C in C_list}
    with open(f'{args.dir}subset_orig_rebuff_dicts/{left_out_policy}/orig_rebuffs.pkl', 'rb') as f:
        orig_dict = pickle.load(f)
    for C in C_list:
        with open(f'{args.dir}subset_stall_dicts/{left_out_policy}/stalls_{C}.pkl', 'rb') as f:
            sim_dict = pickle.load(f)
        for policy in buffer_based_names:
            sim_data[C][policy].append(100 * abs(np.sum(sim_dict[policy]['rebuffs']) / np.sum(sim_dict[policy]['lens']) - np.sum(orig_dict[policy]['rebuffs']) / np.sum(orig_dict[policy]['lens'])))
    downloadtime_hyperparams[left_out_policy] = [C_list[np.argmin(
        [np.mean([sim_data[C][policy] for policy in buffer_based_names if policy != left_out_policy]) for C in C_list])]]

hyperparam_path = f'{args.dir}tuned_hyperparams'
os.makedirs(hyperparam_path, exist_ok=True)
with open(f'{hyperparam_path}/downloadtime.pkl', 'wb') as f:
    pickle.dump(downloadtime_hyperparams, f)
