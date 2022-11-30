import numpy as np
import argparse
import pickle
import os
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="source directory")
args = parser.parse_args()
policy_names = ['bola_basic_v2', 'bola_basic_v1', 'puffer_ttp_cl', 'puffer_ttp_20190202', 'linear_bba']
buffer_based_names = ['bola_basic_v2', 'bola_basic_v1', 'linear_bba']
buffer_hyperparams = {policy: [] for policy in buffer_based_names}

C_list = ['0.05', '0.1', '0.5', '1.0', '5.0', '10.0', '15.0', '20.0', '25.0', '30.0', '40.0']

keys = ['buff']
for left_out_policy in buffer_based_names:
    sim_EMDs = {key: {'val': []} for key in keys}
    expert_EMDs = {key: {'val': []} for key in keys}
    sl_EMDs = {key: {'val': []} for key in keys}
    for idx, key in enumerate(keys):
        for C in C_list:
            with open(f'{args.dir}subset_EMDs/{left_out_policy}/sim_{key}_{C}.pkl', 'rb') as f:
                sim_dict = pickle.load(f)
            with open(f'{args.dir}subset_EMDs/{left_out_policy}/expert_{key}_{C}.pkl', 'rb') as f:
                expert_dict = pickle.load(f)
            with open(f'{args.dir}subset_EMDs/{left_out_policy}/sl_{key}_{C}.pkl', 'rb') as f:
                sl_dict = pickle.load(f)
            sim_EMDs[key]['val'].append(np.mean(
                [sim_dict[source][target] for (source, target) in itertools.product(policy_names, buffer_based_names) if
                 target != left_out_policy and source != left_out_policy]))
            expert_EMDs[key]['val'].append(np.mean(
                [expert_dict[source][target] for (source, target) in itertools.product(policy_names, buffer_based_names)
                 if target != left_out_policy and source != left_out_policy]))
            sl_EMDs[key]['val'].append(np.mean(
                [sl_dict[source][target] for (source, target) in itertools.product(policy_names, buffer_based_names) if
                 target != left_out_policy and source != left_out_policy]))

    expert_data = {policy: [] for policy in buffer_based_names}
    sl_data = {policy: [] for policy in buffer_based_names}
    sim_data = {C: {policy: [] for policy in buffer_based_names} for C in C_list}
    with open(f'{args.dir}subset_expert_ssim_dicts/{left_out_policy}/expert_ssims.pkl', 'rb') as f:
        expert_ssims = pickle.load(f)
    with open(f'{args.dir}subset_sl_ssim_dicts/{left_out_policy}/ssims.pkl', 'rb') as f:
        sl_ssims = pickle.load(f)
    with open(f'{args.dir}subset_orig_ssim_dicts/{left_out_policy}/orig_ssims.pkl', 'rb') as f:
        orig_ssims = pickle.load(f)
    for policy in buffer_based_names:
        expert_data[policy].append(abs(expert_ssims[policy] - orig_ssims[policy]))
    for policy in buffer_based_names:
        sl_data[policy].append(abs(sl_ssims[policy] - orig_ssims[policy]))
    for C in C_list:
        with open(f'{args.dir}subset_ssim_dicts/{left_out_policy}/ssims_{C}.pkl', 'rb') as f:
            sim_ssims = pickle.load(f)
        for policy in buffer_based_names:
            sim_data[C][policy].append(abs(sim_ssims[policy] - orig_ssims[policy]))
    buffer_hyperparams[left_out_policy].append(C_list[np.argmin(
        [np.mean([sim_data[C][policy] for policy in buffer_based_names if policy != left_out_policy]) for C in C_list])])
    buffer_hyperparams[left_out_policy].append(C_list[np.argmin(sim_EMDs['buff']['val'])])

hyperparam_path = f'{args.dir}tuned_hyperparams'
os.makedirs(hyperparam_path, exist_ok=True)
with open(f'{hyperparam_path}/buffer.pkl', 'wb') as f:
    pickle.dump(buffer_hyperparams, f)
