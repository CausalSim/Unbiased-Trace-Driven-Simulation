import numpy as np
import matplotlib.pyplot as plt
import datetime
import argparse
import pickle
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="source directory")
parser.add_argument("--C", type=float, help="discriminator loss coefficient")
parser.add_argument("--left_out_policy", type=str, help="left out policy")
parser.add_argument("--causalsim_model_number", type=int, help="CausalSim's saved model epoch number", default=5000)
parser.add_argument("--slsim_model_number", type=int, help="SLSim's saved model epoch number", default=10000)
args = parser.parse_args()
C = args.C
NUMBER_OF_BINS = 10000
DISCRIMINATOR_EPOCH = 10
left_out_text = f'_{args.left_out_policy}'
PERIOD_TEXT = f'2020-07-27to2021-06-01{left_out_text}'
cf_type = 'buff'
sim_key_text = '_buffs'
sl_key_text = '_buffs'
expert_key_text = '_buffs'

EMD_dir = f'{args.dir}subset_EMDs/{args.left_out_policy}'
os.makedirs(EMD_dir, exist_ok=True)

policy_names = ['bola_basic_v2', 'bola_basic_v1', 'puffer_ttp_cl', 'puffer_ttp_20190202', 'linear_bba']
buffer_based_names = ['bola_basic_v2', 'bola_basic_v1', 'linear_bba']
EMDs = {source: {target: None for target in buffer_based_names} for source in policy_names}
expert_EMDs = {source: {target: None for target in buffer_based_names} for source in policy_names}
sl_EMDs = {source: {target: None for target in buffer_based_names} for source in policy_names}

start_date = datetime.date(2020, 7, 27)
end_date = datetime.date(2021, 6, 1)
all_days = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]
all_days = [day for day in all_days if day not in [datetime.date(2019, 5, 12), datetime.date(2019, 5, 13),
                                                   datetime.date(2019, 5, 15), datetime.date(2019, 5, 17),
                                                   datetime.date(2019, 5, 18), datetime.date(2019, 5, 19),
                                                   datetime.date(2019, 5, 25), datetime.date(2019, 5, 27),
                                                   datetime.date(2019, 5, 30), datetime.date(2019, 6, 1),
                                                   datetime.date(2019, 6, 2), datetime.date(2019, 6, 3),
                                                   datetime.date(2019, 7, 2), datetime.date(2019, 7, 3),
                                                   datetime.date(2019, 7, 4), datetime.date(2020, 7, 7),
                                                   datetime.date(2020, 7, 8), datetime.date(2021, 6, 2),
                                                   datetime.date(2021, 6, 2), datetime.date(2021, 6, 3),
                                                   datetime.date(2022, 1, 31), datetime.date(2022, 2, 1),
                                                   datetime.date(2022, 2, 2), datetime.date(2022, 2, 3),
                                                   datetime.date(2022, 2, 4), datetime.date(2022, 2, 5),
                                                   datetime.date(2022, 2, 6), datetime.date(2022, 2, 7)]]

plt.figure(figsize=(24, 15))
orig_dict = {source_policy: [] for source_policy in policy_names}
expert_dict = {source_policy: {target_policy: [] for target_policy in buffer_based_names} for source_policy in
               policy_names}
sl_dict = {source_policy: {target_policy: [] for target_policy in buffer_based_names} for source_policy in policy_names}
sim_dict = {source_policy: {target_policy: [] for target_policy in buffer_based_names} for source_policy in
            policy_names}
cooked_path = f'{args.dir}cooked'
for today in tqdm(all_days):
    date_string = "%d-%02d-%02d" % (today.year, today.month, today.day)
    ids = np.load(f'{cooked_path}/{date_string}_ids_translated.npy', allow_pickle=True)
    sim_path = f'{args.dir}{PERIOD_TEXT}_{cf_type}_cfs/inner_loop_{DISCRIMINATOR_EPOCH}/C_{C}/cfs/' \
               f'model_{args.causalsim_model_number}'
    bba_preds = np.load(f'{sim_path}/{date_string}_linear_bba{sim_key_text}.npy', allow_pickle=True)
    bola1_preds = np.load(f'{sim_path}/{date_string}_bola1{sim_key_text}.npy', allow_pickle=True)
    bola2_preds = np.load(f'{sim_path}/{date_string}_bola2{sim_key_text}.npy', allow_pickle=True)
    orig_trajs = np.load(f'{cooked_path}/{date_string}_trajs.npy', allow_pickle=True)
    expert_path = f'{args.dir}2020-07-27to2021-06-01_expert_predictions'
    expert_bba_preds = np.load(f'{expert_path}/{date_string}_linear_bba{expert_key_text}.npy', allow_pickle=True)
    expert_bola1_preds = np.load(f'{expert_path}/{date_string}_bola1{expert_key_text}.npy', allow_pickle=True)
    expert_bola2_preds = np.load(f'{expert_path}/{date_string}_bola2{expert_key_text}.npy', allow_pickle=True)
    sl_path = f'{args.dir}{PERIOD_TEXT}_sl_cfs/cfs/model_{args.slsim_model_number}'
    sl_bba_preds = np.load(f'{sl_path}/{date_string}_linear_bba{sl_key_text}.npy', allow_pickle=True)
    sl_bola1_preds = np.load(f'{sl_path}/{date_string}_bola1{sl_key_text}.npy', allow_pickle=True)
    sl_bola2_preds = np.load(f'{sl_path}/{date_string}_bola2{sl_key_text}.npy', allow_pickle=True)
    for idx, policy_name in enumerate(ids):
        sim_dict[policy_name]['bola_basic_v1'].append(bola1_preds[idx])
        sim_dict[policy_name]['bola_basic_v2'].append(bola2_preds[idx])
        sim_dict[policy_name]['linear_bba'].append(bba_preds[idx])
        sl_dict[policy_name]['bola_basic_v1'].append(sl_bola1_preds[idx])
        sl_dict[policy_name]['bola_basic_v2'].append(sl_bola2_preds[idx])
        sl_dict[policy_name]['linear_bba'].append(sl_bba_preds[idx])
        expert_dict[policy_name]['bola_basic_v1'].append(expert_bola1_preds[idx])
        expert_dict[policy_name]['bola_basic_v2'].append(expert_bola2_preds[idx])
        expert_dict[policy_name]['linear_bba'].append(expert_bba_preds[idx])
        orig_dict[policy_name].append(orig_trajs[idx][:-1, 0])
for source in policy_names:
    orig_dict[source] = np.concatenate(orig_dict[source])
    for target in buffer_based_names:
        sim_dict[source][target] = np.concatenate(sim_dict[source][target])
        expert_dict[source][target] = np.concatenate(expert_dict[source][target])
        sl_dict[source][target] = np.concatenate(sl_dict[source][target])
for source_policy in policy_names:
    for target_policy in buffer_based_names:
        source = orig_dict[source_policy]
        target = orig_dict[target_policy]
        simulated = sim_dict[source_policy][target_policy]
        expert = expert_dict[source_policy][target_policy]
        sl = sl_dict[source_policy][target_policy]
        simulated = simulated[~np.isinf(simulated)]
        min_range = min([min(source), min(target), min(simulated), min(expert), min(sl)])
        max_range = max([max(source), max(target), max(simulated), max(expert), max(sl)])
        n_target, bin_target, _ = plt.hist(
            target, bins=NUMBER_OF_BINS, density=True, histtype='step',
            cumulative=True, label=f'Target ({target_policy})', range=(min_range, max_range))
        n_pred, bin_pred, _ = plt.hist(
            simulated, bins=NUMBER_OF_BINS, density=True, histtype='step',
            cumulative=True, label=f'CausalSim (Simulated {target_policy})', range=(min_range, max_range))
        n_expert, bin_expert, _ = plt.hist(
            expert, bins=NUMBER_OF_BINS, density=True, histtype='step',
            cumulative=True, label=f'ExpertSim (Simulated {target_policy})', range=(min_range, max_range))
        n_sl, bin_sl, _ = plt.hist(
            sl, bins=NUMBER_OF_BINS, density=True, histtype='step',
            cumulative=True, label=f'SLSim (Simulated {target_policy})', range=(min_range, max_range))
        n_source, bin_source, _ = plt.hist(
            source, bins=NUMBER_OF_BINS, density=True, histtype='step',
            cumulative=True, label=f'Source ({source_policy})', range=(min_range, max_range))
        simulation_EMD = np.sum(np.abs(n_pred - n_target) * (bin_target[1:] - bin_target[:-1]))
        expert_EMD = np.sum(np.abs(n_expert - n_target) * (bin_target[1:] - bin_target[:-1]))
        sl_EMD = np.sum(np.abs(n_sl - n_target) * (bin_target[1:] - bin_target[:-1]))
        EMDs[source_policy][target_policy] = simulation_EMD
        expert_EMDs[source_policy][target_policy] = expert_EMD
        sl_EMDs[source_policy][target_policy] = sl_EMD

with open(f'{EMD_dir}/expert_buff_{C}.pkl', 'wb') as f:
    pickle.dump(expert_EMDs, f)
with open(f'{EMD_dir}/sl_buff_{C}.pkl', 'wb') as f:
    pickle.dump(sl_EMDs, f)
with open(f'{EMD_dir}/sim_buff_{C}.pkl', 'wb') as f:
    pickle.dump(EMDs, f)
plt.close()
