from typing import List
import numpy as np
import matplotlib.pyplot as plt
import datetime
import argparse
import pickle
from tqdm import tqdm
import os


def get_stall(traj: np.ndarray or List[int]) -> float:
    if len(traj) > 1:
        assert traj[1] > 0, traj
        return traj[-1] - traj[1]
    else:
        assert traj[-1] == 0, traj
        return 0

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="source directory")
parser.add_argument("--left_out_policy", type=str, help="left out policy")
args = parser.parse_args()
NUMBER_OF_BINS = 10000
left_out_text = f'_{args.left_out_policy}'
PERIOD_TEXT = f'2020-07-27to2021-06-01{left_out_text}'
policy_names = ['bola_basic_v2', 'bola_basic_v1', 'puffer_ttp_cl', 'puffer_ttp_20190202', 'linear_bba']
buffer_based_names = ['bola_basic_v2', 'bola_basic_v1', 'linear_bba']
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
plt.figure()
orig_data = {target_policy: {'lens': [], 'rebuffs': []} for target_policy in buffer_based_names}
expert_data = {target_policy: {'lens': [], 'rebuffs': []} for target_policy in buffer_based_names}
cooked_path = f'{args.dir}cooked'
expert_path = f'{args.dir}2020-07-27to2021-06-01_expert_predictions'
for today in tqdm(all_days):
    date_string = "%d-%02d-%02d" % (today.year, today.month, today.day)
    ids = np.load(f'{cooked_path}/{date_string}_ids_translated.npy', allow_pickle=True)
    orig_trajs = np.load(f'{cooked_path}/{date_string}_trajs.npy', allow_pickle=True)
    expert_bba_rebuffs = np.load(f'{expert_path}/{date_string}_linear_bba_rebuffs.npy', allow_pickle=True)
    expert_bola1_rebuffs = np.load(f'{expert_path}/{date_string}_bola1_rebuffs.npy', allow_pickle=True)
    expert_bola2_rebuffs = np.load(f'{expert_path}/{date_string}_bola2_rebuffs.npy', allow_pickle=True)
    for idx, policy_name in enumerate(ids):
        if policy_name in buffer_based_names:
            orig_data[policy_name]['rebuffs'].append(
                np.sum(np.maximum(orig_trajs[idx][1:-1, 6] - orig_trajs[idx][1:-1, 0], 0)))
            orig_data[policy_name]['lens'].append((orig_trajs[idx].shape[0] - 1) * 2.002 + orig_data[policy_name]['rebuffs'][-1])
        if policy_name != args.left_out_policy:
            expert_data['bola_basic_v1']['rebuffs'].append(get_stall(expert_bola1_rebuffs[idx]))
            expert_data['bola_basic_v2']['rebuffs'].append(get_stall(expert_bola2_rebuffs[idx]))
            expert_data['linear_bba']['rebuffs'].append(get_stall(expert_bba_rebuffs[idx]))
            expert_data['bola_basic_v1']['lens'].append((len(expert_bola1_rebuffs[idx]) - 1) * 2.002 + expert_data['bola_basic_v1']['rebuffs'][-1])
            expert_data['bola_basic_v2']['lens'].append((len(expert_bola2_rebuffs[idx]) - 1) * 2.002 + expert_data['bola_basic_v2']['rebuffs'][-1])
            expert_data['linear_bba']['lens'].append((len(expert_bba_rebuffs[idx]) - 1) * 2.002 + expert_data['linear_bba']['rebuffs'][-1])


orig_stall_path = f'{args.dir}subset_orig_rebuff_dicts/{args.left_out_policy}'
expert_stall_path = f'{args.dir}subset_expert_rebuff_dicts/{args.left_out_policy}'

os.makedirs(orig_stall_path, exist_ok=True)
os.makedirs(expert_stall_path, exist_ok=True)

with open(f'{orig_stall_path}/orig_rebuffs.pkl', 'wb') as f:
    pickle.dump(orig_data, f)
with open(f'{expert_stall_path}/expert_rebuffs.pkl', 'wb') as f:
    pickle.dump(expert_data, f)