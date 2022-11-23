import numpy as np
import matplotlib.pyplot as plt
import datetime
import argparse
import pickle
from tqdm import tqdm
import os

MIN_SSIM = 0
MAX_SSIM = 60


def ssim_db(ssim: np.ndarray) -> np.ndarray:
    return np.where(ssim == 1, MAX_SSIM, np.clip(-10 * np.log10(1 - ssim), a_min=MIN_SSIM, a_max=MAX_SSIM))


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
orig_ssims = {target_policy: [] for target_policy in buffer_based_names}
expert_ssims = {target_policy: [] for target_policy in buffer_based_names}
cooked_path = f'{args.dir}cooked'
expert_path = f'{args.dir}2020-07-27to2021-06-01_expert_predictions'
for today in tqdm(all_days):
    date_string = "%d-%02d-%02d" % (today.year, today.month, today.day)
    ids = np.load(f'{cooked_path}/{date_string}_ids_translated.npy', allow_pickle=True)
    orig_trajs = np.load(f'{cooked_path}/{date_string}_trajs.npy', allow_pickle=True)
    expert_bba_ssims = np.load(f'{expert_path}/{date_string}_linear_bba_ssims.npy', allow_pickle=True)
    expert_bola1_ssims = np.load(f'{expert_path}/{date_string}_bola1_ssims.npy', allow_pickle=True)
    expert_bola2_ssims = np.load(f'{expert_path}/{date_string}_bola2_ssims.npy', allow_pickle=True)
    for idx, policy_name in enumerate(ids):
        if policy_name in buffer_based_names:
            orig_ssims[policy_name].append(orig_trajs[idx][:-1, 8])
    for idx, policy_name in enumerate(ids):
        if policy_name != args.left_out_policy:
            expert_ssims['bola_basic_v1'].append(expert_bola1_ssims[idx])
            expert_ssims['bola_basic_v2'].append(expert_bola2_ssims[idx])
            expert_ssims['linear_bba'].append(expert_bba_ssims[idx])
for target in buffer_based_names:
    orig_ssims[target] = np.concatenate(orig_ssims[target])
    orig_ssims[target] = np.mean(orig_ssims[target])
    orig_ssims[target] = ssim_db(orig_ssims[target])
    expert_ssims[target] = np.concatenate(expert_ssims[target])
    expert_ssims[target] = np.mean(expert_ssims[target])
    expert_ssims[target] = ssim_db(expert_ssims[target])

orig_ssim_path = f'{args.dir}subset_orig_ssim_dicts/{args.left_out_policy}'
expert_ssim_path = f'{args.dir}subset_expert_ssim_dicts/{args.left_out_policy}'

os.makedirs(orig_ssim_path, exist_ok=True)
os.makedirs(expert_ssim_path, exist_ok=True)

with open(f'{orig_ssim_path}/orig_ssims.pkl', 'wb') as f:
    pickle.dump(orig_ssims, f)
with open(f'{expert_ssim_path}/expert_ssims.pkl', 'wb') as f:
    pickle.dump(expert_ssims, f)