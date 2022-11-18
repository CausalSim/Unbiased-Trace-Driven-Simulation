import numpy as np
import datetime
import argparse
import pickle
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="source directory")
parser.add_argument("--C", type=float, help="discriminator loss coefficient")
parser.add_argument("--left_out_policy", type=str, help="left out policy")
parser.add_argument("--model_number", type=int, help="saved model epoch number", default=5000)
args = parser.parse_args()
NUMBER_OF_BINS = 10000
left_out_text = f'_{args.left_out_policy}'
PERIOD_TEXT = f'2020-07-27to2021-06-01{left_out_text}'
DISCRIMINATOR_EPOCH = 10
C = args.C
policy_names = ['bola_basic_v2', 'bola_basic_v1', 'puffer_ttp_cl', 'puffer_ttp_20190202', 'linear_bba']
buffer_based_names = ['bola_basic_v2', 'bola_basic_v1', 'linear_bba']

sim_data = {target_policy: {'rebuffs': [], 'lens': []} for target_policy in buffer_based_names}
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
cooked_path = f'{args.dir}cooked'
for today in tqdm(all_days):
    date_string = "%d-%02d-%02d" % (today.year, today.month, today.day)
    ids = np.load(f'{cooked_path}/{date_string}_ids_translated.npy', allow_pickle=True)
    cf_path = f'{args.dir}{PERIOD_TEXT}_dt_cfs/inner_loop_{DISCRIMINATOR_EPOCH}/C_{C}/cfs/' \
              f'model_{args.model_number}'
    bba_rebuffs = np.load(f'{cf_path}/{date_string}_linear_bba_rebuffs.npy', allow_pickle=True)
    bola1_rebuffs = np.load(f'{cf_path}/{date_string}_bola1_rebuffs.npy', allow_pickle=True)
    bola2_rebuffs = np.load(f'{cf_path}/{date_string}_bola2_rebuffs.npy', allow_pickle=True)
    for idx, policy_name in enumerate(ids):
        if policy_name != args.left_out_policy:
            sim_data['bola_basic_v1']['rebuffs'].append(np.sum(bola1_rebuffs[idx][1:]))
            sim_data['bola_basic_v2']['rebuffs'].append(np.sum(bola2_rebuffs[idx][1:]))
            sim_data['linear_bba']['rebuffs'].append(np.sum(bba_rebuffs[idx][1:]))
            sim_data['bola_basic_v1']['lens'].append(len(bola1_rebuffs[idx]) * 2.002 + sim_data['bola_basic_v1']['rebuffs'][-1])
            sim_data['bola_basic_v2']['lens'].append(len(bola2_rebuffs[idx]) * 2.002 + sim_data['bola_basic_v2']['rebuffs'][-1])
            sim_data['linear_bba']['lens'].append(len(bba_rebuffs[idx]) * 2.002 + sim_data['linear_bba']['rebuffs'][-1])
stall_path = f'{args.dir}subset_stall_dicts/{args.left_out_policy}'
os.makedirs(stall_path, exist_ok=True)

with open(f'{stall_path}/stalls_{C}.pkl', 'wb') as f:
    pickle.dump(sim_data, f)
