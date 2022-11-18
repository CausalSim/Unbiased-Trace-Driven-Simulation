import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
import os
import datetime
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="root directory")
args = parser.parse_args()

source_policies = ['bola_basic_v2', 'bola_basic_v1', 'puffer_ttp_cl', 'puffer_ttp_20190202', 'linear_bba']
target_policies = ['bola_basic_v2', 'bola_basic_v1', 'linear_bba']

mape_dict = {source: {target: {'diff': 0, 'number': 0, 'average': 0, 'mad': 0} for target in target_policies} for
             source in source_policies}
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
for target in target_policies:
    if target == 'linear_bba':
        name = 'linear_bba'
    elif target == 'bola_basic_v1':
        name = 'bola1'
    elif target == 'bola_basic_v2':
        name = 'bola2'
    PERIOD_TEXT = f'2020-07-27to2021-06-01_{target}'
    sl_path = f'{args.dir}{PERIOD_TEXT}_sl_cfs/cfs/model_10000'
    for today in tqdm(all_days):
        date_string = "%d-%02d-%02d" % (today.year, today.month, today.day)
        ids = np.load(f'{cooked_path}/{date_string}_ids_translated.npy', allow_pickle=True)
        orig_trajs = np.load(f'{cooked_path}/{date_string}_trajs.npy', allow_pickle=True)
        counterfactual_actions = np.load(f'{sl_path}/{date_string}_{name}_actions.npy', allow_pickle=True)
        for idx, action_trajectory in enumerate(counterfactual_actions):
            factual_actions = orig_trajs[idx][:-1, 7]
            assert len(action_trajectory) == len(factual_actions), f'{len(action_trajectory)}, {factual_actions.shape}'
            source = ids[idx]
            mape_dict[source][target]['number'] += len(action_trajectory)
            mape_dict[source][target]['diff'] += np.sum(
                np.abs((action_trajectory - factual_actions) / factual_actions * 100))
            mape_dict[source][target]['mad'] += np.sum(np.abs(action_trajectory - factual_actions))
for source in source_policies:
    for target in target_policies:
        mape_dict[source][target]['average'] = mape_dict[source][target]['diff'] / mape_dict[source][target]['number']
        mape_dict[source][target]['mad'] = mape_dict[source][target]['mad'] / mape_dict[source][target]['number']
        del mape_dict[source][target]['diff']
        del mape_dict[source][target]['number']
with open(f'{args.dir}tuned_hyperparams/buffer.pkl', 'rb') as f:
    f = pickle.load(f)
    Cs = [f[policy][1] for policy in target_policies]
data = {source: {target: {'sim_EMD': None, 'sl_EMD': None, 'expert_EMD': None, 'mape': None, 'mad': None} for target in
                 target_policies} for source in source_policies}
for index, left_out_policy in enumerate(target_policies):
    with open(f'{args.dir}subset_EMDs/{left_out_policy}/sim_buff_{Cs[index]}.pkl', 'rb') as f:
        sim_dict = pickle.load(f)
    with open(f'{args.dir}subset_EMDs/{left_out_policy}/sl_buff_{Cs[index]}.pkl', 'rb') as f:
        sl_dict = pickle.load(f)
    with open(f'{args.dir}subset_EMDs/{left_out_policy}/expert_buff_{Cs[index]}.pkl', 'rb') as f:
        expert_dict = pickle.load(f)
    for source in source_policies:
        data[source][left_out_policy]['sim_EMD'] = sim_dict[source][left_out_policy]
        data[source][left_out_policy]['sl_EMD'] = sl_dict[source][left_out_policy]
        data[source][left_out_policy]['expert_EMD'] = expert_dict[source][left_out_policy]
        data[source][left_out_policy]['mape'] = mape_dict[source][left_out_policy]['average']
        data[source][left_out_policy]['mad'] = mape_dict[source][left_out_policy]['mad'] * 8 / 1e6

plt.figure(figsize=(3.25, 2.25))
x = np.array(
    [data[source][left_out]['mad'] for source in source_policies for index, left_out in enumerate(target_policies)])
y = np.array(
    [data[source][left_out]['sim_EMD'] for source in source_policies for index, left_out in enumerate(target_policies)])
r = np.polyfit(x, y, deg=1)
plt.plot([np.min(x), np.max(x)], [r[0] * np.min(x) + r[1], r[0] * np.max(x) + r[1]], label='CausalSim', color='green')
y = np.array([data[source][left_out]['expert_EMD'] for source in source_policies for index, left_out in
              enumerate(target_policies)])
r = np.polyfit(x, y, deg=1)
plt.plot([np.min(x), np.max(x)], [r[0] * np.min(x) + r[1], r[0] * np.max(x) + r[1]], label='ExpertSim', color='blue')
y = np.array(
    [data[source][left_out]['sl_EMD'] for source in source_policies for index, left_out in enumerate(target_policies)])
r = np.polyfit(x, y, deg=1)
plt.plot([np.min(x), np.max(x)], [r[0] * np.min(x) + r[1], r[0] * np.max(x) + r[1]], label='SLSim', color='red')
plt.legend()
for source in source_policies:
    for index, left_out in enumerate(target_policies):
        plt.scatter(data[source][left_out]['mad'], data[source][left_out]['sim_EMD'], label='CausalSim', color='green',
                    marker='>')
        plt.scatter(data[source][left_out]['mad'], data[source][left_out]['expert_EMD'], label='ExpertSim',
                    color='blue', marker='s')
        plt.scatter(data[source][left_out]['mad'], data[source][left_out]['sl_EMD'], label='SLSim', color='red',
                    marker='o')

fig_path = f'{args.dir}plots'
os.makedirs(fig_path, exist_ok=True)
plt.savefig(f'{fig_path}/fig7b.pdf', format='pdf')