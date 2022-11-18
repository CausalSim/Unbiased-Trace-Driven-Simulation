import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="root directory")
args = parser.parse_args()

policy_names = ['bola_basic_v2', 'bola_basic_v1', 'puffer_ttp_cl', 'puffer_ttp_20190202', 'linear_bba']
buffer_based_names = ['bola_basic_v2', 'bola_basic_v1', 'linear_bba']
sl_EMDs, expert_EMDs, sim_EMDs = [], [], []
with open(f'{args.dir}tuned_hyperparams/buffer.pkl', 'rb') as f:
    f = pickle.load(f)
    bf_C = {policy: f[policy][1] for policy in buffer_based_names}
for left_out_policy in buffer_based_names:
    with open(f'{args.dir}subset_EMDs/{left_out_policy}/sim_buff_{bf_C[left_out_policy]}.pkl', 'rb') as f:
        sim_dict = pickle.load(f)
    with open(f'{args.dir}subset_EMDs/{left_out_policy}/expert_buff_{bf_C[left_out_policy]}.pkl', 'rb') as f:
        expert_dict = pickle.load(f)
    with open(f'{args.dir}subset_EMDs/{left_out_policy}/sl_buff_{bf_C[left_out_policy]}.pkl', 'rb') as f:
        sl_dict = pickle.load(f)
    sl_EMDs.extend([sl_dict[source][left_out_policy] for source in policy_names if source != left_out_policy])
    expert_EMDs.extend([expert_dict[source][left_out_policy] for source in policy_names if source != left_out_policy])
    sim_EMDs.extend([sim_dict[source][left_out_policy] for source in policy_names if source != left_out_policy])
plt.figure(figsize=(3.25, 2.25))
sl_EMDs = np.sort(sl_EMDs)
expert_EMDs = np.sort(expert_EMDs)
sim_EMDs = np.sort(sim_EMDs)
plt.plot(sl_EMDs, [i/12*100 for i in range(1, 13)], label='SLSim')
plt.plot(expert_EMDs, [i/12*100 for i in range(1, 13)], label='ExpertSim')
plt.plot(sim_EMDs, [i/12*100 for i in range(1, 13)], label='CausalSim')
plt.legend()
plt.ylabel('CDF %')
plt.xlabel('EMD')

fig_path = f'{args.dir}plots'
os.makedirs(fig_path, exist_ok=True)
plt.savefig(f'{fig_path}/fig7a.pdf', format='pdf')