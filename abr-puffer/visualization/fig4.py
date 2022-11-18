import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="root directory")
args = parser.parse_args()

plt.rcParams.update({
    "text.usetex": True,
    'legend.fontsize': 6,
    'font.family': 'serif',
    'font.serif': ['Times']
})
buffer_based_names = ['bola_basic_v2', 'bola_basic_v1', 'linear_bba']
color_dict = {'linear_bba': 'C2', 'bola_basic_v1': 'C1', 'bola_basic_v2': 'C3'}
marker_dict = {'expert': 'v', 'orig': 'o', 'sl': 's', 'causal': '*'}
with open(f'{args.dir}tuned_hyperparams/buffer.pkl', 'rb') as f:
    f = pickle.load(f)
    bf_C = {policy: f[policy][0] for policy in buffer_based_names}
with open(f'{args.dir}tuned_hyperparams/downloadtime.pkl', 'rb') as f:
    f = pickle.load(f)
    dt_C = {policy: f[policy][0] for policy in buffer_based_names}

plt.figure(figsize=(3.25, 2.25))
#Original
for policy in buffer_based_names:
    with open(f'{args.dir}subset_orig_ssim_dicts/{policy}/orig_ssims.pkl', 'rb') as f:
        orig_ssim = pickle.load(f)
    with open(f'{args.dir}subset_orig_rebuff_dicts/{policy}/orig_rebuffs.pkl', 'rb') as f:
        orig_stall = pickle.load(f)
    plt.scatter(100 * np.sum(orig_stall[policy]['rebuffs']) / np.sum(orig_stall[policy]['lens']), orig_ssim[policy], color=color_dict[policy], marker=marker_dict['orig'], label=f'orig_{policy}', s=25, zorder=500)
    print(f'orig_{policy}', 100 * np.sum(orig_stall[policy]['rebuffs']) / np.sum(orig_stall[policy]['lens']), orig_ssim[policy])
#Expert
for policy in buffer_based_names:
    with open(f'{args.dir}subset_expert_rebuff_dicts/{policy}/expert_rebuffs.pkl', 'rb') as f:
        expert_stall = pickle.load(f)
    with open(f'{args.dir}subset_expert_ssim_dicts/{policy}/expert_ssims.pkl', 'rb') as f:
        expert_ssim = pickle.load(f)
    plt.scatter(100 * np.sum(expert_stall[policy]['rebuffs']) / np.sum(expert_stall[policy]['lens']), expert_ssim[policy], color=color_dict[policy], marker=marker_dict['expert'], label=f'expert_{policy}', s=25, zorder=500)
    print(f'expert_{policy}', 100 * np.sum(expert_stall[policy]['rebuffs']) / np.sum(expert_stall[policy]['lens']), expert_ssim[policy])
#SL
for policy in buffer_based_names:
    with open(f'{args.dir}subset_sl_stall_dicts/{policy}/stalls.pkl', 'rb') as f:
        sl_stall = pickle.load(f)
    with open(f'{args.dir}subset_sl_ssim_dicts/{policy}/ssims.pkl', 'rb') as f:
        sl_ssim = pickle.load(f)
    plt.scatter(100 * np.sum(sl_stall[policy]['rebuffs']) / np.sum(sl_stall[policy]['lens']), sl_ssim[policy], color=color_dict[policy], marker=marker_dict['sl'], label=f'sl_{policy}', s=25, zorder=500)
    print(f'sl_{policy}', 100 * np.sum(sl_stall[policy]['rebuffs']) / np.sum(sl_stall[policy]['lens']), sl_ssim[policy])
#Causal
for policy in buffer_based_names:
    with open(f'{args.dir}subset_stall_dicts/{policy}/stalls_{dt_C[policy]}.pkl', 'rb') as f:
        causal_stall = pickle.load(f)
    with open(f'{args.dir}subset_ssim_dicts/{policy}/ssims_{bf_C[policy]}.pkl', 'rb') as f:
        causal_ssim = pickle.load(f)
    plt.scatter(100 * np.sum(causal_stall[policy]['rebuffs']) / np.sum(causal_stall[policy]['lens']), causal_ssim[policy], color=color_dict[policy], marker=marker_dict['causal'], label=f'causal_{policy}', s=35, zorder=500)
    print(f'causal_{policy}', 100 * np.sum(causal_stall[policy]['rebuffs']) / np.sum(causal_stall[policy]['lens']), causal_ssim[policy])
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color='w', markerfacecolor='w', markeredgecolor='k', marker='o'),
          Line2D([0], [0], color='w', markerfacecolor='w', markeredgecolor='k', marker='*', ms=8),
          Line2D([0], [0], color='w', markerfacecolor='w', markeredgecolor='k', marker='v'),
          Line2D([0], [0], color='w', markerfacecolor='w', markeredgecolor='k', marker='s'),]
plt.legend(handles, ['Ground Truth', 'CausalSim', 'ExpertSim', 'SLSim'], ncol=4, loc='lower left', bbox_to_anchor=(-0.066, 1.1, 1.1, 0.25), borderaxespad=0, borderpad=0.5, mode="expand", handletextpad=0.2)
plt.xlabel(r'Time Spent Stalled (\%)', fontsize=8)
plt.ylabel(r'Average SSIM (dB)', fontsize=8)
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.tight_layout()
plt.xlim([0.5, 9.1])
plt.ylim([14.81, 15.69])
plt.gca().invert_xaxis()
plt.grid(zorder=550)

fig_path = f'{args.dir}plots'
os.makedirs(fig_path, exist_ok=True)
plt.savefig(f'{fig_path}/fig4.pdf', format='pdf')