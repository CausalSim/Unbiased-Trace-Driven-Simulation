import numpy as np
import matplotlib.pyplot as plt
import argparse
from env.abr import ssim_db


# Adapted from https://oco-carbon.com/metrics/find-pareto-frontiers-in-python/
def pareto_frontier(x_s, y_s, map_x=False, map_y=True):
    my_list = sorted([[x_s[i], y_s[i]] for i in range(len(x_s))], reverse=map_x)
    p_front = [my_list[0]]
    for pair in my_list[1:]:
        if map_y:
            if pair[1] >= p_front[-1][1]:
                p_front.append(pair)
        else:
            if pair[1] <= p_front[-1][1]:
                p_front.append(pair)
    rem = []
    for start in p_front:
        for end in p_front:
            for mid in p_front:
                if start != end and start != mid and end != mid:
                    if start[0] < mid[0] < end[0] and \
                            (mid[0]-start[0])/(end[0]-start[0])*(end[1]-start[1])+start[1] > mid[1]:
                        if mid not in rem:
                            rem.append(mid)
    for pair in rem:
        p_front.remove(pair)
    p_front_x = [pair[0] for pair in p_front]
    p_front_y = [pair[1] for pair in p_front]
    return np.array(p_front_x), np.array(p_front_y)


def main():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dir', type=str, required=True, help='Output folder')
    parser.add_argument('--sim_mode', type=str, required=True, choices=['causalsim', 'expertsim'], help='Simulator')
    parser.add_argument('--policies', type=str, nargs='+', required=True, choices=['bba', 'bola1', 'bola2'],
                        help='Policies to plot')
    parser.add_argument('--annotate_frontier', action='store_true',
                        help='Annotate the parameters for the frontier as (cushion, reservoir)')
    config = parser.parse_args()

    plt.figure(figsize=(14, 10))

    dict_col = {
        'bba': 'C2',
        'bola1': 'C1',
        'bola2': 'C3',
    }

    for pol in config.policies:
        run_stats = np.load(f'{config.dir}/tests/gp_{pol}_{config.sim_mode}/run_stats.npy')
        print(f'There are {len(run_stats)} points')
        s_s = ssim_db(run_stats[:, [3]])
        r_s = run_stats[:, [4]]
        accept = np.logical_and(r_s < 0.07, s_s > 14)

        x, y = pareto_frontier(r_s[accept], s_s[accept])
        plt.scatter(r_s[accept] * 100, s_s[accept], marker='o', color=dict_col[pol], label=pol)
        plt.fill_between(x*100, y, s_s[accept].min(), color=dict_col[pol], alpha=0.1)
        plt.plot(x*100, y, color=dict_col[pol])

        if config.annotate_frontier:
            b, t = plt.gca().get_ylim()
            r, l = plt.gca().get_xlim()
            hd = (t-b)/100
            wd = (r-l)/100
            for i in range(len(x)):
                i_x = np.where((r_s == x[i]) & (s_s == y[i]))[0]
                assert len(i_x) == 1
                i_x = i_x[0]
                plt.annotate(f'({run_stats[i_x, 0]:.1f}, {run_stats[i_x, 1]:.1f})', xy=(x[i]*100+wd, y[i]+hd))

    plt.legend()
    plt.grid()
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel(r'Time Spent Stalled (%)')
    plt.ylabel(r'Average SSIM (dB)')
    ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(f'{config.dir}/pareto_{config.sim_mode}_{"_".join(config.policies)}.pdf', format='pdf')


if __name__ == '__main__':
    main()
