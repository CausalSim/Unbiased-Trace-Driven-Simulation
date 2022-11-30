import datetime
import itertools
import os
import shutil

import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

from env.abr import ssim_db

parser = ArgumentParser(description='Puffer RL dataset parameters')
parser.add_argument('--dir', type=str, required=True, help='Puffer trace path')
parser.add_argument('--buf_latent_dir', type=str, required=True, help='Path to load latent buffers from')
parser.add_argument('--dt_latent_dir', type=str, required=True, help='Path to load download time latents from')
config = parser.parse_args()


def main():
    # Each traces is a numpy array with the following:
    #         1) Buffer hidden feature,
    #         2) DT hidden feature,
    #         3) factual throughput,
    #         4) time between actions,
    #         5) min rtt,
    #         6-17) bitrate choices,
    #         18-29) ssim of choices in db

    os.makedirs(f"{config.dir}/gp_cooked/", exist_ok=True)

    all_ts = 0
    all_wts = []

    # 235298 traces
    start_date = datetime.date(2020, 7, 27)
    end_date = datetime.date(2021, 6, 1)
    all_days = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]

    for today in tqdm(all_days):
        all_traces = []

        date_string = "%d-%02d-%02d" % (today.year, today.month, today.day)
        trajs = np.load(f"{config.dir}/cooked/{date_string}_trajs.npy", allow_pickle=True)
        lat_buf = np.load(f"{config.buf_latent_dir}/{date_string}_features.npy", allow_pickle=True)
        lat_dt = np.load(f"{config.dt_latent_dir}/{date_string}_features.npy", allow_pickle=True)

        assert len(trajs) == len(lat_buf)
        assert len(trajs) == len(lat_dt)

        for traj, lat_buf_traj, lat_dt_traj in zip(trajs, lat_buf, lat_dt):
            assert len(lat_buf_traj) == len(lat_dt_traj)
            assert len(traj) > 1
            buf_lat_traj = np.array(lat_buf_traj).squeeze()
            dt_lat_traj = np.array(lat_dt_traj).squeeze()
            assert buf_lat_traj.shape == dt_lat_traj.shape
            fact_thr_traj = traj[:-1, 7] / traj[:-1, 6]
            act_inter_times_traj = traj[1:, 9]
            min_rtt_traj = traj[:-1, 14] / 1000
            size_s = traj[:-1, 16:28]
            ssim_s = ssim_db(traj[:-1, 28:40])
            all_traces.append(np.c_[buf_lat_traj, dt_lat_traj, fact_thr_traj, act_inter_times_traj, min_rtt_traj,
                                    size_s, ssim_s])
            all_ts += 1

            watch_time_no_stall = 2.002 * len(traj) - traj[-1, 0]
            assert watch_time_no_stall > 0
            all_wts.append(watch_time_no_stall)

        np.save(f"{config.dir}/gp_cooked/{date_string}_trc.npy", np.array(all_traces, dtype=object))

    np.save(f"{config.dir}/gp_cooked/wts.npy", np.array(all_wts))
    print(f"There were {all_ts} traces!!!")

    src_stats = f"{config.dir}/2020-07-27to2021-06-01_no_filter_data/"
    stats = ['mean', 'std']
    tags = [
        ('buffs', 'buffer'),
        ('chats', 'c_hat'),
        ('actions', 'chosen_chunk_size'),
        ('dts', 'download_time'),
        ('next_buffs', 'next_buffer'),
    ]

    for stat, (tag_src, tag_dst) in itertools.product(stats, tags):
        shutil.copyfile(f"{src_stats}/{tag_src}_{stat}.npy", f"{config.dir}/gp_cooked/{tag_dst}_{stat}.npy")

    print(f"Copied normalization statistics!!!")


if __name__ == '__main__':
    main()
