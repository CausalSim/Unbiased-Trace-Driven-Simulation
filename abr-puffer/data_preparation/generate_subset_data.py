import numpy as np
import datetime
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="root directory")
args = parser.parse_args()

policy_names = ['bola_basic_v2', 'bola_basic_v1', 'puffer_ttp_cl', 'puffer_ttp_20190202', 'linear_bba']
target_policy_names = ['bola_basic_v2', 'bola_basic_v1', 'linear_bba']


def convert_id_to_number(id, current_policies):
    return current_policies.index(id)


def whiten(raw_data):
    mean = np.mean(raw_data)
    std = np.std(raw_data)
    white_data = (raw_data - mean) / std
    return white_data, mean, std


def save(dir, white_data, mean, std, name):
    np.save(f'{dir}/white_{name}s.npy', white_data)
    np.save(f'{dir}/{name}s_mean.npy', mean)
    np.save(f'{dir}/{name}s_std.npy', std)


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
load_dir = f'{args.dir}cooked'
for left_out_policy in target_policy_names:
    buffers = []
    next_buffers = []
    chosen_chunk_sizes = []
    c_hats = []
    numbers = []
    download_times = []
    current_policies = policy_names.copy()
    current_policies.remove(left_out_policy)
    print('left_out: ', left_out_policy, 'training_data: ', current_policies)
    for today in tqdm(all_days):
        date_string = "%d-%02d-%02d" % (today.year, today.month, today.day)
        trajs = np.load(f"{load_dir}/{date_string}_trajs.npy", allow_pickle=True)
        ids = np.load(f"{load_dir}/{date_string}_ids_translated.npy", allow_pickle=True)
        for idx, traj in enumerate(trajs):
            policy_name = ids[idx]
            if policy_name != left_out_policy:
                number = convert_id_to_number(policy_name, current_policies)
                c_hats.extend(np.divide(traj[:-1, 7], traj[:-1, 6]))
                chosen_chunk_sizes.extend(traj[:-1, 7])
                buffers.extend(traj[:-1, 0])
                next_buffers.extend(traj[1:, 0])
                download_times.extend(traj[:-1, 6])
                numbers.extend([number for _ in traj[:-1, 0]])
    buffers = np.array(buffers)
    next_buffers = np.array(next_buffers)
    chosen_chunk_sizes = np.array(chosen_chunk_sizes)
    numbers = np.array(numbers)
    c_hats = np.array(c_hats)
    download_times = np.array(download_times)
    save_dir = f'{args.dir}subset_data/{left_out_policy}'
    os.makedirs(save_dir, exist_ok=True)

    white_buffs, buffs_mean, buffs_std = whiten(buffers)
    white_next_buffs, next_buffs_mean, next_buffs_std = whiten(next_buffers)
    white_chats, chats_mean, chats_std = whiten(c_hats)
    white_actions, actions_mean, actions_std = whiten(chosen_chunk_sizes)
    white_dts, dts_mean, dts_std = whiten(download_times)

    assert len(white_buffs) == len(white_next_buffs) == len(white_chats) == len(white_actions) == len(white_dts)

    save(dir=save_dir, white_data=white_buffs, mean=buffs_mean, std=buffs_std, name='buff')
    save(dir=save_dir, white_data=white_next_buffs, mean=next_buffs_mean, std=next_buffs_std, name='next_buff')
    save(dir=save_dir, white_data=white_chats, mean=chats_mean, std=chats_std, name='chat')
    save(dir=save_dir, white_data=white_actions, mean=actions_mean, std=actions_std, name='action')
    save(dir=save_dir, white_data=white_dts, mean=dts_mean, std=dts_std, name='dt')

    np.save(f'{save_dir}/policy_numbers.npy', np.array(numbers))
