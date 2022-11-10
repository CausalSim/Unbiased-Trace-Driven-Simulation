from typing import Callable, Tuple, List, Dict
import pandas as pd
import numpy as np
import os.path
import os
from collections import Counter
import wget
import time
import datetime
from tqdm import tqdm
import json
import istarmap
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser(description='Auto experiment launch')
parser.add_argument('--dir', type=str, required=True, help='Puffer dataset directory')

args = parser.parse_args()

fmt_list = ['426x240-26', '640x360-26', '640x360-24', '854x480-26', '854x480-24', '854x480-22', '1280x720-26',
            '1280x720-24', '1280x720-22', '1920x1080-24', '1280x720-20', '1920x1080-22']
fmt_map = {fmt: i for i, fmt in enumerate(fmt_list)}
num_fmts = len(fmt_list)

one_day = datetime.timedelta(days=1)


def puffer_to_traj(df: pd.DataFrame) -> np.ndarray:
    # This function extracts a trajectory from a dataframe with all file sizes
    # Buffer, Rebuffer,
    # Download time, Chosen file size, Chosen file SSIM, Time, Timestamp,
    # Delivery rate, cwnd, in_flight, min_rtt, rtt
    # all file sizes, all ssims

    # There are 4 empty columns that used to be approximations of buffer and rebuffer, but they are no longer needed.
    # Their space is kept empty for consistency
    traj = np.zeros((df.shape[0], 16 + num_fmts + num_fmts + 1))
    traj[:, 0] = df['buffer']
    traj[:, 1] = df['cum_rebuf']
    traj[:, 6] = df['transmition_time (ns)'] / 1e9
    traj[:, 7] = df['size']
    traj[:, 8] = df['ssim_index']
    traj[:, 9] = df['time (ns GMT)_x'] / 1e9
    traj[:, 9] -= traj[0, 9]
    traj[:, 10] = df['video_ts']
    traj[:, 11] = df['delivery_rate']
    traj[:, 12] = df['cwnd']
    traj[:, 13] = df['in_flight']
    traj[:, 14] = df['min_rtt']
    traj[:, 15] = df['rtt']
    for index in range(num_fmts):
        traj[:, 16 + index] = df['size_%d' % index]
    for index in range(num_fmts):
        traj[:, 16 + num_fmts + index] = df['ssim_index_%d' % index]
    nan_size = np.isnan(traj[:, 16:16 + num_fmts])
    nan_ssim = np.isnan(traj[:, 16 + num_fmts:16 + num_fmts * 2])
    nan_total = np.logical_or(nan_size, nan_ssim)
    traj[:, 16 + num_fmts * 2] = num_fmts - np.count_nonzero(nan_total, axis=1)
    return traj


def client_traj(df: pd.DataFrame, df_server: pd.DataFrame) -> np.ndarray:
    # This function extracts a trajectory from a client dataframe
    # Buffer, Rebuffer, Time
    traj = np.zeros((df.shape[0], 1 + 1 + 1))
    traj[:, 0] = df['buffer']
    traj[:, 1] = df['cum_rebuf']
    traj[:, 2] = df['time (ns GMT)'] / 1e9
    traj[:, 2] -= df_server['time (ns GMT)_x'].iloc[0] / 1e9
    return traj


def add_sizes(df: pd.DataFrame, size_data_grp: pd.core.groupby.generic.DataFrameGroupBy,
              ssim_data_grp: pd.core.groupby.generic.DataFrameGroupBy) -> pd.DataFrame:
    start_time = df['time (ns GMT)_x'].iloc[0]
    assert start_time == df['time (ns GMT)_x'].min()
    first_grp = size_data_grp.get_group(df['format'].iloc[0])
    size_times = first_grp.loc[(first_grp['video_ts'] == df['video_ts'].iloc[0]) & (
                first_grp['channel'] == df['channel'].iloc[0]), 'time (ns GMT)']
    size_times = size_times[size_times < start_time]
    assert len(size_times) != 0
    begin_time = size_times.max() - 1e10
    end_time = df['time (ns GMT)_x'].iloc[-1]
    assert end_time == df['time (ns GMT)_x'].max()
    last_grp = size_data_grp.get_group(df['format'].iloc[-1])
    size_times = last_grp.loc[(last_grp['video_ts'] == df['video_ts'].iloc[-1]) & (
                last_grp['channel'] == df['channel'].iloc[-1]), 'time (ns GMT)']
    size_times = size_times[size_times < end_time]
    finish_time = size_times.max() + 1e10
    # This function adds all file sizes to a dataframe
    # Can be applied to the entire dataset to speed up computations
    for index, fmt in enumerate(fmt_list):
        size_fmt_limit = size_data_grp.get_group(fmt)[['size', 'video_ts', 'channel', 'time (ns GMT)']]
        size_fmt_limit = size_fmt_limit[size_fmt_limit['time (ns GMT)'].between(begin_time, finish_time,
                                                                                inclusive='both')]
        size_fmt_limit.drop(columns='time (ns GMT)', inplace=True)

        ssim_fmt_limit = ssim_data_grp.get_group(fmt)[['ssim_index', 'video_ts', 'channel', 'time (ns GMT)']]
        ssim_fmt_limit = ssim_fmt_limit[ssim_fmt_limit['time (ns GMT)'].between(begin_time, finish_time,
                                                                                inclusive='both')]
        ssim_fmt_limit.drop(columns='time (ns GMT)', inplace=True)

        df_before = df
        df = pd.merge(df, size_fmt_limit, how='left', on=['video_ts', 'channel'],
                      suffixes=("", "_%d" % index))
        df = pd.merge(df, ssim_fmt_limit, how='left', on=['video_ts', 'channel'],
                      suffixes=("", "_%d" % index))
        assert df.shape[0] == df_before.shape[0], "was %d, is %d now" % (df_before.shape[0], df.shape[0])
    return df


def fix_sizes(df: pd.DataFrame, sent_data: pd.DataFrame, print_stats: Callable = None) -> pd.DataFrame:
    fmt_identify = []
    for i in range(len(fmt_list)):
        if not df['size_%d' % i].isnull().values.all():
            fmt_identify.append(i)

    for index in fmt_identify:
        if df['size_%d' % index].isnull().values.any():
            nans = df[df['size_%d' % index].isna()]
            for k in range(nans.shape[0]):
                start_time = nans['time (ns GMT)_x'].iloc[k] - 1e11
                end_time = nans['time (ns GMT)_x'].iloc[k] + 1e10
                matching_sizes = sent_data[(sent_data['video_ts'] == nans.iloc[k]['video_ts']) & \
                                           (sent_data['channel'] == nans.iloc[k]['channel']) & \
                                           (sent_data['format'] == fmt_list[index])]
                matching_sizes = matching_sizes[(matching_sizes['time (ns GMT)'] >= start_time) & \
                                                (matching_sizes['time (ns GMT)'] <= end_time)]
                if len(matching_sizes) > 0:
                    df.loc[nans.index[k], 'size_%d' % index] = matching_sizes['size'].iloc[0]
                    if print_stats:
                        print_stats('Filled a missing value')
                    assert len(np.unique(matching_sizes['size'])) == 1
        if df['ssim_index_%d' % index].isnull().values.any():
            nans = df[df['size_%d' % index].isna()]
            for k in range(nans.shape[0]):
                start_time = nans['time (ns GMT)_x'].iloc[k] - 1e11
                end_time = nans['time (ns GMT)_x'].iloc[k] + 1e10
                matching_sizes = sent_data[(sent_data['video_ts'] == nans.iloc[k]['video_ts']) & \
                                           (sent_data['channel'] == nans.iloc[k]['channel']) & \
                                           (sent_data['format'] == fmt_list[index])]
                matching_sizes = matching_sizes[(matching_sizes['time (ns GMT)'] >= start_time) & \
                                                (matching_sizes['time (ns GMT)'] <= end_time)]
                if len(matching_sizes) > 0:
                    df.loc[nans.index[k], 'ssim_index_%d' % index] = matching_sizes['ssim_index'].iloc[0]
                    if print_stats:
                        print_stats('Filled a missing value\n')
                    assert len(np.unique(matching_sizes['size'])) == 1
    return df


def download_data(today: datetime.datetime, path_orig: str) -> Tuple[str, str]:
    tomorrow = today + one_day
    yesterday = today - one_day

    period = "%d-%02d-%02dT11_%d-%02d-%02dT11" % (today.year, today.month, today.day,
                                                  tomorrow.year, tomorrow.month, tomorrow.day)
    period_yesterday = "%d-%02d-%02dT11_%d-%02d-%02dT11" % (yesterday.year, yesterday.month, yesterday.day,
                                                            today.year, today.month, today.day)

    # Download Data
    acked_path = 'video_acked_%s.csv' % period
    if not os.path.exists(f"{path_orig}/" + acked_path):
        acked_url = 'https://storage.googleapis.com/puffer-data-release/%s/%s' % (period, acked_path)
        wget.download(acked_url, f"{path_orig}/" + acked_path, bar=None)

    sent_path = 'video_sent_%s.csv' % period
    if not os.path.exists(f"{path_orig}/" + sent_path):
        sent_url = 'https://storage.googleapis.com/puffer-data-release/%s/%s' % (period, sent_path)
        wget.download(sent_url, f"{path_orig}/" + sent_path, bar=None)

    buf_sent_path = 'buf_video_sent_%s.csv' % period
    if not os.path.exists(f"{path_orig}/" + buf_sent_path):
        buf_sent_url = 'https://storage.googleapis.com/puffer-data-release/buf_video_sent/%s' % (buf_sent_path)
        wget.download(buf_sent_url, f"{path_orig}/" + buf_sent_path, bar=None)

    size_path = 'video_size_%s.csv' % period
    if not os.path.exists(f"{path_orig}/" + size_path):
        size_url = 'https://storage.googleapis.com/puffer-data-release/%s/%s' % (period, size_path)
        wget.download(size_url, f"{path_orig}/" + size_path, bar=None)

    size_path_yesterday = 'video_size_%s_%s.csv' % (period, period_yesterday)
    size_url_yesterday = 'video_size_%s.csv' % period_yesterday
    if not os.path.exists(f"{path_orig}/" + size_path_yesterday):
        size_url_yesterday = 'https://storage.googleapis.com/puffer-data-release/%s/%s' % (
        period_yesterday, size_url_yesterday)
        wget.download(size_url_yesterday, f"{path_orig}/" + size_path_yesterday, bar=None)

    ssim_path = 'ssim_%s.csv' % period
    if not os.path.exists(f"{path_orig}/" + ssim_path):
        ssim_url = 'https://storage.googleapis.com/puffer-data-release/%s/%s' % (period, ssim_path)
        wget.download(ssim_url, f"{path_orig}/" + ssim_path, bar=None)

    ssim_path_yesterday = 'ssim_%s_%s.csv' % (period, period_yesterday)
    ssim_url_yesterday = 'ssim_%s.csv' % period_yesterday
    if not os.path.exists(f"{path_orig}/" + ssim_path_yesterday):
        ssim_url_yesterday = 'https://storage.googleapis.com/puffer-data-release/%s/%s' % (
        period_yesterday, ssim_url_yesterday)
        wget.download(ssim_url_yesterday, f"{path_orig}/" + ssim_path_yesterday, bar=None)

    client_path = 'client_buffer_%s.csv' % period
    if not os.path.exists(f"{path_orig}/" + client_path):
        client_url = 'https://storage.googleapis.com/puffer-data-release/%s/%s' % (period, client_path)
        wget.download(client_url, f"{path_orig}/" + client_path, bar=None)

    return period, period_yesterday


def remove_data(today: datetime.datetime, path_orig: str):
    tomorrow = today + one_day
    yesterday = today - one_day

    period = "%d-%02d-%02dT11_%d-%02d-%02dT11" % (today.year, today.month, today.day,
                                                  tomorrow.year, tomorrow.month, tomorrow.day)
    period_yesterday = "%d-%02d-%02dT11_%d-%02d-%02dT11" % (yesterday.year, yesterday.month, yesterday.day,
                                                            today.year, today.month, today.day)

    # Remove Data
    os.remove(f"{path_orig}/video_acked_%s.csv" % period)
    os.remove(f"{path_orig}/video_sent_%s.csv" % period)
    os.remove(f"{path_orig}/buf_video_sent_%s.csv" % period)

    os.remove(f"{path_orig}/video_size_%s.csv" % period)
    os.remove(f"{path_orig}/video_size_%s_%s.csv" % (period, period_yesterday))
    os.remove(f"{path_orig}/ssim_%s.csv" % period)
    os.remove(f"{path_orig}/ssim_%s_%s.csv" % (period, period_yesterday))

    os.remove(f"{path_orig}/client_buffer_%s.csv" % period)


def sort_merge_by_common_keys(filtered_sent_data: pd.DataFrame, filtered_acked_data: pd.DataFrame,
                              client_data: pd.DataFrame,
                              print_stats: Callable = None) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    # Group filtered data
    grouped_filtered_sent_data = filtered_sent_data.groupby(['session_id', 'index'])
    grouped_filtered_acked_data = filtered_acked_data.groupby(['session_id', 'index'])
    grouped_client_data = client_data.groupby(['session_id', 'index'])

    # Find common groups present in both and remove streams that didn't start from zero
    common_keys = set(grouped_filtered_sent_data.groups).intersection(grouped_filtered_acked_data.groups)
    if print_stats:
        print_stats(f"{len(common_keys)} present in both ack and sent streams" + '\n')
    start_events = [grouped_client_data.get_group(key).sort_values('time (ns GMT)')['buffer'].iloc[0] for key in
                    common_keys]
    if print_stats:
        print_stats(str(Counter(start_events).keys()) + '\n')
        print_stats(str(Counter(start_events).values()) + '\n')
    start_events = [grouped_client_data.get_group(key).sort_values('time (ns GMT)')['event'].iloc[0] for key in
                    common_keys]
    if print_stats:
        print_stats(str(Counter(start_events).keys()) + '\n')
        print_stats(str(Counter(start_events).values()) + '\n')
    common_keys = [key for key in common_keys if
                   grouped_client_data.get_group(key).sort_values('time (ns GMT)')['event'].iloc[0] == 'init']
    if print_stats:
        print_stats(f"{len(common_keys)} that start from init" + '\n')

    # Split filtered data into a list of data frames for each stream
    stream_sent_data_list = [grouped_filtered_sent_data.get_group(x).sort_values('time (ns GMT)') for x in common_keys]
    stream_acked_data_list = [grouped_filtered_acked_data.get_group(x).sort_values('time (ns GMT)') for x in
                              common_keys]
    stream_client_data_list = [grouped_client_data.get_group(x).sort_values('time (ns GMT)') for x in common_keys]

    merged_data_list = [pd.merge(stream_sent_data_list[i],
                                 stream_acked_data_list[i].drop(columns=['session_id', 'index', 'expt_id', 'channel']),
                                 on='video_ts') for i in range(len(stream_sent_data_list))]

    to_remove = []
    for i, merged_stream in enumerate(merged_data_list):
        merged_stream = merged_stream.sort_values('video_ts')
        diff_vts = merged_stream['video_ts'].to_numpy()[1:] - merged_stream['video_ts'].to_numpy()[:-1]
        if np.any(np.abs(diff_vts - 180180) > 1e-3):
            a = diff_vts
            to_remove.append(i)
        if len(merged_stream) == 0 or np.abs(merged_stream['buffer'].iloc[0]) > 1e-5 or np.abs(
                merged_stream['cum_rebuf'].iloc[0]) > 1e-5:
            to_remove.append(i)

    if print_stats:
        print_stats(f"Must remove {len(to_remove)} keys that have non-uniform video timestamp" + '\n')

    merged_data_list = [merged_data_list[i] for i in range(len(merged_data_list)) if i not in to_remove]
    stream_client_data_list = [stream_client_data_list[i] for i in range(len(stream_client_data_list)) if
                               i not in to_remove]

    return merged_data_list, stream_client_data_list


def add_transmission_c_hat(merged_data: pd.DataFrame):
    merged_data['transmition_time (ns)'] = merged_data['time (ns GMT)_y'] - merged_data['time (ns GMT)_x']
    merged_data['chat'] = merged_data['size'] / merged_data['transmition_time (ns)'] * 1e9
    assert merged_data['buffer'].min() >= 0
    assert merged_data['cum_rebuf'].min() >= 0


def get_mapper(path_orig: str) -> Dict[int, str]:
    path_set = f"{path_orig}/expt_settings.log"
    url_set = 'https://storage.googleapis.com/puffer-data-release/2022-09-19T11_2022-09-20T11/logs/expt_settings'
    if os.path.exists(path_set):
        os.remove(path_set)
    wget.download(url_set, path_set)
    print()

    mapper = {}
    with open(path_set, "r") as f:
        for line in f.readlines():
            pieces = line.split(" ", 1)
            policy_settings = json.loads(pieces[1])
            if 'abr_name' in policy_settings:
                mapper[int(pieces[0])] = policy_settings['abr_name']
            elif 'abr' in policy_settings:
                mapper[int(pieces[0])] = policy_settings['abr']
            else:
                print('Ignoring setting %s' % pieces[0])

    return mapper


def get_all_length_day(today: datetime.datetime, mapper: Dict[int, str], path_cooked: str, path_orig: str):
    t0 = time.time()

    period, period_yesterday = download_data(today, path_orig)

    file_log = open(f'{path_cooked}/%d-%02d-%02d_extraction.log' % (today.year, today.month, today.day), 'w')

    # Load ACK and SEND logs
    acked_data = pd.read_csv(f"{path_orig}/video_acked_%s.csv" % period)
    sent_data = pd.read_csv(f"{path_orig}/video_sent_%s.csv" % period)

    # If either log is empty, discard
    if len(sent_data) == 0 or len(acked_data) == 0:
        remove_data(today, path_orig)
        file_log.write('Empty file!!!\n')
        file_log.write('%s took %d seconds.\n' % (period, time.time() - t0))
        file_log.close()
        print("Empty file!!!")
        print('%s took %d seconds.' % (period, time.time() - t0))
        return

    # Append buffer and cum_buffer to ACK-SEND
    buf_sent_data = pd.read_csv(f"{path_orig}/buf_video_sent_%s.csv" % period)
    assert len(sent_data) == len(buf_sent_data)
    sent_data = pd.concat((sent_data, buf_sent_data), axis=1)
    assert len(sent_data) == len(buf_sent_data)
    del buf_sent_data
    assert 'buffer' in sent_data.columns
    assert 'cum_rebuf' in sent_data.columns

    # Load CLIENT logs
    client_data = pd.read_csv(f"{path_orig}/client_buffer_%s.csv" % period)

    # Remove Streams with average delivery rate below 6 Mbps
    filtered_sent_data = sent_data.groupby(['session_id', 'index'])
    file_log.write(f"{len(filtered_sent_data)} traces, cutoff is 6Mbps...\n")
    filtered_sent_data = filtered_sent_data.filter(lambda stream: stream['delivery_rate'].mean() <= 6000000 / 8)

    # Merge ACK, SEND and CLIENT logs by common keys
    merged_data_list, stream_client_data_list = sort_merge_by_common_keys(filtered_sent_data, acked_data,
                                                                          client_data, print_stats=file_log.write)

    # Sometimes cum_rebuffer is negative in logs, we follow puffer statistics codebase and discard them:
    # https://github.com/StanfordSNR/puffer-statistics/blob/c389402fea75173abf140ada4f568e017cf07e18/csv_to_stream_stats.cc#L517
    to_remove = []
    for i, merged_data in enumerate(merged_data_list):
        if merged_data['cum_rebuf'].min() < 0:
            to_remove.append(i)
            file_log.write(f"Session {merged_data['session_id'].iloc[0]} has negative cum_rebuf, removing...\n")
            print(f"Session {merged_data['session_id'].iloc[0]} has negative cum_rebuf, removing...")
    merged_data_list = [merged_data_list[i] for i in range(len(merged_data_list)) if i not in to_remove]
    stream_client_data_list = [stream_client_data_list[i] for i in range(len(stream_client_data_list)) if i not in
                               to_remove]
    file_log.write(f"{len(to_remove)} trajectories removed due to negative cum_rebuf...\n")

    # Add download time and observed bandwidth ({m_t}, trace)
    for merged_data in merged_data_list:
        add_transmission_c_hat(merged_data)

    # Collect size table, by concatenating today and yesterday SIZE logs
    sent_data = pd.read_csv(f"{path_orig}/video_sent_%s.csv" % period)
    size_data = pd.read_csv(f"{path_orig}/video_size_%s.csv" % period)
    size_data_yesterday = pd.read_csv(f"{path_orig}/video_size_%s_%s.csv" % (period, period_yesterday))
    size_data = pd.concat([size_data, size_data_yesterday])
    grouped_fmt_size_data = size_data.groupby('format')

    # Collect ssim table, by concatenating today and yesterday SSIM logs
    ssim_data = pd.read_csv(f"{path_orig}/ssim_%s.csv" % period)
    ssim_data_yesterday = pd.read_csv(f"{path_orig}/ssim_%s_%s.csv" % (period, period_yesterday))
    ssim_data = pd.concat([ssim_data, ssim_data_yesterday])
    grouped_fmt_ssim_data = ssim_data.groupby('format')

    # If tables are empty, discard this day
    if len(ssim_data) == 0 or len(size_data) == 0:
        remove_data(today, path_orig)
        file_log.write('Empty file!!!\n')
        file_log.write('%s took %d seconds.\n' % (period, time.time() - t0))
        file_log.close()
        print("Empty size file!!!")
        print('%s took %d seconds.' % (period, time.time() - t0))
        return

    # Use SENT logs to fill missing data with chosen chunks in SENT logs
    size_sent_data = sent_data.groupby(['format', 'channel', 'video_ts', 'size', 'ssim_index']).first().reset_index()
    to_remove = []
    for i, x in enumerate(merged_data_list):
        try:
            y = add_sizes(x, grouped_fmt_size_data, grouped_fmt_ssim_data)
            merged_data_list[i] = fix_sizes(y, size_sent_data)
        except:
            file_log.write(f"No sizes for trajectory {i}, removing...\n")
            print(f"No sizes for trajectory {i}, removing...")
            to_remove.append(i)

    cfil_df_list = [merged_data_list[i] for i in range(len(merged_data_list)) if i not in to_remove]
    cfil_client_list = [stream_client_data_list[i] for i in range(len(stream_client_data_list)) if i not in to_remove]
    file_log.write(f"{len(to_remove)} trajctories removed...\n")

    # If no stream is left after all this processing, discard the day.
    if len(cfil_df_list) == 0:
        remove_data(today, path_orig)
        file_log.write('No trajectories left!!!\n')
        file_log.write('%s took %d seconds.\n' % (period, time.time() - t0))
        file_log.close()
        print("No trajectories left!!!")
        print('%s took %d seconds.' % (period, time.time() - t0))
        return

    trajs = np.array([puffer_to_traj(x) for x in cfil_df_list], dtype=object)

    # Add other sizes
    to_remove = []
    for i, x in enumerate(trajs):
        if np.any(x[:, -1] == 0):
            file_log.write(f"No sizes in trajectory {i} for {np.sum(x[:, -1] == 0)} chunks, removing...\n")
            print(f"No sizes in trajectory {i} for {np.sum(x[:, -1] == 0)} chunks, removing...")
            to_remove.append(i)

    trajs = np.array([trajs[i] for i in range(len(trajs)) if i not in to_remove], dtype=object)
    cfil_df_list = [cfil_df_list[i] for i in range(len(cfil_df_list)) if i not in to_remove]
    cfil_client_list = [cfil_client_list[i] for i in range(len(cfil_client_list)) if i not in to_remove]

    # Save the IDs and indices for each stream
    passed_keys = [(x['session_id'].iloc[0], x['index'].iloc[0]) for x in cfil_df_list]
    num_valids = np.concatenate([x[:, -1] for x in trajs])
    file_log.write(f'Least valids were {np.min(num_valids)}, most valids were {np.max(num_valids)}, '
                   f'average valids were {np.mean(num_valids)}\n')
    ctrajs = np.array([client_traj(x, y) for x, y in zip(cfil_client_list, cfil_df_list)], dtype=object)
    ctrajs_events = np.array([x['event'] for x in cfil_client_list], dtype=object)
    # Save the policy tag for each stream
    ids = [merged_data['expt_id'].iloc[-1] for merged_data in cfil_df_list]
    ids_translated = [mapper[name] for name in ids]

    assert len(trajs) == len(passed_keys)
    assert len(trajs) == len(ctrajs)
    assert len(trajs) == len(ctrajs_events)
    assert len(trajs) == len(ids)

    np.save(f'{path_cooked}/%d-%02d-%02d_keys_pre.npy' % (today.year, today.month, today.day), passed_keys)
    np.save(f'{path_cooked}/%d-%02d-%02d_trajs.npy' % (today.year, today.month, today.day), trajs)
    np.save(f'{path_cooked}/%d-%02d-%02d_ctrajs.npy' % (today.year, today.month, today.day), ctrajs)
    np.save(f'{path_cooked}/%d-%02d-%02d_ctrajs_events.npy' % (today.year, today.month, today.day), ctrajs_events)
    np.save(f'{path_cooked}/%d-%02d-%02d_ids_translated.npy' % (today.year, today.month, today.day), ids_translated)

    remove_data(today, path_orig)

    file_log.write('%s took %d seconds.\n' % (period, time.time() - t0))
    file_log.close()
    print('%s took %d seconds.' % (period, time.time() - t0))


def get_extent_day(today: datetime.datetime, path_cooked: str):
    s_tr = np.load(f'{path_cooked}/%d-%02d-%02d_trajs.npy' % (today.year, today.month, today.day), allow_pickle=True)
    c_tr = np.load(f'{path_cooked}/%d-%02d-%02d_ctrajs.npy' % (today.year, today.month, today.day), allow_pickle=True)
    c_e = np.load(f'{path_cooked}/%d-%02d-%02d_ctrajs_events.npy' % (today.year, today.month, today.day),
                  allow_pickle=True)
    tr_ext = []

    for tr, ev, ser_tr in zip(c_tr, c_e, s_tr):
        ev = ev.to_numpy()

        t_last_ev = None
        t_last_play = None
        t_startup = None
        t_last_slow = None
        last_buf = None
        startup_cum_rebuf = None
        last_cum_rebuf = None
        last_play_rebuf = None
        playing = False
        entry = None
        for i in range(len(tr)):
            if t_last_ev is not None and tr[i, 2] - t_last_ev > 8:
                entry = [None, i, t_last_ev, 1, None, None, None]
                break

            if tr[i, 0] > 0.3:
                t_last_slow = None
            else:
                if t_last_slow is None:
                    t_last_slow = tr[i, 2]

            if t_last_slow is not None and tr[i, 2] - t_last_slow > 20:
                entry = [None, i, t_last_ev, 2, None, None, None]
                break

            if tr[i, 0] > 5 and last_buf is not None and last_buf > 5 and last_cum_rebuf is not None and tr[
                i, 1] > last_cum_rebuf + 0.15:
                entry = [None, -1, -1, 3, None, None, None]
                break

            if ev[i] == 'startup':
                if t_startup is None:
                    t_startup = tr[i, 2]
                    startup_cum_rebuf = tr[i, 1]
            if ev[i] == 'rebuffer':
                playing = False
            if ev[i] in ['startup', 'play']:
                t_last_play = tr[i, 2]
                playing = True
                last_play_rebuf = tr[i, 1]
            if ev[i] == 'timer':
                if playing:
                    t_last_play = tr[i, 2]
                    last_play_rebuf = tr[i, 1]

            t_last_ev = tr[i, 2]
            last_buf = tr[i, 0]
            last_cum_rebuf = tr[i, 1]

        if t_startup is None:
            entry = [None, -1, -1, 4, None, None, None]
        elif t_last_play <= t_startup:
            entry = [None, -1, -1, 5, None, None, None]
        elif last_cum_rebuf < startup_cum_rebuf:
            entry = [None, -1, -1, 6, None, None, None]
        elif entry is None:
            entry = [None, len(tr), t_last_ev, 0, None, None, None]

        if entry[1] == -1:
            entry[0] = 0
        elif entry[3] == 0:
            entry[0] = len(ser_tr)
            entry[4] = t_last_play - t_startup
            entry[5] = startup_cum_rebuf
            entry[6] = last_play_rebuf
        else:
            entry[0] = (ser_tr[:, 9] <= entry[2]).sum()
            entry[4] = t_last_play - t_startup
            entry[5] = startup_cum_rebuf
            entry[6] = last_play_rebuf

        tr_ext.append(entry)

    np.save(f'{path_cooked}/%d-%02d-%02d_ext.npy' % (today.year, today.month, today.day), tr_ext)


def apply_extent(today: datetime.datetime, path_cooked: str):
    date_string = "%d-%02d-%02d" % (today.year, today.month, today.day)
    trajs = np.load(f"{path_cooked}/{date_string}_trajs.npy", allow_pickle=True)
    trajs_ids = np.load(f"{path_cooked}/{date_string}_ids_translated.npy", allow_pickle=True)
    trajs_keys = np.load(f"{path_cooked}/{date_string}_keys_pre.npy", allow_pickle=True)
    exts = np.load(f"{path_cooked}/{date_string}_ext.npy", allow_pickle=True)
    trajs_new = []
    ids_new = []
    keys_new = []

    assert len(trajs) == len(exts)

    for traj, traj_id, traj_key, extent in zip(trajs, trajs_ids, trajs_keys, exts):
        ind_max = extent[0]
        traj_new = traj[:ind_max]
        ssim_arr = traj_new[:, 8]
        watch_time = extent[4]

        if ind_max == 0 or ind_max == 1 or np.all(ssim_arr > 0.99999) or np.any(
                traj_new[1:, 0] - traj_new[:-1, 0] > 2.003):
            continue

        if watch_time < (1 << 2):
            continue

        trajs_new.append(traj_new)
        ids_new.append(traj_id)
        keys_new.append(traj_key)

    np.save(f"{path_cooked}/{date_string}_trajs.npy", trajs_new)
    np.save(f"{path_cooked}/{date_string}_ids_translated.npy", ids_new)
    np.save(f"{path_cooked}/{date_string}_keys.npy", keys_new)


def main():
    os.makedirs(f"{args.dir}/orig/", exist_ok=True)
    os.makedirs(f"{args.dir}/cooked/", exist_ok=True)
    mapper = get_mapper(f"{args.dir}/orig/")

    start_date = datetime.date(2020, 7, 27)
    end_date = datetime.date(2021, 6, 1)
    all_days = [start_date + datetime.timedelta(days=x) for x in range((end_date-start_date).days+1)]

    with Pool(32) as pool:
        pool_args = [(day, mapper, f"{args.dir}/cooked/", f"{args.dir}/orig/") for day in all_days]
        for _ in tqdm(pool.istarmap(get_all_length_day, pool_args), total=len(pool_args)):
            pass

        pool_args = [(day, f"{args.dir}/cooked/") for day in all_days]
        for _ in tqdm(pool.istarmap(get_extent_day, pool_args), total=len(pool_args)):
            pass

        for _ in tqdm(pool.istarmap(apply_extent, pool_args), total=len(pool_args)):
            pass


if __name__ == '__main__':
    main()
