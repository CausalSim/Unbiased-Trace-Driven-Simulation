import numpy as np
import datetime
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="root directory")
parser.add_argument("--month", type=int, default=None)
parser.add_argument("--year", type=int, default=None)
args = parser.parse_args()


class LinearBBA(object):
    def __init__(self, ssim_table, size_table, lower=3, upper=13.5):
        self.lower = lower
        self.upper = upper
        assert ssim_table.shape == size_table.shape
        self.num_formats = ssim_table.shape[1]
        self.ssim_table = ssim_table
        self.size_table = size_table

    def select_video_format(self, index_curr, buffer, no_ssim=False):
        invalid_mask = np.logical_or(np.isnan(self.size_table[index_curr]),
                                     np.isnan(self.ssim_table[index_curr]))
        size_arr_valid = np.ma.array(self.size_table[index_curr], mask=invalid_mask)
        ssim_arr_valid = np.ma.array(self.ssim_table[index_curr], mask=invalid_mask)
        min_choice = size_arr_valid.argmin()
        max_choice = size_arr_valid.argmax()
        if buffer < self.lower:
            act = min_choice
        elif buffer >= self.upper:
            act = max_choice
        else:
            ratio = (buffer - self.lower) / float(self.upper - self.lower)
            min_chunk = size_arr_valid[min_choice]
            max_chunk = size_arr_valid[max_choice]
            bitrate = ratio * (max_chunk - min_chunk) + min_chunk
            mask = np.logical_or(invalid_mask, size_arr_valid > bitrate)
            if no_ssim:
                act = np.ma.array(size_arr_valid, mask=mask).argmax()
            else:
                act = np.ma.array(ssim_arr_valid, mask=mask).argmax()
        return act, self.size_table[index_curr][act], self.ssim_table[index_curr][act]


class BolaBasic(object):
    size_ladder_bytes = [44319, 93355, 115601, 142904, 196884, 263965, 353752, 494902, 632193, 889893]
    ssim_index_ladder = [0.91050748, 0.94062527, 0.94806355, 0.95498943, 0.96214503, 0.96717277, 0.97273958, 0.97689813,
                         0.98004106, 0.98332605]
    MIN_BUF_S = 3
    MAX_BUF_S = 15
    MIN_SSIM = 0
    MAX_SSIM = 60
    chunk_length = 2.002

    def __init__(self, version, ssim_table, size_table):
        """

        :param version:
        :type version: int
        :type ssim_table: np.ndarray
        :type size_table: np.ndarray
        """
        assert self.size_ladder_bytes[0] < self.size_ladder_bytes[1]
        assert self.ssim_index_ladder[0] < self.ssim_index_ladder[1]
        assert self.MIN_BUF_S < self.MAX_BUF_S
        assert version in [1, 2]
        self.version = version
        assert ssim_table.shape == size_table.shape
        self.num_formats = ssim_table.shape[1]
        self.ssim_table = ssim_table
        self.size_table = size_table

        smallest = {'size': self.size_ladder_bytes[0],
                    'utility': self.utility(self.ssim_index_ladder[0])}
        second_smallest = {'size': self.size_ladder_bytes[1],
                           'utility': self.utility(self.ssim_index_ladder[1])}
        largest = {'size': self.size_ladder_bytes[-1],
                   'utility': self.utility(self.ssim_index_ladder[-1])}

        size_delta = self.size_ladder_bytes[1] - self.size_ladder_bytes[0]
        if version == 1:
            utility_high = largest['utility']
        else:
            utility_high = self.utility(1)

        size_utility_term = second_smallest['size'] * smallest['utility'] - smallest['size'] * \
                            second_smallest['utility']
        gp_nominator = self.MAX_BUF_S * size_utility_term - utility_high * self.MIN_BUF_S * size_delta
        gp_denominator = ((self.MIN_BUF_S - self.MAX_BUF_S) * size_delta)
        self.gp = gp_nominator / gp_denominator
        self.Vp = self.MAX_BUF_S / (utility_high + self.gp)

    def utility(self, ssim_index):
        """

        :param ssim_index:
        :type ssim_index: float or np.ndarray
        :return:
        :rtype: float or np.ndarray
        """

        if self.version == 1:
            return np.clip(-10 * np.log10(1 - np.where(ssim_index == 1, 1 - 1e-12, ssim_index)),
                           a_min=self.MIN_SSIM, a_max=self.MAX_SSIM)
        else:
            return ssim_index

    def objective(self, utility, size, buffer_in_chunks):
        """

        :param utility:
        :type utility: float or np.ndarray
        :param size:
        :type size: float or np.ndarray
        :param buffer_in_chunks:
        :type buffer_in_chunks: float
        :return:
        :rtype: float or np.ndarray
        """
        return (self.Vp / self.chunk_length * (utility + self.gp) - buffer_in_chunks) / size

    def choose_max_objective(self, format_sizes, format_ssims, buffer_in_chunks):
        """

        :param format_sizes:
        :type format_sizes: np.ndarray
        :param format_ssims:
        :type format_ssims: np.ndarray
        :param buffer_in_chunks:
        :type buffer_in_chunks: float
        :return:
        :rtype: (int, float, float, float)
        """
        objs = self.objective(self.utility(format_ssims), format_sizes, buffer_in_chunks)
        # print("*********")
        # print('format_sizes: ', format_sizes)
        # print('format_ssims: ', format_ssims)
        # print('buffer_in_chunks: ', buffer_in_chunks)
        chosen_index = np.argmax(objs)
        return chosen_index, format_sizes[chosen_index], format_ssims[chosen_index], objs[chosen_index]

    def choose_max_scaled_utility(self, format_sizes, format_ssims):
        """

        :param format_sizes:
        :type format_sizes: np.ndarray
        :param format_ssims:
        :type format_ssims: np.ndarray
        :return:
        :rtype: (int, float, float)
        """
        chosen_index = np.argmax(self.utility(format_ssims) + self.gp)
        return chosen_index, format_sizes[chosen_index], format_ssims[chosen_index]

    def select_video_format(self, index_curr, buffer):
        """

        :param index_curr:
        :type index_curr: int
        :param buffer:
        :type buffer: float
        :return: Return the action index, size and ssim
        :rtype: (int, float, float)
        """
        buffer_in_chunks = buffer / self.chunk_length
        valid_mask = np.logical_not(np.logical_or(np.isnan(self.size_table[index_curr]),
                                                  np.isnan(self.ssim_table[index_curr])))
        size_arr_valid = self.size_table[index_curr][valid_mask]
        ssim_arr_valid = self.ssim_table[index_curr][valid_mask]
        index_arr_valid = np.arange(self.num_formats)[valid_mask]
        max_obj_index, max_obj_size, max_obj_ssim, max_obj = self.choose_max_objective(size_arr_valid,
                                                                                       ssim_arr_valid,
                                                                                       buffer_in_chunks)

        if self.version == 1 or max_obj >= 0:
            return index_arr_valid[max_obj_index], max_obj_size, max_obj_ssim
        else:
            max_util_index, max_util_size, max_util_ssim = self.choose_max_scaled_utility(size_arr_valid,
                                                                                          ssim_arr_valid)
            return index_arr_valid[max_util_index], max_util_size, max_util_ssim


def counterfactual(chat_list, time_list, abr_algo):
    download_time_history = []
    rebuffer_history = [0]
    buf_history = []
    size_history = []
    ssim_history = []

    buffer = 0
    rebuf = 0

    time = time_list[0]
    time_index = 0
    step = 0
    time_finished = False
    while time < time_list[-1] and step < (len(chat_list) - 1):
        if buffer > 15:
            idle_time = buffer - 15
            time = time + idle_time
            if time >= time_list[-1]:
                break
            buffer = 15
            while time_list[time_index + 1] <= time:
                time_index = time_index + 1
        buf_history.append(buffer)
        _, selected_size, selected_ssim = abr_algo.select_video_format(step, buffer)
        time_start_sending = time
        size_history.append(selected_size)
        ssim_history.append(selected_ssim)
        remaining_size = selected_size
        while remaining_size > 1e-4:
            download_time = remaining_size / chat_list[time_index]
            td = time_list[time_index + 1] - time
            if download_time < td:
                if download_time < buffer:
                    buffer = buffer - download_time + 2.002
                else:
                    rebuf = rebuf + download_time - buffer
                    buffer = 2.002
                time = time + download_time
                if time >= time_list[-1]:
                    time_finished = True
                    break
                remaining_size = 0
                while time_list[time_index + 1] <= time:
                    time_index = time_index + 1
            else:
                if td < buffer:
                    buffer = buffer - td
                else:
                    rebuf = rebuf + td - buffer
                    buffer = 0
                time = time + td
                if time >= time_list[-1]:
                    time_finished = True
                    break
                remaining_size = remaining_size - td * chat_list[time_index]
                while time_list[time_index + 1] <= time:
                    time_index = time_index + 1
        download_time_history.append(time - time_start_sending)
        rebuffer_history.append(rebuf)
        if time_finished:
            del rebuffer_history[-1]
            del buf_history[-1]
            del size_history[-1]
            del ssim_history[-1]
            del download_time_history[-1]
        step = step + 1

    assert len(buf_history) == len(rebuffer_history) - 1 == len(download_time_history) == len(size_history)
    return buf_history, size_history, rebuffer_history, download_time_history, ssim_history

PERIOD_TEXT = '2020-07-27to2021-06-01'
new_path = f'{args.dir}{PERIOD_TEXT}_expert_predictions'
os.makedirs(new_path, exist_ok=True)

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
if args.month is not None and args.year is not None:
    all_days = [date for date in all_days if date.month == args.month and date.year == args.year]
cooked_path = f'{args.dir}cooked'

for today in tqdm(all_days):
    date_string = "%d-%02d-%02d" % (today.year, today.month, today.day)
    trajs = np.load(f"{cooked_path}/{date_string}_trajs.npy", allow_pickle=True)
    bba_buffs = []
    bola1_buffs = []
    bola2_buffs = []
    bba_sizes = []
    bola1_sizes = []
    bola2_sizes = []
    bba_ssims = []
    bola1_ssims = []
    bola2_ssims = []
    bba_rebufs = []
    bola1_rebufs = []
    bola2_rebufs = []
    bba_download_times = []
    bola1_download_times = []
    bola2_download_times = []

    for traj_idx, traj in enumerate(trajs):
        linear_bba = LinearBBA(traj[:, 28:40], traj[:, 16:28])
        bola1 = BolaBasic(1, traj[:, 28:40], traj[:, 16:28])
        bola2 = BolaBasic(2, traj[:, 28:40], traj[:, 16:28])
        bba_buff, bba_size_history, bba_rebuf_history, bba_download_time_history, bba_ssim_history = \
            counterfactual(np.divide(traj[:, 7], traj[:, 6]), traj[:, 9], linear_bba)
        bola1_buff, bola1_size_history, bola1_rebuf_history, bola1_download_time_history, bola1_ssim_history = \
            counterfactual(np.divide(traj[:, 7], traj[:, 6]), traj[:, 9], bola1)
        bola2_buff, bola2_size_history, bola2_rebuf_history, bola2_download_time_history, bola2_ssim_history = \
            counterfactual(np.divide(traj[:, 7], traj[:, 6]), traj[:, 9], bola2)
        bba_buffs.append(bba_buff)
        bola1_buffs.append(bola1_buff)
        bola2_buffs.append(bola2_buff)
        bba_sizes.append(bba_size_history)
        bola1_sizes.append(bola1_size_history)
        bola2_sizes.append(bola2_size_history)
        bba_ssims.append(bba_ssim_history)
        bola1_ssims.append(bola1_ssim_history)
        bola2_ssims.append(bola2_ssim_history)
        bba_rebufs.append(bba_rebuf_history)
        bola1_rebufs.append(bola1_rebuf_history)
        bola2_rebufs.append(bola2_rebuf_history)
        bba_download_times.append(bba_download_time_history)
        bola1_download_times.append(bola1_download_time_history)
        bola2_download_times.append(bola2_download_time_history)

    np.save(f'{new_path}/{date_string}_linear_bba_buffs.npy', bba_buffs)
    np.save(f'{new_path}/{date_string}_bola1_buffs.npy', bola1_buffs)
    np.save(f'{new_path}/{date_string}_bola2_buffs.npy', bola2_buffs)
    np.save(f'{new_path}/{date_string}_linear_bba_actions.npy', bba_sizes)
    np.save(f'{new_path}/{date_string}_bola1_actions.npy', bola1_sizes)
    np.save(f'{new_path}/{date_string}_bola2_actions.npy', bola2_sizes)
    np.save(f'{new_path}/{date_string}_linear_bba_ssims.npy', bba_ssims)
    np.save(f'{new_path}/{date_string}_bola1_ssims.npy', bola1_ssims)
    np.save(f'{new_path}/{date_string}_bola2_ssims.npy', bola2_ssims)
    np.save(f'{new_path}/{date_string}_linear_bba_rebuffs.npy', bba_rebufs)
    np.save(f'{new_path}/{date_string}_bola1_rebuffs.npy', bola1_rebufs)
    np.save(f'{new_path}/{date_string}_bola2_rebuffs.npy', bola2_rebufs)
    np.save(f'{new_path}/{date_string}_linear_bba_dts.npy', bba_download_times)
    np.save(f'{new_path}/{date_string}_bola1_dts.npy', bola1_download_times)
    np.save(f'{new_path}/{date_string}_bola2_dts.npy', bola2_download_times)
