from termcolor import colored
import pickle
import os
import torch


def set_omp_thrs(num: int = 1):
    assert num > 0
    assert num <= 256
    os.environ['OMP_NUM_THREADS'] = f'{num}'
    torch.set_num_threads(num)


def save_args(config, path: str):
    args_dict = vars(config)
    with open(f'{path}/args.pkl', 'wb') as handle:
        pickle.dump(args_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_folders(output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    exit_run = False

    if os.path.exists(f'{output_folder}/rewards_train.npy'):
        exit_run = True

    if exit_run:
        print(colored('Results already exist in output folder', 'red'))
        print(colored('Possibility of overwrite, exiting', 'red'))
        exit(1)
