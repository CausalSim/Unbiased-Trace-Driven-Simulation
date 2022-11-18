import numpy as np
import torch
import datetime
import torch.nn as nn
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--dir", help="root directory")
parser.add_argument("--C", type=float, help="discriminator loss coefficient")
parser.add_argument("--left_out_policy", type=str, help="left out policy")
parser.add_argument("--month", type=int, default=None)
parser.add_argument("--year", type=int, default=None)
parser.add_argument("--model_number", type=int, help="saved model epoch number", default=5000)
args = parser.parse_args()


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_sizes, activation):
        super().__init__()
        self.predict = mlp(sizes=[input_dim] + list(hidden_sizes) + [output_dim], activation=activation,
                           output_activation=nn.Identity)

    def forward(self, raw_input):
        prediction = self.predict(raw_input)
        return prediction


def extract(chat_list, chosen_size_list, feature_extractor, chat_mean, chat_std, size_mean, size_std):
    extracted_latents = []
    for step in range(len(chat_list)):
        chat = chat_list[step]
        orig_chosen_size = chosen_size_list[step]
        chat_white = (chat - chat_mean) / chat_std
        orig_chosen_size_white = (orig_chosen_size - size_mean) / size_std
        input_numpy = np.array([chat_white, orig_chosen_size_white])
        input_numpy = np.expand_dims(input_numpy, axis=0)
        input_tensor = torch.as_tensor(input_numpy, dtype=torch.float32, device=torch.device('cpu'))
        with torch.no_grad():
            feature_tensor = feature_extractor(input_tensor)
        extracted_feature = feature_tensor.cpu().numpy()
        extracted_latents.append(extracted_feature)
    return extracted_latents


DISCRIMINATOR_EPOCH = 10
C = args.C
left_out_text = f'_{args.left_out_policy}'
PERIOD_TEXT = f'2020-07-27to2021-06-01{left_out_text}'
latent_path = f'{args.dir}{PERIOD_TEXT}_features/inner_loop_{DISCRIMINATOR_EPOCH}/C_{C}/' \
              f'model_{args.model_number}'
os.makedirs(latent_path, exist_ok=True)

data_path = f'{args.dir}subset_data/{args.left_out_policy}'
chats_mean = np.load(f'{data_path}/chats_mean.npy')
actions_mean = np.load(f'{data_path}/actions_mean.npy')
chats_std = np.load(f'{data_path}/chats_std.npy')
actions_std = np.load(f'{data_path}/actions_std.npy')

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
model_path = f'{args.dir}{PERIOD_TEXT}_trained_models/inner_loop_{DISCRIMINATOR_EPOCH}/C_{C}'
feature_extractor = torch.load(f"{model_path}/{args.model_number}_feature_extractor.pth",
                               map_location=torch.device('cpu')).cpu()
cooked_path = f'{args.dir}cooked'

for today in tqdm(all_days):
    date_string = "%d-%02d-%02d" % (today.year, today.month, today.day)
    trajs = np.load(f"{cooked_path}/{date_string}_trajs.npy", allow_pickle=True)
    latents = []
    for traj in trajs:
        features = extract(np.divide(traj[:-1, 7], traj[:-1, 6]), traj[:-1, 7], feature_extractor, chats_mean,
                           chats_std, actions_mean, actions_std)
        latents.append(features)
    np.save(f'{latent_path}/{date_string}_features.npy', latents)
    del latents