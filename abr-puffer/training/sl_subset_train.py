import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="root directory")
parser.add_argument("--left_out_policy", type=str, help="left out policy")
parser.add_argument("--device", type=str, help="Compute device", default='cuda:0')
parser.add_argument("--batch_size", type=int, default=17)
args = parser.parse_args()
BATCH_SIZE = 2 ** args.batch_size
device = torch.device(args.device)
left_out_text = f'_{args.left_out_policy}'
PERIOD_TEXT = f'2020-07-27to2021-06-01{left_out_text}'

torch.manual_seed(10)
np.random.seed(10)


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


new_path = f'{args.dir}{PERIOD_TEXT}_SL_trained_models'
os.makedirs(new_path, exist_ok=True)
data_dir = f'{args.dir}subset_data/{args.left_out_policy}'
dts = np.load(f'{data_dir}/white_dts.npy')
buffs = np.load(f'{data_dir}/white_buffs.npy')
next_buffs = np.load(f'{data_dir}/white_next_buffs.npy')
policy_numbers = np.load(f'{data_dir}/policy_numbers.npy')
chats = np.load(f'{data_dir}/white_chats.npy')
actions = np.load(f'{data_dir}/white_actions.npy')
data_size = len(chats)
dt_tensors = torch.as_tensor(dts, dtype=torch.float32, device=device)
del dts
buff_tensors = torch.as_tensor(buffs, dtype=torch.float32, device=device)
next_buff_tensors = torch.as_tensor(next_buffs, dtype=torch.float32, device=device)
del buffs, next_buffs
chat_tensors = torch.as_tensor(chats, dtype=torch.float32, device=device)
action_tensors = torch.as_tensor(actions, dtype=torch.float32, device=device)
policy_number_tensors = torch.as_tensor(policy_numbers, dtype=torch.long, device=device)
del chats, actions, policy_numbers

predictor = MLP(input_dim=3, output_dim=2, hidden_sizes=[128, 128], activation=nn.ReLU).to(device=device)
huber_loss = nn.HuberLoss(delta=0.2)
predictor_optimizer = torch.optim.Adam(predictor.parameters())
writer_train = SummaryWriter(log_dir=f"{args.dir}logs/subset_{args.left_out_policy}_SL")
for epoch in tqdm(range(10000)):
    # Predictor training:
    idx = np.random.choice(data_size, size=BATCH_SIZE)
    predictor_optimizer.zero_grad()
    pred_tensors = predictor(torch.cat((buff_tensors[idx].unsqueeze(1),
                                        action_tensors[idx].unsqueeze(1), chat_tensors[idx].unsqueeze(1)), dim=1))
    dt_pred_tensors = pred_tensors[:, 1:2]
    dt_gt_tensors = dt_tensors[idx].unsqueeze(1)
    buff_pred_tensors = pred_tensors[:, 0:1]
    buff_gt_tensors = next_buff_tensors[idx].unsqueeze(1)
    assert buff_pred_tensors.shape == buff_gt_tensors.shape, f'{buff_pred_tensors.shape}, {buff_gt_tensors.shape}'
    assert dt_pred_tensors.shape == dt_gt_tensors.shape, f'{dt_pred_tensors.shape}, {dt_gt_tensors.shape}'
    buff_pred_loss = huber_loss(buff_pred_tensors, buff_gt_tensors)
    dt_pred_loss = huber_loss(dt_pred_tensors, dt_gt_tensors)
    pred_loss = (dt_pred_loss + buff_pred_loss) / 2
    writer_train.add_scalar("predictor_loss/dt_prediction", dt_pred_loss.cpu().detach().item(), epoch)
    writer_train.add_scalar("predictor_loss/buff_prediction", buff_pred_loss.cpu().detach().item(), epoch)
    pred_loss.backward()
    predictor_optimizer.step()
    if epoch % 100 == 99:
        torch.save(predictor, f'{new_path}/{epoch + 1}_predictor.pth')
