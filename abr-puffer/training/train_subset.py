import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="root directory")
parser.add_argument("--C", type=float, help="discriminator loss coefficient")
parser.add_argument("--left_out_policy", type=str, help="left out policy")
parser.add_argument("--device", type=str, help="Compute device", default='cuda:0')
parser.add_argument("--batch_size", type=int, default=17)
args = parser.parse_args()

torch.manual_seed(10)
np.random.seed(10)

BATCH_SIZE = 2 ** args.batch_size
DISCRIMINATOR_EPOCH = 10
C = args.C
device = torch.device(args.device)
left_out_text = f'_{args.left_out_policy}'
PERIOD_TEXT = f'2020-07-27to2021-06-01{left_out_text}'


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


new_path = f'{args.dir}{PERIOD_TEXT}_trained_models/inner_loop_{DISCRIMINATOR_EPOCH}/C_{C}'
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

feature_extractor = MLP(input_dim=2, output_dim=1, hidden_sizes=[128, 128], activation=nn.ReLU).to(device=device)
predictor = MLP(input_dim=3, output_dim=2, hidden_sizes=[128, 128], activation=nn.ReLU).to(device=device)
discriminator = MLP(input_dim=1, output_dim=5, hidden_sizes=[128, 128], activation=nn.ReLU).to(device=device)
huber_loss = nn.HuberLoss(delta=0.2)
ce_loss = nn.CrossEntropyLoss()
feature_extractor_optimizer = torch.optim.Adam(feature_extractor.parameters())
predictor_optimizer = torch.optim.Adam(predictor.parameters())
discriminator_optimizer = torch.optim.Adam(discriminator.parameters())
writer_train = SummaryWriter(log_dir=f"{args.dir}logs/subset_{args.left_out_policy}/"
                                     f"inner_loop_{DISCRIMINATOR_EPOCH}/C_{C}")
for epoch in tqdm(range(5000)):
    t1 = time.time()
    # Discriminator inner training loop:
    train_loss_list = []
    for discriminator_epoch in range(DISCRIMINATOR_EPOCH):
        discriminator_optimizer.zero_grad()
        idx = np.random.choice(data_size, size=BATCH_SIZE)
        feature_tensors = feature_extractor(
            torch.cat((chat_tensors[idx].unsqueeze(1), action_tensors[idx].unsqueeze(1)), dim=1))
        policy_gt_tensors = policy_number_tensors[idx]
        discriminated_tensors = discriminator(feature_tensors)
        assert discriminated_tensors.shape[0] == policy_gt_tensors.shape[0]
        discriminator_loss = ce_loss(discriminated_tensors, policy_gt_tensors)
        discriminator_loss.backward()
        discriminator_optimizer.step()
        train_loss_list.append(discriminator_loss.cpu().detach().item())
    writer_train.add_scalar("discriminator_loss", min(train_loss_list), epoch)
    writer_train.add_scalar("training/elapsed_disc", time.time() - t1, epoch)

    t1 = time.time()
    # Predictor training:
    idx = np.random.choice(data_size, size=BATCH_SIZE)
    feature_extractor_optimizer.zero_grad()
    predictor_optimizer.zero_grad()
    feature_tensors = feature_extractor(
        torch.cat((chat_tensors[idx].unsqueeze(1), action_tensors[idx].unsqueeze(1)), dim=1))
    pred_tensors = predictor(
        torch.cat((buff_tensors[idx].unsqueeze(1), action_tensors[idx].unsqueeze(1), feature_tensors), dim=1))
    dt_pred_tensors = pred_tensors[:, 1:2]
    dt_gt_tensors = dt_tensors[idx].unsqueeze(1)
    buff_pred_tensors = pred_tensors[:, 0:1]
    buff_gt_tensors = next_buff_tensors[idx].unsqueeze(1)
    assert buff_pred_tensors.shape == buff_gt_tensors.shape, f'{buff_pred_tensors.shape}, {buff_gt_tensors.shape}'
    assert dt_pred_tensors.shape == dt_gt_tensors.shape, f'{dt_pred_tensors.shape}, {dt_gt_tensors.shape}'
    buff_pred_loss = huber_loss(buff_pred_tensors, buff_gt_tensors)
    dt_pred_loss = huber_loss(dt_pred_tensors, dt_gt_tensors)
    pred_loss = (dt_pred_loss + buff_pred_loss) / 2
    discriminated_tensors = discriminator(feature_tensors)
    policy_gt_tensors = policy_number_tensors[idx]
    assert policy_gt_tensors.shape[0] == discriminated_tensors.shape[0]
    fool_loss = ce_loss(discriminated_tensors, policy_gt_tensors)
    total_loss = pred_loss - C * fool_loss
    writer_train.add_scalar("predictor_loss/dt_prediction", dt_pred_loss.cpu().detach().item(), epoch)
    writer_train.add_scalar("predictor_loss/buff_prediction", buff_pred_loss.cpu().detach().item(), epoch)
    writer_train.add_scalar("predictor_loss/discriminator", fool_loss.cpu().detach().item(), epoch)
    writer_train.add_scalar("predictor_loss/total", total_loss.cpu().detach().item(), epoch)
    writer_train.add_scalar("training/elapsed_pred", time.time() - t1, epoch)
    total_loss.backward()
    feature_extractor_optimizer.step()
    predictor_optimizer.step()
    if epoch % 100 == 99:
        torch.save(feature_extractor, f'{new_path}/{epoch + 1}_feature_extractor.pth')
        torch.save(predictor, f'{new_path}/{epoch + 1}_predictor.pth')
        torch.save(discriminator, f'{new_path}/{epoch + 1}_discriminator.pth')
