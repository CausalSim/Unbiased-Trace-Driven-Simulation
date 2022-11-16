import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from nn_util import MLP

BATCH_SIZE = 2**13  # Maximum without exceeding GPU memory limit


def train_causal_sim(
    datapath,
    kappa,
    no_policies,
    r=1,
    DISCRIMINATOR_EPOCH=10,
    models_path="models",
    BATCH_SIZE=2**13,
    N=int(5 * 1e6),
):
    path_models = f"{models_path}/causalsim/"

    try:
        os.makedirs(path_models)
    except:
        pass
    log_path = f"{path_models}/logs"
    try:
        os.makedirs(log_path)
    except:
        pass

    if torch.cuda.is_available():
        device = torch.device(f"cuda:0")
    else:
        device = torch.device(f"cpu")
    inputs_train = np.load(f"{datapath}/white_train_inputs.npy")[:N, :]  #
    # [:, (processing_time, one hot encoder for action (8))]

    outputs_train = np.load(f"{datapath}/white_train_outputs.npy")[:N, :]
    # [:, (processing_time, policy_label)]

    val_size = int(inputs_train.shape[0] * 0.1)
    train_idx, val_idx = train_test_split(
        np.arange(len(inputs_train)), test_size=val_size, train_size=None
    )

    train_input_tensors = torch.as_tensor(
        inputs_train[train_idx], dtype=torch.float32, device=device
    )
    train_output_tensors = torch.as_tensor(
        outputs_train[train_idx], dtype=torch.float32, device=device
    )

    val_input_tensors = torch.as_tensor(
        inputs_train[val_idx], dtype=torch.float32, device=device
    )
    val_output_tensors = torch.as_tensor(
        outputs_train[val_idx], dtype=torch.float32, device=device
    )

    feature_extractor = MLP(
        input_dim=9, output_dim=r, hidden_sizes=[64, 64], activation=nn.ReLU
    ).to(device)
    action_factor_net = MLP(
        input_dim=8, output_dim=r, hidden_sizes=[], activation=nn.ReLU
    ).to(device)

    ## discriminator input
    discriminator1 = MLP(
        input_dim=r, output_dim=no_policies, hidden_sizes=[128, 128], activation=nn.ReLU
    ).to(device)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    action_factor_optimizer = torch.optim.Adam(
        action_factor_net.parameters(),
    )
    feature_extractor_optimizer = torch.optim.Adam(
        feature_extractor.parameters(),
    )
    discriminator1_optimizer = torch.optim.Adam(
        discriminator1.parameters(),
    )

    writer_train = SummaryWriter(
        log_dir=f"{log_path}/inner_loop_%d/kappa_%d/training"
        % (DISCRIMINATOR_EPOCH, kappa)
    )
    best_loss = np.inf
    for epoch in tqdm(range(10000)):

        # Discriminator inner training loop:
        train_loss_list = []
        for discriminator_epoch in range(DISCRIMINATOR_EPOCH + 1):
            idx = np.random.choice(np.arange(len(train_input_tensors)), size=BATCH_SIZE)
            batch_input_tensors = train_input_tensors[idx, :]
            batch_output_tensors1 = train_output_tensors[idx, 1]
            discriminator1_optimizer.zero_grad()
            feature_tensors = feature_extractor(batch_input_tensors)
            discriminated1_tensors = discriminator1(feature_tensors)
            discriminator1_loss = ce_loss(
                discriminated1_tensors, batch_output_tensors1.long()
            )
            discriminator_loss = discriminator1_loss
            discriminator_loss.backward()
            discriminator1_optimizer.step()
            train_loss_list.append(discriminator_loss.cpu().detach().numpy())

        writer_train.add_scalar("discriminators_loss", min(train_loss_list), epoch)

        # mediator predictor training:
        idx = np.random.choice(np.arange(len(train_input_tensors)), size=BATCH_SIZE)
        batch_input_tensors = train_input_tensors[idx]
        batch_output_tensors = train_output_tensors[idx]
        feature_extractor_optimizer.zero_grad()
        action_factor_optimizer.zero_grad()

        action_factor = action_factor_net(batch_input_tensors[:, 1:])
        feature_tensors = feature_extractor(batch_input_tensors[:, :])
        discriminated_tensors = discriminator1(feature_tensors)

        predicted_pt = torch.mul(feature_tensors, action_factor)
        predicted_pt = torch.matmul(
            predicted_pt, torch.ones([r, 1], dtype=torch.float32, device=device)
        )

        pred_loss = mse_loss(predicted_pt, batch_output_tensors[:, :1])
        fool_loss = ce_loss(discriminated_tensors, batch_output_tensors[:, 1].long())
        total_loss = pred_loss - kappa * fool_loss
        writer_train.add_scalar(
            "predictor_loss/prediction", pred_loss.cpu().detach().numpy(), epoch
        )
        writer_train.add_scalar(
            "predictor_loss/discriminator", fool_loss.cpu().detach().numpy(), epoch
        )
        writer_train.add_scalar(
            "predictor_loss/total", total_loss.cpu().detach().numpy(), epoch
        )
        total_loss.backward()
        action_factor_optimizer.step()
        feature_extractor_optimizer.step()

        if epoch % 1000 == 999:
            batch_input_tensors = val_input_tensors[:]
            batch_output_tensors = val_output_tensors[:]

            action_factor = action_factor_net(batch_input_tensors[:, 1:])
            feature_tensors = feature_extractor(batch_input_tensors[:, :])
            discriminated_tensors = discriminator1(feature_tensors)

            predicted_pt = torch.mul(feature_tensors, action_factor)
            predicted_pt = torch.matmul(
                predicted_pt, torch.ones([r, 1], dtype=torch.float32, device=device)
            )

            pred_loss = mse_loss(predicted_pt, batch_output_tensors[:, :1])
            fool_loss = ce_loss(
                discriminated_tensors, batch_output_tensors[:, 1].long()
            )
            total_loss = pred_loss - kappa * fool_loss

            print(
                f"Val loss: epoch {epoch}, prediction loss {total_loss.cpu().detach().numpy()}, disc_loss {fool_loss.cpu().detach().numpy()} "
            )
            if best_loss > total_loss:
                best_loss = total_loss
                print(f"saving ... best losses: {best_loss}")
                torch.save(
                    feature_extractor,
                    f"{path_models}/best_feature_" "extractor.pth",
                )
                torch.save(
                    action_factor_net, f"{path_models}/best_action_factor" ".pth"
                )
                torch.save(discriminator1, f"{path_models}/best_discriminator.pth")
