from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear

import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim

# Third Party
from thop import clever_format, profile
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

warnings.filterwarnings("ignore")

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description="GPT4TS")

parser.add_argument("--model_id", type=str, required=True, default="test")
parser.add_argument("--checkpoints", type=str, default="./checkpoints/")

parser.add_argument("--root_path", type=str, default="./dataset/traffic/")
parser.add_argument("--data_path", type=str, default="traffic.csv")
parser.add_argument("--data", type=str, default="custom")
parser.add_argument("--features", type=str, default="M")
parser.add_argument("--freq", type=int, default=1)
parser.add_argument("--target", type=str, default="OT")
parser.add_argument("--embed", type=str, default="timeF")
parser.add_argument("--percent", type=int, default=10)

parser.add_argument("--seq_len", type=int, default=512)
parser.add_argument("--pred_len", type=int, default=96)
parser.add_argument("--label_len", type=int, default=48)

parser.add_argument("--decay_fac", type=float, default=0.75)
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=10)
parser.add_argument("--train_epochs", type=int, default=10)
parser.add_argument("--lradj", type=str, default="type1")
parser.add_argument("--patience", type=int, default=3)

parser.add_argument("--gpt_layers", type=int, default=3)
parser.add_argument("--is_gpt", type=int, default=1)
parser.add_argument("--e_layers", type=int, default=3)
parser.add_argument("--d_model", type=int, default=768)
parser.add_argument("--n_heads", type=int, default=16)
parser.add_argument("--d_ff", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--enc_in", type=int, default=862)
parser.add_argument("--c_out", type=int, default=862)
parser.add_argument("--patch_size", type=int, default=16)
parser.add_argument("--kernel_size", type=int, default=25)

parser.add_argument("--loss_func", type=str, default="mse")
parser.add_argument("--pretrain", type=int, default=1)
parser.add_argument("--freeze", type=int, default=1)
parser.add_argument("--model", type=str, default="model")
parser.add_argument("--stride", type=int, default=8)
parser.add_argument("--max_len", type=int, default=-1)
parser.add_argument("--hid_dim", type=int, default=16)
parser.add_argument("--tmax", type=int, default=10)

parser.add_argument("--itr", type=int, default=3)
parser.add_argument("--cos", type=int, default=0)


args = parser.parse_args()

SEASONALITY_MAP = {
    "minutely": 1440,
    "10_minutes": 144,
    "half_hourly": 48,
    "hourly": 24,
    "daily": 7,
    "weekly": 1,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1,
}

mses = []
maes = []


def calculate_flops(model, train_dls):
    try:
        # no_batches = len(train_dls)
        # num_patch = (
        #     max(context_length, patch_length) - patch_length
        # ) // patch_length + 1

        # batch size = 32
        # s_input = torch.rand(32, len(forecast_columns), num_patch, patch_length)
        # s_input = torch.rand(32, context_length, len(forecast_columns))
        for x in train_dls:
            s_input = x[0]

            if s_input is None:
                raise Exception("")

            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                s_input = s_input.float().to(device)

            macs_per_bs, params = profile(
                model, inputs=(s_input, args.itr), verbose=True
            )
            break

        macs = macs_per_bs * len(train_dls)
        # macs, _ = clever_format([macs_per_bs * no_batches, params], "%.3f")
        macs, macs_per_bs, params = clever_format([macs, macs_per_bs, params], "%.3f")
        return params, macs, macs_per_bs
    except Exception as e:
        # print("flops calculation failed", str(e))
        print("flops calculation failed", e)
        return None, None, None


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


for ii in range(args.itr):
    setting = "{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}".format(
        args.model_id,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.gpt_layers,
        args.d_ff,
        args.embed,
        ii,
    )
    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)

    if args.freq == 0:
        args.freq = "h"

    train_data, train_loader = data_provider(args, "train")
    vali_data, vali_loader = data_provider(args, "val")
    test_data, test_loader = data_provider(args, "test")

    if args.freq != "h":
        args.freq = SEASONALITY_MAP[test_data.freq]
        print("freq = {}".format(args.freq))

    device = torch.device("cuda:0")

    time_now = time.time()
    train_steps = len(train_loader)

    breakpoint()
    if args.model == "PatchTST":
        model = PatchTST(args, device)
        model.to(device)
    elif args.model == "DLinear":
        model = DLinear(args, device)
        model.to(device)
    else:
        model = GPT4TS(args, device)

    print("Model params that has gradient")
    total_params_manual = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params_ = param.numel()
            total_params_manual += num_params_
            print(
                f"{name}: requires_grad={param.requires_grad} : #params={num_params_}"
            )
    print("Total #params manual =", total_params_manual)
    # mse, mae = test(model, test_data, test_loader, args, device, ii)

    breakpoint()

    params = model.parameters()
    model_optim = torch.optim.Adam(params, lr=args.learning_rate)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    if args.loss_func == "mse":
        criterion = nn.MSELoss()
    elif args.loss_func == "smape":

        class SMAPE(nn.Module):
            def __init__(self):
                super(SMAPE, self).__init__()

            def forward(self, pred, true):
                return torch.mean(
                    200
                    * torch.abs(pred - true)
                    / (torch.abs(pred) + torch.abs(true) + 1e-8)
                )

        criterion = SMAPE()

    # Model complexity analysis
    num_model_params, macs, macs_per_bs = calculate_flops(
        model=model, train_dls=train_loader
    )
    print("-" * 50, "MACs across 10% train data =", macs, "nparams =", num_model_params)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("self vs thop trainable params", trainable_params, num_model_params)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model_optim, T_max=args.tmax, eta_min=1e-8
    )

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
            enumerate(train_loader)
        ):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)

            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            outputs = model(batch_x, ii)

            outputs = outputs[:, -args.pred_len :, :]
            batch_y = batch_y[:, -args.pred_len :, :].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 1000 == 0:
                print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()
                    )
                )
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print(
                    "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time)
                )
                iter_count = 0
                time_now = time.time()
            loss.backward()
            model_optim.step()
            MAX_MEMORY = convert_size(torch.cuda.max_memory_allocated(device=device))

        calculated_epoch_time = time.time() - epoch_time
        print(
            "-" * 50,
            "Epoch: {} cost time: {}".format(epoch + 1, calculated_epoch_time),
        )

        train_loss = np.average(train_loss)
        vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
        # test_loss = vali(model, test_data, test_loader, criterion, args, device, ii)
        # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}, Test Loss: {4:.7f}".format(
        #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        print(
            "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss
            )
        )

        if args.cos:
            scheduler.step()
            print("lr = {:.10f}".format(model_optim.param_groups[0]["lr"]))
        else:
            adjust_learning_rate(model_optim, epoch + 1, args)
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    best_model_path = path + "/" + "checkpoint.pth"
    model.load_state_dict(torch.load(best_model_path))
    print("------------------------------------")
    test_time = time.time()
    mse, mae = test(model, test_data, test_loader, args, device, ii)
    calculated_test_time = time.time() - test_time
    print("-" * 50, "TEST time =", calculated_test_time)
    mses.append(mse)
    maes.append(mae)

mses = np.array(mses)
maes = np.array(maes)
print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))

print("*" * 100)
print("=" * 40, "Runtime metrics", "=" * 40)
print("MACs =", macs)
print("params =", num_model_params)
print("total params =", clever_format(total_params, "%.3f"))
print("epoch time =", calculated_epoch_time)
print("max memory =", MAX_MEMORY)
print("test time =", calculated_test_time)

print("*" * 100)
