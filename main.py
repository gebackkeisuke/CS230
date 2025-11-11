
# 1.必要なライブラリをインポート
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import yaml
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim

# user defined functions
import data_load as dl
from mydataset import MyDataset
from myunet import MyUnet
from train import train_model

with open("config/config_NN.yaml", "r") as f:
    config = yaml.safe_load(f)

# print(config)

# 2.データを準備（Dataset, DataLoader)
import_groups   = [str(key) for key, value in config["dataset_group"].items() if value]
group           = import_groups[0]
# ほかのグループのデータも読み込めるように調整する

df_params       = dl.load_simulation_parameters(group) # Load simulation parameters
cases           = df_params["case"].tolist()

# inputとoutputのリストを作成
list_of_inputs = []
list_of_outputs = []
for i, case in enumerate(cases):
    if i >= 100:
        break
    inputs, outputs = dl.load_input_output_features(df_params, case, i, group)
    list_of_inputs.append(inputs)
    list_of_outputs.append(outputs)  

dataset = MyDataset(list_of_inputs, list_of_outputs, config)
loader  = DataLoader(dataset, batch_size=4, shuffle=True)

# for idx_batch, batch in zip(loader.batch_sampler, loader):
#     inputs, outputs = batch
#     print("Batch indices:", idx_batch) # [34, 76, 44, 91]
#     print("Input shape:", inputs.shape) # [4, 4, Nx, Ny]
#     print("Output shape:", outputs.shape) # [4, 1, Nx, Ny]
#     break

# 3.モデルを定義（nn.Moduleを継承）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

N_input = len(config["channels"]["input"])
N_output = len(config["channels"]["output"])

model = MyUnet(N_input, N_output, config)
model.to(device)

# 5.学習ループ（Training loop）
train_model(model, dataset, config, device)
torch.save(model.state_dict(), "model.pth")


