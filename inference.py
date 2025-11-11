
# 1.必要なライブラリをインポート
import sys
import torch
import yaml
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt

# user defined functions
import data_load as dl
from mydataset import MyDataset
from myunet import MyUnet

with open("config/config_NN.yaml", "r") as f:
    config = yaml.safe_load(f)

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

# 6.評価（Inference/Validation)

# Notes: These variables can be inside the function.
N_input = len(config["channels"]["input"])
N_output = len(config["channels"]["output"])

def inference(model_path, x_input, device):
    """
    Inference function for the model.

    """

    # モデル定義
    model = MyUnet(N_input, N_output, config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 推論
    with torch.no_grad():
        x_input = x_input.unsqueeze(0)  # バッチ次元を追加
        x_input = x_input.to(device)
        print("x_input.shape:", x_input.shape)
        output = model(x_input)

    return output

# Case specification
case = 5
x, y = dataset[case]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prediction
pred = inference("model.pth", x, device)
print(pred.shape)

# Visualization
y_pred = pred[0].cpu().squeeze(0)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Ground Truth
axes[0].set_title("Ground Truth")
im0 = axes[0].imshow(y[0].cpu(), cmap='jet', vmin=0, vmax=1)
axes[0].axis('off')

# Prediction
axes[1].set_title("Prediction")
im1 = axes[1].imshow(y_pred, cmap='jet', vmin=0, vmax=1)
axes[1].axis('off')

# 共通カラーバー (右側に)
cbar = fig.colorbar(
    im1,                     # どちらかの画像を渡せばOK
    ax=axes,
    fraction=0.046,
    pad=0.05
)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # ← colorbar の領域を確保
plt.show()

# To be implemented:
# ケースを選んで推論を実行
# 自分でケースを作って推論を実行
# 訓練に使った全てのケースで推論を実行し、評価指標を計算
# Validation データで推論を実行し、評価指標を計算