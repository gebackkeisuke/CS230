
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


class MyDataset(Dataset):
    def __init__(self, list_of_inputs, list_of_outputs, config): # Case 5

        self.list_of_inputs     = list_of_inputs
        self.list_of_outputs    = list_of_outputs

        self.preprocess         = config.get("preprocess", {})
        self.input_channels     = config.get("channels", []).get("input", [])
        self.output_channels    = config.get("channels", []).get("output", [])
        self.flatten            = config.get("flatten", False)

        self.mean = {}
        self.std = {}
        self.min = {}
        self.max = {}

        for c in self.input_channels:
            if self.preprocess.get(c, None) == "standardize":
                self._compute_channel_stats(c, config)
            elif self.preprocess.get(c, None) == "min-max":
                self._min_max_channel_stats(c, config)

    def _min_max_channel_stats(self, channel, config):
        """
        指定したチャンネルごとにmin/maxを計算して保存
        """
        self.min[channel] = config.get("min", {}).get(channel, None)
        self.max[channel] = config.get("max", {}).get(channel, None)
        if self.min[channel] is not None and self.max[channel] is not None:
            return
        
        arr_list = [np.asarray(inputs[channel], dtype=np.float32)
                    for inputs in self.list_of_inputs]
        stacked = np.stack(arr_list)  # (num_cases, Nx, Ny)
        self.min[channel] = stacked.min()
        self.max[channel] = stacked.max()

    def _compute_channel_stats(self, channel, config):
        """
        指定したチャンネルごとにmean/stdを計算して保存
        """
        self.mean[channel] = config.get("mean", {}).get(channel, None)
        self.std[channel]  = config.get("std", {}).get(channel, None)
        if self.mean[channel] is not None and self.std[channel] is not None:
            return
        
        arr_list = [np.asarray(inputs[channel], dtype=np.float32)
                    for inputs in self.list_of_inputs]
        stacked = np.stack(arr_list)  # (num_cases, Nx, Ny)
        self.mean[channel] = stacked.mean()
        self.std[channel]  = stacked.std() + 1e-6

    def _pack_inputs(self, inputs):
        """
        dictの各2D行列を（C, Nx, Ny）のスタック
        """
        arr_list = []

        for c in self.input_channels:
            arr = np.asarray(inputs[c], dtype=np.float32)
            if self.preprocess.get(c, None) == "standardize":
                arr = (arr - self.mean[c]) / self.std[c]
            elif self.preprocess.get(c, None) == "min-max":
                arr = (arr - self.min[c]) / (self.max[c] - self.min[c])
            arr_list.append(arr)
        
        x = np.stack(arr_list, axis=0) # shale -> (Nx, Ny) to (N_keys, Nx, Ny)
        x = torch.tensor(x, dtype=torch.float32)

        if self.flatten:
            x = x.view(-1)  # Flatten to 1D
        
        return x
    
    def _pack_outputs(self, outputs):
        """
        dictの各2D行列を（C, Nx, Ny）のスタック
        """
        arr_list = []
        for key in self.output_channels:
            arr = np.asarray(outputs[key], dtype=np.float32)
            if self.preprocess.get(key, None) == "standardize":
                arr = (arr - self.mean[key]) / self.std[key]
            elif self.preprocess.get(key, None) == "min-max":
                arr = (arr - self.min[key]) / (self.max[key] - self.min[key])
            arr_list.append(arr)
        
        y = np.stack(arr_list, axis=0) # shale -> (Nx, Ny) to (N_keys, Nx, Ny)
        return torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.list_of_inputs) # Case 3

    def __getitem__(self, idx):
        x = self._pack_inputs(self.list_of_inputs[idx])
        y = self._pack_outputs(self.list_of_outputs[idx])
        return x, y

    def print_stats(self):
        """
        チャンネルごとの前処理統計量を表示
        使用方法
        dataset.print_stats()
        """

        for c in self.channels:
            if c in self.mean:
                print(f"Standardize Channel: {c}, Mean: {self.mean[c]:.4f}, Std: {self.std[c]:.4f}")

            if c in self.min:
                print(f"Min-Max Channel: {c}, Min: {self.min[c]:.4f}, Max: {self.max[c]:.4f}")
            
            else:
                print(f"No preprocessing for Channel: {c}")

    def plot_histogram(self, bins=50, log_scale=False):
        """
        前処理済み input features のヒストグラムをチャンネルごとに描画

        使用方法
        dataset.plot_histogram(log_scale=True)

        """
        
        # 全データをまとめて取得
        all_inputs = [self._pack_inputs(inp) for inp in self.list_of_inputs]
        all_outputs = [self._pack_outputs(out) for out in self.list_of_outputs]
        all_inputs = torch.stack(all_inputs)  # (num_samples, channels, Nx, Ny)
        all_outputs = torch.stack(all_outputs)  # (num_samples, channels, Nx, Ny)
        all = torch.cat([all_inputs, all_outputs], dim=1)  # Combine inputs and outputs

        num_samples, num_channels, Nx, Ny = all.shape
        flat_inputs = all.view(num_samples, num_channels, -1)

        for i, c in enumerate(self.input_channels + self.output_channels):
            plt.figure()
            plt.hist(flat_inputs[:, i, :].numpy().flatten(), bins=bins, color='skyblue', edgecolor='black')
            plt.title(f"Histogram of {c}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            if log_scale:
                plt.yscale("log")
            plt.show()