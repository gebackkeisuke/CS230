import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
    
# user defined functions
from utils import *    

def load_simulation_results_grid(param, case, group):
    """
    Load 2D simulation result data from a numpy file, reshape it, and exclude boundary values.
    """
    dir  = "ProcessedData/group_" + group
    fn = param + "_" + case + ".npy"

    data = np.load(dir + "/" + fn)                  # Load numpy array (time, z, y, x)
    data = data.reshape(-1, data.shape[-1])         # Reshape to 2D array (y, x)
    data = data[1:-1,1:-1]                          # Exclude boundary (y, x)
    data = data.T                                   # Transpose to (x, y)
    return data

def load_simulation_results_1D(case, group):
    """
    Load 1D simulation result data from a numpy file.
    """
    dir = "ProcessedData/group_" + group
    fn  = "summary_" + case + ".csv"

    data = pd.read_csv(dir + "/" + fn, delimiter=",")   # Load CSV file
    return data

def load_simulation_parameters(group):
    """
    Load simulation parameters from a CSV file.
    """
    fn_parameters       = "Parameters/parameters_" + group + ".csv"

    df_params = pd.read_csv(fn_parameters, delimiter=",") # Load simulation parameters
    df_params["case"] = np.linspace(0, len(df_params)-1, len(df_params)).astype(int)
    df_params["case"] = df_params["case"].astype(str).str.zfill(3)

    return df_params


def load_permeability(case_k):
    """
    Load permeability data
    Note
    - permeability shoud be (x,y). We don't have to transpose it.
    
    """
    fn = "INCLUDE/03_PERMX/k_" + case_k + ".inc"
    perm_x = pd.read_csv(fn, delimiter="\t", skiprows=1, header=None)
    perm_x = perm_x.iloc[:-1, :]
    perm_x = perm_x.values.flatten()
    perm_x = perm_x.reshape(202, 202) # (x, y)
    perm_x = perm_x[1:-1, 1:-1]
    perm_x = perm_x.astype(float)

    return perm_x

def create_2D_input_features(df_params, Nx, Ny, N_wells, flag_debug=False):    
    """
    Create 2D input features from simulation parameters.
    """
    location    = np.zeros((Nx,Ny))
    rate        = np.zeros((Nx,Ny))
    timing      = np.zeros((Nx,Ny))

    for j in range(N_wells):
        x = df_params.loc[f"x_{j}"]
        y = df_params.loc[f"y_{j}"]
        location[x-1, y-1]  = 1  # Mark well location
        rate[x-1, y-1]      = df_params.loc[f"rate_{j}"] 
        timing[x-1, y-1]    = df_params.loc[f"year_{j}"] - 2030 + 1 # Adjust timing to start from 1

    if flag_debug:
        plt.imshow(location, cmap='gray')
        plt.colorbar()
        plt.show()

        plt.imshow(rate, cmap='gray')
        plt.colorbar()
        plt.show()

        plt.imshow(timing, cmap='gray')
        plt.colorbar()
        plt.show()

    return location, rate, timing

def load_input_output_features(df_params, case, i, group):
    """
    Load input and output features for a given simulation case.
    """
    config = load_config("config/config.yaml")
    Nx = config["Nx"] - 2
    Ny = config["Ny"] - 2 
    N_wells     = sum(df_params.columns.str.startswith("x_"))


    # Input features
    location, rate, timing = create_2D_input_features(df_params.iloc[i], Nx, Ny, N_wells)
    case_k = "0"
    perm_k = load_permeability(case_k)

    # Output features
    Sg          = load_simulation_results_grid("Sg", case, group) # Load 2D simulation results (i.e., Sg)
    df_1D       = load_simulation_results_1D(case, group) # Load 1D simulation results (i.e., BHP and Rate)
        
    inputs = {
        "perm_k":   perm_k,
        "location": location,
        "rate":     rate,
        "timing":   timing,
    }

    outputs = {
        "Sg": Sg,
        "df_1D": df_1D
    }

    return inputs, outputs

if __name__ == "__main__":
    """
    必要なコード
    ・Sgのロード >>>>Done
    ・BHPとRateのロード >>>>Done
    ・Permeabiltyのロード >>>>Done
    ・シミュレーション条件のロード >>>>Done
    ・上記のロード機能を全て関数にする >>>>Done
    ・井戸位置などをバイナリ2Dに変える >>>>Done
    ・浸透率のマップの番号を引数で指定 
    ・Summaryファイルの読み込みは変数の数が変わる可能性があることを考慮して関数化する
    ・groupごとにINPUTの数が違うことに対応したい
    """

    group = sys.argv[1]  # Get group identifier from command line argument

    df_params   = load_simulation_parameters(group) # Load simulation parameters
    cases       = df_params["case"].tolist()
    
    for i, case in enumerate(cases):
        print("Loading case: ", case)
        if i > 1:
            break

        inputs, outputs = load_input_output_features(df_params, case, i, group)

    print(inputs)
    print(outputs)