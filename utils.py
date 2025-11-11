
import yaml
import numpy as np
import pandas as pd

def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.
    
    Parameters:
    - config_path: Path to the YAML configuration file.
    
    Returns:
    - config: Dictionary containing the configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def extract_DP_Sg(grid_data, t):
    """
    グリッドデータを抽出した後、使いやすい形に直す
    """
    
    # --- INPUT Parameters ---
    config = load_config("config/config.yaml")
    Nx = config.get('Nx')              # Number of grid cells in x-direction [-]
    Ny = config.get('Ny')              # Number of grid cells in y-direction [-]
    Nz = config.get('Nz')              # Number of grid cells in z-direction [-]
    
    # --- Pressure [bar] ---
    Pg_ini  = np.array(grid_data[0]["PRESSURE"]) #　NOTENOTENOTENOTE: ここが遅いと思う
    Pg_ini  = Pg_ini.reshape(Nz, Ny, Nx)
    Pg_end  = np.array(grid_data[t]["PRESSURE"])
    Pg_end  = Pg_end.reshape(Nz, Ny, Nx)
    
    DP      = Pg_end - Pg_ini   # Pressure change [bar]
    
    # --- CO2 saturation [-] ---
    Sg      = np.array(grid_data[t]["SGAS"]) 
    Sg      = Sg.reshape(Nz, Ny, Nx)

    # --- Brine volumetric rate [m3/day] ---
    FLRWATI = np.array(grid_data[t]["FLRWATI+"])
    FLRWATI = FLRWATI.reshape(Nz, Ny, Nx)
    FLRWATJ = np.array(grid_data[t]["FLRWATJ+"])
    FLRWATJ = FLRWATJ.reshape(Nz, Ny, Nx)
    if Nz > 1:
        FLRWATK = np.array(grid_data[t]["FLRWATK+"])
        FLRWATK = FLRWATK.reshape(Nz, Ny, Nx)
    else:
        FLRWATK = None

    # --- Water density [kg/m3] ---
    DENW    = np.array(grid_data[t]["DENW"])
    DENW    = DENW.reshape(Nz, Ny, Nx)

    return DP, Sg, FLRWATI, FLRWATJ, FLRWATK, DENW

def set_well_and_project_information(x_well_locations):
    """
    Set well and project information

    Returns:
    - df_well: DataFrame containing well information
    - projects: List of unique project names
    - new_projects_names: List of new project names

    Note:
    - new_projects_namesは途中から坑井を追加するプロジェクトの名前のリスト
    - 途中から坑井を追加するかどうかは、最適化の決定変数の中に入る可能性がある。その場合、この関数の出力を調整する
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    - プロジェクトの開始時期をどこかで設定する必要がある -> INPUT.DATAから読み取る形がベスト -> すべての坑井の開始時期を記録し、データフレームに格納する
    - ひとまずは、手打ちで設定するのが良いか。
    - new_wells.csvとexisting_wells.csvというファイルは良くないかも。データフレームで一元管理するのが良い。

    """
    # 1. Load configuration data
    df_existing_wells   = pd.read_csv("config/existing_wells.csv")  # Load the existing well information
    df_new_wells        = pd.read_csv("config/new_wells.csv")       # Load the new well information

    # 2. Set the project names
    new_projects_names  = df_new_wells['project'].unique()          # Project names which have new wells

    # UOFの可変パラメータ
    N_variable_wells   = len(df_new_wells)  # 変数井戸の数を取得
    N_variable_x = len(x_well_locations)/2
    if N_variable_wells != N_variable_x:
        raise ValueError(f"Number of variable wells ({N_variable_wells}) does not match the number of variable x ({N_variable_x}). Please check your input data.")
    for i in range(N_variable_wells):
        df_new_wells.loc[i, 'I'] = x_well_locations[2*i]    # 井戸のI座標を更新
        df_new_wells.loc[i, 'J'] = x_well_locations[2*i+1]

    df_well         = pd.concat([df_existing_wells, df_new_wells], ignore_index=True)  # 井戸情報を追加

    df_well["I(0)"] = df_well["I"] - 1 # Convert to 0-based index
    df_well["J(0)"] = df_well["J"] - 1 # Convert to 0-based index

    projects    = df_well['project'].unique()                           # Get the project names

    return df_well, projects