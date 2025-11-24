"""
Optimized and Engineering-Oriented Script for Deep Learning Model Training
on Atmospheric Optical Depth Prediction using ResChannelAttention.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Custom modules
from DL_MODEL_Attachment import ResChannelAttention, train_model_Jacobian,convert_JOD_to_Jnorm
from TauDL_Create_Attachment import (
    preprocess_data_Channel,
    preprocess_data_Channel_inverse,
    load_tau_data,
    build_predictors,
    compute_optical_depth,
    read_od_tl_file
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------
# Configuration
# ----------------------------
PROJECT_ROOT = Path(".")
MODEL_ROOT = Path("./DL_MODEL")
JACOBIAN_ROOT = Path("./DL_Jacobian")  # Requires export from ARMS
DATA_DIR = PROJECT_ROOT / "TauData"
ATM_DIR = PROJECT_ROOT / "Atmosphere"

# Model paths
MODEL_LOAD_PATH = MODEL_ROOT / "Best_Model_EarlyStop.pth"
MODEL_SAVE_PATH = MODEL_ROOT / "Model_best_Jacobian.pth"

# Data file paths
TRAIN_DATA = {
    "tau": DATA_DIR / "MWHS_FY3F_V2_1850.TotTauProfile.nc",
    "atm": ATM_DIR / "IFS_NUM_1850.AtmProfile.nc"
}
VAL_DATA = {
    "tau": DATA_DIR / "MWHS_FY3F_V2_83.TotTauProfile.nc",
    "atm": ATM_DIR / "IFS_137_101LVL_83ATM.AtmProfile.nc"
}

# Jacobian file paths
TRAIN_J_DATA = {
    "Q": JACOBIAN_ROOT / "OD_TL_Q_FY3F.txt",
    "T": JACOBIAN_ROOT / "OD_TL_T_FY3F.txt"
}
VAL_J_DATA = {
    "Q": JACOBIAN_ROOT / "OD_TL_Q_FY3F_Val.txt",
    "T": JACOBIAN_ROOT / "OD_TL_T_FY3F_Val.txt"
}
# Model hyperparameters
HIDDEN_SIZE = 256
NUM_LAYERS = 3
BATCH_SIZE = 64
NUM_EPOCHS = 5  # Set to 5 for testing; typically 80 in production
INPUT_PREDICTORS = 3  # water vapor, temperature, zenith angle
N_CHANNEL = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# Main Execution
# ----------------------------

def main() -> None:
    logger.info("Starting optical depth prediction training...")
    # -------------------------------
    # Load and preprocess training data
    # -------------------------------
    tot_tau_train, atm_train = load_tau_data(TRAIN_DATA["tau"], TRAIN_DATA["atm"])
    n_profile_train = tot_tau_train['Sensor_TotTau'].shape[0]
    n_ang_train = tot_tau_train['Sensor_TotTau'].shape[1]
    n_lay = tot_tau_train['Sensor_TotTau'].shape[2] - 1
    n_channel = tot_tau_train['Sensor_TotTau'].shape[3]

    predictors_train = build_predictors(tot_tau_train, atm_train, n_profile_train, n_ang_train, n_lay)
    optical_depth_train = compute_optical_depth(
        tot_tau_train['Sensor_TotTau'].values, n_profile_train, n_ang_train, n_lay, n_channel
    )

    X_train, Y_train, scaler_X, scaler_Y = preprocess_data_Channel(predictors_train, optical_depth_train)
    logger.info(f"Training data prepared: X={X_train.shape}, Y={Y_train.shape}")

    # -------------------------------
    # Load and preprocess validation data
    # -------------------------------
    tot_tau_val, atm_val = load_tau_data(VAL_DATA["tau"], VAL_DATA["atm"])
    n_profile_val = tot_tau_val['Sensor_TotTau'].shape[0]
    n_ang_val = tot_tau_val['Sensor_TotTau'].shape[1]

    predictors_val = build_predictors(tot_tau_val, atm_val, n_profile_val, n_ang_val, n_lay)
    optical_depth_val = compute_optical_depth(
        tot_tau_val['Sensor_TotTau'].values, n_profile_val, n_ang_val, n_lay, n_channel
    )

    X_val, Y_val = preprocess_data_Channel_inverse(predictors_val, optical_depth_val, scaler_X, scaler_Y)
    logger.info(f"Validation data prepared: X_val={X_val.shape}, Y_val={Y_val.shape}")

    # Extract scale factors from scalers
    scalerX_std = np.array([[scaler_X[lay].scale_[pred] for pred in range(INPUT_PREDICTORS)]
                            for lay in range(n_lay)])
    scalerY_std = np.array([[scaler_Y[ch][lay].scale_[0] for ch in range(N_CHANNEL)]
                            for lay in range(n_lay)])
    # -------------------------------
    # Load and process Jacobian reference data
    # -------------------------------
    logger.info("Loading Jacobian sensitivity data from ARMS...")
    Q_TL_train = read_od_tl_file(TRAIN_J_DATA["Q"])  # Water vapor sensitivity
    T_TL_train = read_od_tl_file(TRAIN_J_DATA["T"])  # Temperature sensitivity

    Q_TL_val = read_od_tl_file(VAL_J_DATA["Q"])  # Water vapor sensitivity
    T_TL_val = read_od_tl_file(VAL_J_DATA["T"])  # Temperature sensitivity

    # Scale Jacobians using normalization parameters
    J_train_tensor = convert_JOD_to_Jnorm(Q_TL_train,T_TL_train,n_ang_train,scalerX_std,scalerY_std)
    J_val_tensor = convert_JOD_to_Jnorm(Q_TL_val,T_TL_val,n_ang_val,scalerX_std,scalerY_std)

    J_ref_1 = J_train_tensor[::15, :, :, 0].clone()  # [N_sub, H, W] -> 每个样本上输入通道1的雅可比
    J_ref_2 = J_train_tensor[::15, :, :, 1].clone()  # [N_sub, H, W]

    # 计算每个空间位置 (h,w) 上的 min/max
    max_map_1 = torch.max(J_ref_1, dim=0)[0]  # [H, W]
    min_map_1 = torch.min(J_ref_1, dim=0)[0]

    max_map_2 = torch.max(J_ref_2, dim=0)[0]
    min_map_2 = torch.min(J_ref_2, dim=0)[0]

    # 构造 Module 保存多个边界
    class BoundsModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.min_map_1 = torch.nn.Parameter(min_map_1, requires_grad=False)
            self.max_map_1 = torch.nn.Parameter(max_map_1, requires_grad=False)
            self.min_map_2 = torch.nn.Parameter(min_map_2, requires_grad=False)
            self.max_map_2 = torch.nn.Parameter(max_map_2, requires_grad=False)
        
        def forward(self, x):
            return x

    # trace 并保存
    model_bounds = BoundsModule()
    example_input = torch.randn(1, 1)
    scripted_bounds = torch.jit.trace(model_bounds, example_input)
    scripted_bounds.save("FY3F_jacobian_clamp_bounds.pt")
    
    # Convert scalers to tensor for Jacobian regularization
    scaler_x_tensor = torch.tensor(scalerX_std, dtype=torch.float32)  # (L, P)
    scaler_y_tensor = torch.tensor(scalerY_std, dtype=torch.float32)  # (L, C)
    with torch.no_grad():
        OD_MIN = J_train_tensor.amin(dim=(0, 1))  # (output_size, input_size)
        OD_MAX = J_train_tensor.amax(dim=(0, 1))
    # -------------------------------
    # Initialize and load model
    # -------------------------------
    logger.info("Initializing ResChannelAttention model...")
    model = ResChannelAttention(INPUT_PREDICTORS, HIDDEN_SIZE, NUM_LAYERS, N_CHANNEL).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, weights_only=True))
    logger.info(f"Model loaded from {MODEL_LOAD_PATH}")

    # -------------------------------
    # Train model with Jacobian regularization
    # -------------------------------
    logger.info("Starting model training with H¹ regularization...")
    model_trained, train_losses, val_losses = train_model_Jacobian(
        device=DEVICE,
        X_train=X_train,
        Y_train=Y_train,
        J_train=J_train_tensor,
        X_val=X_val,
        Y_val=Y_val,
        J_val=J_val_tensor,
        J_MIN=OD_MIN,
        J_MAX=OD_MAX,
        model=model,
        scalerX_std=scaler_x_tensor,
        scalerY_std=scaler_y_tensor,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        λ=0.005  # Tune as needed
    )

    # -------------------------------
    # Save final model
    # -------------------------------
    torch.save(model_trained.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Model training completed and saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()