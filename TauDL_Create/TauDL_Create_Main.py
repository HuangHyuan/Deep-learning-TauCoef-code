"""
Optimized and Engineering-Oriented Script for Deep Learning Model Training
on Atmospheric Optical Depth Prediction using ResChannelAttention.
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm
import logging
from pathlib import Path
# Custom modules
from DL_MODEL_Attachment import ResChannelAttention, train_model
from TauDL_Create_Attachment import preprocess_data_Channel, preprocess_data_Channel_inverse, save_scaler_binary,load_tau_data,build_predictors,compute_optical_depth

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------
# Configuration
# ----------------------------
PROJECT_ROOT = Path(".")
MODEL_ROOT = Path("./DL_MODEL")
DATA_DIR = PROJECT_ROOT / "TauData"
ATM_DIR = PROJECT_ROOT / "Atmosphere"
MODEL_SAVE_PATH = MODEL_ROOT / "model_best.pth"

TRAIN_DATA = {
    "tau": DATA_DIR / "MWHS_FY3F_V2_1850.TotTauProfile.nc",
    "atm": ATM_DIR / "IFS_NUM_1850.AtmProfile.nc"
}
Val_DATA = {
    "tau": DATA_DIR / "MWHS_FY3F_V2_83.TotTauProfile.nc",
    "atm": ATM_DIR / "IFS_137_101LVL_83ATM.AtmProfile.nc"
}

SCALER_X_PATH = "scaler_X_FY3F_V4.bin"
SCALER_Y_PATH = "scaler_Y_FY3F_V4.bin"

# Model hyperparameters
HIDDEN_SIZE = 256
NUM_LAYERS = 3
BATCH_SIZE = 32
NUM_EPOCHS = 5 #Here, it is set to 5 for testing purposes. Generally, it can be set to 200
INPUT_PREDICTORS = 3  # water vapor, temperature, zenith angle
N_CHANNEL = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# Main Execution
# ----------------------------
def main():
    logger.info("Starting optical depth prediction ...")
    # --------------------------------------
    # --- Load and process training data ---
    # --------------------------------------

    tot_tau_train, atm_train = load_tau_data(TRAIN_DATA["tau"], TRAIN_DATA["atm"])
    n_profile_train = tot_tau_train['Sensor_TotTau'].shape[0]
    n_ang_train = tot_tau_train['Sensor_TotTau'].shape[1]
    n_lay = tot_tau_train['Sensor_TotTau'].shape[2] - 1
    n_channel = tot_tau_train['Sensor_TotTau'].shape[3]

    predictors_train = build_predictors(tot_tau_train, atm_train, n_profile_train, n_ang_train, n_lay)
    optical_depth_train = compute_optical_depth(
        tot_tau_train['Sensor_TotTau'].values, n_profile_train, n_ang_train, n_lay, n_channel
    )
    
    # Standardize the data structure required for the model
    X, Y, scaler_X, scaler_Y = preprocess_data_Channel(predictors_train, optical_depth_train)
    logger.info(f"Training data shape: X={X.shape}, Y={Y.shape}")

    # ---------------------------------
    # --- Load and process val data ---
    # ---------------------------------
    tot_tau_val, atm_val = load_tau_data(Val_DATA["tau"], Val_DATA["atm"])
    n_profile_val = tot_tau_val['Sensor_TotTau'].shape[0]
    n_ang_val = tot_tau_val['Sensor_TotTau'].shape[1]  # Hardcoded in original

    predictors_val = build_predictors(tot_tau_val, atm_val, n_profile_val, n_ang_val, n_lay)
    optical_depth_val = compute_optical_depth(
        tot_tau_val['Sensor_TotTau'].values, n_profile_val, n_ang_val, n_lay, n_channel
    )

    X_val, Y_val = preprocess_data_Channel_inverse(predictors_val, optical_depth_val, scaler_X, scaler_Y)
    logger.info(f"Validation data shape: X_val={X_val.shape}, Y_val={Y_val.shape}")

    # -----------------------------------------------
    # --- Save scalers for external use (AMRS_DL) ---
    # -----------------------------------------------
    save_scaler_binary(scaler_X, SCALER_X_PATH, is_y=False)
    save_scaler_binary(scaler_Y, SCALER_Y_PATH, is_y=True)

    # -------------------------------------
    # --- Model Definition and Training ---
    # -------------------------------------
    logger.info("Initializing ResChannelAttention model...")
    model = ResChannelAttention(INPUT_PREDICTORS, HIDDEN_SIZE, NUM_LAYERS, N_CHANNEL).to(DEVICE)

    # Train model using your existing train_model function
    logger.info("Starting model training...")
    model_trained, train_losses, val_losses = train_model(
        device=DEVICE,
        X_train=X, Y_train=Y,
        X_val=X_val, Y_val=Y_val,
        model=model,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS
    )

    # Optional: Save final model
    torch.save(model_trained.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Model training completed and saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()