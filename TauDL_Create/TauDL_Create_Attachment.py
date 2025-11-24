from sklearn.preprocessing import StandardScaler
import numpy as np
import xarray as xr
import torch
import struct
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# ----------------------------
# Utility Functions
# ----------------------------
def validate_files(files: dict):
    """Check if required files exist."""
    for key, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")


def preprocess_data_Channel(Predictors, OpticalDepth):
    """
    Preprocess input data for training a deep learning model.

    This function:
    - Reshapes data from (N_Profile, N_Ang, N_Lay, N_Feature) to (N_Profile * N_Ang, N_Lay, N_Feature)
      to treat each viewing angle as an independent sample.
    - Applies per-layer standardization (zero mean, unit variance) to both predictors and targets.
    - Returns fitted scalers for later inverse transformation (e.g., post-processing).

    Args:
        Predictors (np.ndarray): Input features of shape (N_Profile, N_Ang, N_Lay, N_Predictor).
                                E.g., atmospheric state variables (T, q, O3, p, etc.).
        OpticalDepth (np.ndarray): Target output of shape (N_Profile, N_Ang, N_Lay, N_Channel).
                                   E.g., channel-wise optical depth per layer.

    Returns:
        X (torch.Tensor): Standardized predictors, shape (N_Samples, N_Lay, N_Predictor)
        Y (torch.Tensor): Standardized optical depths, shape (N_Samples, N_Lay, N_Channel)
        scaler_X (list of StandardScaler): One scaler per vertical layer for input features.
        scaler_Y (list of list of StandardScaler): scaler_Y[ichannel][lay] for inverse transform.
    """
    # Extract dimensions
    N_Profile, N_Ang, N_Lay, N_Predictor = Predictors.shape
    _, _, _, N_Channel = OpticalDepth.shape  # Reuse N_Profile, N_Ang, N_Lay

    # Reshape: merge profile and angle dimensions -> treat each (profile, angle) as a sample
    # This increases effective batch size and assumes azimuthal symmetry or independent angles
    Predictors = Predictors.reshape(N_Profile * N_Ang, N_Lay, N_Predictor)  # (B, T, D_x)
    OpticalDepth = OpticalDepth.reshape(N_Profile * N_Ang, N_Lay, N_Channel)  # (B, T, D_y)

    # Initialize scalers and scaled data arrays
    scaler_X = []  # One StandardScaler per vertical layer (for input features)
    scaler_Y = [[] for _ in range(N_Channel)]  # Each channel has a list of scalers (one per layer)

    Predictors_scaled = np.zeros((N_Profile * N_Ang, N_Lay, N_Predictor), dtype=np.float32)
    OpticalDepth_scaled = np.zeros((N_Profile * N_Ang, N_Lay, N_Channel), dtype=np.float32)

    # Apply per-layer standardization (independent normalization across layers)
    for lay in range(N_Lay):
        # Fit and transform predictors at layer `lay`
        scaler_X_k = StandardScaler()
        Predictors_scaled[:, lay, :] = scaler_X_k.fit_transform(Predictors[:, lay, :])
        scaler_X.append(scaler_X_k)

        # Fit and transform each output channel at layer `lay`
        for ichannel in range(N_Channel):
            scaler_Y_ki = StandardScaler()
            OpticalDepth_scaled[:,lay,ichannel]= scaler_Y_ki.fit_transform(OpticalDepth[:,lay,ichannel].reshape(-1, 1)).flatten()
            scaler_Y[ichannel].append(scaler_Y_ki)

    # Convert to PyTorch tensors
    X = torch.tensor(Predictors_scaled, dtype=torch.float32)  # (B, T, D_x)
    Y = torch.tensor(OpticalDepth_scaled, dtype=torch.float32)  # (B, T, D_y)

    return X, Y, scaler_X, scaler_Y


def preprocess_data_Channel_inverse(Predictors, OpticalDepth, scaler_X, scaler_Y):
    """
    Preprocess data using pre-fitted scalers (for validation/test or inverse modeling).

    Unlike the training version, this function uses already-fitted scalers (no .fit() called).
    Useful for:
    - Preprocessing test data with training scalers.
    - Inverse models where inputs were previously standardized.

    Args:
        Predictors (np.ndarray): Input features, shape (N_Profile, N_Ang, N_Lay, N_Predictor)
        OpticalDepth (np.ndarray): Target/output, shape (N_Profile, N_Ang, N_Lay, N_Channel)
        scaler_X (list of StandardScaler): Fitted scalers for input features (one per layer)
        scaler_Y (list of list of StandardScaler): Fitted scalers for outputs, [ichannel][lay]

    Returns:
        X (torch.Tensor): Transformed predictors, shape (N_Samples, N_Lay, N_Predictor)
        Y (torch.Tensor): Transformed optical depths, shape (N_Samples, N_Lay, N_Channel)
    """
    # Extract dimensions
    N_Profile, N_Ang, N_Lay, N_Predictor = Predictors.shape
    _, _, _, N_Channel = OpticalDepth.shape

    # Reshape: merge profile and angle dimensions
    Predictors = Predictors.reshape(N_Profile * N_Ang, N_Lay, N_Predictor)
    OpticalDepth = OpticalDepth.reshape(N_Profile * N_Ang, N_Lay, N_Channel)

    # Initialize arrays for scaled data
    Predictors_scaled = np.zeros((N_Profile * N_Ang, N_Lay, N_Predictor))
    OpticalDepth_scaled = np.zeros((N_Profile * N_Ang, N_Lay, N_Channel))

    # Apply pre-fitted scalers (no fitting, only transformation)
    for lay in range(N_Lay):
        # Transform predictors using pre-fitted scaler for layer `lay`
        scaler_X_k = scaler_X[lay]
        Predictors_scaled[:, lay, :] = scaler_X_k.transform(Predictors[:, lay, :])

        # Transform each channel using pre-fitted scaler
        for ichannel in range(N_Channel):
            scaler_Y_ki = scaler_Y[ichannel][lay]
            OpticalDepth_scaled[:,lay,ichannel]= scaler_Y_ki.transform(OpticalDepth[:,lay,ichannel].reshape(-1, 1)).flatten()


    # Convert to PyTorch tensors
    X = torch.tensor(Predictors_scaled, dtype=torch.float32)
    Y = torch.tensor(OpticalDepth_scaled, dtype=torch.float32)

    return X, Y

def Read_TotauData(file_name):

    DryTotTauData = xr.open_dataset(file_name)

    n_Profile = DryTotTauData.Sensor_TotTau.values.shape[0]
    n_Angle = DryTotTauData.Sensor_TotTau.values.shape[1]
    n_level = 101
    n_channel = 15

    ToTalTau = np.ones((n_Profile, n_Angle, n_level, n_channel))
    ToTalTau[:,:,1:,:] = DryTotTauData.Sensor_TotTau.values
    ToTalTau_dataset = xr.Dataset(
        {
            'Sensor_TotTau': (('n_Profile','n_Ang','n_levels','n_Channel') , ToTalTau),
            'Angle': ('n_Ang',DryTotTauData.Angle.values),
        },
        coords={
            'n_profiles': np.arange(ToTalTau.shape[0]),
            'n_levels': np.arange(ToTalTau.shape[2]),
            'n_layers': np.arange(ToTalTau.shape[2]-1),
            'n_Ang': np.arange(ToTalTau.shape[1]),
            'n_Channel': np.arange(ToTalTau.shape[3]),
        }
    )

    return ToTalTau_dataset

def save_scaler_binary(scaler_dict, filepath: str, is_y: bool = True):
    """
    Save scaler mean and scale to binary file in big-endian double format.
    
    Args:
        scaler_dict: dict of scalers (layer-wise for X, channel-layer for Y)
        filepath: output binary file path
        is_y: True if saving Y (channel, layer), False for X (layer, predictor)
    """
    with open(filepath, 'wb+') as f:
        if is_y:
            # Shape: [N_Channel, N_Lay] -> each has .mean_[0], .scale_[0]
            N_Channel = 15
            N_Lay = 100
            for lay in range(N_Lay):
                for ichannel in range(N_Channel):
                    f.write(struct.pack('>d', scaler_dict[ichannel][lay].mean_[0]))
                    f.write(struct.pack('>d', scaler_dict[ichannel][lay].scale_[0]))
        else:
            # Shape: [N_Lay], each scaler has multiple predictors
            N_Lay = 100
            N_Predictors = 3
            for lay in range(N_Lay):
                for pred_idx in range(N_Predictors):
                    f.write(struct.pack('>d',scaler_dict[lay].mean_[pred_idx]))
                    f.write(struct.pack('>d',scaler_dict[lay].scale_[pred_idx]))
    logger.info(f"Saved scaler data to {filepath}")

def load_tau_data(tau_path, atm_path):
    """Load and preprocess  data."""
    logger.info("Loading tau data...")
    validate_files({"tau": tau_path, "atm": atm_path})

    tot_tau = Read_TotauData(tau_path)
    atm = xr.open_dataset(atm_path)

    # Replace zero tau with small positive value
    tot_tau['Sensor_TotTau'] = tot_tau['Sensor_TotTau'].where(
        tot_tau['Sensor_TotTau'] != 0, np.float64(2e-16)
    )

    return tot_tau, atm

def build_predictors(tot_tau_data, atm_data, n_profile: int, n_ang: int, n_lay: int):
    """
    Construct input predictors: layer gas, temperature, and zenith angle.
    """
    zenith_angle = tot_tau_data.Angle.values
    predictors = np.zeros((n_profile, n_ang, n_lay, 3), dtype=np.float64)

    # Flip vertical dimension (top-down to bottom-up)
    layer_gases = np.flip(atm_data.variables['layer_absorber'][:,:,:], axis=2).values
    layer_temperature = np.flip(atm_data.variables['layer_temperature'][:], axis=1).values 

    # Assign features
    predictors[..., 0] = layer_gases[:, 0, :].reshape(n_profile, 1, n_lay)  # water vapor
    predictors[..., 1] = layer_temperature.reshape(n_profile, 1, n_lay)     # temperature
    for iang in range(n_ang):
        predictors[:, iang, :, 2] = zenith_angle[iang]  # zenith angle broadcast

    return predictors

def compute_optical_depth(tot_tau_values, n_profile: int, n_ang: int, n_lay: int, n_channel: int):
    """Compute optical depth from total transmittance."""
    tau_top = tot_tau_values[:, :n_ang, :n_lay, :]
    tau_bottom = tot_tau_values[:, :n_ang, 1:n_lay + 1, :]
    optical_depth = -np.log(tau_bottom / tau_top, dtype=np.float64)

    # Clip invalid values
    optical_depth[np.isnan(optical_depth)] = 0.0
    optical_depth[optical_depth < 0.0] = 0.0
    # Optional: clip extreme values
    # optical_depth = np.clip(optical_depth, 0.0, Common.MAX_TotOD)

    return optical_depth

def read_od_tl_file(filename):
    """
    Read Jacobian sensitivity data from OD_TL output file.

    Format:
        ### OD_TL output at Profile X, channel Y
        value_0
        value_1
        ...
        value_99

    Returns:
        Array of shape (N_profiles, N_layers=100, N_channels=15)
    """
    data = []  # List to collect each channel-layer profile

    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("### OD_TL output at Profile"):
            try:
                parts = line.split(',')
                profile_id = int(parts[0].split("Profile")[1]) - 1
                channel_id = int(parts[1].split("channel")[1]) - 1

                # Read next 100 values
                values = [float(lines[i + j].strip()) for j in range(1, 101)]
                data.append((profile_id, channel_id, values))
                i += 101
            except (IndexError, ValueError) as e:
                logger.warning(f"Failed to parse block at line {i}: {e}")
                i += 1
                continue
        else:
            i += 1

    if not data:
        raise ValueError(f"No valid data found in {filename}")

    # Determine max profile index
    max_profile = max(d[0] for d in data)
    output = np.zeros((max_profile + 1, 100, 15), dtype=np.float64)

    for profile_id, channel_id, values in data:
        output[profile_id, :, channel_id] = values

    return output  # shape: (N_profiles, N_Lay=100, N_Channel=15)