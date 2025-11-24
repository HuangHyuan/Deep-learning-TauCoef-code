import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from scipy.stats import zscore
#Function

def load_aam_bbm(nlev):
    if nlev == 91:
        return aam_91,bbm_91
    else:
        return aam_137,bbm_137
    
def ec_p(spres,nlev,aam,bbm):
#  Computes vertical pressure grid
#  associated to the input surface pressure
#  All pressures are in Pa
    """
    Vertical pressure grid based on input surface pressure
    
    Parameters:
    spres (float): Surface pressure, unit: Pa
    
    return:
    pap (numpy array): full layer pressure , unit: Pa
    paph (numpy array): Half layer pressure, unit: Pa
    """

    paph = [aam[j] + bbm[j] * spres for j in range(nlev + 1)]

    pap = [(paph[j] + paph[j + 1]) / 2 for j in range(nlev)]
    
    return np.array(pap), np.array(paph)

def generating_weight():
    """
    Generate the weight functions needed for different variables
    return:
       temp_weights (float): Temperature weight.
       humidity_weights (float): Humidity weight.
       ozone_weights (float): Ozone weight.
    """

    # Define ozone weight range [100, 1]
    lower_bound = 100  
    upper_bound = 1 

    #  Initialize the weight array
    temp_weights = np.zeros_like(pressure_level_101, dtype=float)
    humidity_weights = np.zeros_like(pressure_level_101, dtype=float)
    ozone_weights = np.zeros_like(pressure_level_101, dtype=float)

    #Define weight
    temp_weights = np.tile(1/101, 101) #Temperature pressure level equal weight
    humidity_weights = np.linspace(0, 1, num=len(pressure_level_101)) #Water vapor lower level high weight

    # Ozone stratospheric weights
    for i, p in enumerate(pressure_level_101):
        if lower_bound >= p >= upper_bound:  # Weights are only calculated in the [10e2,10e0] range
            # Map pressure values to the [-1, 1] range
            mapped_value = 2 * (np.log10(p) - np.log10(upper_bound)) / (np.log10(lower_bound) - np.log10(upper_bound)) - 1
            
            # The cosine function is used to calculate the weights
            weight = (np.cos(mapped_value * np.pi) + 1) / 2  #Normalized to [0, 1]
            ozone_weights[i] = weight

    return temp_weights,humidity_weights,ozone_weights


def interpolate_to_101_levels(dataset):
    """
    The profile interpolated to level 101 and extrapolated according to the rules:
    1、Temperature and ozone are extrapolated at constant values
    2、The water vapor mixing ratio is extrapolated from the dry adiabatic lapse rate
    Parameters:
        dataset (xr.Dataset): Contains the data sets of level_temperature and level_absorber.
        
    return:
       new_dataset (xr.Dataset): Interpolated 101-layer data set.
    """
    # Get original data
    temperature = dataset['level_temperature'].values  # temperature (n_profiles, n_levels)
    humidity = dataset['level_absorber'].values[:, 0, :]  # water vapor mixing ratio (n_profiles, n_levels)
    ozone = dataset['level_absorber'].values[:, 1, :]  # ozone mixing ratio (n_profiles, n_levels)

    # Initializes the interpolated array
    interpolated_temperature = np.zeros((temperature.shape[0], len(pressure_level_101)))
    interpolated_humidity = np.zeros_like(interpolated_temperature)
    interpolated_ozone = np.zeros_like(interpolated_temperature)

    # Cycle each profile
    for i in range(temperature.shape[0]):
        original_levels = dataset['level_pressure'].values[i, :]  # original level pressure (n_levels,)
        surface_pressure = dataset['surface_pressure'].values[i]  # surface pressure (hPa)

        # Current profile temperature and absorption gas
        temp_profile = temperature[i]
        humidity_profile = humidity[i]
        ozone_profile = ozone[i]

        # Find the area below the surface pressure
        mask_surface = original_levels < surface_pressure
        valid_levels = original_levels[mask_surface]
        valid_temp = temp_profile[mask_surface]
        valid_humidity = humidity_profile[mask_surface]
        valid_ozone = ozone_profile[mask_surface]

        # Create interpolation function
        temp_interp = interp1d(valid_levels, valid_temp, fill_value="extrapolate", bounds_error=False)
        humidity_interp = interp1d(valid_levels, valid_humidity, fill_value="extrapolate", bounds_error=False)
        ozone_interp = interp1d(valid_levels, valid_ozone, fill_value="extrapolate", bounds_error=False)

        # Interpolate to the new 101 layer
        mask_101 = pressure_level_101 < max(valid_levels)  # Interpolation range
        interpolated_temperature[i, mask_101] = temp_interp(np.array(pressure_level_101)[mask_101])
        interpolated_humidity[i, mask_101] = humidity_interp(np.array(pressure_level_101)[mask_101])
        interpolated_ozone[i, mask_101] = ozone_interp(np.array(pressure_level_101)[mask_101])

        # Extrapolation part: out-of-scope processing
        if surface_pressure < 1100:
            mask_extrap = ~mask_101  # Extrapolation range
            interpolated_temperature[i, mask_extrap] = valid_temp[-1]  # Temperature is extrapolated to a constant value
            interpolated_ozone[i, mask_extrap] = valid_ozone[-1]  # Same as temperature

            # Water vapor mixing ratio is extrapolated from the dry adiabatic lapse rate
            dT_dz = -9.8 / 1000  # Dry adiabatic lapse rate (K/m)            
            Rv = 461.5  # Water vapor gas constant (J/kg/K)            formula:  q(z) = q0·exp(-(g·▲Z)/(Rv·T(z)))
            g = 9.8  # Acceleration of gravity (m/s^2)

            for j in np.where(mask_extrap)[0]:
                delta_z = (np.log(surface_pressure) - np.log(pressure_level_101[j])) * Rv * interpolated_temperature[i, j - 1] / g
                new_temp = interpolated_temperature[i, j - 1] + dT_dz * delta_z
                interpolated_humidity[i, j] = valid_humidity[-1] * np.exp(-g * delta_z / (Rv * new_temp))

    # Update level_pressure data
    interpolated_pressure = np.tile(pressure_level_101, (temperature.shape[0], 1))

    # Create a new Dataset
    dataset_interpolated = xr.Dataset(
        {
            'level_temperature': (('n_profiles', 'n_levels'), interpolated_temperature),
            'layer_temperature': (('n_profiles', 'n_layers'), (interpolated_temperature[:,:-1]+interpolated_temperature[:,1:])/2),
            'level_absorber': (('n_profiles', 'n_absorbers', 'n_levels'),
                               np.stack([interpolated_humidity, interpolated_ozone], axis=1)),
            'layer_absorber': (('n_profiles', 'n_absorbers', 'n_layers'),
                               np.stack([(interpolated_humidity[:,:-1]+interpolated_humidity[:,1:])/2, (interpolated_ozone[:,:-1]+interpolated_ozone[:,1:])/2], axis=1)),
            'absorber_id': (('n_absorbers'), np.array([1, 3])), 
            'absorber_units_id': (('n_absorbers'), np.array([3, 1])),  
            'surface_altitude': ('n_profiles', dataset.surface_altitude.values),
            'level_pressure': (('n_profiles', 'n_levels'), interpolated_pressure),
            'latitude': ('n_profiles', dataset.latitude.values),
            'longitude': ('n_profiles', dataset.longitude.values),
            'year': ('n_profiles', dataset.year.values),
            'month': ('n_profiles', dataset.month.values),
            'day': ('n_profiles', dataset.day.values),
        },
        coords={
            'n_profiles': np.arange(interpolated_temperature.shape[0]),
            'n_levels': pressure_level_101,
            'n_layers': np.arange(len(pressure_level_101)-1),
            'n_absorbers': np.arange(2),
        }
    )

    return dataset_interpolated

def detect_outlier_profiles_by_layer(dataset, threshold=3):
    """
    Detects which profiles contain outliers, based on the Z-score of each level.
    
    Parameters:
        dataset (xr.Dataset):  Contains the data sets of level_temperature and level_absorber.
        threshold (float): Z-score threshold. The default value is 3.
    
    return:
        outlier_mask (np.ndarray): Boolean array, True indicates that the profile is an outlier profile
    """
    # Extract temperature, humidity and ozone concentration.
    temperature = dataset.level_temperature.values  # (profiles, pressure_levels)
    humidity = dataset.level_absorber[:, 0, :].values  # (profiles, pressure_levels)
    ozone = dataset.level_absorber[:, 1, :].values  # (profiles, pressure_levels)

    # Combine three variables into a three-dimensional array. (profiles, pressure_levels, features)
    data = np.stack([temperature, humidity, ozone], axis=-1)  #  (profiles, pressure_levels, 3)
    num_profiles, num_layers, num_features = data.shape
    
    # Initialize the Z-score matrix.
    z_scores = np.zeros_like(data)
    
    # Z-scores are calculated separately for each level.
    for layer in range(num_layers):
        layer_data = data[:, layer, :]  # (profiles, features)
        z_scores[:, layer, :] = np.abs(zscore(layer_data, axis=0))  # Calculate z-scores
    
    # Find out the profile of any level and any feature beyond the threshold value.
    outlier_mask = np.any(z_scores > threshold, axis=(1, 2))  # Judging by profile dimension

    # Screening non-outlier profiles
    selected_temperature = temperature[~outlier_mask]
    selected_humidity = humidity[~outlier_mask]
    selected_ozone = ozone[~outlier_mask]

    # Gets filtering results for other auxiliary variables
    best_selection = ~outlier_mask

    # Create a new Dataset
    dataset_clean = xr.Dataset(
        {
            'level_temperature': (('n_profiles', 'n_levels'), selected_temperature),
            'layer_temperature': (('n_profiles', 'n_layers'), dataset.layer_temperature.values[best_selection,:]),
            'level_absorber': (('n_profiles', 'n_absorbers', 'n_levels'),
                            np.stack([selected_humidity, selected_ozone], axis=1)),
            'layer_absorber': (('n_profiles', 'n_absorbers', 'n_layers'),
                               dataset.layer_absorber.values[best_selection, :,:]),
            'absorber_id': (('n_absorbers'), np.array([1, 3])),
            'absorber_units_id': (('n_absorbers'), np.array([3, 1])),
            'surface_altitude': ('n_profiles', dataset.surface_altitude.values[best_selection]),
            'level_pressure': (('n_profiles', 'n_levels'), dataset.level_pressure.values[best_selection, :]),
            'latitude': ('n_profiles', dataset.latitude.values[best_selection]),
            'longitude': ('n_profiles', dataset.longitude.values[best_selection]),
            'year': ('n_profiles', dataset.year.values[best_selection]),
            'month': ('n_profiles', dataset.month.values[best_selection]),
            'day': ('n_profiles', dataset.day.values[best_selection]),
        },
        coords={
            'n_profiles': np.arange(selected_temperature.shape[0]),
            'n_levels': np.arange(selected_temperature.shape[1]),
            'n_layers': np.arange(selected_temperature.shape[1]-1),
            'n_absorbers': np.arange(2),
        }
    )
    return dataset_clean,outlier_mask

def detect_outlier_profiles_by_ozone(dataset, is_91=False):
    """
    Detects which profiles contain outliers based on the ozone concentration within the pressure levels [1, 100].
    
    Parameters:
        dataset (xr.Dataset): Contains the data sets of level_temperature and level_absorber.
        is_91 (bool): If True, apply a special condition for the range [20, 60].
    
    Returns:
        dataset_clean (xr.Dataset): A new dataset with outlier profiles removed.
        outlier_mask (np.ndarray): Boolean array, True indicates that the profile is an outlier profile.
    """
    # Extract ozone concentration data
    ozone = dataset.level_absorber[:, 1, :].values  # (profiles, pressure_levels)

    # Get pressure level information
    pressure_levels = dataset.level_pressure.values[0, :]  # Assume all profiles share the same pressure levels

    # Determine the valid pressure level range [0.005, 1]
    valid_levels_mask = (pressure_levels >= 0.005) & (pressure_levels <= 1)
    if is_91:
        # Add the range [20, 60] when is_91 is True
        valid_levels_mask = ((pressure_levels >= 0.005) & (pressure_levels <= 1)) | \
                            ((pressure_levels >= 20) & (pressure_levels <= 50))
    valid_levels_indices = np.where(valid_levels_mask)[0]

    # Extract ozone concentration data for valid pressure levels
    ozone_valid_levels = ozone[:, valid_levels_indices]  # (profiles, valid_levels)

    # Calculate the mean ozone concentration for each valid pressure level
    ozone_mean_per_layer = np.mean(ozone_valid_levels, axis=0)  # (valid_levels,)

    # Identify profiles where ozone concentration exceeds thresholds
    if is_91:
        # Split the valid levels into two ranges: [0.005, 1] and [20, 60]
        range_1_mask = (pressure_levels[valid_levels_indices] >= 0.005) & (pressure_levels[valid_levels_indices] <= 1)
        range_2_mask = (pressure_levels[valid_levels_indices] >= 20) & (pressure_levels[valid_levels_indices] <= 60)

        # Outlier detection for range [0.005, 1]: threshold = 3 * mean
        outlier_mask_range_1 = np.any(ozone_valid_levels[:, range_1_mask] > 3 * ozone_mean_per_layer[range_1_mask], axis=1)

        # Outlier detection for range [20, 60]: threshold = 2 * mean
        outlier_mask_range_2 = np.any(ozone_valid_levels[:, range_2_mask] > 2 * ozone_mean_per_layer[range_2_mask], axis=1)

        # Combine outlier masks from both ranges
        outlier_mask = outlier_mask_range_1 | outlier_mask_range_2
    else:
        # Default outlier detection for all valid levels: threshold = 3 * mean
        outlier_mask = np.any(ozone_valid_levels > 3 * ozone_mean_per_layer, axis=1)

    # Get the selection mask for auxiliary variables
    best_selection = ~outlier_mask

    # Create a cleaned dataset
    dataset_clean = xr.Dataset(
        {
            'level_temperature': (('n_profiles', 'n_levels'), dataset.level_temperature.values[best_selection, :]),
            'layer_temperature': (('n_profiles', 'n_layers'), 
                                  dataset.layer_temperature.values[best_selection, :]),
            'level_absorber': (('n_profiles', 'n_absorbers', 'n_levels'),
                               np.stack([dataset.level_absorber[:, 0, :].values[best_selection, :], dataset.level_absorber[:, 1, :].values[best_selection, :]], axis=1)),
            'layer_absorber': (('n_profiles', 'n_absorbers', 'n_layers'),
                               dataset.layer_absorber.values[best_selection, :, :]),
            'absorber_id': (('n_absorbers'), np.array([1, 3])),
            'absorber_units_id': (('n_absorbers'), np.array([3, 1])),
            'surface_altitude': ('n_profiles', dataset.surface_altitude.values[best_selection]),
            'level_pressure': (('n_profiles', 'n_levels'), dataset.level_pressure.values[best_selection, :]),
            'latitude': ('n_profiles', dataset.latitude.values[best_selection]),
            'longitude': ('n_profiles', dataset.longitude.values[best_selection]),
            'year': ('n_profiles', dataset.year.values[best_selection]),
            'month': ('n_profiles', dataset.month.values[best_selection]),
            'day': ('n_profiles', dataset.day.values[best_selection]),
        },
        coords={
            'n_profiles': np.arange(sum(best_selection)),
            'n_levels': np.arange(ozone.shape[1]),
            'n_layers': np.arange(ozone.shape[1] - 1),
            'n_absorbers': np.arange(2),
        }
    )
    return dataset_clean, outlier_mask

#Unit conversion

def specific_humidity_to_mixing_ratio(hum):
    """
    Convert specific humidity from kg/kg to mass mixing ratio g/kg.
    
    Parameters:
        hum (float): specific humidity, unit: kg/kg
        
    return:
        mass_mix_ratio (float): mass mixing ratio, unit: g/kg
    """

    if np.all(hum >= 1) or np.all(hum <= 0):

        raise ValueError("The specific humidity value  should be between 0 and 1.")
    
    mass_mix_ratio = (hum / (1 - hum)) * 1000  

    return mass_mix_ratio


def ozone_kg_per_kg_to_ppmv(c_kg_per_kg):
    """
    Convert ozone concentration from kg/kg to ppmv.
    
    Parameters:
        c_kg_per_kg (float): ozone concentration, unit: kg/kg
        
    return:
        ppmv (float): volume mixing ratio, unit: ppmv
    """

    M_air = 28.97  # Molar mass of dry air (g/mol)

    M_ozone = 48.00  # Molar mass of dry ozone (g/mol)

    ppmv = c_kg_per_kg * (M_air / M_ozone) * 1e6 

    return ppmv


#Common constant

aam_137 = [
    0.000000, 2.000365, 3.102241, 4.666084, 6.827977, 9.746966, 13.605424, 18.608931, 24.985718,
    32.985710, 42.879242, 54.955463, 69.520576, 86.895882, 107.415741, 131.425507, 159.279404,
    191.338562, 227.968948, 269.539581, 316.420746, 368.982361, 427.592499, 492.616028, 564.413452,
    643.339905, 729.744141, 823.967834, 926.344910, 1037.201172, 1156.853638, 1285.610352, 1423.770142,
    1571.622925, 1729.448975, 1897.519287, 2076.095947, 2265.431641, 2465.770508, 2677.348145, 2900.391357,
    3135.119385, 3381.743652, 3640.468262, 3911.490479, 4194.930664, 4490.817383, 4799.149414, 5119.895020,
    5452.990723, 5798.344727, 6156.074219, 6526.946777, 6911.870605, 7311.869141, 7727.412109, 8159.354004,
    8608.525391, 9076.400391, 9562.682617, 10065.978516, 10584.631836, 11116.662109, 11660.067383, 12211.547852,
    12766.873047, 13324.668945, 13881.331055, 14432.139648, 14975.615234, 15508.256836, 16026.115234, 16527.322266,
    17008.789062, 17467.613281, 17901.621094, 18308.433594, 18685.718750, 19031.289062, 19343.511719, 19620.042969,
    19859.390625, 20059.931641, 20219.664062, 20337.863281, 20412.308594, 20442.078125, 20425.718750, 20361.816406,
    20249.511719, 20087.085938, 19874.025391, 19608.572266, 19290.226562, 18917.460938, 18489.707031, 18006.925781,
    17471.839844, 16888.687500, 16262.046875, 15596.695312, 14898.453125, 14173.324219, 13427.769531, 12668.257812,
    11901.339844, 11133.304688, 10370.175781, 9617.515625, 8880.453125, 8163.375000, 7470.343750, 6804.421875,
    6168.531250, 5564.382812, 4993.796875, 4457.375000, 3955.960938, 3489.234375, 3057.265625, 2659.140625,
    2294.242188, 1961.500000, 1659.476562, 1387.546875, 1143.250000, 926.507812, 734.992188, 568.062500, 424.414062,
    302.476562, 202.484375, 122.101562, 62.781250, 22.835938, 3.757813, 0.000000, 0.000000
]

aam_91 = [
    0.000000, 2.000040, 3.980832, 7.387186, 12.908319, 21.413612,
    33.952858, 51.746601, 76.167656, 108.715561, 150.986023, 204.637451,
    271.356506, 352.824493, 450.685791, 566.519226, 701.813354, 857.945801,
    1036.166504, 1237.585449, 1463.163940, 1713.709595, 1989.874390,
    2292.155518, 2620.898438, 2976.302246, 3358.425781, 3767.196045,
    4202.416504, 4663.776367, 5150.859863, 5663.156250, 6199.839355,
    6759.727051, 7341.469727, 7942.926270, 8564.624023, 9208.305664,
    9873.560547, 10558.881836, 11262.484375, 11982.662109, 12713.897461,
    13453.225586, 14192.009766, 14922.685547, 15638.053711, 16329.560547,
    16990.623047, 17613.281250, 18191.029297, 18716.968750, 19184.544922,
    19587.513672, 19919.796875, 20175.394531, 20348.916016, 20434.158203,
    20426.218750, 20319.011719, 20107.031250, 19785.357422, 19348.775391,
    18798.822266, 18141.296875, 17385.595703, 16544.585938, 15633.566406,
    14665.645508, 13653.219727, 12608.383789, 11543.166992, 10471.310547,
    9405.222656, 8356.252930, 7335.164551, 6353.920898, 5422.802734,
    4550.215820, 3743.464355, 3010.146973, 2356.202637, 1784.854614,
    1297.656128, 895.193542, 576.314148, 336.772369, 162.043427,
    54.208336, 6.575628, 0.003160, 0.000000]

bbm_137 = [
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000007, 0.000024, 0.000059, 0.000112, 0.000199, 
    0.000340, 0.000562, 0.000890,0.001353, 0.001992, 0.002857, 0.003971, 0.005378, 0.007133, 0.009261, 
    0.011806, 0.014816, 0.018318,0.022355, 0.026964, 0.032176, 0.038026, 0.044548, 0.051773, 0.059728, 
    0.068448, 0.077958, 0.088286,0.099462, 0.111505, 0.124448, 0.138313, 0.153125, 0.168910, 0.185689, 
    0.203491, 0.222333, 0.242244,0.263242, 0.285354, 0.308598, 0.332939, 0.358254, 0.384363, 0.411125, 
    0.438391, 0.466003, 0.493800,0.521619, 0.549301, 0.576692, 0.603648, 0.630036, 0.655736, 0.680643, 
    0.704669, 0.727739, 0.749797,0.770798, 0.790717, 0.809536, 0.827256, 0.843881, 0.859432, 0.873929, 
    0.887408, 0.899900, 0.911448,0.922096, 0.931881, 0.940860, 0.949064, 0.956550, 0.963352, 0.969513, 
    0.975078, 0.980072, 0.984542,0.988500, 0.991984, 0.995003, 0.997630, 1.000000
]

bbm_91 = [
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000014, 
    0.000055, 0.000131, 0.000279, 0.000548, 0.001000,0.001701, 
    0.002765, 0.004267, 0.006322, 0.009035, 0.012508,0.016860, 
    0.022189, 0.028610, 0.036227, 0.045146, 0.055474,0.067316, 
    0.080777, 0.095964, 0.112979, 0.131935, 0.152934,0.176091, 
    0.201520, 0.229315, 0.259554, 0.291993, 0.326329,0.362203,
    0.399205, 0.436906, 0.475016, 0.513280, 0.551458,0.589317, 
    0.626559, 0.662934, 0.698224, 0.732224, 0.764679,0.795385, 
    0.824185, 0.850950, 0.875518, 0.897767, 0.917651,0.935157, 
    0.950274, 0.963007, 0.973466, 0.982238, 0.989153,0.994204, 
    0.997630, 1.000000]

pressure_level_101 = [
    0.005, .016, .038, .077, .137, .224, .345, .506, .714, .975,
    1.297, 1.687, 2.153, 2.701, 3.340, 4.077, 4.920, 5.878, 6.957, 8.165,
    9.512, 11.004, 12.649, 14.456, 16.432, 18.585, 20.922, 23.453, 26.183,
    29.121, 32.274, 35.651, 39.257, 43.100, 47.188, 51.528, 56.126, 60.989,
    66.125, 71.540, 77.240, 83.231, 89.520, 96.114, 103.017, 110.237, 117.777,
    125.646, 133.846, 142.385, 151.266, 160.496, 170.078, 180.018, 190.320,
    200.989, 212.028, 223.441, 235.234, 247.408, 259.969, 272.919, 286.262,
    300.000, 314.137, 328.675, 343.618, 358.966, 374.724, 390.893, 407.474,
    424.470, 441.882, 459.712, 477.961, 496.630, 515.720, 535.232, 555.167,
    575.525, 596.306, 617.511, 639.140, 661.192, 683.667, 706.565, 729.886,
    753.628, 777.790, 802.371, 827.371, 852.788, 878.620, 904.866, 931.524,
    958.591, 986.067, 1013.948, 1042.232, 1070.917, 1100.000
]
