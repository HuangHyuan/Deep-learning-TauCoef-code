import numpy as np
import xarray as xr
from sklearn.cluster import MiniBatchKMeans,KMeans
from concurrent.futures import ThreadPoolExecutor
from Common_Attachment import generating_weight,pressure_level_101

def select_profiles_with_clustering_and_ga(ds, bins=2000,n_clusters=80 ,num_profiles=2000, population_size=500, generations=100, mutation_rate=0.4):
    """
    The optimal D value is found by clustering and genetic algorithm
    
    Parameters:
    ds (xr.dataset): the profiles dataset used for clustering, which recommended to be the dataset after data cleaning.
    bins (int): corresponding to the number of profiles to be selected.
    num_profiles (int): The number of the selected profiles.
    population_size (int): the number of population.
    generations (int): the generation numbers of multiplication. 
    mutation_rate (float): the probability of mutation. 
    return:
    selected_dataset (xr.dataset)
    best_D (int): the Minimum of D values
    best_selection (list): selected profile label
    """

    #  Extract variable data
    temperature = ds.level_temperature.values  # (profiles, pressure_levels)
    humidity = ds.level_absorber[:, 0, :].values  # (profiles, pressure_levels)
    ozone = ds.level_absorber[:, 1, :].values  # (profiles, pressure_levels)
    month = ds.month.values.reshape(-1,1)
    lat = ds.latitude.values.reshape(-1,1)
    lon = ds.longitude.values.reshape(-1,1)
    # Data dimension check
    profiles, pressure_levels = temperature.shape
    assert humidity.shape == (profiles, pressure_levels), "Humidity shape mismatch!"
    assert ozone.shape == (profiles, pressure_levels), "Ozone shape mismatch!"
    assert num_profiles <= profiles, "Number of profiles to select cannot exceed the total number of samples!"
    # Separate containers for each level
    def get_level_bin_indices(data, bins):
        """
        Each level is partitioned separately, and the partitioned index of each level is returned.

        data: (profiles, pressure_levels)
        bins: Number of bins per level
        """
        bin_indices = np.zeros_like(data, dtype=int)
        for level in range(pressure_levels):
            bin_edges = np.linspace(np.min(data[:, level]), np.max(data[:, level]), bins + 1)
            bin_indices[:, level] = np.digitize(data[:, level], bin_edges) - 1  #Returns the bin index to which each data point belongs (starting from 0)
        return bin_indices

    temp_bins = get_level_bin_indices(temperature, bins)
    hum_bins = get_level_bin_indices(humidity, bins)
    oz_bins = get_level_bin_indices(ozone, bins)

    # Define a function that evaluates the value of D
    def calculate_D(selected_indices):
        """
        Calculation of D value: based on the box distribution of selected samples at each level.
        """
        D = 0
        for level in range(pressure_levels):
            temp_counts = np.bincount(temp_bins[selected_indices, level], minlength=bins)
            hum_counts = np.bincount(hum_bins[selected_indices, level], minlength=bins)
            oz_counts = np.bincount(oz_bins[selected_indices, level], minlength=bins)

            D += (
                np.sum(np.abs(temp_counts - 1)) +
                np.sum(np.abs(hum_counts - 1)) +
                np.sum(np.abs(oz_counts - 1))
            )
        return D

    # Using K-Means clustering
    features = np.hstack([temperature, humidity, ozone])  # Feature matrix, optional:month,lat,lon
    n_clusters = n_clusters#min(num_profiles, profiles)
    kmeans = KMeans(n_clusters=10, random_state=42)#MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    cluster_labels = kmeans.fit_predict(features)

    # Initialize the population: Randomly select samples from each cluster
    def initialize_population(cluster_labels, num_profiles, population_size):
        clusters = np.unique(cluster_labels)
        population = []
        for _ in range(population_size):
            selected_indices = set()
            for cluster in clusters:
                cluster_samples = np.where(cluster_labels == cluster)[0]
                selected_from_cluster = np.random.choice(
                    cluster_samples,
                    size=min(len(cluster_samples), num_profiles // len(clusters)),
                    replace=False
                )
                selected_indices.update(selected_from_cluster)
            # If the number of samples is insufficient, replenish the remaining samples
            if len(selected_indices) < num_profiles:
                remaining_samples = np.setdiff1d(np.arange(profiles), list(selected_indices))
                additional_samples = np.random.choice(
                    remaining_samples,
                    size=num_profiles - len(selected_indices),
                    replace=False
                )
                selected_indices.update(additional_samples)
            population.append(np.array(list(selected_indices)))
        return population

    population = initialize_population(cluster_labels, num_profiles, population_size)

    # Genetic algorithm main loop
    best_D = float('inf')
    best_selection = None

    for generation in range(generations):
        # Compute fitness
        def fitness_wrapper(individual):
            return 1 / (1 + calculate_D(individual)) # Fitness function: A smaller D value corresponds to a higher fitness

        with ThreadPoolExecutor() as executor:
            fitness = list(executor.map(fitness_wrapper, population))

        # Updated optimal solution
        current_best_fitness = max(fitness)
        current_best_idx = fitness.index(current_best_fitness)
        current_best_individual = population[current_best_idx]
        current_best_D = calculate_D(current_best_individual)

        if current_best_D < best_D:
            best_D = current_best_D
            best_selection = current_best_individual

        # Normalized fitness
        fitness = np.array(fitness)
        fitness /= np.sum(fitness)

        # Selection (Roulette selection)
        selected_indices = np.random.choice(
            range(population_size),
            size=population_size,
            p=fitness
        )
        selected_population = [population[i] for i in selected_indices]

        # Cross (single point cross)
        new_population = []
        for i in range(0, population_size, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1] if i + 1 < population_size else selected_population[0]

            # Select the intersection at random
            crossover_point = np.random.randint(1, num_profiles)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

            # Remove and complete the missing index
            child1 = np.unique(child1)
            child2 = np.unique(child2)
            missing1 = np.setdiff1d(np.arange(profiles), child1)
            missing2 = np.setdiff1d(np.arange(profiles), child2)
            child1 = np.concatenate([child1, np.random.choice(missing1, size=num_profiles - len(child1), replace=False)])
            child2 = np.concatenate([child2, np.random.choice(missing2, size=num_profiles - len(child2), replace=False)])

            new_population.extend([child1, child2])

        # Mutation
        for individual in new_population:
            if np.random.rand() < mutation_rate:
                # Swap two positions at random
                idx1, idx2 = np.random.choice(num_profiles, size=2, replace=False)
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

            # Forced weight removal
            individual[:] = np.unique(individual)
            missing = np.setdiff1d(np.arange(profiles), individual)
            individual = np.concatenate([individual, np.random.choice(missing, size=num_profiles - len(individual), replace=False)])
        # Regeneration population
        population = new_population

        # Print the current optimal solution
        print(f"Generation {generation + 1}/{generations}, Best D: {best_D}")

    # Build dataset
    selected_temperature = temperature[best_selection, :]
    selected_humidity = humidity[best_selection, :]
    selected_ozone = ozone[best_selection, :]

    selected_dataset = xr.Dataset(
        {
            'level_temperature': (('n_profiles', 'n_levels'), selected_temperature),
            'layer_temperature': (('n_profiles', 'n_layers'), (selected_temperature[:,:-1]+selected_temperature[:,1:])/2),
            'level_absorber': (('n_profiles', 'n_absorbers', 'n_levels'),
                               np.stack([selected_humidity, selected_ozone], axis=1)),
            'layer_absorber': (('n_profiles', 'n_absorbers', 'n_layers'),
                               np.stack([(selected_humidity[:,:-1]+selected_humidity[:,1:])/2, (selected_ozone[:,:-1]+selected_ozone[:,1:])/2], axis=1)),
            'absorber_id': (('n_absorbers'), np.array([1, 3])),
            'absorber_units_id': (('n_absorbers'), np.array([3, 1])),
            'surface_altitude': ('n_profiles', ds.surface_altitude.values[best_selection]),
            'level_pressure': (('n_profiles', 'n_levels'), ds.level_pressure.values[best_selection, :]),
            'latitude': ('n_profiles', ds.latitude.values[best_selection]),
            'longitude': ('n_profiles', ds.longitude.values[best_selection]),
            'year': ('n_profiles', ds.year.values[best_selection]),
            'month': ('n_profiles', ds.month.values[best_selection]),
            'day': ('n_profiles', ds.day.values[best_selection]),
        },
        coords={
            'n_profiles': np.arange(num_profiles),
            'n_levels': np.arange(pressure_levels),
            'n_layers': np.arange(pressure_levels-1),
            'n_absorbers': np.arange(2),
        }
    )

    return selected_dataset, best_D, best_selection

def find_min_max_profiles(ds,is_normalization=False):
    """
    Find the feature maximum and minimum values of the profile in this dataset
    
    Parameters:
        ds (xr.Dataset): Arbitrary profile data set
        
    return:
        min_profile (float): Profile minimum feature
        max_profile (float): Profile maximum feature
    """
    # Extract variable 
    temperature = ds.level_temperature.values  # (profiles, pressure_levels)
    humidity = ds.level_absorber[:, 0, :].values  # (profiles, pressure_levels)
    ozone = ds.level_absorber[:, 1, :].values  # (profiles, pressure_levels)

    # Initializes the weight array
    temp_weights,humidity_weights,ozone_weights = generating_weight()
    
    if is_normalization:# Normalized temperature, humidity and ozone
        temperature = (temperature - temperature.min()) / (temperature.max() - temperature.min())
        humidity = (humidity - humidity.min()) / (humidity.max() - humidity.min())
        ozone = (ozone - ozone.min()) / (ozone.max() - ozone.min())
    
    # Calculate the weighted average
    temp_weighted_mean = np.average(temperature, axis=1, weights=temp_weights)
    hum_weighted_mean = np.average(humidity, axis=1, weights=humidity_weights)
    oz_weighted_mean = np.average(ozone, axis=1, weights=ozone_weights)

    # Synthetic eigenvalue
    combined_feature = hum_weighted_mean # optional: temp_weighted_mean + oz_weighted_mean,here only consider the feature of humidity

    # Find the characteristic maximum and minimum values of the profile
    min_profile = np.sort(combined_feature)[0]
    max_profile = np.sort(combined_feature)[-1]

    return min_profile, max_profile


def map_indices_to_original(best_selection, original_ds, clean_ds):
    """
    Map best_selection from clean_ds to the index of original_ds.
    Parameters:
    best_selection (list): The profile lead selected in clean_ds.
    original_ds (xr.Dataset): indicates the original complete dataset.
    clean_ds (xr.Dataset): indicates the cleaned data set.

    return:
    mapped_indices (list): Index corresponding to original_ds.
    """
    # Find the correspondence between clean_ds and original_ds
    mapped_indices = []
    for idx in best_selection:
        # Match based on certain unique identifiers (such as latitude, longitude, etc.)
        lat_match = np.isclose(clean_ds.latitude.values[idx], original_ds.latitude.values)
        lon_match = np.isclose(clean_ds.longitude.values[idx], original_ds.longitude.values)
        year_match = np.isclose(clean_ds.year.values[idx], original_ds.year.values)
        temp_values_match = np.isclose(clean_ds.level_temperature.values[idx,0], original_ds.level_temperature.values[:,0])
        hum_values_match = np.isclose(clean_ds.level_absorber[:, 0, :].values[idx,0], original_ds.level_absorber[:, 0, :].values[:,0])
        # ozone_values_match = np.isclose(clean_ds.level_absorber[:, 1, :].values[idx,0], original_ds.level_absorber[:, 1, :].values[:,0])
        match = np.where(lat_match & lon_match & temp_values_match & year_match & hum_values_match)[0]
        if len(match) > 0:
            mapped_indices.append(match[0])

    return mapped_indices


def find_uniform_profiles(original_ds, min_profile, max_profile, best_selection, total_profiles_needed,is_normalization=False):
    """
    A specified number of profiles are added to between the two intervals [min_profile, min_profile(original_ds(best_selection))] and [max_profile, max_profile(original_ds(best_selection))] .

    Parameters:
    original_ds (xr.Dataset): indicates the original complete dataset.
    min_profile (int): Minimum profile clue (found in EC83).
    max_profile (int): Maximum profile clue (found in EC83).
    best_selection (list): indicates the selected profile lead.
    num_profiles_per_interval (int): specifies the number of profiles to be added for each interval.

    return:
    uniform_profile_indices (list): Uniform_profile_indices (list): uniformly distributed profile clues.
    """
    # Extract variable data
    temperature = original_ds.level_temperature.values  # (profiles, pressure_levels)
    humidity = original_ds.level_absorber[:, 0, :].values  # (profiles, pressure_levels)
    ozone = original_ds.level_absorber[:, 1, :].values  # (profiles, pressure_levels)

    
    # Initializes the weight array
    temp_weights,humidity_weights,ozone_weights = generating_weight()

    if is_normalization:# Normalized temperature, humidity and ozone
        temperature = (temperature - temperature.min()) / (temperature.max() - temperature.min())
        humidity = (humidity - humidity.min()) / (humidity.max() - humidity.min())
        ozone = (ozone - ozone.min()) / (ozone.max() - ozone.min())
    
    # Calculate the weighted average
    temp_weighted_mean = np.average(temperature, axis=1, weights=temp_weights)
    hum_weighted_mean = np.average(humidity, axis=1, weights=humidity_weights)
    oz_weighted_mean = np.average(ozone, axis=1, weights=ozone_weights)

    # Combine feature
    combined_feature = hum_weighted_mean  # The addition of humidity and ozone can be adjusted according to requirements

    # Find the maximum and minimum values in best_selection
    best_combined_feature = combined_feature[best_selection]
    best_min_profile_index = best_selection[np.argmin(best_combined_feature)]
    best_max_profile_index = best_selection[np.argmax(best_combined_feature)]

    # Filter out the profiles of the intervals [min_profile, best_min_profile_index] and [best_max_profile_index, max_profile]
    mask_lower_interval = (combined_feature >= min_profile) & \
                          (combined_feature <= combined_feature[best_min_profile_index])
    mask_upper_interval = (combined_feature >= combined_feature[best_max_profile_index]) & \
                          (combined_feature <= max_profile)

    profiles_lower_interval = np.where(mask_lower_interval)[0]
    profiles_upper_interval = np.where(mask_upper_interval)[0]

    # Remove the selected profile
    available_profiles_lower = set(profiles_lower_interval) - set(best_selection)
    available_profiles_upper = set(profiles_upper_interval) - set(best_selection)

    # Check the number of candidate profiles
    lower_available_count = len(available_profiles_lower)
    upper_available_count = len(available_profiles_upper)

    # Initialize the outline thread of the selected outline
    selected_lower_indices = []
    selected_upper_indices = []

    # Dynamic allocation of requirements
    remaining_demand = total_profiles_needed

    # Step 1: Allocate the requirements of the lower interval
    lower_target = min(lower_available_count, remaining_demand // 2)
    if lower_target > 0:
        sorted_lower_indices = sorted(available_profiles_lower, key=lambda idx: combined_feature[idx])
        sample_lower_indices = np.linspace(0, len(sorted_lower_indices) - 1, num=lower_target).astype(int)
        selected_lower_indices = [sorted_lower_indices[i] for i in sample_lower_indices]
        remaining_demand -= len(selected_lower_indices)

    # Step 2: Allocate the requirements for the upper interval
    upper_target = min(upper_available_count, remaining_demand)
    if upper_target > 0:
        sorted_upper_indices = sorted(available_profiles_upper, key=lambda idx: combined_feature[idx])
        sample_upper_indices = np.linspace(0, len(sorted_upper_indices) - 1, num=upper_target).astype(int)
        selected_upper_indices = [sorted_upper_indices[i] for i in sample_upper_indices]
        remaining_demand -= len(selected_upper_indices)

    # Step 3: If there are still remaining demands, supplement them across regions
    if remaining_demand > 0:
        if lower_available_count > len(selected_lower_indices):
            #Supplement the remaining requirements from the lower interval
            additional_lower_indices = sorted(
                list(available_profiles_lower - set(selected_lower_indices)),
                key=lambda idx: combined_feature[idx]
            )
            sample_additional_lower = additional_lower_indices[:remaining_demand]
            selected_lower_indices.extend(sample_additional_lower)
            remaining_demand -= len(sample_additional_lower)

        if remaining_demand > 0 and upper_available_count > len(selected_upper_indices):
            # Supplement the remaining requirements from the upper interval
            additional_upper_indices = sorted(
                list(available_profiles_upper - set(selected_upper_indices)),
                key=lambda idx: combined_feature[idx]
            )
            sample_additional_upper = additional_upper_indices[:remaining_demand]
            selected_upper_indices.extend(sample_additional_upper)
            remaining_demand -= len(sample_additional_upper)

    # Merge results
    uniform_profile_indices = selected_lower_indices + selected_upper_indices

    # Output information
    print(f"Selected {len(selected_lower_indices)} profiles from lower interval.")
    print(f"Selected {len(selected_upper_indices)} profiles from upper interval.")
    print(f"Total profiles selected: {len(uniform_profile_indices)}")

    return uniform_profile_indices

def generate_average_profile(ds):
    """
    Generate an average profile based on the selected profile dataset.
    
    Parameters:
        ds (xr.Dataset): the selected profile dataset
        
    return:
        average_profile (dict): average profile data.
    """
    # Extract variable data
    temperature = ds.level_temperature.values # (selected_profiles, pressure_levels)
    humidity = ds.level_absorber[:, 0, :].values # (selected_profiles, pressure_levels)
    ozone = ds.level_absorber[:, 1, :].values  # (selected_profiles, pressure_levels)

    # Compute the mean of variable
    avg_temperature = np.mean(temperature, axis=0)
    avg_humidity = np.mean(humidity, axis=0)
    avg_ozone = np.mean(ozone, axis=0)

    # build the mean profile
    average_profile = {
        'level_temperature': avg_temperature,
        'level_absorber': np.stack([avg_humidity, avg_ozone], axis=0),
    }

    return average_profile

def combine_profiles(ds, best_selection,  uniform_profile_indices, average_profile):
    """
    Merge all profiles, to ensure no duplication.

    Parameters:
    ds (xr.Dataset): indicates the original uncleaned data set.
    best_selection (list):  selection clues.
    uniform_profile_indices (list): evenly distributed profile clues are cited.
    average_profile (dict): average profile data.

    return:
    combine_dataset (xr.Dataset): A new dataset.
    """
    # Merge profile cue
    selected_indices = list(set(best_selection))
    selected_indices += list(uniform_profile_indices)

    # duplicate removal
    selected_indices = list(set(selected_indices))

    # Add an average profile
    avg_temperature = average_profile['level_temperature']
    avg_humidity = average_profile['level_absorber'][0]
    avg_ozone = average_profile['level_absorber'][1]

    # Extract variable data
    temperature = ds.level_temperature.values[selected_indices]  # (selected_profiles, pressure_levels)
    humidity = ds.level_absorber[:, 0, :].values[selected_indices]  # (selected_profiles, pressure_levels)
    ozone = ds.level_absorber[:, 1, :].values[selected_indices]  # (selected_profiles, pressure_levels)

    # Add the average profile to the end
    selected_temperature = np.vstack([temperature, avg_temperature])
    selected_humidity = np.vstack([humidity, avg_humidity])
    selected_ozone = np.vstack([ozone, avg_ozone])

    # Create a new Dataset
    combine_dataset = xr.Dataset(
        {
            'level_temperature': (('n_profiles', 'n_levels'), selected_temperature),
            'layer_temperature': (('n_profiles', 'n_layers'), (selected_temperature[:,:-1]+selected_temperature[:,1:])/2),
            'level_absorber': (('n_profiles', 'n_absorbers', 'n_levels'),
                               np.stack([selected_humidity, selected_ozone], axis=1)),
            'layer_absorber': (('n_profiles', 'n_absorbers', 'n_layers'),
                               np.stack([(selected_humidity[:,:-1]+selected_humidity[:,1:])/2, (selected_ozone[:,:-1]+selected_ozone[:,1:])/2], axis=1)),
            'absorber_id': (('n_absorbers'), np.array([1, 3])),
            'absorber_units_id': (('n_absorbers'), np.array([3, 1])),
            'surface_altitude': ('n_profiles', np.append(ds.surface_altitude.values[selected_indices], 0)),
            'level_pressure': (('n_profiles', 'n_levels'), np.vstack([ds.level_pressure.values[selected_indices], ds.level_pressure.values[0]])),
            'latitude': ('n_profiles', np.append(ds.latitude.values[selected_indices], 0)),  
            'longitude': ('n_profiles', np.append(ds.longitude.values[selected_indices], 0)),  
            'year': ('n_profiles', np.append(ds.year.values[selected_indices], 2014)),  
            'month': ('n_profiles', np.append(ds.month.values[selected_indices], 1)),  
            'day': ('n_profiles', np.append(ds.day.values[selected_indices], 1)),
        },
        coords={
            'n_profiles': np.arange(selected_temperature.shape[0]),
            'n_levels': pressure_level_101,
            'n_layers': np.arange(selected_temperature.shape[1]-1),
            'n_absorbers': np.arange(2),
        }
    )

    return combine_dataset

def Combine_Profile_Dataset(total_profiles_needed,find_min_max_dataset,best_selection,original_dataset):

    # Step 1: Find the outline clues for the maximum and minimum features in the original dataset
    ec_min_profile, ec_max_profile = find_min_max_profiles(find_min_max_dataset)

    # Step 2: Map best_selection from IFS_137_101_clean to the index of IFS_137_101
    mapped_best_selection = map_indices_to_original(best_selection, find_min_max_dataset, original_dataset)

    # Step 3: Find uniformly distributed profiles
    uniform_profile_indices = find_uniform_profiles(
        find_min_max_dataset,
        ec_min_profile,
        ec_max_profile,
        mapped_best_selection,
        total_profiles_needed
    )

    # Step 4: Generate the average profile
    average_profile = generate_average_profile(find_min_max_dataset)

    # Step 5: Merge all the profiles
    final_dataset = combine_profiles(
        find_min_max_dataset,
        mapped_best_selection,
        uniform_profile_indices,
        average_profile
    )
    print(f"Final dataset with {len(final_dataset.n_profiles.values)} profiles created successfully!")

    return final_dataset