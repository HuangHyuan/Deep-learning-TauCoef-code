import numpy as np
import xarray as xr
from Read_Profile_Attachment import read_seeborV5,read_saf
from Select_Profile_Attachment import Combine_Profile_Dataset,select_profiles_with_clustering_and_ga
from Common_Attachment import pressure_level_101,interpolate_to_101_levels,detect_outlier_profiles_by_layer,detect_outlier_profiles_by_ozone


#Read IFS-91 and IFS-137 Dataset
IFS_91 = read_saf(91)
IFS_137 = read_saf(137)

#Interp to 101 level
IFS_91_101 = interpolate_to_101_levels(IFS_91)
IFS_137_101 = interpolate_to_101_levels(IFS_137)

#exclude extreme value profiles
IFS_137_101_clean_by_ozone,_ = detect_outlier_profiles_by_ozone(IFS_137_101)
IFS_91_101_clean_by_ozone,_ = detect_outlier_profiles_by_ozone(IFS_91_101,is_91=True)

#concat the profile dataset
IFS_101_Dataset = xr.concat([IFS_137_101_clean_by_ozone, IFS_91_101_clean_by_ozone], dim='n_profiles')
IFS_101_Dataset_clean,_ = detect_outlier_profiles_by_layer(IFS_101_Dataset,3)
#Select dataset by clustering and genetic algorithm
Select_Dataset, D, best_selection = select_profiles_with_clustering_and_ga(
    IFS_101_Dataset_clean, bins=1300, n_clusters=80, num_profiles=1300
)

#Adding the Range
Final_Dataset = Combine_Profile_Dataset(500,IFS_101_Dataset,best_selection,IFS_101_Dataset_clean)

#Place the lower level on the first (Like EC83)
Final_Dataset = Final_Dataset.sortby('n_levels', ascending=False)
Final_Dataset = Final_Dataset.sortby('n_layers', ascending=False)

#Save to NC
Final_Dataset.to_netcdf(f'./IFS_101_NUM_{len(Final_Dataset.n_profiles.values)}.nc')

print('NC Write ' + f'./IFS_101_NUM_{len(Final_Dataset.n_profiles.values)}.nc' + ' Succeed')