import numpy as np
import xarray as xr
from Common_Attachment import specific_humidity_to_mixing_ratio,ozone_kg_per_kg_to_ppmv,ec_p,load_aam_bbm
from Common_Attachment import pressure_level_101


def read_saf(nlev):
    # Read 91-level or 137-level diverse profile dataset from ECMWF
    # save the dataset into nc
    Temperature, Humidity, Ozone = [], [], []
    years, months, days, lats, longs, elevs, paps, lsms, psurfs = [], [], [], [], [], [], [], [], []
    nlev = nlev  # number of vertical levels
    aam ,bbm = load_aam_bbm(nlev)
    for cvar in ['t', 'q', 'oz',]:

        atm_filename = f'./Profile/saf{nlev}/nwp_saf_{cvar}_sampled.atm'
        sfc_filename = f'./Profile/saf{nlev}/nwp_saf_{cvar}_sampled.sfc'

        with open(atm_filename, 'r') as atm_file, open(sfc_filename, 'r') as sfc_file:
            i = 0

            while True:
                try:
                    # Reading atmospheric data
                    line = atm_file.readline()
                    if not line:
                        break
                    fields = list(map(float, line.split()))

                    temp = np.array(fields[:nlev])          # 1) Temperature [K]                         
                    hum = np.array(fields[nlev:nlev*2])     # 2) Humidity [kg/kg]                        
                    ozo = np.array(fields[nlev*2:nlev*3])   # 3) Ozone [kg/kg]                            
                    cc = np.array(fields[nlev*3:nlev*4])    # 4) Cloud Cover [0-1]                       
                    clw = np.array(fields[nlev*4:nlev*5])   # 5) C Liquid W [kg/kg]                   
                    ciw = np.array(fields[nlev*5:nlev*6])   # 6) C Ice W [kg/kg]                         
                    rain = np.array(fields[nlev*6:nlev*7])  # 7) Rain [kg/(m2 *s)]                        
                    snow = np.array(fields[nlev*7:nlev*8])  # 8) Snow [kg/(m2 *s)]                      
                    w = np.array(fields[nlev*8:nlev*9])     # 9) Vertical Velocity [Pa/s]             
                    lnpsurf = fields[nlev*9]                #10) Ln of Surf Pressure in [Pa]              
                    z0 = fields[nlev*9+1]                   #11) Surface geopotential [m2/s2]           
                    tsurf = fields[nlev*9+2]                #12) Surface Skin Temperature [K]           
                    t2m = fields[nlev*9+3]                  #13) 2m Temperature [K]                       
                    td2m = fields[nlev*9+4]                 #14) 2m Dew point temperature [K]   
                    n=4
                    if nlev==91:
                        hum2m = fields[nlev*9+5]            #15) 2m Specific Humidity [kg/kg]             
                        n=5          
                    u10 = fields[nlev*9+n+1]                #15) 10m wind speed U component [m/s]       
                    v10 = fields[nlev*9+n+2]                #16) 10m wind speed V component [m/s]         
                    stratrsrf = fields[nlev*9+n+3]          #17) Stratiform precipitation at surface [m] 
                    convrsrf = fields[nlev*9+n+4]           #18) Convective precipitation at surface [m] 
                    snowsurf = fields[nlev*9+n+5]           #19) Snowfall at surface [m] (water equival.) 
                    lsm = fields[nlev*9+n+6]                #20) Land/sea Mask [0-1]                      
                    lat = fields[nlev*9+n+7]                #21) Latitude [deg]                          
                    long = fields[nlev*9+n+8]               #22) Longitude [deg]                         
                    year = int(fields[nlev*9+n+9])          #23) Year                                     
                    month = int(fields[nlev*9+n+10])        #24) Month                                 
                    day = int(fields[nlev*9+n+11])          #25) Day                                     
                    step = int(fields[nlev*9+n+12])         #26) Step                                    
                    gpoint = int(fields[nlev*9+n+13])       #27) Grid point [1-2140702]                   
                    ind = int(fields[nlev*9+n+14])          #28) Index (rank-sorted)                     

                    # Reading surface data
                    line = sfc_file.readline()
                    fields = list(map(float, line.split()))
                    alb = fields[0]                    # 1) Albedo [0-1]                             (1)
                    sr = fields[1]                     # 2) Roughness [m]                            (2)                    
                    tvl = fields[2]                    # 3) Low  vegetation type [0-20]              (3)                    
                    tvh = fields[3]                    # 4) High vegetation type [0-19]              (4)
                    cvl = fields[4]                    # 5) Low  vegetation fractional cover [0-1]   (5)
                    cvh = fields[5]                    # 6) High vegetation fractional cover [0-1]   (6)
                    seaice = fields[6]                 # 7) SeaIce cover [0-1]                       (7)
                    asn = fields[7]                    # 8) Snow albedo [0-1]                        (8)
                    rsn = fields[8]                    # 9) Snow density [kg/m3]                     (9)
                    tsn = fields[9]                    #10) Snow temperature [K]                    (10)
                    sd = fields[10]                    #11) Snow depth [m]                          (11)
                    stl1 = fields[11]                  #12) Soil Layer 1 temperature [K]            (12)
                    stl2 = fields[12]                  #13) Soil Layer 2 temperature [K]            (13)
                    stl3 = fields[13]                  #14) Soil Layer 3 temperature [K]            (14)
                    stl4 = fields[14]                  #15) Soil Layer 4 temperature [K]            (15)
                    swvl1 = fields[15]                 #16) Volumetric Soil Water Layer 1 [m3/m3]   (16)
                    swvl2 = fields[16]                 #17) Volumetric Soil Water Layer 2 [m3/m3]   (17)
                    swvl3 = fields[17]                 #18) Volumetric Soil Water Layer 3 [m3/m3]   (18)
                    swvl4 = fields[18]                 #19) Volumetric Soil Water Layer 4 [m3/m3]   (19)
                    istl1 = fields[19]                 #20) Ice temperature Layer 1 [K]             (20)
                    istl2 = fields[20]                 #21) Ice temperature Layer 2 [K]             (21)
                    istl3 = fields[21]                 #22) Ice temperature Layer 3 [K]             (22)
                    istl4 = fields[22]                 #23) Ice temperature Layer 4 [K]             (23)
                    #The following variables also appear in .atm file
                    '''
                    lsm = fields[23]                   #24) Land/sea Mask [0-1]                     (24)
                    lat = fields[24]                   #25) Latitude [deg]                          (25)
                    long = fields[25]                  #26) Longitude [deg]                         (26)
                    year = int(fields[26])             #27) Year                                    (27)
                    month = int(fields[27])            #28) Month                                   (28)
                    day = int(fields[28])              #29) Day                                     (29)
                    step = int(fields[29])             #30) Step                                    (30)
                    gpoint = int(fields[30])           #31) Grid point [1-2140702]                  (31)
                    ind = int(fields[31])              #32) Index (rank-sorted)                     (32) 
                    assert lat == lat_sfc, "Mismatch in structure lengths"'
                    '''
                    # Calculation of surface pressure [Pa]
                    psurf = np.exp(lnpsurf)
                    
                    # Calculating surface geometric height [m]
                    elev = z0 / 9.80665
                    
                    # Calculate full layer pressure (pap) and Half layer pressure (paph) [Pa]

                    pap, paph = ec_p(psurf,nlev,aam,bbm)
                    
                    i += 1

                    #Reserved array
                    Temperature.append(temp.reshape(1, nlev))
                    Humidity.append(specific_humidity_to_mixing_ratio(hum).reshape(1, nlev))
                    Ozone.append(ozone_kg_per_kg_to_ppmv(ozo).reshape(1, nlev))
                    lsms.append(lsm)
                    years.append(year)
                    months.append(month)
                    days.append(day)
                    lats.append(lat)
                    longs.append(long)
                    elevs.append(elev)
                    psurfs.append(psurf/100)
                    paps.append(pap/100)
                except Exception as e:
                    print(f"Error reading file at profile {i}: {e}")
                    break  

    # Combined data
    Temperature = np.concatenate(Temperature, axis=0)
    Humidity = np.concatenate(Humidity, axis=0) 
    Ozone = np.concatenate(Ozone, axis=0)
    paps = np.stack(paps, axis=0)

    # Build xarray.Dataset
    saf = xr.Dataset(
        {
            'level_temperature': (('n_profiles', 'n_levels'), Temperature),
            'level_absorber': (('n_profiles','n_absorbers', 'n_levels'), np.concatenate([Humidity[:, np.newaxis,:],Ozone[:, np.newaxis,:]],axis=1)),
            'absorber_id': (('n_absorbers'), np.array([1,3])),
            'absorber_units_id': (('n_absorbers'), np.array([3,1])),
            #'surface_type': ('n_profiles', lsms),
            'surface_altitude': ('n_profiles', elevs),
            'surface_pressure': ('n_profiles', psurfs),
            'level_pressure': (('n_profiles', 'n_levels'), paps),
            'latitude': ('n_profiles', lats),
            'longitude': ('n_profiles', longs),
            'year': ('n_profiles', years),
            'month': ('n_profiles', months),
            'day': ('n_profiles', days),
        },
        coords={
            'n_profiles': np.arange(Temperature.shape[0]),
            'n_levels': np.arange(nlev),
            'n_absorbers':np.arange(2),
        }
    )

    return saf

def read_seeborV5():
    #read seeBorv5.0

    file_path = './Profile/SeeBorV5.0/SeeBorV5.1_Training_data_Emis10inf2004.bin'

    # The total number of bytes per profile
    line_size_bytes = 338

    # Total number of profile
    total_profiles = 15704

    # save the dataset into nc
    Temperature, Mixr, Ozone = [], [], []
    years, months, days, lats, longs, elevs, paps = [], [], [], [], [], [], []
    data_pressure = pressure_level_101

    with open(file_path, 'rb') as file:
        for _ in range(total_profiles):
            # Read all data for a profile
            line_data = np.fromfile(file, dtype=np.float32, count=line_size_bytes)
            
            # 
            temperature_profile=line_data[:101],             # Temperature [K] 
            mixing_ratio_profile= line_data[101:202]*1000,   # Mixing ratio [kg/kg] Convert to g/kg
            ozone_profile= line_data[202:303],               # Ozone [ppmv]
            latitude= line_data[303],                        # latitude
            longitude= line_data[304],                       # longitude
            surface_pressure= line_data[305],                # surface pressure [hPa]
            skin_temperature= line_data[306],                # skin temperature [K]
            wind_speed= line_data[307],                      # wind speed (m/s) - value used for finding seawater emissivity
            tpw= line_data[308],                             # tpw [cm]
            ecosystem= int(line_data[309]),                  # ecosystem, igbp classification
            elevation= line_data[310],                       # elevation [m]
            fraction_land= line_data[311],                   # fraction land
            year= int(line_data[312]),                       # year
            month= int(line_data[313]),                      # month
            day= int(line_data[314]),                        # day
            hour= int(line_data[315]),                       # hour
            profile_type= int(line_data[316]),               # profile type:
                                                             # 1 NOAA-88b 2 TIGR-3 3 Radiosondes 4 Ozonesondes 5 ECMWF
            frequency_emis_hinge_points= line_data[317:327], # frequency (wavenumber) of emissivity at 10 BF emis hinge points
            emissivity_spectra= line_data[327:337],          # emissivity spectra
            spare= line_data[337]

            #Reserved array
            Temperature.append(temperature_profile[0].reshape(1, 101))
            Mixr.append(mixing_ratio_profile[0].reshape(1, 101))
            Ozone.append(ozone_profile[0].reshape(1, 101))
            years.append(year[0])
            months.append(month[0])
            days.append(day[0])
            lats.append(latitude[0])
            longs.append(longitude[0])
            elevs.append(elevation[0])
            paps.append(np.array(data_pressure))

    # Combined data
    Temperature = np.concatenate(Temperature, axis=0)
    Mixr = np.concatenate(Mixr, axis=0) 
    Ozone = np.concatenate(Ozone, axis=0)
    paps = np.stack(paps, axis=0)

    # Build xarray.Dataset
    SeeBor = xr.Dataset(
        {
            'level_temperature': (('n_profiles', 'n_levels'), Temperature),
            'layer_temperature': (('n_profiles', 'n_layers'), (Temperature[:,:-1]+Temperature[:,1:])/2),
            'level_absorber': (('n_profiles','n_absorbers', 'n_levels'), np.concatenate([Mixr[:, np.newaxis,:],Ozone[:, np.newaxis,:]],axis=1)),
            'layer_absorber': (('n_profiles', 'n_absorbers', 'n_layers'),
                               np.stack([(Mixr[:,:-1]+Mixr[:,1:])/2, (Ozone[:,:-1]+Ozone[:,1:])/2], axis=1)),
            'absorber_id': (('n_absorbers'), np.array([1,3])),
            'absorber_units_id': (('n_absorbers'), np.array([3,1])),
            'surface_altitude': ('n_profiles', elevs),
            'level_pressure': (('n_profiles', 'n_levels'), paps),
            'latitude': ('n_profiles', lats),
            'longitude': ('n_profiles', longs),
            'year': ('n_profiles', years),
            'month': ('n_profiles', months),
            'day': ('n_profiles', days),
        },
        coords={
            'n_profiles': np.arange(Temperature.shape[0]),
            'n_levels': np.arange(Temperature.shape[1]),
            'n_layers': np.arange(Temperature.shape[1]-1),
            'n_absorbers':np.arange(2),
        }
    )

    return SeeBor

