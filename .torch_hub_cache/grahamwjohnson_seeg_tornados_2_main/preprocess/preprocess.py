"""
@author: grahamwjohnson
Developed between 2023-2025

This script processes raw EEG/SEEG data by performing the following steps:

1. **File Preparation and Filtering:**
   - It reads the raw data from EDF files for a given patient, filtering the data based on a specified montage (e.g., Bipolar), frequency bands, and unit expectations.
   - The raw data is pickled and saved to a specified directory for further use, with options for scaling and filtering before storing.
   - It applies a predefined set of channel names to ignore (e.g., non-relevant channels or artifacts).

2. **Path and Directory Setup:**
   - Constructs the directory paths required for saving the processed data, ensuring that directories exist before writing data.
   - Copies the current script into the output directory for record-keeping, allowing reproducibility of the preprocessing steps.

3. **Scaling Parameters Acquisition:**
   - Acquires normalization and scaling parameters for the data, including the choice of scaling method (Linear, HyperTan, CubeRoot, or Histogram Equalization).
   - If scaling parameters were previously computed and saved, it loads those parameters for reuse.
   - If not, it computes the necessary scaling factors or linear interpolations based on the raw data, saving these parameters for later use.

4. **Data Normalization:**
   - Applies the computed or loaded scaling parameters to normalize the data, resulting in scaled data epochs suitable for machine learning or other downstream analysis.
   - This normalization is done by utilizing the scaling factors or interpolated values for each channel and frequency band.
   - Data is saved to the specified directory, with file timestamps and other metadata included to facilitate future data management.

### Key Parameters:
- `pat_id`: Patient identifier (e.g., 'Epat38') to find the relevant data.
- `montage`: The montage used for the bipolar channel configuration.
- `freq_bands`: Frequency bands to apply to the data, or empty for the full frequency range.
- `num_channels`: Total number of channels to process, typically 144 for bipolar data.
- `resamp_freq`: Sampling frequency after resampling, default is 512 Hz.
- `scale_type`: Type of normalization applied to the data (e.g., 'HistEqualScale', 'LinearScale').
- `scale_epoch_hours`: Time window (in hours) used for scaling the data.
- `buffer_start_hours`: Buffer time (in hours) to account for any setup issues or data irregularities.
- `scale_epoch_type`: The type of scaling normalization method to apply (e.g., 'data_normalized_to_first').
- `histo_min`/`histo_max`: Histogram equalization scaling range in microvolts.
- `file_buffer_sec`: Buffer in seconds to ignore at the beginning and end of each file during processing.

### Output:
- Scaled data is saved to a directory structured by the patient ID, montage, scaling method, and frequency band.
- Metadata and code used during the preprocessing are also saved for reproducibility and transparency.

### Dependencies:
- `utils_functions`: Custom utility functions for data manipulation and file handling.
- `preprocess_aquire_params`: Module to acquire scaling parameters based on the data.
- `preprocess_employ_norms`: Module to apply normalization and scaling to the data.

### Assumptions:
- The script assumes a specific file naming convention for EDF files (e.g., '<patname>_MMDDYYYY_HHMMSSss').
- The data is processed in a way that all EDF files from the same patient are combined into a single set of pickles for that patient.

### Notes:
- The function is highly configurable and can accommodate various scaling methods and frequency bands.
- The script ensures that the entire pipeline is run only when necessary, checking whether previous pickling and scaling steps have already been completed to save computation time.
- Make sure to update the `ch_names_to_ignore` list when there are additional or irrelevant channels for a new dataset.

"""

# TODO - add stats output to SEE: percent overlap, any gaps in time

import glob, os
import datetime
import shutil
from utilities import utils_functions
from preprocess import preprocess_aquire_params, preprocess_employ_norms
import pickle

# TODO pull out usable channels e.g. Epat02...

# Settings
previous_big_pickles = False
previous_scale_aquired = False
PROCESS_FILE_DEBUG_LIST = []

ignore_channel_units = False  # Need to bypass units check for NK aquisitions. TODO: updatr to check units after channels to ignore are deleted

atd_file = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/data/all_time_data_01092023_112957.csv'

# Define the patient directory that contains all EDF files
# ASSUMPTION: files will be named "<patname>_MMDDYYYY_HHMMSSss" where S = seconds, s = milliseconds
pat_id = 'Epat38' 
montage = 'BIPOLE'
freq_bands = [] # [[1, 12], [12, 59], [61, 179]], []
if freq_bands == []: freq_bands_str = 'wholeband'
else: freq_bands_str = f"{freq_bands}".replace("], [", "Hz_").replace(", ", "to").replace("[[","").replace("]]","Hz")
num_channels = 144  # BIPOLAR!
if freq_bands != []: num_channels = num_channels * len(freq_bands)
bipole_or_monopole = 'Bipole_datasets'
resamp_freq = 512
big_pickle_dir = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/data/Resamp_' + str(resamp_freq) + f'_Hz/bipole_filtered_{freq_bands_str}_unscaled_big_pickles/{pat_id}'
channel_scale_style = 'By_Channel_Scale' # 'Same_Scale_For_All_Channels'  # 'By_Channel_Scale'    Should probably do by_channel for sure if using multiple freq_bands
scale_type = 'HistEqualScale' # 'LinearScale', 'HyperTanScaling', 'CubeRootScale', 'HistEqualScale'
scale_epoch_hours = 24
buffer_start_hours = 2 # allow for setup weirdness to pass
scale_epoch_type= 'data_normalized_to_first'  # data_normalized_to_first, data_normalized_to_first_seizure_centered
out_dur = 1024 # seconds
out_stride = 896 # seconds
duration_stride_str = 'DurStr_' + str(out_dur) + 's' + str(out_stride) + 's_epoched_datasets/'
expected_unit = 'uV'
desired_samp_freq = 512 # Hz
dir_edf = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/data/Raw_EMU_copies/' + pat_id

ch_names_to_ignore = ['-', 'FP1', 'FP2', 'Fp1','F7','T7','P7','O1','Fp2','F8','T8','P8','O2','F3','C3','P3','F4','C4','P4','Fz','Cz','Pz','Fz','F8','T8','T2','F7','T7','T1','63', '64', 'DC01','DC02','DC03','DC04','E', 'EEGMark1', 'EEGMark2','EEG Mark1', 'EEG Mark2', 'Cz', 'Pz', 'EKG1', 'EKG2', 'EKG3', 'EKG4', 'Events/Markers', 'L56','L57','L58','L59','L60','L61','L62','L63','L64','B21','B22','B23','B24','B25','B26','B27','B28','B29','B30','B31','B32','B33','B34','B35','B36','B37','B38','B39','B40','B41','B42','B43','B44','B45','B46','B47','B48','B49','B50','B51','B52','B53','B54','B55','B56','B57','B58','B59','B60','B61','B62','B63','B64']

# Histogram parameters (only used for 'HistEqualScale')
histo_min = -10000 # uV
histo_max = 10000 # uV
num_bins = 100001

# ### CREATE BIG PICKLES IF NOT MADE ALREADY
if not previous_big_pickles:
    print("Pickling the EDF files, this will take a very long time")
    if not os.path.exists(big_pickle_dir): os.makedirs(big_pickle_dir)

    utils_functions.montage_filter_pickle_edfs(
        pat_id=pat_id, 
        dir_edf=dir_edf,
        save_dir=big_pickle_dir,
        desired_samp_freq=resamp_freq,
        freq_bands=freq_bands,
        expected_unit='uV',
        montage=montage,
        ch_names_to_ignore=ch_names_to_ignore,
        ignore_channel_units=ignore_channel_units)

# Use the directory assembler function to ensure valid path is created for output data epochs
root_dir = utils_functions.assemble_model_save_path(
    base_path='/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results',
    bipole_or_monopole=bipole_or_monopole,
    channel_scale_style=channel_scale_style,
    freq_bands_str=freq_bands_str,
    scale_type=scale_type,
    scale_epoch_str=scale_epoch_type + "_" + str(scale_epoch_hours) + "_hours",
    duration_stride_str=duration_stride_str)
    # pat_id=pat_id)

file_buffer_sec = 120 # number of seconds to ignore from beginning and end of each file

save_dir = root_dir + f'{pat_id}/scaled_data_epochs'
if not os.path.exists(save_dir): os.makedirs(save_dir)

code_savepath = save_dir + '/metadata/code_used/'
if not os.path.exists(code_savepath): os.makedirs(code_savepath)
shutil.copy(__file__,code_savepath) # Save code copy

# Get a list of the pickle files for this patient
files = glob.glob(big_pickle_dir + '/' + pat_id + '*.pkl')

# Get start datetime objects for each file start timestamp
file_starts_dt = []
for file in files:
    file_splits = file.split("/")[-1].split("_")
    file_starts_dt.append(
        datetime.datetime(
            year=int(file_splits[1][4:8]), 
            month=int(file_splits[1][0:2]), 
            day=int(file_splits[1][2:4]),
            hour=int(file_splits[2][0:2]), 
            minute=int(file_splits[2][2:4]), 
            second = int(file_splits[2][4:6]),
            microsecond=int((float(file_splits[2][6:8])/100)*1e6)))
    

# ### AQUIRE NORMALIZATION PARAMETERS

# Read in previously aquired scaling variables if already aquired
if previous_scale_aquired:
    if (scale_type == 'LinearScale') | (scale_type == 'HyperTanScaling') | (scale_type == 'CubeRootScale'):
        linear_interp_by_ch = [] # do not use
        file_path = save_dir + '/metadata/scaling_metadata/channel_scale_factors.pkl'
        with open(file_path, "rb") as f:
            scale_factors = pickle.load(f)

    elif scale_type ==  'HistEqualScale':
        scale_factors = [] # do not use
        file_path = save_dir + '/metadata/scaling_metadata/linear_interpolations_by_channel.pkl'
        with open(file_path, "rb") as f:
            linear_interp_by_ch = pickle.load(f)


# Need to get the norm params from scratch
else:
    # Will also write the scaling output to the directory, can read in for use later
    scale_factors, linear_interp_by_ch = preprocess_aquire_params.aquire_scale_params(
        pat_id=pat_id,
        num_channels=num_channels,
        big_pickle_dir=big_pickle_dir,
        files=files,
        file_starts_dt=file_starts_dt,
        atd_file=atd_file,
        save_dir=save_dir,
        scale_epoch_type=scale_epoch_type,
        scale_epoch_hours=scale_epoch_hours,
        buffer_start_hours=buffer_start_hours,
        channel_scale_style=channel_scale_style,
        scale_type=scale_type,
        resamp_freq=resamp_freq,
        histo_min=histo_min,
        histo_max=histo_max,   
        num_bins=num_bins)


# ### EMPLOY NORMALIZATION

preprocess_employ_norms.employ_norm(
    files=files  ,                    
    file_starts_dt=file_starts_dt,
    num_channels=num_channels,
    file_buffer_sec=file_buffer_sec,
    resamp_freq=resamp_freq,
    save_dir=save_dir,
    scale_type=scale_type,
    scale_factors=scale_factors,
    linear_interp_by_ch=linear_interp_by_ch,
    out_dur=out_dur,
    out_stride=out_stride,
    montage=montage,
    savename_base=file_splits[0],
    PROCESS_FILE_DEBUG_LIST=PROCESS_FILE_DEBUG_LIST)

print("Script complete")