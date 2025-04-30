# TODO - add stats output to SEE: percent overlap, any gaps in time
import sys
sys.path.append("/home/ghassanmakhoul/Documents/Tornadoes_v1/")
import glob, os
import datetime
import shutil
from utilities import utils_functions
import preprocess_aquire_params, preprocess_employ_norms
import pickle
import ipdb

# TODO pull out usable channels e.g. Epat02...

# Settings
previous_big_pickles = False
previous_scale_aquired = False
micro_recordings = True
micro_channels = []
PROCESS_FILE_DEBUG_LIST = []
user_dir = '/home/ghassanmakhoul/Documents'
ignore_channel_units = False  # Need to bypass units check for NK aquisitions. TODO: updatr to check units after channels to ignore are deleted

atd_file = '/home/ghassanmakhoul/Documents/tornadoes_v1/all_time_data_01092023_112957.csv'

# Define the patient directory that contains all EDF files
# All varliable defined here
# ASSUMPTION: files will be named "<patname>_MMDDYYYY_HHMMSSss" where S = seconds, s = milliseconds
pat_id = 'Spat115' 
montage = 'BIPOLE'
freq_bands = [] # [[1, 12], [12, 59], [61, 179]], [] - LEave
if freq_bands == []: freq_bands_str = 'wholeband'
else: freq_bands_str = f"{freq_bands}".replace("], [", "Hz_").replace(", ", "to").replace("[[","").replace("]]","Hz")
num_channels = 183  # BIPOLAR! - for Spat 113, remember that we got rid of channels for micro-recordings on this Spat
if freq_bands != []: num_channels = num_channels * len(freq_bands)
bipole_or_monopole = 'Bipole_datasets'
resamp_freq = 512
#big pickle is a bit of a preprocessed set of signals
#hasn't chopped up the recordings for training yet. 
big_pickle_dir = os.path.join(user_dir,'data/Resamp_' + str(resamp_freq) + f'_Hz/bipole_filtered_{freq_bands_str}_unscaled_big_pickles/{pat_id}')
channel_scale_style = 'By_Channel_Scale' # 'Same_Scale_For_All_Channels'  # 'By_Channel_Scale'    Should probably do by_channel for sure if using multiple freq_bands
scale_type = 'HistEqualScale' # 'LinearScale', 'HyperTanScaling', 'CubeRootScale', 'HistEqualScale'
scale_epoch_hours = 24
buffer_start_hours = 2 # allow for setup weirdness to pass

# uses the first 24 hours of data to set scale for hist-norm. Make sure this time
# matches at least as much data as you will be using for training
scale_epoch_type= 'data_normalized_to_first'  # data_normalized_to_first, data_normalized_to_first_seizure_centered
out_dur = 1024 # seconds, how big window is for training
out_stride = 896 # seconds, stride for window generation
duration_stride_str = 'DurStr_' + str(out_dur) + 's' + str(out_stride) + 's_epoched_datasets/'
expected_unit = 'uV'
desired_samp_freq = 512 # Hz
dir_edf = os.path.join(user_dir, 'data',pat_id)

ch_names_to_ignore = ['-', 'FP1', 'FP2', 'Fp1','F7','T7','P7','O1','Fp2','F8','T8','P8','O2','F3','C3','P3','F4','C4','P4','Fz','Cz','Pz','Fz','F8','T8','T2','F7','T7','T1','63', '64', 'DC01','DC02','DC03','DC04','E', 'EEGMark1', 'EEGMark2','EEG Mark1', 'EEG Mark2', 'Cz', 'Pz', 'EKG1', 'EKG2', 'EKG3', 'EKG4', 'Events/Markers', 'L56','L57','L58','L59','L60','L61','L62','L63','L64','B21','B22','B23','B24','B25','B26','B27','B28','B29','B30','B31','B32','B33','B34','B35','B36','B37','B38','B39','B40','B41','B42','B43','B44','B45','B46','B47','B48','B49','B50','B51','B52','B53','B54','B55','B56','B57','B58','B59','B60','B61','B62','B63','B64', 'TRIG']
if micro_recordings:
    ch_names_to_ignore = ch_names_to_ignore + micro_channels
# Histogram parameters (only used for 'HistEqualScale')
#NOTE assumes that fs is multiple of 512!
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

# NOTE change root dir
# Use the directory assembler function to ensure valid path is created for output data epochs
root_dir = utils_functions.assemble_model_save_path(
    base_path=os.path.join(user_dir, 'results'),
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
shutil.copy(__file__,code_savepath) # Save code copyquit

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
#NOTE , now we've montaged and filtered the big pickles
# Read in previously aquired scaling variables if already aquired
# Prev acquired is used for inference in real-time
# need to employ hist equalizatin in real time
# 
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