# -*- coding: utf-8 -*-
"""
@author: grahamwjohnson
Developed between 2023-2025

"""
import string
import subprocess
import time
import hashlib
import random
import datetime
import pandas as pd
import gc
import glob
import pyedflib
import numpy as np
import scipy
import pickle
import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from .latent_plotting import plot_latent
import matplotlib.pylab as pl
pl.switch_backend('agg')
from torchinfo import summary
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import chardet
import codecs
import torch
import shutil
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import multiprocessing as mp
import math
import matplotlib.colors as colors
from sklearn.metrics import confusion_matrix
import matplotlib.colors as mcolors
import cmasher as cmr
from scipy.stats import gaussian_kde

# Local imports
from models.GMVAE import print_models_flow


# PREPROCESSING 

def create_bip_mont(channels: list[str], pat_id: str, ch_names_to_ignore: list, save_dir: str):
    bip_names = []
    mont_idxs = []

    # Delete unused channels, whole strings must match
    ch_idx_to_delete = []
    ch_names_found_to_delete = []
    for j in range(len(channels)):
        for i in range(len(ch_names_to_ignore)):
            if ch_names_to_ignore[i] == channels[j]:
                ch_idx_to_delete = ch_idx_to_delete + [j]
                ch_names_found_to_delete = ch_names_found_to_delete + [channels[j]]
                continue
    
    # TODO Should be sorting channel names now to deal with Edge cases for patients collected on NK 
    # where 2 channels are listed out of order, but this may actually introduce MORE errors than 
    # if we leave it where we assume channels are in order

    # Find all numbers at ends of channel labels
    nums = np.ones(len(channels), dtype=int)*-1
    for i in range(0,len(channels)):

        # Skip unused channels
        if i in ch_idx_to_delete: 
            continue

        str_curr = channels[i]
        still_number = True
        ch_idx = -1
        while still_number:
            curr_chunk = str_curr[ch_idx:]
            if not curr_chunk.isnumeric():
                nums[i] = str_curr[ch_idx+1:]
                still_number = False
            ch_idx = ch_idx - 1
    
    # Base the lead change on when numbers switch because this is more
    # robust to weird naming strategies that use numbers in base name
    for i in range(0,len(nums) - 1):
        if nums[i] + 1 == nums[i+1]:
            # Valid monotonically increasing bipolar pair
            bip_names.append(channels[i] + channels[i+1])
            mont_idxs.append([i,i+1])

    # Save a CSV to output directory with bip names and mont_idxs 
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    df_bipmont = pd.DataFrame({'mont_idxs': mont_idxs, 
                       'bip_names': bip_names})
    df_bipmont.to_csv(save_dir + '/' + pat_id + '_bipolar_montage_names_and_indexes_from_rawEDF.csv')

    return mont_idxs, bip_names

def highpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 8):
    sos = scipy.signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def bandstop(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 8):
    sos = scipy.signal.butter(poles, edges, 'bandstop', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 8):
    sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def apply_wholeband_filter(y0, fs):

    # Hardcoded filter values, Hz - This is done before splitting into desired freq ranges
    FILT_HP = 1
    FILT_BS_RANGE_1 = [59, 61]
    FILT_BS_RANGE_2 = [119, 121]
    FILT_LP = 179
    
    y1 = highpass(y0, FILT_HP, fs)
    y2 = bandstop(y1, FILT_BS_RANGE_1, fs)
    y3 = bandstop(y2, FILT_BS_RANGE_2, fs)
    y4 = lowpass(y3, FILT_LP, fs)

    return y4

def apply_banded_filter(y0, freq_bands, fs, arbitrary_scaling_perc_range=[1, 99]):
    tmp = np.empty((len(freq_bands), y0.shape[0]), dtype=np.float64)
    for b in range(len(freq_bands)):
        curr_range = freq_bands[b]
        y1 = highpass(y0, curr_range[0], fs)
        y2 = lowpass(y1, curr_range[1], fs)

        # Resccale to maximize float16 format
        min_perc_y2 = np.percentile(np.abs(y2), arbitrary_scaling_perc_range[0])
        max_perc_y2 = np.percentile(np.abs(y2), arbitrary_scaling_perc_range[1])

        tmp[b] = y2/(max_perc_y2 - min_perc_y2)

    return tmp.astype(np.float16)

def read_channel(f, i):
    return f.readSignal(i)

def montage_filter_pickle_edfs(pat_id: str, dir_edf: str, save_dir: str, desired_samp_freq: int, freq_bands: list, expected_unit: str, montage: str, ch_names_to_ignore: list, ignore_channel_units: bool):
                    
        files =  glob.glob(dir_edf + "/*.EDF")

        # CREATE BIPOLAR MONTAGE
        # Find a suitable file to mnake the montage first because of "c_label" problem
        # Find a file WITHOUT clabel in the name
        print("Finding file without 'clabel' problem to make bipolar montage")
        file_idx = -1
        total_files = len(files)
        for file in files:
            file_idx += 1
            print("[" + str(file_idx) + '/' + str(total_files) + ']: ' + file)
            if "clabel" not in file.split("/")[-1]:
                print(f"Found file [{file}], calculating bipolar montage")
                # Use PyEDFLib to read in files
                # (MNE broke for large files) 
                f = pyedflib.EdfReader(file)
                channels = f.getSignalLabels()
                mont_idxs, bip_names = create_bip_mont(channels, pat_id, ch_names_to_ignore, save_dir + '/metadata')
                print("Montage created")
                f._close()
                del f
                break


        file_idx = -1
        total_files = len(files)
        print("Processing all files:")
        for file in files:
            file_idx += 1
            print("[" + str(file_idx) + '/' + str(total_files) + ']: ' + file)
            print("Reading in EDF")
            gc.collect()
     
            # Use PyEDFLib to read in files
            # (MNE broke for large files) 
            f = pyedflib.EdfReader(file)
            n = f.signals_in_file
            channels = f.getSignalLabels()
            all_samp_freqs = np.array(f.getSampleFrequencies())
                        
            # Ensure all channel units are the same
            if not ignore_channel_units:
                sig_headers = f.getSignalHeaders()
                ch_units = np.empty(len(channels), dtype=object)
                for i in range(0,len(channels)):
                    ch_units[i] = sig_headers[i]['dimension']
                if not np.all(ch_units == ch_units[0]): raise Exception("Not all channel units are the same for file: " + file)  
                # Compare this unit to the expected file units
                if ch_units[0] != expected_unit: raise Exception("Current file's channel units do not equal expected units: " + expected_unit + ". Current file: " + file)
                # Store this unit to compare to next EDF file
            
            start_t = time.time()
            raw_data = np.zeros((n, f.getNSamples()[0]), dtype=np.float16)
            for i in np.arange(n):
                raw_data[i, :] = f.readSignal(i)
            print(f"Time read EDF in: {time.time()-start_t}")
            f._close()
            del f
            print("EDF read in")

            # Ensure sampling freq are all equal to eachother
            if not np.all(all_samp_freqs == all_samp_freqs[0]): raise Exception("Not all sampling freqs units are the same for file: " + file)
            
            # Check sampling frequency is as desired, resample if not:
            fs = all_samp_freqs[0] # all should be equal, can just pull first one
            if fs != desired_samp_freq: 
                # raise Exception(f"Sampling frequency resampling to {desired_samp_freq} not yet coded, needed for file: {file}, this file was sampled at {fs}" )
                print(f"Resampling to {desired_samp_freq} for file: {file}, this file was sampled at {fs} Hz")
                if fs%desired_samp_freq == 0:
                    fs_mult = int(fs/desired_samp_freq)
                    print(f"Sampling frequency of file [{fs}] is exact multiple of desired sampling frequency [{desired_samp_freq}], so simply indexing at interval of {fs_mult}")
                    # Simply index at the multiple
                    raw_data = raw_data[:, 0::fs_mult]
                    fs = desired_samp_freq
                else:
                    raise Exception(f"FS [{fs}] NOT MULTIPLE OF DESIRED SAMPLING FREQUENCY - NOT CODED UP YET")
            else:
                print(f"Sampling frequency confirmed to be {fs} Hz")

            # If doing bipolar montage
            if montage == 'BIPOLE':
                print("Assigning bipolar montage")
                # Check that bip names match based on already created montage from previous files
                new_bip_names = ["" for x in range(len(bip_names))]
                for i in range(0,len(bip_names)):
                    new_bip_names[i] = channels[mont_idxs[i][0]] + channels[mont_idxs[i][1]]
                if new_bip_names != bip_names: 
                    print("WARNING: Current file's created bip montage does not exactly equal original file montage. Current file: " + file)
                    print("Assuming that channels are in proper order, just with wrong names (i.e. bad Natus template)")

                # If we made it this far, then the montage aligns across EDF files
                # Re-assign data to bipolar montage (i.e. subtraction: A-B)
                new_raw_data = np.empty([len(bip_names),raw_data.shape[1]], dtype=np.float16)
                for i in range(0,len(bip_names)):
                    new_raw_data[i,:] = raw_data[mont_idxs[i][0],:] - raw_data[mont_idxs[i][1],:]
                raw_data = new_raw_data
                del new_raw_data
                gc.collect()

                # Re-assign channel names to bipolar
                channels = bip_names
                print("Bipolar montage assignment complete")

            # Filter with IIR zero phase sosfiltfilt pass bands)
            # Do each channel sepreately (#TODO: parallelize)
            # filt_data = np.empty(raw_data.shape, dtype=np.float16)
            print("Filtering the data")
            filt_data = np.asarray([apply_wholeband_filter(raw_data[i,:], fs) for i in range(0, len(channels))], dtype=np.float16)

            print("Data wholeband filtered, with line noise notch filters")
            if freq_bands == []:
                filt_data_banded = filt_data
            else:
                print(f"Filtering into subbands {freq_bands} Hz")
                # filt_data_banded = np.empty(raw_data.shape[0] * len(freq_bands), raw_data.shape[1], dtype=np.float16) 
                filt_data_banded = np.concatenate([apply_banded_filter(filt_data[i,:], freq_bands, fs) for i in range(0, len(channels))], axis=0).astype(np.float16)

            del raw_data
            del filt_data
            gc.collect()

            # Save the entire UNSCALED filtered data as a pickle file
            freq_str = f"{freq_bands}".replace("], [", "Hz_").replace(", ", "to").replace("[[","").replace("]]","Hz")
            if montage == 'BIPOLE': save_name = save_dir + '/' + file.split('/')[-1].replace('.EDF','_resampled_' + str(fs) + f'_Hz_bipole_filtered_{freq_str}.pkl')
            if montage == 'MONOPOLE': save_name = save_dir + '/' + file.split('/')[-1].replace('.EDF','_resampled_' + str(fs) + f'_Hz_monopole_filtered_{freq_str}.pkl')
            with open(save_name, "wb") as f:
                pickle.dump(filt_data_banded, f)
            print("Big pickle saved")

def fill_hist_by_channel(data_in: np.ndarray, histo_bin_edges: np.ndarray, zero_island_delete_idxs: list):

    """
    Computes per-channel histograms for a multi-channel dataset, with optional removal of specified data segments.

    This function calculates histograms for each channel in the input `data_in` using the provided `histo_bin_edges`.
    It also allows for the exclusion of specific index ranges (`zero_island_delete_idxs`) before histogram computation.

    Parameters:
    -----------
    data_in : np.ndarray (shape `[num_channels, num_samples]`)
        The input multi-channel data array where each row corresponds to a channel.

    histo_bin_edges : np.ndarray (shape `[num_bins + 1]`)
        The edges defining the histogram bins.

    zero_island_delete_idxs : list of tuples
        List of index ranges `(start_idx, end_idx)` specifying segments to remove from all channels before computing histograms.

    Returns:
    --------
    histo_bin_counts : np.ndarray (shape `[num_channels, num_bins]`)
        A 2D array containing histogram counts for each channel.

    Notes:
    ------
    - If `zero_island_delete_idxs` is not empty, specified index ranges are removed before histogram computation.
    - Uses `np.histogram` to compute bin counts for each channel.
    - Outputs progress to `sys.stdout` to indicate per-channel processing status.
    """

    if zero_island_delete_idxs != []:
        zero_island_delete_idxs.sort()
        # Delete zero islands
        for ir in reversed(range(len(zero_island_delete_idxs))):
            data_in = data_in[:, np.concatenate([np.arange(0,zero_island_delete_idxs[ir][0]), np.arange(zero_island_delete_idxs[ir][1]+1, data_in.shape[1])], axis=0)]

    # initialize output 2D array
    num_channels = data_in.shape[0]
    histo_bin_counts = np.zeros([num_channels, len(histo_bin_edges)-1]) # Save a histo count for every channel (will just be sum if scaling same for all channels)

    # Store the counts of datapoints within the histogram bins of interest
    for ch_idx in range(0, num_channels):
        sys.stdout.write("\rComputing histogram for channel ID: " + str(ch_idx) + '/' + str(num_channels-1))
        sys.stdout.flush()    
        # Pull out this channel's data
        data_ch = data_in[ch_idx,:]
        histo_bin_counts[ch_idx, :] = np.histogram(data_ch, histo_bin_edges)[0][:]
    
    return histo_bin_counts


# INITIALIZATIONS

def print_model_summary(model):
    print("Calculating model summary")
    summary(model, num_classes=1)
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = (mem_params + mem_bufs) / 1e9  # in bytes
    print("Expected GPU memory requirement (parameters + buffers): " + str(mem) +" GB")

def random_animal(rand_name_json, **kwargs):
    # Read in animal names, pull random name
    with open(rand_name_json) as json_file:
        json_data = json.load(json_file)
    
    np.random.seed(seed=None) # should replace with Generator for newer code
    rand_idx = np.random.randint(0, len(json_data))
    rand_name = json_data[rand_idx]

    return f"{rand_name}"

def random_filename_string(length=10):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(length))

def run_script_from_shell(env_python_path, script_path, *args):
    """
    Runs a Python script using the shell.

    Args:
        script_path (str): Path to the Python script.
        *args: Arguments to pass to the script.

    Returns:
        tuple: A tuple containing the return code, standard output, and standard error.
    """
    try:
        command = ['python', script_path, args[0], args[1], args[2]] # args: tmp_dir, fnames.csv, num_rand_hashes
        # Run the command and suppress output
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # print("DEBUGGING SLEEP!!!")
        # time.sleep(9999999)

        return process

    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout, e.stderr
    except FileNotFoundError:
        return -1, "", "File not found"

def prepare_dataloader(dataset: Dataset, batch_size: int, droplast=False, num_workers=0):

    if num_workers > 0:
        persistent_workers=True
        print("WARNING: num workers >0, have experienced odd errors...")

    else:
        persistent_workers=False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,    
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        drop_last=droplast,
        persistent_workers=persistent_workers
    )

def initialize_directories(
        run_notes,
        cont_train_model_dir,
        **kwargs):

    # *** CONTINUE EXISTING RUN initialization ***

    if kwargs['continue_existing_training']:

        kwargs['model_dir'] = cont_train_model_dir
        kwargs['pic_dataset_dir'] = kwargs['model_dir'] + '/dataset_bargraphs'
        kwargs['log_dir'] =  kwargs['model_dir'] + '/data_logs'

        # Find the epoch to start training
        check_dir = kwargs['model_dir'] + "/checkpoints"
        epoch_dirs = glob.glob(check_dir + '/Epoch*')
        epoch_nums = [int(f.split("/")[-1].replace("Epoch_","")) for f in epoch_dirs]

        # Find the highest epoch already trained
        max_epoch = max(epoch_nums)
        print(f"Resuming training after saved epoch: {str(max_epoch)}")
        
        # Construct the proper file names to get CORE state dicts
        kwargs['gmvae_state_dict_prev_path'] = check_dir + f'/Epoch_{str(max_epoch)}/posterior_checkpoints/checkpoint_epoch{str(max_epoch)}_gmvae.pt'
        kwargs['gmvae_opt_state_dict_prev_path'] = check_dir + f'/Epoch_{str(max_epoch)}/posterior_checkpoints/checkpoint_epoch{str(max_epoch)}_gmvae_opt.pt'

        # Proper names for running latents 
        kwargs['running_mean_path'] = check_dir + f'/Epoch_{str(max_epoch)}/posterior_checkpoints/checkpoint_epoch{str(max_epoch)}_running_means.pkl'
        kwargs['running_logvar_path'] = check_dir + f'/Epoch_{str(max_epoch)}/posterior_checkpoints/checkpoint_epoch{str(max_epoch)}_running_logvars.pkl'
        kwargs['running_zmeaned_path'] = check_dir + f'/Epoch_{str(max_epoch)}/posterior_checkpoints/checkpoint_epoch{str(max_epoch)}_running_zmeaned.pkl'
        kwargs['running_mogpreds_path'] = check_dir + f'/Epoch_{str(max_epoch)}/posterior_checkpoints/checkpoint_epoch{str(max_epoch)}_running_mogpreds.pkl' 
        kwargs['running_patidxs_path'] = check_dir + f'/Epoch_{str(max_epoch)}/posterior_checkpoints/checkpoint_epoch{str(max_epoch)}_running_patidxs.pkl' 

        # Set the start epoch 1 greater than max trained
        kwargs['start_epoch'] = (max_epoch + 1) 
        

    # *** NEW RUN initialization ***  

    else:
        # Make run directories
        kwargs['model_dir'] = append_timestamp(kwargs['root_save_dir'] + '/trained_models/' + kwargs['run_params_dir_name'] + '/' + run_notes + '_')
        os.makedirs(kwargs['model_dir'])
        kwargs['pic_dataset_dir'] = kwargs['model_dir'] + '/dataset_bargraphs'
        os.makedirs(kwargs['pic_dataset_dir'])
        kwargs['log_dir'] =  kwargs['model_dir'] + '/data_logs'
        os.makedirs(kwargs['log_dir'])

        # Fresh run 
        kwargs['start_epoch'] = 0

    return kwargs

def run_setup(**kwargs):
    # Print to console
    print("\n\n***************** MAIN START " + datetime.datetime.now().strftime("%I:%M%p-%B/%d/%Y") + "******************\n\n")        
    os.environ['KMP_DUPLICATE_LIB_OK']='True'    
    mp.set_start_method('spawn', force=True)
    mp_lock = mp.Lock()

    # Clean tmp directories
    tmp_dirs = glob.glob('/dev/shm/tornado_tmp_*')
    for t in tmp_dirs: 
        shutil.rmtree(t)
        print(f"Deleted tmp directory: {t}")

    # All Time Data file to get event timestamps
    kwargs['run_params_dir_name'] = get_training_dir_name(**kwargs)

    # Set world size to number of GPUs in system available to CUDA
    world_size = torch.cuda.device_count()
        
    # Random animal name
    run_notes = random_animal(**kwargs) 

    # Call the initialization script to start new run or continue existing run
    kwargs = initialize_directories(run_notes=run_notes, **kwargs)

    # Print the model forward pass sizes
    fake_data = torch.rand(kwargs['max_batch_size'], kwargs['transformer_seq_length'], kwargs['padded_channels'], kwargs['encode_token_samples']) 
    print_models_flow(x=fake_data, **kwargs)

    # Get the timestamp ID for this run (will be used to resume wandb logging if this is a restarted training)
    s = kwargs['model_dir'].split("/")[-1]
    kwargs['timestamp_id'] = ''.join(map(str, s))
    kwargs['run_name'] = '_'.join(map(str,s.split('_')[0:2]))

    # Save the post-processed kwargs to a file in model directory
    savedir = f"{kwargs['model_dir']}/config"
    if not os.path.exists(savedir): os.makedirs(savedir)
    save_name = f"{savedir}/kwargs_execd_epoch{kwargs['start_epoch']}.pkl"
    with open(save_name, "wb") as f: pickle.dump(kwargs, f)

    # Export the conda environment
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    if env_name:
        output_file = f"{savedir}/{env_name}_environment.yml"
        try:
            subprocess.run(["conda", "env", "export", "--file", output_file], check=True)
            print(f"Conda environment '{env_name}' exported to {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error exporting Conda environment: {e}")
    else:
        print("No active Conda environment detected.")

    return world_size, kwargs


# FILE I/O

def exec_kwargs(kwargs):
    # exec all kwargs in case there is python code
    for k in kwargs:
            if isinstance(kwargs[k], str):
                if kwargs[k][0:6] == 'kwargs':
                    exec(kwargs[k])

    return kwargs

def get_num_channels(pat_id, pat_num_channels_LUT):

    df = pd.read_csv(pat_num_channels_LUT)

    return int(df.loc[df['pat_id'] == pat_id, 'num_channels'].iloc[0])

def filename_to_datetimes(list_file_names):
        start_datetimes = [datetime.datetime.min]*len(list_file_names)
        stop_datetimes = [datetime.datetime.min]*len(list_file_names)
        for i in range(0, len(list_file_names)):
            splits = list_file_names[i].split('_')
            aD = splits[1]
            aT = splits[2]
            start_datetimes[i] = datetime.datetime(int(aD[4:8]), int(aD[0:2]), int(aD[2:4]), int(aT[0:2]), int(aT[2:4]), int(aT[4:6]), int(int(aT[6:8])*1e4))
            bD = splits[4]
            bT = splits[5]
            stop_datetimes[i] = datetime.datetime(int(bD[4:8]), int(bD[0:2]), int(bD[2:4]), int(bT[0:2]), int(bT[2:4]), int(bT[4:6]), int(int(bT[6:8])*1e4))
        return start_datetimes, stop_datetimes

def delete_old_checkpoints(dir: str, curr_epoch: int, KL_divergence_stall_epochs, KL_divergence_epochs_AT_max, KL_divergence_epochs_TO_max, **kwargs):

    # SAVE_KEYWORDS = ["hdbscan", "pacmap"]

    all_dir_names = glob.glob(f"{dir}/Epoch*")

    epoch_nums = [int(f.split("/")[-1].replace("Epoch_","")) for f in all_dir_names]

    # Add current epoch to save files
    save_epochs = [curr_epoch]

    # KEYWORD Save
    # for i in range(len(all_dir_names)):
    #     subepoch_dirs = [f.split("/")[-1] for f in glob.glob(all_dir_names[i] + "/*")]
    #     for f in subepoch_dirs:
    #         if any(substr in f for substr in SAVE_KEYWORDS):
    #             save_epochs.append(epoch_nums[i])
    #             break

    # KL_divergence Epoch cycle save - save the last epoch in a KL_divergence annealing cycle
    mod_val = KL_divergence_stall_epochs + KL_divergence_epochs_AT_max + KL_divergence_epochs_TO_max - 1
    for x in epoch_nums:
        if (x % mod_val == 0) & (curr_epoch > 0):
            save_epochs.append(x)

    [shutil.rmtree(all_dir_names[i]) if epoch_nums[i] not in save_epochs else print(f"saved: {all_dir_names[i].split('/')[-1]}") for i in range(len(epoch_nums))]

    return

def digest_SPES_notes(spes_file):

    # For gen1 of this code, we want a single epoch for the ENTIRE stim session of a single bipole pair (all current levels in one sungle epoch)
    spes_style = -1 # -1 is style not found yet. 0 is 'old' style with 'Closed Relay' in line. 1 is 'new' style with more specific 'Start Stimulation' for EACH current level
    spes_master_EDF_creation_datetime = []
    spes_stim_pair_names = []
    spes_start_datetimes = []
    spes_stop_datetimes = []


    # Read file to digest stim paairs
    with open(spes_file, 'rb') as file:
        encoding = chardet.detect(file.read())['encoding']

    stim_pairs_count = 0
    seeking_state = 0 # 0 need initial params, 1 seeking next stim pair, 2 found stim pair, need end time
    with codecs.open(spes_file, encoding=encoding) as f:
        for line in f:
            # Parse each line manually
            
            # STATE 0: Catch the inital parameters needed to parse the rest of the file
            if seeking_state == 0:
                if ('Creation Date' in line) & (spes_master_EDF_creation_datetime == []):
                    raw_timestr = line[15:-2]
                    spes_master_EDF_creation_datetime = datetime.datetime.strptime(raw_timestr, '%H:%M:%S %b %d, %Y')
                elif spes_style == -1:
                    if 'Closed relay' in line: 
                        spes_style = 0
                        seeking_state = 1
                    elif 'Start Stimulation' in line: 
                        spes_style = 1
                        seeking_state = 1
            
            # STATE 1: Looking for next stim pair
            if seeking_state == 1: # Not an 'elif' to catch the first relay pair
                if spes_style == 0:
                    if 'Closed relay' in line:
                        spes_stim_pair_names.append(line[29:-2])
                        timestr = line[3:11] + ' ' + spes_master_EDF_creation_datetime.strftime('%d/%m/%Y')
                        datetime_curr = datetime.datetime.strptime(timestr, '%H:%M:%S %d/%m/%Y')
                        if datetime_curr < spes_master_EDF_creation_datetime: datetime_curr = datetime_curr + datetime.timedelta(days=1)
                        spes_start_datetimes.append(datetime_curr)
                        stim_pairs_count += 1
                        seeking_state = 2

                elif spes_style == 1:
                    if 'Start Stimulation' in line: 
                        raise Exception("Need to code up style 2, might not be same indexes as style 1")

                        stim_pairs_count += 1
                        seeking_state = 2
            
            # STATE 2: Looking for next stim pair
            elif seeking_state == 2:
                if 'Closed relay' in line: raise Exception("ERROR: 'Closed relay' found in following line before an enddate for previous stim pair was found: " + line)

                elif spes_style == 0:
                    if 'Opened relay' in line: 
                        timestr = line[3:11] + ' ' + spes_master_EDF_creation_datetime.strftime('%d/%m/%Y')
                        datetime_curr = datetime.datetime.strptime(timestr, '%H:%M:%S %d/%m/%Y')
                        if datetime_curr < spes_master_EDF_creation_datetime: datetime_curr = datetime_curr + datetime.timedelta(days=1)
                        spes_stop_datetimes.append(datetime_curr)
                        seeking_state = 1

                elif spes_style == 1:
                    raise Exception("Need to code up style 2, complicated logic to get the end of stim pair session")
                
    # Read file in again to find the session start and stop times
    if spes_style == 0:
        stim_session_start = []
        stim_session_stop = []
        with open(spes_file, 'rb') as file:
            encoding = chardet.detect(file.read())['encoding']

        with codecs.open(spes_file, encoding=encoding) as f:
            for line in f:
                if 'Beginning of Recording' in line:
                    if stim_session_start != []: raise Exception('ERROR: stim session start already found')
                    timestr = line[3:11] + ' ' + spes_master_EDF_creation_datetime.strftime('%d/%m/%Y')
                    stim_session_start = datetime.datetime.strptime(timestr, '%H:%M:%S %d/%m/%Y')
                if 'End of Study' in line:
                    if stim_session_stop != []: raise Exception('ERROR: stim session stop already found')
                    timestr = line[3:11] + ' ' + spes_master_EDF_creation_datetime.strftime('%d/%m/%Y')
                    stim_session_stop = datetime.datetime.strptime(timestr, '%H:%M:%S %d/%m/%Y')
    
    elif spes_style == 1:
        raise Exception("Not coded up yet")

    return spes_stim_pair_names, spes_start_datetimes, spes_stop_datetimes, stim_session_start, stim_session_stop

def get_training_dir_name(train_val_pat_perc, **kwargs):
    
    train_str = str(train_val_pat_perc[0]*100)
    val_str = str(train_val_pat_perc[1]*100)
    run_params_dir_name = "dataset_train" + train_str + "_val" + val_str 

    return run_params_dir_name

def get_pat_seiz_datetimes(
    pat_id, 
    atd_file, 
    FBTC_bool=True, 
    FIAS_bool=True, 
    FAS_to_FIAS_bool=True,
    FAS_bool=True, 
    subclinical_bool=True, 
    focal_unknown_bool=True,
    unknown_bool=True, 
    non_electro_bool=False,
    **kwargs
    ):

    # # Debugging
    # print(pat_id)

    # Original ATD file from Derek was tab seperated
    atd_df = pd.read_csv(atd_file, sep=',', header='infer')
    pat_seizure_bool = (atd_df['Pat ID'] == pat_id) & (atd_df['Type'] == "Seizure")
    pat_seizurebool_AND_desiredTypes = pat_seizure_bool
    
    # Look for each seizure type individually & delete if not desired
    # seiz_type_list = ['FBTC', 'FIAS', 'FAS_to_FIAS', 'FAS', 'Subclinical', 'Focal, unknown awareness', 'Unknown', 'Non-electrographic']
    seiz_type_list = ['FBTC', 'FIAS', 'FAS_to_FIAS', 'FAS', 'Subclinical', 'Focal unknown awareness', 'Unknown', 'Non-electrographic']
    delete_seiz_type_bool_list = [FBTC_bool, FIAS_bool, FAS_to_FIAS_bool, FAS_bool, subclinical_bool, focal_unknown_bool, unknown_bool, non_electro_bool]
    for i in range(0,len(seiz_type_list)):
        if delete_seiz_type_bool_list[i]==False:
            find_str = seiz_type_list[i]
            curr_bool = pat_seizure_bool & (atd_df.loc[:,'Seizure Type (FAS; FIAS; FBTC; Non-electrographic; Subclinical; Unknown)'] == find_str)
            pat_seizurebool_AND_desiredTypes[curr_bool] = False

    df_subset = atd_df.loc[pat_seizurebool_AND_desiredTypes, ['Type','Seizure Type (FAS; FIAS; FBTC; Non-electrographic; Subclinical; Unknown)', 'Date (MM:DD:YYYY)', 'Onset String (HH:MM:SS)', 'Offset String (HH:MM:SS)']]
    
    pat_seiz_startdate_str = df_subset.loc[:,'Date (MM:DD:YYYY)'].astype(str).values.tolist() 
    pat_seiz_starttime_str = df_subset.loc[:,'Onset String (HH:MM:SS)'].astype(str).values.tolist()
    pat_seiz_stoptime_str = df_subset.loc[:,'Offset String (HH:MM:SS)'].astype(str).values.tolist()
    pat_seiz_types_str = df_subset.loc[:,'Seizure Type (FAS; FIAS; FBTC; Non-electrographic; Subclinical; Unknown)'].astype(str).values.tolist()

    # Skip any lines that have nan/none or unknown time entries
    delete_list_A = [i for i, val in enumerate(pat_seiz_starttime_str) if (val=='nan' or val=='Unknown')]
    delete_list_B = [i for i, val in enumerate(pat_seiz_stoptime_str) if (val=='nan' or val=='Unknown')]
    delete_list = list(set(delete_list_A + delete_list_B))
    delete_list.sort()
    if len(delete_list) > 0:
        print(f"WARNING: deleting {len(delete_list)} seizure(s) out of {len(pat_seiz_startdate_str)} due to 'nan'/'none'/'Unknown' in master time sheet")
        print(f"Delete list is: {delete_list}")
        [pat_seiz_startdate_str.pop(del_idx) for del_idx in reversed(delete_list)]
        [pat_seiz_starttime_str.pop(del_idx) for del_idx in reversed(delete_list)]
        [pat_seiz_stoptime_str.pop(del_idx) for del_idx in reversed(delete_list)]
        [pat_seiz_types_str.pop(del_idx) for del_idx in reversed(delete_list)]

    # Initialize datetimes
    pat_seiz_start_datetimes = [0]*len(pat_seiz_starttime_str)
    pat_seiz_stop_datetimes = [0]*len(pat_seiz_stoptime_str)

    for i in range(0,len(pat_seiz_startdate_str)):
        sD_splits = pat_seiz_startdate_str[i].split(':')
        sT_splits = pat_seiz_starttime_str[i].split(':')
        start_time = datetime.time(
                            int(sT_splits[0]),
                            int(sT_splits[1]),
                            int(sT_splits[2]))
        pat_seiz_start_datetimes[i] = datetime.datetime(int(sD_splits[2]), # Year
                                            int(sD_splits[0]), # Month
                                            int(sD_splits[1]), # Day
                                            int(sT_splits[0]), # Hour
                                            int(sT_splits[1]), # Minute
                                            int(sT_splits[2])) # Second
        
        sTstop_splits = pat_seiz_stoptime_str[i].split(':')
        stop_time = datetime.time(
                            int(sTstop_splits[0]),
                            int(sTstop_splits[1]),
                            int(sTstop_splits[2]))

        if stop_time > start_time: # if within same day (i.e. the TIME advances, no date included), assign same date to datetime, otherwise assign next day
            pat_seiz_stop_datetimes[i] = datetime.datetime.combine(pat_seiz_start_datetimes[i], stop_time)
        else: 
            pat_seiz_stop_datetimes[i] = datetime.datetime.combine(pat_seiz_start_datetimes[i] + datetime.timedelta(days=1), stop_time)

    return pat_seiz_start_datetimes, pat_seiz_stop_datetimes, pat_seiz_types_str

def get_desired_fnames(
        gpu_id: int,
        pat_id: str, 
        atd_file: str, 
        data_dir: str, 
        intrapatient_dataset_style: list, 
        hour_dataset_range: list, 
        dataset_pic_dir: str,
        print_dataset_bargraphs: bool,
        **kwargs
        ):
    
    # This will have all of the desired file names before splitting into train/val/test
    curr_fnames = []  

    # Get all data without SPES
    if intrapatient_dataset_style == 1:
        curr_fnames = glob.glob(data_dir + '/all_epochs_woSPES/*.pkl') # relies of directory naming to be consistent
        curr_fnames = sort_filenames(curr_fnames)
    
    # Get all data with SPES
    elif intrapatient_dataset_style == 2:
        curr_fnames = glob.glob(data_dir + '/*/*.pkl')
        curr_fnames = sort_filenames(curr_fnames)

    # Get ONLY SPES data
    elif intrapatient_dataset_style == 3:
        curr_fnames = glob.glob(data_dir + '/SPES/*.pkl')
        curr_fnames = sort_filenames(curr_fnames)

    else:
        raise Exception(f"[GPU{str(gpu_id)}] Invalid 'dataset_style' choice, must be 1, 2, or 3")


    # Now split up the data according to dataset range
    end_names = [x.split('/')[-1] for x in curr_fnames]
    start_dts, end_dts = filename_to_datetimes(end_names)


    if (hour_dataset_range[0] == -1) & (hour_dataset_range[1] == -1):
        curr_fnames = curr_fnames # Do nothing

    elif hour_dataset_range[1] == -1:
        # Start time givem, but end is -1 meaning give all
        found_hours = False
        for i in range(len(start_dts)):
            if start_dts[i] > (start_dts[0] + datetime.timedelta(hours=hour_dataset_range[0])):
                curr_fnames = curr_fnames[i:-1]
                found_hours = True
                break
        if not found_hours: raise Exception("Hours desired not found")
    
    elif hour_dataset_range[0] == -1:
        # Run from beginning to given end hours
        found_hours = False
        print("ToDO")
        for i in range(len(end_dts)):
            if end_dts[i] > (start_dts[0] + datetime.timedelta(hours=hour_dataset_range[1])):
                curr_fnames = curr_fnames[0:i-1]
                found_hours = True
                break
        if not found_hours: raise Exception("Hours desired not found")

    else:
        # Should have 2 valid numbers in range
        found_hours = False
        for i in range(len(start_dts)):
            if start_dts[i] > (start_dts[0] + datetime.timedelta(hours=hour_dataset_range[0])):
                for j in range(i, len(end_dts)):
                    if end_dts[j] > (start_dts[0] + datetime.timedelta(hours=hour_dataset_range[1])):
                        curr_fnames = curr_fnames[i:j-1]
                        found_hours = True
                        break
                break
        if not found_hours: raise Exception("Hours desired not found")

    if (gpu_id == 0) & print_dataset_bargraphs:
        print_dataset_bargraphs(pat_id, curr_fnames, curr_fnames, dataset_pic_dir, atd_file=atd_file)

    return curr_fnames

def sort_filenames(file_list):
    # Ensure just have the filename and not whole path
    fnames = [x.split("/")[-1] for x in file_list]
    all_datetimes = filename_to_datetimes(fnames)
    all_start_seconds = [int((x - x.min).total_seconds()) for x in all_datetimes[0]]

    sort_idxs = np.argsort(all_start_seconds)

    sorted_file_list = [file_list[i] for i in sort_idxs]

    return sorted_file_list

def append_timestamp(filename):
    ts = time.asctime(time.localtime(time.time()))
    ts = ts.replace(" ", "_")
    ts = ts.replace(":", "_")
    return filename + ts


# LEARNING RATES & LOSS WEIGHTS

def LR_subfunction(iter_curr, LR_min, LR_max, epoch, manual_gamma, manual_step_size, epoch_stall, LR_epochs_TO_max, LR_epochs_AT_max, iters_per_epoch, LR_rise_first=True):

    """
    Computes the learning rate (LR) based on the current iteration and epoch, with a cyclical learning rate schedule.

    This function applies a piecewise learning rate schedule that includes an initial rise, followed by a plateau at the maximum value. 
    The learning rate adjusts based on manual parameters such as gamma, step size, and the number of epochs to reach maximum learning rate.

    Parameters:
    -----------
    iter_curr : int
        The current iteration within the current epoch.

    LR_min : float
        The minimum learning rate value.

    LR_max : float
        The maximum learning rate value.

    epoch : int
        The current epoch number.

    manual_gamma : float
        The gamma value used to scale the learning rate over epochs.

    manual_step_size : int
        The step size for adjusting the learning rate.

    epoch_stall : int
        The epoch at which the learning rate schedule starts.

    LR_epochs_TO_max : int
        The number of epochs to increase the learning rate to its maximum value.

    LR_epochs_AT_max : int
        The number of epochs the learning rate remains at its maximum value.

    iters_per_epoch : int
        The number of iterations per epoch.

    LR_rise_first : bool, optional (default=True)
        Determines whether the learning rate rises first before reaching the maximum value. If False, the implementation is incomplete.

    Returns:
    --------
    LR_val : float
        The computed learning rate for the current iteration and epoch.

    Notes:
    ------
    - The function assumes that the learning rate rises first (i.e., from `LR_min` to `LR_max`) unless `LR_rise_first` is set to False.
    - The learning rate is adjusted by gamma after a number of epochs specified by `manual_step_size`.
    - The function raises an exception if `LR_rise_first` is set to False, as that part is not yet implemented.
    """

    # Shift the epoch by the stall:
    if epoch < epoch_stall:
        if LR_rise_first: 
            return LR_min
        else:
            return LR_max
    
    else: # Shift epoch so that new epoch 0 is equal to epoch stall, schedule starts from there
        epoch = epoch - epoch_stall
    
    assert epoch >=0, f"Error in epoch stall, calculated shifted epoch to be {epoch}"
    
    # Adjust max and min based on gamma value
    LR_gamma_iter = np.floor(epoch / manual_step_size)
    gamma_curr = manual_gamma ** LR_gamma_iter 
    LR_max_curr = LR_max * gamma_curr
    LR_min_curr = LR_min * gamma_curr
    LR_range = LR_max_curr - LR_min_curr

    # Get current residual
    LR_epoch_period = LR_epochs_TO_max + LR_epochs_AT_max
    LR_epoch_residual = epoch % LR_epoch_period

    # START with rise
    if LR_rise_first:    

        if LR_epoch_residual < LR_epochs_TO_max:
            LR_floor = LR_min_curr + ( LR_range * (LR_epoch_residual/LR_epochs_TO_max) )
            LR_ceil = LR_min_curr + ( LR_range * (LR_epoch_residual+1) /LR_epochs_TO_max)
            LR_val = LR_floor + iter_curr/iters_per_epoch * (LR_ceil - LR_floor) 

        else: 
            LR_val = LR_max_curr
        
    else:
        # LR_ceil = LR_max_curr - ( LR_range * (LR_epoch_residual/LR_epochs_cycle) )
        # LR_floor = LR_ceil - ( LR_range * (LR_epoch_residual + 1) /LR_epochs_cycle)
        # LR_val = LR_ceil - iter/iters_per_epoch * (LR_ceil - LR_floor)        
        raise Exception("ERROR: not coded up")

    return LR_val

def LR_and_weight_schedules(
        epoch, iter_curr, iters_per_epoch, 
        mse_weight_min, mse_weight_max, mse_weight_stall_epochs, mse_weight_epochs_TO_max, mse_weight_epochs_AT_max,
        KL_divergence_max_weight, KL_divergence_min_weight, KL_divergence_epochs_TO_max, KL_divergence_epochs_AT_max, KL_divergence_stall_epochs,
        gp_weight_max, gp_weight_min, gp_weight_stall_epochs, gp_weight_epochs_TO_max, gp_weight_epochs_AT_max, 
        posterior_mogpreds_entropy_weight_max, posterior_mogpreds_entropy_weight_min, posterior_mogpreds_entropy_weight_taper_epochs,
        classifier_weight, 
        classifier_alpha_max, classifier_alpha_min, classifier_epochs_AT_max, classifier_epochs_TO_max, classifier_rise_first,
        LR_min_classifier, 
        mean_match_static_weight, 
        logvar_match_static_weight, 
        posterior_mogpreds_intersequence_diversity_weight_min, posterior_mogpreds_intersequence_diversity_weight_max, posterior_mogpreds_intersequence_diversity_weight_taper_epochs,
        LR_max_posterior, LR_min_posterior, LR_epochs_stall_posterior, LR_epochs_TO_max_posterior, LR_epochs_AT_max_posterior,  manual_gamma_posterior, manual_step_size_posterior,
        LR_max_prior, LR_min_prior, LR_epochs_stall_prior, LR_epochs_TO_max_prior, LR_epochs_AT_max_prior,  manual_gamma_prior, manual_step_size_prior,
        mse_weight_rise_first=True, KL_divergence_rise_first=True, gp_weight_rise_first=True, LR_rise_first=True, **kwargs):

    """
    Computes learning rate (LR) schedules, KL divergence schedule, classifier weights, and Gaussian Process (GP) prior weights at each epoch for training.

    This function dynamically adjusts various learning rates, weights, and schedules over the course of training. It includes:
    - KL divergence weight scheduling with rise, stall, and plateau phases.
    - Gaussian Process (GP) prior weight scheduling with similar dynamic adjustments.
    - Posterior MoG prediction entropy and inter-sequence diversity weighting.
    - Classifier weight and alpha value adjustments over training epochs.
    - Learning rate scheduling for posterior and prior components with manual or automated control.

    The schedules are piecewise with rise and plateau phases, designed to smoothly transition the model through different learning stages.

    Parameters:
    -----------
    epoch : int
        The current epoch number.

    iter_curr : int
        The current iteration within the epoch.

    iters_per_epoch : int
        The number of iterations per epoch.

    KL_divergence_max_weight : float
        Maximum weight for KL divergence.

    KL_divergence_min_weight : float
        Minimum weight for KL divergence.

    KL_divergence_epochs_TO_max : int
        Number of epochs to increase KL divergence to its maximum weight.

    KL_divergence_epochs_AT_max : int
        Number of epochs to hold KL divergence at its maximum weight.

    KL_divergence_stall_epochs : int
        Number of stall epochs before KL divergence scheduling begins.

    gp_weight_max : float
        Maximum weight for the Gaussian Process prior.

    gp_weight_min : float
        Minimum weight for the Gaussian Process prior.

    gp_weight_stall_epochs : int
        Number of stall epochs before GP weight scheduling begins.

    gp_weight_epochs_TO_max : int
        Number of epochs to increase GP weight to its maximum value.

    gp_weight_epochs_AT_max : int
        Number of epochs to hold GP weight at its maximum value.

    posterior_mogpreds_entropy_weight_max : float
        Maximum weight for posterior MoG prediction entropy.

    posterior_mogpreds_entropy_weight_min : float
        Minimum weight for posterior MoG prediction entropy.

    posterior_mogpreds_entropy_weight_taper_epochs : int
        Number of epochs for tapering the entropy weight.

    classifier_weight : float
        Static weight for the classifier.

    classifier_alpha_max : float
        Maximum alpha value for the classifier.

    classifier_alpha_min : float
        Minimum alpha value for the classifier.

    classifier_epochs_AT_max : int
        Number of epochs to maintain classifier alpha at its maximum.

    classifier_epochs_TO_max : int
        Number of epochs to increase classifier alpha to its maximum.

    classifier_rise_first : bool
        Whether to increase classifier alpha first before plateauing.

    LR_min_classifier : float
        Minimum learning rate for the classifier.

    mean_match_static_weight : float
        Static weight for mean matching loss.

    logvar_match_static_weight : float
        Static weight for log variance matching loss.

    posterior_mogpreds_intersequence_diversity_weight_min : float
        Minimum diversity weight for posterior MoG prediction.

    posterior_mogpreds_intersequence_diversity_weight_max : float
        Maximum diversity weight for posterior MoG prediction.

    posterior_mogpreds_intersequence_diversity_weight_taper_epochs : int
        Number of epochs for tapering diversity weight.

    LR_max_posterior : float
        Maximum learning rate for the posterior.

    LR_min_posterior : float
        Minimum learning rate for the posterior.

    LR_epochs_stall_posterior : int
        Number of stall epochs before posterior LR scheduling begins.

    LR_epochs_TO_max_posterior : int
        Number of epochs to increase posterior LR to its maximum.

    LR_epochs_AT_max_posterior : int
        Number of epochs to hold posterior LR at its maximum.

    manual_gamma_posterior : float
        Manual gamma value for the posterior LR schedule.

    manual_step_size_posterior : int
        Manual step size for the posterior LR schedule.

    LR_max_prior : float
        Maximum learning rate for the prior.

    LR_min_prior : float
        Minimum learning rate for the prior.

    LR_epochs_stall_prior : int
        Number of stall epochs before prior LR scheduling begins.

    LR_epochs_TO_max_prior : int
        Number of epochs to increase prior LR to its maximum.

    LR_epochs_AT_max_prior : int
        Number of epochs to hold prior LR at its maximum.

    manual_gamma_prior : float
        Manual gamma value for the prior LR schedule.

    manual_step_size_prior : int
        Manual step size for the prior LR schedule.

    KL_divergence_rise_first : bool, optional (default=True)
        Whether to increase KL divergence weight first before plateauing.

    gp_weight_rise_first : bool, optional (default=True)
        Whether to increase GP weight first before plateauing.

    LR_rise_first : bool, optional (default=True)
        Whether to increase learning rates first before plateauing.

    Returns:
    --------
    mean_match_static_weight : float
        Static weight for mean matching loss.

    logvar_match_static_weight : float
        Static weight for log variance matching loss.

    KL_divergence_val : float
        Computed KL divergence weight for the current epoch.

    gp_weight_val : float
        Computed Gaussian Process prior weight for the current epoch.

    LR_val_posterior : float
        Computed learning rate for the posterior.

    LR_val_prior : float
        Computed learning rate for the prior.

    LR_val_cls : float
        Computed learning rate for the classifier.

    mogpred_entropy_val : float
        Computed entropy weight for posterior MoG predictions.

    mogpred_diversity_val : float
        Computed diversity weight for posterior MoG predictions.

    classifier_weight : float
        Classifier weight for the current epoch.

    classifier_val : float
        Computed classifier alpha value for the current epoch.

    Notes:
    ------
    - The function ensures smooth transitions between different training phases.
    - Gaussian Process prior weight is now dynamically scheduled alongside KL divergence and learning rates.
    - Schedules are cyclical, resetting after each full period unless otherwise specified.
    """

    # MSE
    if epoch < mse_weight_stall_epochs: mse_val = mse_weight_min
    else: # After stall
        mse_weight_epoch_period = mse_weight_epochs_TO_max + mse_weight_epochs_AT_max
        mse_weight_epoch_residual = (epoch - mse_weight_stall_epochs) % mse_weight_epoch_period # Shift for the stall epochs
        mse_weight_range = mse_weight_max - mse_weight_min
        if mse_weight_rise_first: # START with rise
            if mse_weight_epoch_residual < mse_weight_epochs_TO_max:
                mse_weight_state_length = mse_weight_epochs_TO_max 
                mse_weight_floor = mse_weight_min + mse_weight_range * (mse_weight_epoch_residual/mse_weight_state_length)
                mse_weight_ceil = mse_weight_floor + mse_weight_range * (1) /mse_weight_state_length
                mse_val = mse_weight_floor + (iter_curr/iters_per_epoch) * (mse_weight_ceil - mse_weight_floor)
            else: mse_val = mse_weight_max
        else: raise Exception("ERROR: not coded up")

    # Posterior MoG Prediction Entropy TAPER
    if epoch >= posterior_mogpreds_entropy_weight_taper_epochs:
        mogpred_entropy_val = posterior_mogpreds_entropy_weight_min
    else:
        posterior_mogpreds_entropy_range = posterior_mogpreds_entropy_weight_max - posterior_mogpreds_entropy_weight_min
        total_posterior_mogpreds_entropy_iters = posterior_mogpreds_entropy_weight_taper_epochs * iters_per_epoch
        iter_mogpred_curr = epoch * iters_per_epoch + iter_curr
        mogpred_entropy_val = posterior_mogpreds_entropy_weight_max - posterior_mogpreds_entropy_range * (iter_mogpred_curr / total_posterior_mogpreds_entropy_iters) 

    # Posterior MoG Prediction Batchwise-Diversity TAPER  posterior_mogpreds_intersequence_diversity_weight_min, posterior_mogpreds_intersequence_diversity_weight_max, posterior_mogpreds_intersequence_diversity_weight_taper_epochs
    if epoch >= posterior_mogpreds_intersequence_diversity_weight_taper_epochs:
        mogpred_diversity_val = posterior_mogpreds_intersequence_diversity_weight_min
    else:
        posterior_mogpreds_diversity_range = posterior_mogpreds_intersequence_diversity_weight_max - posterior_mogpreds_intersequence_diversity_weight_min
        total_posterior_mogpreds_diversity_iters = posterior_mogpreds_intersequence_diversity_weight_taper_epochs * iters_per_epoch
        iter_mogpred_curr = epoch * iters_per_epoch + iter_curr
        mogpred_diversity_val = posterior_mogpreds_intersequence_diversity_weight_max - posterior_mogpreds_diversity_range * (iter_mogpred_curr / total_posterior_mogpreds_diversity_iters) 

    # *** Classifier Weight ###
    classifier_weight = classifier_weight # Dummy pass

    # *** Classifier LR ###
    LR_val_cls = LR_min_classifier

    # *** Classifier Alpha ###
    classifier_range = classifier_alpha_max - classifier_alpha_min
    classifier_epoch_period = classifier_epochs_TO_max + classifier_epochs_AT_max
    if classifier_rise_first: classifier_epoch_residual = epoch % classifier_epoch_period 
    else: classifier_epoch_residual = (epoch + classifier_epochs_TO_max) % classifier_epoch_period 
    
    # Calculate alpha value for where in period we are
    if classifier_epoch_residual < classifier_epochs_TO_max:
        classifier_state_length = classifier_epochs_TO_max 
        classifier_floor = classifier_alpha_min + classifier_range * (classifier_epoch_residual/classifier_state_length)
        classifier_ceil = classifier_floor + classifier_range * (1) /classifier_state_length
        classifier_val = classifier_floor + (iter_curr/iters_per_epoch) * (classifier_ceil - classifier_floor)
    else: classifier_val = classifier_alpha_max

    # *** KL_divergence SCHEDULE ***
    # If within the stall, send out KL_divergence_min_weight
    if epoch < KL_divergence_stall_epochs: KL_divergence_val = KL_divergence_min_weight
    else: # After stall
        KL_divergence_epoch_period = KL_divergence_epochs_TO_max + KL_divergence_epochs_AT_max
        KL_divergence_epoch_residual = (epoch - KL_divergence_stall_epochs) % KL_divergence_epoch_period # Shift for the stall epochs
        KL_divergence_range = KL_divergence_max_weight - KL_divergence_min_weight
        if KL_divergence_rise_first: # START with rise
            if KL_divergence_epoch_residual < KL_divergence_epochs_TO_max:
                KL_divergence_state_length = KL_divergence_epochs_TO_max 
                KL_divergence_floor = KL_divergence_min_weight + KL_divergence_range * (KL_divergence_epoch_residual/KL_divergence_state_length)
                KL_divergence_ceil = KL_divergence_floor + KL_divergence_range * (1) /KL_divergence_state_length
                KL_divergence_val = KL_divergence_floor + (iter_curr/iters_per_epoch) * (KL_divergence_ceil - KL_divergence_floor)
            else: KL_divergence_val = KL_divergence_max_weight
        else: raise Exception("ERROR: not coded up")

    # *** Guassian Process Prior SCHEDULE ***
    # If within the stall, send out min
    if epoch < gp_weight_stall_epochs: gp_weight_val = gp_weight_min
    else: # After stall
        gp_weight_epoch_period = gp_weight_epochs_TO_max + gp_weight_epochs_AT_max
        gp_weight_epoch_residual = (epoch - gp_weight_stall_epochs) % gp_weight_epoch_period # Shift for the stall epochs
        gp_weight_range = gp_weight_max - gp_weight_min
        if gp_weight_rise_first: # START with rise
            if gp_weight_epoch_residual < gp_weight_epochs_TO_max:
                gp_weight_state_length = gp_weight_epochs_TO_max 
                gp_weight_floor = gp_weight_min + gp_weight_range * (gp_weight_epoch_residual/gp_weight_state_length)
                gp_weight_ceil = gp_weight_floor + gp_weight_range * (1) /gp_weight_state_length
                gp_weight_val = gp_weight_floor + (iter_curr/iters_per_epoch) * (gp_weight_ceil - gp_weight_floor)
            else: gp_weight_val = gp_weight_max
        else: raise Exception("ERROR: not coded up")
                
    # *** POSTERIOR LR ***
    LR_val_posterior = LR_subfunction(
        iter_curr=iter_curr,
        LR_min=LR_min_posterior,
        LR_max=LR_max_posterior,
        epoch=epoch, 
        manual_gamma=manual_gamma_posterior, 
        manual_step_size=manual_step_size_posterior, 
        epoch_stall=LR_epochs_stall_posterior,
        LR_epochs_TO_max=LR_epochs_TO_max_posterior,  
        LR_epochs_AT_max=LR_epochs_AT_max_posterior, 
        iters_per_epoch=iters_per_epoch,
        LR_rise_first=LR_rise_first)

    # *** PRIOR LR ***
    LR_val_prior = LR_subfunction(
        iter_curr=iter_curr,
        LR_min=LR_min_prior,
        LR_max=LR_max_prior,
        epoch=epoch, 
        manual_gamma=manual_gamma_prior, 
        manual_step_size=manual_step_size_prior, 
        epoch_stall=LR_epochs_stall_prior,
        LR_epochs_TO_max=LR_epochs_TO_max_prior,  
        LR_epochs_AT_max=LR_epochs_AT_max_prior, 
        iters_per_epoch=iters_per_epoch,
        LR_rise_first=LR_rise_first)

    return  mean_match_static_weight, logvar_match_static_weight, mse_val, KL_divergence_val, gp_weight_val, LR_val_posterior, LR_val_prior, LR_val_cls, mogpred_entropy_val, mogpred_diversity_val, classifier_weight, classifier_val


# DECODER HASHING

def hash_to_vector(input_string, num_channels, padded_channels, latent_dim, modifier, hash_output_range):
    """
    Generates a vector from a hash of the input string, scaled to an arbitrary range.

    Args:
        input_string (str): The input string to hash.
        num_channels (int): The number of channels (used for generating the ordered vector).
        padded_channels (int): The total number of channels after padding with -1.
        latent_dim (int): The dimensionality of the output vector.
        modifier (str or int): A modifier to vary the output for the same input string.
        hash_output_range (tuple): The desired output range (e.g., (0, 2)).

    Returns:
        torch.Tensor: A vector of size latent_dim, scaled to the specified range.
        list: A shuffled list of numbers from 0 to num_channels-1, padded with -1 up to padded_channels.
    """
    # Incorporate the modifier into the input string to vary the output
    modified_input = f"{input_string}_{modifier}"

    # Generate a SHA-256 hash from the modified input string
    hash_object = hashlib.sha256(modified_input.encode('utf-8'))
    hash_digest = hash_object.digest()  # 32 bytes (256 bits)

    # If latent_dim > 256, repeat the hash digest to ensure we have enough data
    extended_hash = (hash_digest * ((latent_dim // 32) + 1))[:latent_dim]  # Repeat and slice to exactly latent_dim bytes
    
    # Generate a vector of size latent_dim
    hashed_vector = np.zeros(latent_dim)

    # Define the output range
    a, b = hash_output_range

    for i in range(latent_dim):
        # Use the i-th byte from the extended hash digest
        byte_value = extended_hash[i]
        
        # Normalize the byte value to the range [0, 1]
        normalized_value = byte_value / 255.0  # Normalize to [0, 1]
        
        # Scale the normalized value to the desired range [a, b]
        hashed_vector[i] = a + (b - a) * normalized_value

    # Convert hashed_vector to a PyTorch tensor
    hashed_vector_tensor = torch.tensor(hashed_vector, dtype=torch.float32)

    # Generate a vector of shuffled numbers 0 to num_channels-1
    ordered_vector = list(range(num_channels))
    
    # Set the seed for deterministic shuffling based on the hash of the modified input string
    random.seed(int.from_bytes(hash_digest[:8], 'big'))  # Use first 8 bytes of hash as the seed
    random.shuffle(ordered_vector)  # Shuffle the list in place

    # Pad the shuffled list with -1 up to padded_channels
    padded_vector = ordered_vector.copy()
    while len(padded_vector) < padded_channels:
        # Insert -1 at a random position, but deterministically based on the hash
        random.seed(int.from_bytes(hash_digest[len(padded_vector) % 8: (len(padded_vector) % 8) + 8], 'big'))
        insert_index = random.randint(0, len(padded_vector))
        padded_vector.insert(insert_index, -1)

    return hashed_vector_tensor, padded_vector
 

# PLOTTING

def plot_prior(prior_means, prior_logvars, prior_weights, savedir, epoch, **kwargs):
    """
    Plot the MoG prior parameters (means, log variances, and weights) in a combined plot.

    Args:
        prior_means (np.ndarray): Means of the MoG components, shape (K, D).
        prior_logvars (np.ndarray): Log variances of the MoG components, shape (K, D).
        prior_weights (np.ndarray): Weights of the MoG components, shape (K,).
        savedir (str): Directory to save the plot.
        epoch (int): Current epoch (for labeling the plot).
        **kwargs: Additional arguments (e.g., titles, colors).
    """
    # Ensure the save directory exists
    os.makedirs(savedir, exist_ok=True)

    # Extract dimensions
    K, D = prior_means.shape  # K = number of components, D = latent dimension

    # Create the combined plot
    plt.figure(figsize=(15, 5))

    # Subplot 1: Means
    plt.subplot(1, 3, 1)
    plt.imshow(prior_means, cmap=cmr.waterlily, aspect='auto', interpolation='none')
    plt.colorbar(label='Mean Value')
    plt.xlabel("Latent Dimension")
    plt.ylabel("Component")
    plt.yticks(range(K), labels=[str(i) for i in range(K)])  # Fix y-axis labels
    plt.title("MoG Component Means")

    # Subplot 2: Log Variances
    plt.subplot(1, 3, 2)
    plt.imshow(prior_logvars, cmap=sns.cubehelix_palette(as_cmap=True, reverse=True), aspect='auto', interpolation='none')
    plt.colorbar(label='Log Variance')
    plt.xlabel("Latent Dimension")
    plt.ylabel("Component")
    plt.yticks(range(K), labels=[str(i) for i in range(K)])  # Fix y-axis labels
    plt.title("MoG Component Log Variances")

    # Subplot 3: Weights
    plt.subplot(1, 3, 3)
    plt.bar(range(K), prior_weights, color='purple', alpha=0.5)
    plt.xlabel("Component")
    plt.ylabel("Weight")
    plt.xticks(range(K), labels=[str(i) for i in range(K)])  # Fix x-axis labels
    plt.title("MoG Component Weights")

    # Add a suptitle for the combined plot
    plt.suptitle(f"MoG Prior Visualization (Epoch {epoch})")
    plt.tight_layout()
    
    if not os.path.exists(savedir): os.makedirs(savedir)
    savename_jpg = f"{savedir}/Prior_epoch{epoch}.jpg"
    pl.savefig(savename_jpg, dpi=200)
    plt.close()

def plot_posterior(gpu_id, prior_means, prior_logvars, prior_weights, encoder_means, encoder_logvars, 
                   encoder_mogpreds, encoder_zmeaned, savedir, epoch, mean_lims, logvar_lims, 
                   num_accumulated_plotting_dims=5, max_components=8, n_bins=200, threshold_modifier=10, **kwargs):
    """
    Plot distributions of encoder statistics across MoG components using histograms.
    Only plots up to max_components MoG components (default: 8).
    
    Args:
        gpu_id: GPU ID (for logging purposes)
        prior_means: MoG prior means, shape (K, D)
        prior_logvars: MoG prior log-variances, shape (K, D)
        prior_weights: MoG prior weights, shape (K,)
        encoder_means: Encoder means, shape (Batch, K, D)
        encoder_logvars: Encoder log-variances, shape (Batch, K, D)
        encoder_mogpreds: Encoder MoG predictions, shape (Batch, K)
        encoder_zmeaned: Aggregated latent representation, shape (Batch, D)
        savedir: Directory to save the plots
        epoch: Current epoch
        mean_lims: Tuple (min, max) for mean value range
        logvar_lims: Tuple (min, max) for logvar value range
        num_accumulated_plotting_dims: Number of dimensions to visualize (default: 5)
        max_components: Maximum number of MoG components to plot (default: 8)
        n_bins: Number of bins for histograms (default: 100)
        threshold: Threshold for filtering encoder_mogpreds (default: 0.01)
        **kwargs: Additional arguments
    """
    Batch, K, D = encoder_means.shape

    threshold = 1.0 / (K * threshold_modifier)
    
    # Limit number of components to plot
    components_to_plot = min(max_components, K)
    prior_means = prior_means[:components_to_plot]
    prior_logvars = prior_logvars[:components_to_plot]
    prior_weights = prior_weights[:components_to_plot]
    encoder_means = encoder_means[:, :components_to_plot]
    encoder_logvars = encoder_logvars[:, :components_to_plot]
    encoder_mogpreds = encoder_mogpreds[:, :components_to_plot]

    # Validate input shapes
    assert prior_means.shape == (components_to_plot, D), f"prior_means must have shape ({components_to_plot}, D), but got {prior_means.shape}"
    assert prior_logvars.shape == (components_to_plot, D), f"prior_logvars must have shape ({components_to_plot}, D), but got {prior_logvars.shape}"
    assert prior_weights.shape == (components_to_plot,), f"prior_weights must have shape ({components_to_plot},), but got {prior_weights.shape}"
    assert encoder_means.shape == (Batch, components_to_plot, D), f"encoder_means must have shape (Batch, {components_to_plot}, D), but got {encoder_means.shape}"
    assert encoder_logvars.shape == (Batch, components_to_plot, D), f"encoder_logvars must have shape (Batch, {components_to_plot}, D), but got {encoder_logvars.shape}"
    assert encoder_mogpreds.shape == (Batch, components_to_plot), f"encoder_mogpreds must have shape (Batch, {components_to_plot}), but got {encoder_mogpreds.shape}"
    assert encoder_zmeaned.shape == (Batch, D), f"encoder_zmeaned must have shape (Batch, D), but got {encoder_zmeaned.shape}"
    assert num_accumulated_plotting_dims <= D, f"num_accumulated_plotting_dims ({num_accumulated_plotting_dims}) cannot exceed latent dimension D ({D})"

    # Flatten the batch dimension
    encoder_means_flat = encoder_means.reshape(-1, components_to_plot, D)
    encoder_logvars_flat = encoder_logvars.reshape(-1, components_to_plot, D)
    encoder_mogpreds_flat = encoder_mogpreds.reshape(-1, components_to_plot)
    encoder_zmeaned_flat = encoder_zmeaned.reshape(-1, D)

    # Create figure with subplots
    fig = plt.figure(figsize=(5 * num_accumulated_plotting_dims, 20))
    component_colors = plt.cm.tab10.colors[:components_to_plot]

    # Plot 1: Distribution of Encoder Means vs MoG Prior Means
    for d in range(num_accumulated_plotting_dims):
        ax = fig.add_subplot(4, num_accumulated_plotting_dims, d + 1)
        
        hist_vals = []
        for k in range(components_to_plot):
            hist, bin_edges = np.histogram(encoder_means_flat[:, k, d], bins=n_bins, range=mean_lims, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            hist_vals.append(hist)
            ax.hist(encoder_means_flat[:, k, d], bins=n_bins, range=mean_lims, density=True, alpha=0.5, 
                   color=component_colors[k], label=f'Posterior Comp {k}')

        max_hist = np.max([np.max(h) for h in hist_vals])
        x_vals = np.linspace(mean_lims[0], mean_lims[1], 1000)
        kde_vals_all = []
        
        for k in range(components_to_plot):
            samples = np.random.normal(
                loc=prior_means[k, d],
                scale=np.sqrt(np.exp(prior_logvars[k, d])),
                size=int(1e3)
            )
            kde = gaussian_kde(samples)
            kde_vals = kde(x_vals) * prior_weights[k]
            kde_vals_all.append(kde_vals)
        
        max_kde = np.max([np.max(kde_vals) for kde_vals in kde_vals_all])

        for k in range(components_to_plot):
            kde_vals = kde_vals_all[k] * (max_hist / max_kde)
            ax.plot(x_vals, kde_vals, linestyle='--', color=component_colors[k], alpha=0.8, 
                   label=f'Prior Comp {k}' if d == 0 else None)
        
        ax.set_title(f'Encoder Means (Dim {d})')
        ax.set_xlabel('Mean')
        ax.set_ylabel('Frequency')
        ax.set_xlim(mean_lims[0], mean_lims[1])
        if d == 0:
            ax.legend()

    # Plot 2: Distribution of Encoder Logvars vs MoG Prior Logvars
    for d in range(num_accumulated_plotting_dims):
        ax = fig.add_subplot(4, num_accumulated_plotting_dims, num_accumulated_plotting_dims + d + 1)
        for k in range(components_to_plot):
            ax.hist(encoder_logvars_flat[:, k, d], bins=n_bins, range=logvar_lims, alpha=0.5, 
                   color=component_colors[k], label=f'Posterior Comp {k}')
        ax.set_xlabel('Logvar')
        ax.set_ylabel('Frequency')
        ax.set_xlim(logvar_lims[0], logvar_lims[1])
        if d == 0:
            ax.legend()

    # Plot 3: Split into two subplots
    ax_left = plt.subplot2grid((4, num_accumulated_plotting_dims), (2, 0), colspan=1)
    below_threshold_percent = np.mean(encoder_mogpreds_flat < threshold, axis=0) * 100
    ax_left.bar(range(components_to_plot), below_threshold_percent, color=component_colors, alpha=0.5)
    ax_left.set_title(f'% of encoder_mogpreds < {threshold}')
    ax_left.set_xlabel('MoG Component')
    ax_left.set_ylabel('Percentage')
    ax_left.set_ylim(0, 100)

    ax_right = plt.subplot2grid((4, num_accumulated_plotting_dims), (2, 1), colspan=num_accumulated_plotting_dims - 1)
    for k in range(components_to_plot):
        filtered_mogpreds = encoder_mogpreds_flat[:, k][encoder_mogpreds_flat[:, k] >= threshold]
        ax_right.hist(filtered_mogpreds, bins=n_bins, alpha=0.5, color=component_colors[k], 
                     label=f'Posterior Comp {k}')
    ax_right.set_title(f'Encoder MoG Predictions (≥ {threshold})')
    ax_right.set_xlabel('Weight')
    ax_right.set_ylabel('Frequency')
    ax_right.legend()

    # Plot 4: Distribution of encoder_zmeaned
    for d in range(num_accumulated_plotting_dims):
        ax = fig.add_subplot(4, num_accumulated_plotting_dims, 3 * num_accumulated_plotting_dims + d + 1)
        ax.hist(encoder_zmeaned_flat[:, d], bins=n_bins, range=mean_lims, alpha=0.5, color='gray', label=f'Dim {d}')
        ax.set_xlabel('Z-Posterior')
        ax.set_ylabel('Frequency')
        ax.set_xlim(mean_lims[0], mean_lims[1])
        ax.legend()

    if gpu_id == 0: time.sleep(0.5)
    os.makedirs(savedir, exist_ok=True)
    savename_jpg = f"{savedir}/posterior_epoch{epoch}_numForwards{Batch}_gpu{gpu_id}_maxcomp{components_to_plot}.jpg"
    plt.savefig(savename_jpg, dpi=200)
    plt.close(fig)

PATIENT_COLOR_MAP = {} # Initialize a global dictionary to store patient ID to color mappings
def print_patposteriorweights_cumulative(mogpreds, patidxs, patname_list, savedir, epoch, iter_curr, **kwargs):
    """
    Plot stacked bar plots for MoG weights, colored by patient proportions.
    Each xtick shows the component ID, and below it, the number of unique
    patients represented in that bar (only counting patients with at least
    one MoG weight > 1/(num_components * 10)). The threshold is displayed
    in the x-axis label. The first legend entry corresponds to the top
    of the bar stack, and so on.

    Args:
        mogpreds (np.ndarray): Softmaxed MoG weights, shape (num_samples, num_components).
        patidxs (np.ndarray): Patient indices (class labels), shape (num_samples,).
        patname_list (list): List of patient names corresponding to patidxs.
        savedir (str): Directory to save the plot.
        epoch (int): Current epoch.
        iter_curr (int): Current iteration.
        **kwargs: Additional arguments (e.g., component names, colors).
    """
    global PATIENT_COLOR_MAP

    # Filter out samples where patidxs == -1
    valid_mask = (patidxs != -1)
    mogpreds = mogpreds[valid_mask]
    patidxs = patidxs[valid_mask]

    # Get unique patient indices and MoG components
    unique_patidxs = np.unique(patidxs)
    num_components = mogpreds.shape[1]
    weight_threshold = 1.0 / (num_components * 10)

    # Accumulate MoG weights for each component, grouped by patient
    component_sums = np.zeros((num_components, len(unique_patidxs)))
    for i, patidx in enumerate(unique_patidxs):
        mask = (patidxs == patidx)
        component_sums[:, i] = np.sum(mogpreds[mask], axis=0)

    # Create a stacked bar plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors for patients
    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(unique_patidxs) > len(colors):
        colors = plt.cm.tab20.colors  # Use a larger colormap if needed

    # Assign colors to patients (reuse existing mappings or create new ones)
    patient_handles = []
    patient_labels = []
    bottom = np.zeros(num_components)
    for i, patidx in enumerate(unique_patidxs):
        if patidx not in PATIENT_COLOR_MAP:
            PATIENT_COLOR_MAP[patidx] = colors[len(PATIENT_COLOR_MAP) % len(colors)]

        # Get the patient name from patname_list using the index
        patname = patname_list[int(patidx)]
        color = PATIENT_COLOR_MAP[patidx]
        bars = ax.bar(
            range(num_components),
            component_sums[:, i],
            bottom=bottom,
            color=color,
            label=patname  # Use patient name instead of numerical ID
        )
        bottom += component_sums[:, i]
        patient_handles.append(bars[0])  # Store the bar object for the legend
        patient_labels.append(patname)

    # Customize the plot
    ax.set_xlabel(f"MoG Component\n[Unique Patients (Weight > {weight_threshold:.3g})] ")
    ax.set_ylabel("Sum of MoG Weights")
    ax.set_title(f"MoG Component Weights by Patient (Epoch {epoch}, Iter {iter_curr}) - {mogpreds.shape[0]} Embeddings")
    ax.set_xticks(range(num_components))

    # Calculate the number of unique patients contributing to each bar (with threshold)
    unique_patients_per_component = []
    for c in range(num_components):
        contributing_patients_count = 0
        for i, patidx in enumerate(unique_patidxs):
            mask = (patidxs == patidx)
            patient_mog_preds = mogpreds[mask]
            # Check if any single MoG weight for this patient exceeds the threshold
            if np.any(patient_mog_preds[:, c] > weight_threshold):
                contributing_patients_count += 1
        unique_patients_per_component.append(contributing_patients_count)

    # Set the x-tick labels with component ID and unique patient count
    xtick_labels = [f"{i}\n[{count}]" for i, count in enumerate(unique_patients_per_component)]
    ax.set_xticklabels(xtick_labels)

    # Add legend (reversed order)
    ax.legend(patient_handles[::-1], patient_labels[::-1], title="Patient Name", bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
    plt.setp(ax.get_legend().get_title(), fontsize='8')

    # Save the plot
    if not os.path.exists(savedir): os.makedirs(savedir)
    savename_jpg = f"{savedir}/patposteriorweights_epoch{epoch}_iter{iter_curr}.jpg"
    plt.savefig(savename_jpg, bbox_inches='tight', dpi=200)
    plt.close(fig)

def print_latent_singlebatch(mean, logvar, mogpreds, prior_means, prior_logvars, prior_weights, savedir, epoch, iter_curr, mean_lims, logvar_lims, max_components = 8, n_bins=35, **kwargs):
    """
    Plot re-sampled weighted means, logvars, and average MoG weights at the token level for the first 5 dimensions.
    Only shows the first 8 MoG components in the weights plot.
    """
    batch_size, T, K, D = mean.shape
    
    # Limit to first 8 components
    K = min(max_components, K)
    mean = mean[:, :, :K, :]
    logvar = logvar[:, :, :K, :]
    mogpreds = mogpreds[:, :, :K]
    prior_means = prior_means[:K, :]
    prior_logvars = prior_logvars[:K, :]
    prior_weights = prior_weights[:K]

    # Re-sample components using numpy.random.multinomial
    weighted_mean = np.zeros((batch_size, T, D))
    weighted_logvar = np.zeros((batch_size, T, D))

    for b in range(batch_size):
        for t in range(T):
            # Normalize probabilities in case we truncated components
            probs = mogpreds[b, t] / np.sum(mogpreds[b, t])
            component_idx = np.random.choice(K, p=probs)
            weighted_mean[b, t] = mean[b, t, component_idx]
            weighted_logvar[b, t] = logvar[b, t, component_idx]

    # Compute the average MoG weights across all tokens for each batch
    avg_mogpreds = np.mean(mogpreds, axis=1)  # Shape: (batch_size, K)
    ci_mogpreds = 1.96 * np.std(mogpreds, axis=1) / np.sqrt(T)  # Shape: (batch_size, K)

    # Plot the first 5 dimensions
    num_dims = min(5, D)
    fig = plt.figure(figsize=(28, 5 * num_dims))

    # Define color palettes
    batch_colors = plt.cm.tab10.colors[:batch_size]
    prior_kde_colors = plt.cm.Set2.colors[:K]

    # Store legend handles and labels
    legend_handles = []
    legend_labels = []

    for d in range(num_dims):
        # Plot re-sampled weighted means with KDEs (first column)
        ax1 = plt.subplot2grid((num_dims, 7), (d, 0), colspan=1)
        
        # Compute KDEs for each MoG prior component (first 8 only)
        x_vals = np.linspace(-5, 5, 1000)
        kde_vals_all = []
        for k in range(K):
            samples = np.random.normal(
                loc=prior_means[k, d],
                scale=np.sqrt(np.exp(prior_logvars[k, d])),
                size=int(1e3))
            kde = gaussian_kde(samples)
            kde_vals = kde(x_vals) * prior_weights[k]
            kde_vals_all.append(kde_vals)
            line, = ax1.plot(x_vals, kde_vals, linestyle='--', color=prior_kde_colors[k], alpha=0.6)
            if d == 0:
                legend_handles.append(line)
                legend_labels.append(f'Prior Comp {k}')
        
        max_kde = np.max(kde_vals_all)

        # Plot histograms of weighted means
        hist_vals_all = []
        for b in range(batch_size):
            hist, bin_edges = np.histogram(weighted_mean[b, :, d], bins=n_bins, range=(-5, 5), density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            hist_vals_all.append(hist)
        
        max_hist = np.max(hist_vals_all)
        for b in range(batch_size):
            hist_normalized = (hist_vals_all[b] / max_hist) * max_kde
            line, = ax1.plot(bin_centers, hist_normalized, alpha=0.4, color=batch_colors[b])
            if d == 0:
                legend_handles.append(line)
                legend_labels.append(f'Batch {b}')
        
        ax1.set_title(f'Weighted Posterior Mean (Dim {d})')
        ax1.set_xlim(mean_lims[0], mean_lims[1])

        # Plot re-sampled weighted logvars (second column)
        ax2 = plt.subplot2grid((num_dims, 7), (d, 1), colspan=1)
        for b in range(batch_size):
            hist, bin_edges = np.histogram(weighted_logvar[b, :, d], bins=n_bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax2.plot(bin_centers, hist, alpha=0.4, color=batch_colors[b])
        
        ax2.set_title(f'Weighted Posterior Logvar (Dim {d})')
        ax2.set_xlim(logvar_lims[0], logvar_lims[1])

        # Plot average MoG weights (only first 8 components)
        if d == 0:
            ax3 = plt.subplot2grid((num_dims, 7), (d, 2), colspan=5, rowspan=num_dims)
            for b in range(batch_size):
                bar = ax3.bar(np.arange(K) + 0.1 * b, avg_mogpreds[b], width=0.1, alpha=0.4, 
                              color=batch_colors[b])
                ax3.errorbar(np.arange(K) + 0.1 * b, avg_mogpreds[b], yerr=ci_mogpreds[b], fmt='none', 
                             ecolor='darkgray', capsize=3, capthick=1, elinewidth=1)
            ax3.set_title(f'Average MoG Weights (First {K} Components) with 95% CI')
            ax3.set_xticks(np.arange(K))
            ax3.legend(legend_handles, legend_labels, loc='upper right')

    plt.tight_layout()
    os.makedirs(savedir, exist_ok=True)
    plt.savefig(f"{savedir}/RealtimeLatents_epoch{epoch}_iter{iter_curr}.jpg", dpi=200)
    plt.close(fig)

def print_recon_singlebatch(x, x_hat, savedir, epoch, iter_curr, file_name, num_singlebatch_channels_recon, num_recon_samples, **kwargs):

    x_hat = x_hat.detach().cpu().numpy()
    x = x.detach().cpu().numpy()

    # Fuse the sequential decodes/predictions together
    x_fused = np.moveaxis(x, 3, 2)
    x_fused = x_fused.reshape(x_fused.shape[0], x_fused.shape[1] * x_fused.shape[2], x_fused.shape[3])
    x_fused = np.moveaxis(x_fused, 1, 2)

    x_hat_fused = np.moveaxis(x_hat, 3, 2)
    x_hat_fused = x_hat_fused.reshape(x_hat_fused.shape[0], x_hat_fused.shape[1] * x_hat_fused.shape[2], x_hat_fused.shape[3])
    x_hat_fused = np.moveaxis(x_hat_fused, 1, 2)

    batchsize = x_hat.shape[0]

    np.random.seed(seed=None) 
    r = np.arange(0,x_hat_fused.shape[1])
    np.random.shuffle(r)
    random_ch_idxs = r[0:num_singlebatch_channels_recon]

    # Make new grid/fig
    if x_fused.shape[2] > num_recon_samples:
        gs = gridspec.GridSpec(batchsize, num_singlebatch_channels_recon * 2) # *2 because beginning and end of transformer sequence
    else:
        sqrt_num = int(np.ceil(np.sqrt(batchsize * num_singlebatch_channels_recon)))
        gs = gridspec.GridSpec(sqrt_num, sqrt_num) 
        subplot_iter = 0

    fig = pl.figure(figsize=(24, 24))
    palette = sns.cubehelix_palette(n_colors=2, start=3, rot=1) 
    for b in range(0, batchsize):
        for c in range(0,len(random_ch_idxs)):
            if x_fused.shape[2] > num_recon_samples: # If length of recon is bigger than desire visualized length, then plot only start and end of transformer tokens (may be overlap)
                for seq in range(0,2):
                    if seq == 0:
                        x_decode_plot = x_fused[b, random_ch_idxs[c], :num_recon_samples]
                        x_hat_plot = x_hat_fused[b, random_ch_idxs[c], :num_recon_samples]
                        title_str = 'StartOfTransSeq'
                    else:
                        x_decode_plot = x_fused[b, random_ch_idxs[c], -num_recon_samples:]
                        x_hat_plot = x_hat_fused[b, random_ch_idxs[c], -num_recon_samples:]   
                        title_str = 'EndOfTransSeq'             

                    df = pd.DataFrame({
                        "Target": x_decode_plot,
                        "Prediction": x_hat_plot
                    })

                    ax = fig.add_subplot(gs[b, c*2 + seq]) 
                    sns.lineplot(data=df, palette=palette, linewidth=1.5, dashes=False, ax=ax)
                    ax.set_title(f"Ch:{random_ch_idxs[c]}\n{file_name[b]}, {title_str}", fontdict={'fontsize': 12, 'fontweight': 'medium'})

                    pl.ylim(-1, 1) # Set y-axis limit -1 to 1

            else: # Can fit entire seuqence into desired raw signal visualization length
                x_decode_plot = x_fused[b, random_ch_idxs[c], :]
                x_hat_plot = x_hat_fused[b, random_ch_idxs[c], :]

                df = pd.DataFrame({
                    "Target": x_decode_plot,
                    "Prediction": x_hat_plot
                })

                row = int(subplot_iter/sqrt_num)
                col = subplot_iter - (row * sqrt_num)
                ax = fig.add_subplot(gs[row, col]) 
                sns.lineplot(data=df, palette=palette, linewidth=1.5, dashes=False, ax=ax)
                ax.set_title(f"{file_name[b]}", fontdict={'fontsize': 8, 'fontweight': 'medium'})

                pl.ylim(-1, 1) # Set y-axis limit -1 to 1
                subplot_iter = subplot_iter + 1
            
    fig.suptitle(f"Batches 0:{batchsize-1}, Ch:{random_ch_idxs}")
    if not os.path.exists(savedir): os.makedirs(savedir)
    savename_jpg = f"{savedir}/RealtimeRecon_epoch{epoch}_iter{iter_curr}_allbatch.jpg"
    pl.savefig(savename_jpg, dpi=200)
    pl.close(fig)   

    pl.close('all') 

def print_classprobs_singlebatch(class_probs, class_labels, savedir, epoch, iter_curr, file_name, classifier_num_pats, **kwargs):
    batchsize = class_probs.shape[0]

    class_probs_cpu = class_probs.detach().cpu().numpy()
    class_labels_cpu = class_labels.detach().cpu().numpy()

    for b in range(0, batchsize):
        
        # Only print for one batch index at a time
        # class_probs_plot = class_probs_cpu[b, :, :]
        class_probs_plot = class_probs_cpu[b, :]
        class_labels_plot = class_labels_cpu[b]

        # Compute mean and 95% confidence intervals for each class
        # mean_probs = np.mean(class_probs_plot, axis=0)  # Mean probability for each class
        # std_probs = np.std(class_probs_plot, axis=0)  # Standard deviation for each class
        # n = class_probs_plot.shape[0]  # Number of samples
        # confidence_intervals = 1.96 * (std_probs / np.sqrt(n))  # 95% CI

        # Create a DataFrame for Seaborn
        data = pd.DataFrame({
            'Class': np.arange(classifier_num_pats),  # Class indices
            'Mean Probability': class_probs_plot,  # Mean probabilities
            # 'CI': confidence_intervals
            })  # Confidence intervals

        # Plot using Seaborn
        fig = pl.figure(figsize=(12, 6))
        # sns.barplot(x='Class', y='Mean Probability', data=data, yerr=data['CI'], capsize=0.1, color='skyblue')
        bar_plot = sns.barplot(x='Class', y='Mean Probability', data=data, capsize=0.1, color='skyblue')

        # Change the color of the nth bar to orange
        for i, bar in enumerate(bar_plot.patches):
            if i == class_labels_plot:
                bar.set_facecolor('orange')

        # Add labels and title
        pl.xlabel('Class')
        pl.ylabel('Mean Probability')
        pl.title(f'Mean Class Probabilities\nTrue Class: {class_labels_plot}')
        pl.xticks(rotation=90)  # Rotate x-axis labels for better readability
        pl.tight_layout()
        pl.ylim(0, 1) # Set y-axis limit to 1

        if not os.path.exists(savedir): os.makedirs(savedir)
        savename_jpg = f"{savedir}/RealtimeClassProb_epoch{epoch}_iter{iter_curr}_{file_name[b]}_batch{b}.jpg"
        pl.savefig(savename_jpg, dpi=200)
        pl.close(fig)    

        pl.close('all') 

def print_confusion_singlebatch(class_probs, class_labels, savedir, epoch, iter_curr, classifier_num_pats, **kwargs):
    batchsize = class_probs.shape[0]

    class_probs_cpu = class_probs.detach().cpu().numpy()
    true_labels = class_labels.detach().cpu().numpy()
    predicted_labels = np.argmax(class_probs_cpu, axis=1)
    
    cm = confusion_matrix(true_labels, predicted_labels)

    # Create a mask for the diagonal
    mask = np.eye(cm.shape[0], dtype=bool)
    
    fig = pl.figure(figsize=(12, 12))  # Square figure
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", mask=mask, cbar=False)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", mask=~mask, cbar=False)
    pl.xlabel("Predicted Labels")
    pl.ylabel("True Labels")
    pl.title("Confusion Matrix")
    
    if not os.path.exists(savedir): os.makedirs(savedir)
    savename_jpg = f"{savedir}/RealtimeConfusion_epoch{epoch}_iter{iter_curr}.jpg"
    pl.savefig(savename_jpg, dpi=200)
    pl.close(fig)    

    pl.close('all') 

def print_attention_singlebatch(epoch, iter_curr, pat_idxs, scores_byLayer_meanHeads, savedir, diag_mask_buffer_tokens, **kwargs):
    """
    Plot attention weights with diagonal buffer masking.
    
    Args:
        epoch: Current epoch
        iter_curr: Current iteration
        pat_idxs: Patient indices
        scores_byLayer_meanHeads: Attention scores tensor (batch, layers, rows, cols)
        savedir: Directory to save plots
        diag_mask_buffer_tokens: Number of tokens around diagonal that should be masked
        **kwargs: Additional arguments
    """
    scores_byLayer_meanHeads = scores_byLayer_meanHeads.detach().cpu().numpy()
    batchsize, n_layers, rows, cols = scores_byLayer_meanHeads.shape

    # Make new grid/fig for every batch
    for b in range(batchsize):
        gs = gridspec.GridSpec(1, 2) 
        fig = pl.figure(figsize=(20, 14))

        # Only plotting First and Last layer
        for l in range(n_layers):
            ax_curr = fig.add_subplot(gs[0, l]) 
            plot_data = scores_byLayer_meanHeads[b, l, :, :]
            
            # Create mask for diagonal buffer zone
            mask = np.zeros_like(plot_data, dtype=bool)
            for i in range(rows):
                start = max(0, i - diag_mask_buffer_tokens)
                end = min(cols, i + diag_mask_buffer_tokens + 1)
                mask[i, start:end] = True
                
            # Verify masked area sums to 0 (if not, raise Exception)
            masked_sum = np.where(mask, plot_data, 0).sum()
            if not np.isclose(masked_sum, 0, atol=1e-6):
                raise Exception(f"Error: Masked attention weights sum to {masked_sum:.6f} (expected 0) "
                      f"for batch {b}, layer {l}")
            
            # Apply mask (replace with NaN for plotting)
            plot_data = np.where(mask, np.nan, plot_data)

            # Plot the heatmap
            sns.heatmap(
                plot_data, 
                cmap=sns.cubehelix_palette(as_cmap=True), 
                ax=ax_curr, 
                cbar_kws={
                    'label': 'Attention Weights', 
                    'orientation': 'horizontal'
                }
            )
            
            title = "First Layer - Mean of Heads" if l == 0 else "Last Layer - Mean of Heads"
            ax_curr.set_title(title)
            ax_curr.set_aspect('equal', adjustable='box')

        fig.suptitle(f"Attention Weights - Batch:{b}")
        os.makedirs(savedir, exist_ok=True)
        savename_jpg = f"{savedir}/ByLayer_MeanHead_Attention_epoch{epoch}_iter{iter_curr}_batch{b}_patidx{pat_idxs[b].cpu().numpy()}_buffer{diag_mask_buffer_tokens}.jpg"
        pl.savefig(savename_jpg, dpi=200)
        pl.close(fig)   

    pl.close('all')

def print_dataset_bargraphs(pat_id, curr_file_list, curr_fpaths, dataset_pic_dir, atd_file, pre_ictal_taper_sec=120, post_ictal_taper_sec=120, **kwargs):

    # Get the end of the path (i.e. filename)
    potential_fnames = [x.split("/")[-1] for x in curr_fpaths]
    train_fnames = [x.split("/")[-1] for x in curr_file_list]
    val_fnames = []
    test_fnames = []

    # Convert the filenames to datetime objects
    all_datetimes = filename_to_datetimes(potential_fnames)
    first_starttime = all_datetimes[0][0]

    train_datetimes = filename_to_datetimes(train_fnames)
    val_datetimes = filename_to_datetimes(val_fnames)
    test_datetimes = filename_to_datetimes(test_fnames)

    # Convert the datetimes to seconds from first file (files SHOULD be in order, but check along the way)
    all_start_seconds = [int((x - first_starttime).total_seconds()) for x in all_datetimes[0]]
    all_end_seconds = [int((x - first_starttime).total_seconds()) for x in all_datetimes[1]]
    file_seconds = all_end_seconds[0] - all_start_seconds[0]
    file_seconds_nonoverlap = all_start_seconds[1] - all_start_seconds[0]
    # file_seconds_check = [all_start_seconds[i-1] - all_start_seconds[i] for i in range(1, len(all_start_seconds))]
    # if file_seconds_nonoverlap != file_seconds_check: raise Exception("Error: nonoverlap seconds miscalculated, probably due to skipped time periods when making data epochs")
    all_middle_seconds = [int((all_start_seconds[i] + file_seconds_nonoverlap/2)) for i in range(len(all_start_seconds))]
    all_delta_seconds = [all_middle_seconds[i+1]- all_middle_seconds[i] for i in range(0, len(all_middle_seconds)-1)]
    most_common_delta = np.argmax(np.bincount(all_delta_seconds))
    break_idxs = np.append(np.append(0, np.where(all_delta_seconds[:]%most_common_delta != 0)[0] + 1), len(all_middle_seconds)-1)

    # Convert the dataset to seconds
    train_middle_seconds = [int((x - first_starttime).total_seconds() + file_seconds_nonoverlap/2) for x in train_datetimes[0]]
    val_middle_seconds = [int((x - first_starttime).total_seconds() + file_seconds_nonoverlap/2) for x in val_datetimes[0]]
    test_middle_seconds = [int((x - first_starttime).total_seconds() + file_seconds_nonoverlap/2) for x in test_datetimes[0]]

    # Get the seizure seconds from file start
    seiz_start_dt, seiz_stop_dt, seiz_types = get_pat_seiz_datetimes(pat_id, atd_file=atd_file, **kwargs)
    seiz_start_seconds = [(x - first_starttime).total_seconds() for x in seiz_start_dt]
    seiz_stop_seconds = [(x - first_starttime).total_seconds() for x in seiz_stop_dt]

    # Calculate the file time and get all the possible time indexes for graph
    # Construct the x_vals carefully to accomodate shifts in timing due to RAW EDF file splits
    x_vals = np.zeros(0, dtype=int)
    for i in range(len(break_idxs)-1):
        x_vals = np.append(x_vals, np.arange(all_middle_seconds[break_idxs[i]], all_middle_seconds[break_idxs[i+1]], int(file_seconds_nonoverlap), dtype=int))
    # Add the last value in manually
    x_vals = np.append(x_vals, all_middle_seconds[-1])

    # Iterate through all of the possible x_vals and fill with periictal info
    interictal_perc = np.zeros(len(x_vals))
    preictal_perc = np.zeros(len(x_vals))
    ictal_perc = np.zeros(len(x_vals))
    postictal_perc = np.zeros(len(x_vals))
    trainset = np.zeros(len(x_vals))
    valset = np.zeros(len(x_vals))
    testset = np.zeros(len(x_vals))
    for i in range(0, len(x_vals)):
        x_val_curr = x_vals[i]
        if x_val_curr in all_middle_seconds:
            if x_val_curr in train_middle_seconds:
                if trainset[i] != 0: print("WARNING: file in more than one dataset")
                trainset[i] = [v == x_val_curr for v in train_middle_seconds].count(True) * 10
            
            if x_val_curr in val_middle_seconds:
                if trainset[i] != 0: print("WARNING: file in more than one dataset")
                valset[i] = [v == x_val_curr for v in val_middle_seconds].count(True) * 10

            if x_val_curr in test_middle_seconds:
                if trainset[i] != 0: print("WARNING: file in more than one dataset")
                testset[i] =  [v == x_val_curr for v in test_middle_seconds].count(True) * 10

            # Iterate through seizures and determine periictal percentages that this file spans
            for j in range(0, len(seiz_start_seconds)):
                seiz_start = seiz_start_seconds[j]
                seiz_stop = seiz_stop_seconds[j]
                buffered_seiz_start = seiz_start - pre_ictal_taper_sec
                buffered_seiz_stop = seiz_stop + post_ictal_taper_sec

                file_nonoverlap_start = x_val_curr - file_seconds_nonoverlap/2
                file_nonoverlap_stop = x_val_curr + file_seconds_nonoverlap/2

                # No part of buffered seizure within file
                if buffered_seiz_start > file_nonoverlap_stop: continue
                elif buffered_seiz_stop < file_nonoverlap_start: continue

                # Check if ictal totally within
                if (seiz_start > file_nonoverlap_start) & (seiz_stop < file_nonoverlap_stop):
                    # Buffered totally within
                    if (buffered_seiz_start > file_nonoverlap_start) & (buffered_seiz_stop < file_nonoverlap_stop):
                        preictal_perc[i] = preictal_perc[i] + ((seiz_start - buffered_seiz_start)/file_seconds_nonoverlap) * 100
                        postictal_perc[i] = postictal_perc[i] + ((buffered_seiz_stop - seiz_stop)/file_seconds_nonoverlap) * 100
                        ictal_perc[i] = ictal_perc[i] + ((seiz_stop - seiz_start)/file_seconds_nonoverlap) * 100
                        continue

                    # Buffered start begins before file AND buffered stop ends after file
                    elif (buffered_seiz_start < file_nonoverlap_start) & (buffered_seiz_stop > file_nonoverlap_stop):
                        preictal_perc[i] = preictal_perc[i] + ((seiz_start - file_nonoverlap_start)/file_seconds_nonoverlap) * 100
                        postictal_perc[i] = postictal_perc[i] + ((file_nonoverlap_stop - seiz_stop)/file_seconds_nonoverlap) * 100
                        ictal_perc[i] = ictal_perc[i] + ((seiz_stop - seiz_start)/file_seconds_nonoverlap) * 100
                        continue

                    # Thus, not all within and not all over
                    # Buffered ends before file end
                    elif buffered_seiz_stop < file_nonoverlap_stop:
                        preictal_perc[i] = preictal_perc[i] + ((seiz_start - file_nonoverlap_start)/file_seconds_nonoverlap) * 100
                        postictal_perc[i] = postictal_perc[i] + ((buffered_seiz_stop - seiz_stop)/file_seconds_nonoverlap) * 100
                        ictal_perc[i] = ictal_perc[i] + ((seiz_stop - seiz_start)/file_seconds_nonoverlap) * 100    
                        continue     

                    # Buffered start begins after file start
                    elif buffered_seiz_start > file_nonoverlap_start:
                        preictal_perc[i] = preictal_perc[i] + ((seiz_start - buffered_seiz_start)/file_seconds_nonoverlap) * 100
                        postictal_perc[i] = postictal_perc[i] + ((file_nonoverlap_stop - seiz_stop)/file_seconds_nonoverlap) * 100
                        ictal_perc[i] = ictal_perc[i] + ((seiz_stop - seiz_start)/file_seconds_nonoverlap) * 100
                        continue

                    else: raise Exception("Error: Should not be able to get here, A")       

                # Only preictal in file
                elif seiz_start > file_nonoverlap_stop:
                    preictal_perc[i] = preictal_perc[i] + ((file_nonoverlap_stop - buffered_seiz_start)/file_seconds_nonoverlap) * 100
                    continue
                # Only postictal in file
                elif seiz_stop < file_nonoverlap_start:
                    postictal_perc[i] = postictal_perc[i] + ((buffered_seiz_stop - file_nonoverlap_start)/file_seconds_nonoverlap) * 100
                    continue

                # Now only the end OR only the start of seiz should be within
                # End only within
                elif seiz_stop < file_nonoverlap_stop:
                        if buffered_seiz_stop > file_nonoverlap_stop:
                            postictal_perc[i] = postictal_perc[i] + ((file_nonoverlap_stop - seiz_stop)/file_seconds_nonoverlap) * 100
                            ictal_perc[i] = ictal_perc[i] + ((seiz_stop - file_nonoverlap_start)/file_seconds_nonoverlap) * 100
                            continue
                        else:
                            postictal_perc[i] = postictal_perc[i] + ((buffered_seiz_stop - seiz_stop)/file_seconds_nonoverlap) * 100
                            ictal_perc[i] = ictal_perc[i] + ((seiz_stop - file_nonoverlap_start)/file_seconds_nonoverlap) * 100

                # Beginning of seizure only within
                elif seiz_start > file_nonoverlap_start:
                        if buffered_seiz_start > file_nonoverlap_start:
                            preictal_perc[i] = preictal_perc[i] + ((seiz_start - buffered_seiz_start)/file_seconds_nonoverlap) * 100
                            ictal_perc[i] = ictal_perc[i] + ((file_nonoverlap_stop - seiz_start)/file_seconds_nonoverlap) * 100
                            continue
                        else:
                            preictal_perc[i] = preictal_perc[i] + ((seiz_start - file_nonoverlap_start)/file_seconds_nonoverlap) * 100
                            ictal_perc[i] = ictal_perc[i] + ((file_nonoverlap_stop - seiz_start)/file_seconds_nonoverlap) * 100
                            continue
                
                # else: raise Exception("Error: Should not be able to get here, B") 
                else:
                    # Entire file must be seizure...
                    print("WARNING: Entire file is labeled as ictal")
                    ictal_perc[i] = 1 * 100
                    
            # Calculate the interictal percentage
            buffered_perc = preictal_perc[i] + ictal_perc[i] + postictal_perc[i]
            # Not very scientific, but shrink the percents if they add up to over 100 due to back tio back seizures overlapping thee buffer periods
            if buffered_perc > 100:
                preictal_perc[i] = (preictal_perc[i] / buffered_perc) * 100
                ictal_perc[i] = (ictal_perc[i] / buffered_perc) * 100
                postictal_perc[i] = (postictal_perc[i] / buffered_perc) * 100
            else:
                interictal_perc[i] = 100 - buffered_perc
            
                
    # Now that the Train/Val/Test and periictal designations are made, it's time to plot
            
    df = pd.DataFrame({
        'inter': interictal_perc,
        'pre': preictal_perc,
        'ictal': ictal_perc,
        'post': postictal_perc,
        'train': trainset,
        'val': valset,
        'infer': testset
        },
        index=x_vals
        )        

    # create stacked bar chart 
    ax = df.plot(kind='bar', stacked=True, width=1, color=['silver', 'indianred', 'midnightblue', 'cornflowerblue', 'mediumturquoise', 'orchid', 'orange'], figsize=(14,3))
    ax.legend(bbox_to_anchor=(1.005, 1), loc='upper left', ncol=1)
    ax.get_xaxis().set_ticks([])
    
    # labels for x & y axis
    pl.xlabel('Files')
    pl.ylabel('Percent of File')
    
    # title of plot
    pl.title(f'Dataset Breakdown')

    if not os.path.exists(dataset_pic_dir): os.makedirs(dataset_pic_dir)
    savename = dataset_pic_dir + f"/{pat_id}_Dataset_Breakdown.jpg"
    pl.savefig(savename, dpi=200)
    pl.close('all')


