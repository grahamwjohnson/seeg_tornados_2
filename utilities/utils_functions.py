# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 22:05:34 2023

@author: grahamwjohnson
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
from models.VAE import print_models_flow

def fill_hist_by_channel(data_in: np.ndarray, histo_bin_edges: np.ndarray, zero_island_delete_idxs: list):

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

def random_animal(rand_name_json, **kwargs):
    # Read in animal names, pull random name
    with open(rand_name_json) as json_file:
        json_data = json.load(json_file)
    
    np.random.seed(seed=None) # should replace with Generator for newer code
    rand_idx = np.random.randint(0, len(json_data))
    rand_name = json_data[rand_idx]

    return f"{rand_name}"

def get_num_channels(pat_id, pat_num_channels_LUT):

    df = pd.read_csv(pat_num_channels_LUT)

    return int(df.loc[df['pat_id'] == pat_id, 'num_channels'].iloc[0])

def epoch_contains_pacmap_hdbscan(model_dir, epoch, return_paths=False):

    needed_files = [
        f'checkpoint_epoch{epoch}_hdbscan.pkl',
        f'checkpoint_epoch{epoch}_PaCMAP.ann',
        f'checkpoint_epoch{epoch}_PaCMAP.pkl',
        f'checkpoint_epoch{epoch}_PaCMAP_MedDim.ann',
        f'checkpoint_epoch{epoch}_PaCMAP_MedDim.pkl',
        f'checkpoint_epoch{epoch}_PCA.pkl',
        f'checkpoint_epoch{epoch}_pre_PaCMAP_stride_sec.pkl',
        f'checkpoint_epoch{epoch}_pre_PaCMAP_window_sec.pkl',
        f'checkpoint_epoch{epoch}_xy_lims.pkl',
        f'checkpoint_epoch{epoch}_xy_lims_RAW_DIMS.pkl',
        f'checkpoint_epoch{epoch}_xy_lims_PCA.pkl',
    ]

    found_file_paths = [''] * len(needed_files)


    check_dir = model_dir + "/checkpoints"

    for i in range(len(needed_files)):
        curr_file = needed_files[i]
        possible_matches = glob.glob(f'{check_dir}/{curr_file}')
        num_matches = len(possible_matches)
        if num_matches == 1:
            # print(f"Found {curr_file}")
            found_file_paths[i] = possible_matches[0]
        elif num_matches > 1: 
            raise Exception(f"Found more than 1 match for {curr_file}, this should never happen")
        else:
            print(f"Did NOT Find {curr_file}, cannot conduct inference")
            if return_paths: return False, found_file_paths
            else: return False
    
    if return_paths: 
        return True, found_file_paths
    else: 
        return True

def print_model_summary(model):
    print("Calculating model summary")
    summary(model, num_classes=1)
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = (mem_params + mem_bufs) / 1e9  # in bytes
    print("Expected GPU memory requirement (parameters + buffers): " + str(mem) +" GB")

def reset_batch_vars(num_channels, latent_dim, decode_samples, len_train_data, manual_batch_size, gpu_id):
    iters_per_backprop = len_train_data * manual_batch_size
    backprop_iter = 0
    backprop_x = torch.zeros(num_channels, decode_samples * iters_per_backprop).to(gpu_id)
    backprop_xhat = torch.zeros(num_channels,  decode_samples * iters_per_backprop).to(gpu_id)
    backprop_mean = torch.zeros(latent_dim, iters_per_backprop).to(gpu_id)
    backprop_logvar = torch.zeros(latent_dim, iters_per_backprop).to(gpu_id)

    return iters_per_backprop, backprop_iter, backprop_x, backprop_xhat, backprop_mean, backprop_logvar

def initalize_val_vars(gpu_id, batch_size, mini_batch_window_size, mini_batch_stride, decode_samples, num_channels, latent_dim, num_forward_iters, num_data_time_elements):
    # val_label = torch.zeros(num_files, num_forward_iters).detach() 
    val_latent = torch.zeros(batch_size, latent_dim, num_forward_iters, dtype=torch.float32).detach()    
    val_mean = torch.zeros(batch_size, latent_dim, num_forward_iters, dtype=torch.float32).detach()
    val_logvar = torch.zeros(batch_size, latent_dim, num_forward_iters, dtype=torch.float32).detach()
    val_x = torch.zeros(batch_size, num_channels, num_data_time_elements).detach()
    val_xhat = torch.zeros(batch_size, num_channels, num_data_time_elements).detach()
    val_start_datetimes = [datetime.datetime.min]*batch_size
    val_stop_datetimes = [datetime.datetime.min]*batch_size

    return val_x, val_xhat, val_latent, val_mean, val_logvar, val_start_datetimes, val_stop_datetimes

def create_metadata_subtitle(plot_dict):

    return ("\nLR: " + str(plot_dict["LR_curr"]) + 
    "\nReg Multiplier: " + str(round(plot_dict["Reg_multiplier"],4)) + 
    "\nPre/Postictal Color: " + str(plot_dict["plot_preictal_color"]) + "/" + str(plot_dict["plot_postictal_color"]) + " sec" + 
    ", File Attention Pre/Postictal: " + str(plot_dict["preictal_classify_sec"]) + "/" + str(plot_dict["postictal_classify_sec"]) + " sec" + 
    "\nInput Samples: " + str(plot_dict["input_samples"]) + 
    ", Latent Dimensions: " + str(plot_dict["total_latent_dims"]) + 
    ", Decode Samples: " + str(plot_dict["decode_samples"]) + 
    ", Compression: " + str(plot_dict["dec_compression_ratio"]) + 
    ", Input Stride: " + str(plot_dict["input_stride"]))

def LR_subfunction(iter_curr, LR_min, LR_max, epoch, manual_gamma, manual_step_size, LR_epochs_TO_max, LR_epochs_AT_max, iters_per_epoch, LR_rise_first=True):

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
        Reg_max, Reg_min, Reg_epochs_TO_max, Reg_epochs_AT_max, Reg_stall_epochs,
        mogpreds_entropy_weight_max, mogpreds_entropy_weight_min, mogpreds_entropy_weight_taper_epochs,
        classifier_weight, 
        classifier_alpha_max, classifier_alpha_min, classifier_epochs_AT_max, classifier_epochs_TO_max, classifier_rise_first,
        LR_min_classifier, 
        Sparse_max, Sparse_min, Sparse_epochs_TO_max, Sparse_epochs_AT_max, 
        LR_max_core, LR_min_core, 
        LR_epochs_TO_max_core, LR_epochs_AT_max_core, 
        manual_gamma_core, manual_step_size_core,
        Reg_rise_first=True, Sparse_rise_first=True, LR_rise_first=True, **kwargs):
            


    # MoG Prediction Entropy TAPER
    if epoch >= mogpreds_entropy_weight_taper_epochs:
        mogpred_entropy_val = mogpreds_entropy_weight_min
    else:
        mogpreds_entropy_range = mogpreds_entropy_weight_max - mogpreds_entropy_weight_min
        total_mogpreds_entropy_iters = mogpreds_entropy_weight_taper_epochs * iters_per_epoch
        iter_mogpred_curr = epoch * iters_per_epoch + iter_curr
        mogpred_entropy_val = mogpreds_entropy_weight_max - mogpreds_entropy_range * (iter_mogpred_curr / total_mogpreds_entropy_iters) 

    # *** Classifier Weight ###
    classifier_weight = classifier_weight # Dummy pass

    # *** Classifier LR ###
    LR_val_cls = LR_min_classifier

    # *** Classifier Alpha ###
    classifier_range = classifier_alpha_max - classifier_alpha_min
    classifier_epoch_period = classifier_epochs_TO_max + classifier_epochs_AT_max

    if classifier_rise_first: # START with rise
        classifier_epoch_residual = epoch % classifier_epoch_period 
    else:
        classifier_epoch_residual = (epoch + classifier_epochs_TO_max) % classifier_epoch_period 
    
    # Calculate alpha value for where in period we are
    if classifier_epoch_residual < classifier_epochs_TO_max:
        classifier_state_length = classifier_epochs_TO_max 
        classifier_floor = classifier_alpha_min + classifier_range * (classifier_epoch_residual/classifier_state_length)
        classifier_ceil = classifier_floor + classifier_range * (1) /classifier_state_length
        classifier_val = classifier_floor + (iter_curr/iters_per_epoch) * (classifier_ceil - classifier_floor)
    else:
        classifier_val = classifier_alpha_max


    # *** Reg SCHEDULE ***

    # If within the stall, send out Reg_min
    if epoch < Reg_stall_epochs:
        Reg_val = Reg_min

    # After stall
    else:
        Reg_epoch_period = Reg_epochs_TO_max + Reg_epochs_AT_max
        Reg_epoch_residual = (epoch - Reg_stall_epochs) % Reg_epoch_period # Shift for the stall epochs

        # Reg_range = 10**Reg_max - 10**Reg_min
        Reg_range = Reg_max - Reg_min

        # START with rise
        # Logarithmic rise
        if Reg_rise_first: 
            if Reg_epoch_residual < Reg_epochs_TO_max:

                Reg_state_length = Reg_epochs_TO_max 
                Reg_floor = Reg_min + Reg_range * (Reg_epoch_residual/Reg_state_length)
                Reg_ceil = Reg_floor + Reg_range * (1) /Reg_state_length
                Reg_val = Reg_floor + (iter_curr/iters_per_epoch) * (Reg_ceil - Reg_floor)
            else:
                Reg_val = Reg_max

        else:
            raise Exception("ERROR: not coded up")


   # *** Sparse Weight ***

    Sparse_epoch_period = Sparse_epochs_TO_max + Sparse_epochs_AT_max
    Sparse_epoch_residual = epoch % Sparse_epoch_period

    Sparse_range = 10**Sparse_max - 10**Sparse_min
    # Sparse_range = Sparse_max - Sparse_min

    # START with rise
    # Logarithmic rise
    if Sparse_rise_first: 
        if Sparse_epoch_residual < Sparse_epochs_TO_max:
            # Sparse_state_length = Sparse_epochs_AT_max
            # Sparse_ceil = Sparse_max - ( Sparse_range * (Sparse_epoch_residual/Sparse_state_length) )
            # Sparse_floor = Sparse_ceil - ( Sparse_range * (Sparse_epoch_residual + 1) /Sparse_state_length)
            # Sparse_val = Sparse_ceil - iter_curr/iters_per_epoch * (Sparse_ceil - Sparse_floor) 

            Sparse_state_length = Sparse_epochs_TO_max 
            Sparse_floor = 10 ** Sparse_min + Sparse_range * (Sparse_epoch_residual/Sparse_state_length)
            Sparse_ceil = Sparse_floor + Sparse_range * (1) /Sparse_state_length
            Sparse_val = math.log10(Sparse_floor + iter_curr/iters_per_epoch * (Sparse_ceil - Sparse_floor))
        else:
            Sparse_val = Sparse_max

    else:
        raise Exception("ERROR: not coded up")


    # *** LR SCHEDULES ***

    # CORE
    LR_val_core = LR_subfunction(
        iter_curr=iter_curr,
        LR_min=LR_min_core,
        LR_max=LR_max_core,
        epoch=epoch, 
        manual_gamma=manual_gamma_core, 
        manual_step_size=manual_step_size_core, 
        LR_epochs_TO_max=LR_epochs_TO_max_core,  
        LR_epochs_AT_max=LR_epochs_AT_max_core, 
        iters_per_epoch=iters_per_epoch,
        LR_rise_first=LR_rise_first 
    )
            
    return Reg_val, LR_val_core, LR_val_cls, mogpred_entropy_val, Sparse_val, classifier_weight, classifier_val

def get_random_batch_idxs(num_backprops, num_files, num_samples_in_file, past_seq_length, manual_batch_size, stride, decode_samples):
    # Build the output shape: the idea is that you pull out a backprop iter, then you have sequential idxs the size of manual_batch_size for every file within that backprop
    out = np.zeros([num_backprops, num_files, manual_batch_size])

    for i in range(0, num_files):
        rand_backprop_idxs = list(range(0,num_backprops))
        np.random.shuffle(rand_backprop_idxs)
        for j in range(0, num_backprops):
            for k in range(0, manual_batch_size):
                random_frame_shift = int(random.uniform(0, stride-1)) # Pull a new random shift every time so that repeated augment files have differenr frame shifts
                tmp = random_frame_shift + (stride * manual_batch_size) * rand_backprop_idxs[j] + stride * k
                if (tmp + past_seq_length + decode_samples) > num_samples_in_file: raise Exception("Error: [index + past_seq_length + decode_samples] will exceed file length")
                out[j, i, k] = tmp

    return out.astype('int')
                
def in_seizure(file_name, start_idx, end_idx, samp_freq, atd_file):

    pat_id = file_name.split("_")[0]
    start_datetimes, stop_datetimes = filename_to_datetimes([file_name])
    start_datetime, stop_datetime = start_datetimes[0], stop_datetimes[0]
    seiz_start_dt, seiz_stop_dt, seiz_types = get_pat_seiz_datetimes(pat_id, atd_file=atd_file)
    sample_microsec = (1/samp_freq) * 1e6

    curr_datetime_start = start_datetime + start_idx * datetime.timedelta(microseconds=sample_microsec)
    curr_datetime_stop = start_datetime + end_idx * datetime.timedelta(microseconds=sample_microsec)

    for j in range(0, len(seiz_start_dt)):
        seiz_start_dt_curr = seiz_start_dt[j]
        seiz_stop_dt_curr = seiz_stop_dt[j]

        # Start of data epoch is within seizure
        if (curr_datetime_start > seiz_start_dt_curr) & (curr_datetime_start < seiz_stop_dt_curr):
            return True
        
        # End of data epoch is within seizure
        if (curr_datetime_stop > seiz_start_dt_curr) & (curr_datetime_stop < seiz_stop_dt_curr):
            return True

        # All of seizure is within data epoch 
        # NOT checking because data being fed in is on order of milliseconds

    # If made it through, then return False
    return False

def is_ictal_subprocess(i, dict_in):

    start_datetime = dict_in["start_datetime"]
    stop_datetime = dict_in["stop_datetime"]

    inferred_sample_microsec = dict_in["inferred_sample_microsec"]
    data_samples = dict_in["data_samples"]
    seiz_start_dt = dict_in["seiz_start_dt"]
    seiz_stop_dt = dict_in["seiz_stop_dt"]
    curr_datetime = start_datetime + i * datetime.timedelta(microseconds=inferred_sample_microsec)

    in_seizure = 0
    
    for j in range(0, len(seiz_start_dt)):
        seiz_start_dt_curr = seiz_start_dt[j]
        seiz_stop_dt_curr = seiz_stop_dt[j]
        
        # Tmiepoint is within a seizure, both class
        if (curr_datetime >= seiz_start_dt_curr) & (curr_datetime <= seiz_stop_dt_curr): 
            in_seizure = 1
            return in_seizure

    return in_seizure

def get_all_is_ictal(file_name, data_samples, atd_file):
    pat_id = file_name.split("_")[0]
    start_datetimes, stop_datetimes = filename_to_datetimes([file_name])
    start_datetime, stop_datetime = start_datetimes[0], stop_datetimes[0]
    seiz_start_dt, seiz_stop_dt, seiz_types = get_pat_seiz_datetimes(pat_id, atd_file=atd_file)

    inferred_FS = data_samples/(stop_datetime - start_datetime).total_seconds()
    inferred_sample_microsec = (1/inferred_FS) * 1e6

    file_contains_ictal = False
    # First check to see if file contains a seizure at all
    for i in range(0, len(seiz_start_dt)):
        if (start_datetime > seiz_start_dt[i]) & (start_datetime < seiz_stop_dt[i]):
            file_contains_ictal = True
            break

        if (stop_datetime > seiz_start_dt[i]) & (stop_datetime < seiz_stop_dt[i]):
            file_contains_ictal = True
            break

        if (seiz_start_dt[i] > start_datetime) & (seiz_stop_dt[i] < stop_datetime):
            file_contains_ictal = True
            break

    if not file_contains_ictal:
        return [0]*data_samples
    
    else:
        # Iterate through data timepoints and get exact data samples that are ictal
        dict_in = {
            "start_datetime": start_datetime,
            "stop_datetime": stop_datetime,
            "seiz_start_dt": seiz_start_dt, 
            "seiz_stop_dt": seiz_stop_dt, 
            "inferred_sample_microsec": inferred_sample_microsec, 
            "data_samples": data_samples}
            
        return [is_ictal_subprocess(i, dict_in) for i in range(0, data_samples)]
        
def get_summary_miniepoch_label(onehot_miniepoch):
    max_onehot = [torch.max(onehot_miniepoch[i, :, :], axis = 1) for i in range(onehot_miniepoch.shape[0])]

    out = torch.zeros(len(max_onehot))
    
    for i in range(len(max_onehot)):
        curr_max = max_onehot[i][0]
        if curr_max[1] == 1:
            out[i] = 2 # ictal
            continue
        elif curr_max[0] == 1:
            out[i] = 1 # preictal
            continue
        elif curr_max[2] == 1:
            out[i] = 3 # postictal
            continue
        else:
            out[i] = 0 # interictal

    return out

def get_average_label(tensor_1d):
    # Ictal > pre > post > inter
    #   2   >  1  >   3  >   0
    
    a = np.array(tensor_1d)
    if 2 in a:        return 2
    if 1 in a:
        return 1
    if 3 in a:
        return 3
    else:
        return 0

def atd_str2datetime(s):
    # Must account for varying completeness of microseconds (e.g. 2017-10-15 06:40:20 vs. 2017-11-11 08:33:51.000 vs. 2021-01-19 07:00:31.027734)
    if '.' not in s:
        s = f"{s}.000000"
    else:
        micro_str = s.split('.')[-1]
        num_digs = len(micro_str)
        if num_digs < 6:
            buff = '0'*(6-num_digs)
            s = f"{s}{buff}"

    return datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S.%f')

def find_specified_number_motif_2D(array, motif, verbose=False):
    """Finds specified number motifs in a 2D array.

    Args:
    array: A 2D NumPy array.
    motif: A list of numbers that represents the motif to be found.

    Returns:
    Count of motifs found
    """
    num_labels = array.shape[0]
    counts_by_label = [count_motif_1D(i, num_labels, array[i, :], motif, verbose) for i in range(num_labels)]
    count = np.sum(counts_by_label)

    return count

def count_motif_1D(i, num_labels, a, motif, verbose):
    if verbose:
        sys.stdout.write(f"\rFinding Exit-Cluster Transition Motifs (i.e. [... 1, 0 ...]) in Master_Onehot: State {i}/{num_labels - 1}              ")
        sys.stdout.flush() 

    count = 0
    for j in range(a.shape[0] - len(motif) + 1):
        if (a[j:j + len(motif)] == motif).all():
            count = count + 1

    return count


# DATA MANIPULATIONS

def flatten(list_of_lists):
  return [item for sublist in list_of_lists for item in sublist]

def pseudobatch_raw_data(x, token_samples):
    '''
    x [batch, channels, datapoints]
    returns: x_batched [batch * token_lengths, channels, datapoints]
    '''
    x_batched = torch.stack(torch.split(x.transpose(1,2), token_samples, dim=1), dim=1).transpose(2,3)
    x_batched = x_batched.reshape(x_batched.shape[0]*x_batched.shape[1], x_batched.shape[2], x_batched.shape[3])
    
    return x_batched

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
    pl.savefig(savename_jpg)
    plt.close()

def plot_observed(gpu_id, prior_means, prior_logvars, prior_weights, encoder_means, encoder_logvars, encoder_mogpreds, encoder_zmeaned, savedir, epoch, num_accumulated_plotting_dims=5, n_bins=200, **kwargs):
    """
    Plot distributions of encoder statistics across MoG components using histograms.
    Compares encoder statistics with MoG prior state and includes encoder_zmeaned visualization.
    Each dimension is plotted in a separate column for clarity.
    
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
        num_accumulated_plotting_dims: Number of dimensions to visualize (default: 5)
        n_bins: Number of bins for histograms (default: 100)
        **kwargs: Additional arguments
    """
    Batch, K, D = encoder_means.shape  # Batch size, number of MoG components, and latent dimension

    # Validate input shapes
    assert prior_means.shape == (K, D), f"prior_means must have shape (K, D), but got {prior_means.shape}"
    assert prior_logvars.shape == (K, D), f"prior_logvars must have shape (K, D), but got {prior_logvars.shape}"
    assert prior_weights.shape == (K,), f"prior_weights must have shape (K,), but got {prior_weights.shape}"
    assert encoder_means.shape == (Batch, K, D), f"encoder_means must have shape (Batch, K, D), but got {encoder_means.shape}"
    assert encoder_logvars.shape == (Batch, K, D), f"encoder_logvars must have shape (Batch, K, D), but got {encoder_logvars.shape}"
    assert encoder_mogpreds.shape == (Batch, K), f"encoder_mogpreds must have shape (Batch, K), but got {encoder_mogpreds.shape}"
    assert encoder_zmeaned.shape == (Batch, D), f"encoder_zmeaned must have shape (Batch, D), but got {encoder_zmeaned.shape}"
    assert num_accumulated_plotting_dims <= D, f"num_accumulated_plotting_dims ({num_accumulated_plotting_dims}) cannot exceed latent dimension D ({D})"

    # Flatten the batch dimension to treat all batches as a single dataset
    encoder_means_flat = encoder_means.reshape(-1, K, D)  # Shape: (Batch, K, D) -> (Batch * 1, K, D)
    encoder_logvars_flat = encoder_logvars.reshape(-1, K, D)  # Shape: (Batch, K, D) -> (Batch * 1, K, D)
    encoder_mogpreds_flat = encoder_mogpreds.reshape(-1, K)  # Shape: (Batch, K) -> (Batch * 1, K)
    encoder_zmeaned_flat = encoder_zmeaned.reshape(-1, D)  # Shape: (Batch, D) -> (Batch * 1, D)

    # Create a figure with subplots
    fig = plt.figure(figsize=(5 * num_accumulated_plotting_dims, 20))

    # Define a consistent color palette for MoG components
    component_colors = plt.cm.tab10.colors[:K]  # Use tab10 colormap for up to 10 components

    # Plot 1: Distribution of Encoder Means vs MoG Prior Means (for each MoG component, first `num_accumulated_plotting_dims` dims)
    for d in range(num_accumulated_plotting_dims):
        ax = fig.add_subplot(4, num_accumulated_plotting_dims, d + 1)
        
        # Plot encoder means as histograms
        hist_vals = []
        for k in range(K):
            hist, bin_edges = np.histogram(encoder_means_flat[:, k, d], bins=n_bins, range=(-5, 5), density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            hist_vals.append(hist)
            ax.hist(encoder_means_flat[:, k, d], bins=n_bins, range=(-5, 5), density=True, alpha=0.5, color=component_colors[k], label=f'Posterior Comp {k}')

        # Find the tallest histogram for scaling
        max_hist = np.max([np.max(h) for h in hist_vals])

        # Compute KDEs for the prior means
        x_vals = np.linspace(-5, 5, 1000)  # Range for KDE plots
        kde_vals_all = []  # Store all KDE values for scaling
        for k in range(K):
            # Compute KDE for the current MoG component
            samples = np.random.normal(
                loc=prior_means[k, d],
                scale=np.sqrt(np.exp(prior_logvars[k, d])),  # Scale by variance
                size=int(1e3)  # Fixed number of samples for KDE
            )
            kde = gaussian_kde(samples)
            kde_vals = kde(x_vals) * prior_weights[k]  # Scale KDE by prior weight
            kde_vals_all.append(kde_vals)
        
        # Find the maximum KDE value across all components
        max_kde = np.max([np.max(kde_vals) for kde_vals in kde_vals_all])

        # Scale all KDEs to match the tallest histogram while preserving relative scaling
        for k in range(K):
            kde_vals = kde_vals_all[k] * (max_hist / max_kde)
            ax.plot(x_vals, kde_vals, linestyle='--', color=component_colors[k], alpha=0.8, label=f'Prior Comp {k}' if d == 0 else None)
        
        ax.set_title(f'Encoder Means (Dim {d}), Color MoGComp')
        ax.set_xlabel('Mean Value')
        ax.set_ylabel('Frequency')
        ax.set_xlim(-5, 5)  # Set x-axis range from -5 to 5
        if d == 0:
            ax.legend()

    # Plot 2: Distribution of Encoder Logvars vs MoG Prior Logvars (for each MoG component, first `num_accumulated_plotting_dims` dims)
    for d in range(num_accumulated_plotting_dims):
        ax = fig.add_subplot(4, num_accumulated_plotting_dims, num_accumulated_plotting_dims + d + 1)
        for k in range(K):
            ax.hist(encoder_logvars_flat[:, k, d], bins=n_bins, range=(-5, 5), alpha=0.5, color=component_colors[k], label=f'Posterior Comp {k}')
        ax.set_title(f'Encoder Logvars (Dim {d}), Color MoGComp')
        ax.set_xlabel('Logvar Value')
        ax.set_ylabel('Frequency')
        ax.set_xlim(-5, 5)  # Set x-axis range from -5 to 5

    # Plot 3: Distribution of Encoder MoG Predictions vs MoG Prior Weights (for each MoG component)
    # Span all columns for the MoG predictions plot
    ax = fig.add_subplot(4, 1, 3)  # Span all columns in row 3
    for k in range(K):
        ax.hist(encoder_mogpreds_flat[:, k], bins=n_bins, alpha=0.5, color=component_colors[k], label=f'Posterior Comp {k}')
    ax.set_title(f'Encoder MoG Predictions - Color is MoG Component')
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Frequency')
    ax.legend()

    # Plot 4: Distribution of encoder_zmeaned (aggregated latent representation, first `num_accumulated_plotting_dims` dims)
    for d in range(num_accumulated_plotting_dims):
        ax = fig.add_subplot(4, num_accumulated_plotting_dims, 3 * num_accumulated_plotting_dims + d + 1)
        ax.hist(encoder_zmeaned_flat[:, d], bins=n_bins, range=(-5, 5), alpha=0.5, color='gray', label=f'Dim {d}')
        ax.set_title(f'encoder_zmeaned (Dim {d})')
        ax.set_xlabel('Latent Value')
        ax.set_ylabel('Frequency')
        ax.set_xlim(-5, 5)  # Set x-axis range from -5 to 5
        ax.legend()

    if gpu_id == 0: time.sleep(0.5) # avoid collisions
    if not os.path.exists(savedir): os.makedirs(savedir)
    savename_jpg = f"{savedir}/ObservedLatents_epoch{epoch}_numForwards{Batch}_gpu{gpu_id}.jpg"
    plt.savefig(savename_jpg)
    plt.close(fig)

def print_patmogweights_realtime(mogpreds, patidxs, savedir, epoch, iter_curr, **kwargs):
    """
    Plot stacked bar plots for MoG weights, colored by patient proportions.

    Args:
        mogpreds (np.ndarray): Softmaxed MoG weights, shape (num_samples, num_components).
        patidxs (np.ndarray): Patient indices (class labels), shape (num_samples,).
        savedir (str): Directory to save the plot.
        epoch (int): Current epoch.
        iter_curr (int): Current iteration.
        file_name (str): Base name for the saved plot file.
        **kwargs: Additional arguments (e.g., component names, colors).
    """

    # Get unique patient indices and MoG components
    unique_patidxs = np.unique(patidxs)
    num_components = mogpreds.shape[1]

    # Accumulate MoG weights for each component, grouped by patient
    component_sums = np.zeros((num_components, len(unique_patidxs)))
    for i, patidx in enumerate(unique_patidxs):
        mask = (patidxs == patidx)
        component_sums[:, i] = np.sum(mogpreds[mask], axis=0)

    # Sort patients by their total contribution (largest at the top)
    sorted_indices = np.argsort(np.sum(component_sums, axis=0))[::-1]  # Reverse the order
    component_sums = component_sums[:, sorted_indices]

    # Create a stacked bar plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors for patients
    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(unique_patidxs) > len(colors):
        colors = plt.cm.tab20.colors  # Use a larger colormap if needed

    # Plot each component
    bottom = np.zeros(num_components)
    for i, patidx in enumerate(unique_patidxs[sorted_indices]):
        ax.bar(
            range(num_components),
            component_sums[:, i],
            bottom=bottom,
            color=colors[i % len(colors)],
            label=f"Patient {patidx}"
        )
        bottom += component_sums[:, i]

    # Customize the plot
    ax.set_xlabel("MoG Component")
    ax.set_ylabel("Sum of MoG Weights")
    ax.set_title(f"MoG Component Weights by Patient (Epoch {epoch}, Iter {iter_curr}) - {mogpreds.shape[0]} Embeddings")
    ax.set_xticks(range(num_components))
    ax.set_xticklabels([f"Comp {i}" for i in range(num_components)])
    # ax.legend(title="Patient ID", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the plot
    if not os.path.exists(savedir): os.makedirs(savedir)
    savename_jpg = f"{savedir}/patmogweights_epoch{epoch}_iter{iter_curr}.jpg"
    pl.savefig(savename_jpg)
    pl.close(fig)

def print_latent_realtime(mean, logvar, mogpreds, prior_means, prior_logvars, prior_weights, savedir, epoch, iter_curr,  mean_lims, logvar_lims, n_bins=35, **kwargs):
    """
    Plot re-sampled weighted means, logvars, and average MoG weights at the token level for the first 5 dimensions.
    Aggregates tokens across all batches, with separate histograms for each batch index.
    Re-samples components using numpy.random.multinomial to properly weight the means and logvars.

    Args:
        mean: Encoder means, shape (batch_size, T, K, D)
        logvar: Encoder log-variances, shape (batch_size, T, K, D)
        mogpreds: MoG component probabilities, shape (batch_size, T, K)
        prior_means: MoG prior means, shape (K, D)
        prior_logvars: MoG prior log-variances, shape (K, D)
        prior_weights: MoG prior weights, shape (K,)
        savedir: Directory to save the plots
        epoch: Current epoch
        iter_curr: Current iteration
        n_bins: Number of bins for histograms (default: 35)
        **kwargs: Additional arguments
    """
    batch_size, T, K, D = mean.shape

    # Re-sample components using numpy.random.multinomial
    weighted_mean = np.zeros((batch_size, T, D))  # Shape: (batch_size, T, D)
    weighted_logvar = np.zeros((batch_size, T, D))  # Shape: (batch_size, T, D)

    assert np.min(weighted_mean) > mean_lims[0], f"got mean of {np.min(weighted_mean)}, which is lower than stated limit of {mean_lims[0]}"
    assert np.max(weighted_mean) < mean_lims[1],  f"got mean of {np.max(weighted_mean)}, which is higher than stated limit of {mean_lims[1]}"
    assert np.min(weighted_logvar) > logvar_lims[0], f"got logvar of {np.min(weighted_logvar)}, which is lower than stated limit of {logvar_lims[0]}"
    assert np.max(weighted_logvar) < logvar_lims[1], f"got logvar of {np.max(weighted_logvar)}, which is higher than stated limit of {logvar_lims[1]}"

    for b in range(batch_size):
        for t in range(T):
            # Sample a component index based on MoG probabilities
            component_idx = np.random.choice(K, p=mogpreds[b, t])
            # Use the sampled component's mean and logvar
            weighted_mean[b, t] = mean[b, t, component_idx]
            weighted_logvar[b, t] = logvar[b, t, component_idx]

    # Compute the average MoG weights across all tokens for each batch
    avg_mogpreds = np.mean(mogpreds, axis=1)  # Shape: (batch_size, K)

    # Compute the 95% confidence intervals for the MoG weights
    ci_mogpreds = 1.96 * np.std(mogpreds, axis=1) / np.sqrt(T)  # Shape: (batch_size, K)

    # Plot the first 5 dimensions
    num_dims = min(5, D)
    fig = plt.figure(figsize=(28, 5 * num_dims))  # Wider figure to accommodate 7 columns

    # Define a consistent color palette for batches
    batch_colors = plt.cm.tab10.colors[:batch_size]  # Use tab10 colormap for up to 10 batches

    # Define a distinct color palette for prior KDEs
    prior_kde_colors = plt.cm.Set2.colors[:K]  # Use Set2 colormap for up to 8 components

    # Store legend handles and labels for the right plot
    legend_handles = []
    legend_labels = []

    for d in range(num_dims):
        # Plot re-sampled weighted means with KDEs for each MoG prior component (first column)
        ax1 = plt.subplot2grid((num_dims, 7), (d, 0), colspan=1)
        
        # Compute KDEs for each MoG prior component (scaled by prior weights)
        x_vals = np.linspace(-5, 5, 1000)  # Range for KDE plots
        kde_vals_all = []  # Store all KDE values for scaling
        for k in range(K):
            # Compute KDE for the current MoG component
            samples = np.random.normal(
                loc=prior_means[k, d],
                scale=np.sqrt(np.exp(prior_logvars[k, d])),  # Scale by variance
                size=int(1e3)  # Fixed number of samples for KDE
            )
            kde = gaussian_kde(samples)
            kde_vals = kde(x_vals) * prior_weights[k]  # Scale KDE by prior weight
            kde_vals_all.append(kde_vals)
            line, = ax1.plot(x_vals, kde_vals, linestyle='--', color=prior_kde_colors[k], alpha=0.6)
            if d == 0:  # Add KDE legend entries only once
                legend_handles.append(line)
                legend_labels.append(f'Prior Comp {k}')
        
        # Compute the maximum KDE value for scaling
        max_kde = np.max(kde_vals_all)

        # Plot re-sampled weighted means as histograms (first column)
        hist_vals_all = []  # Store all histogram values for normalization
        for b in range(batch_size):
            hist, bin_edges = np.histogram(weighted_mean[b, :, d], bins=n_bins, range=(-5, 5), density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            hist_vals_all.append(hist)
        
        # Normalize histograms by their maximum value and scale to match KDEs
        max_hist = np.max(hist_vals_all)
        for b in range(batch_size):
            hist_normalized = (hist_vals_all[b] / max_hist) * max_kde  # Normalize and scale
            line, = ax1.plot(bin_centers, hist_normalized, alpha=0.4, color=batch_colors[b])
            if d == 0:  # Add batch legend entries only once
                legend_handles.append(line)
                legend_labels.append(f'Batch {b}')
        
        ax1.set_title(f'Re-sampled Weighted Mean (Dim {d})')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.set_xlim(-5, 5)  # Set x-axis range from -5 to 5

        # Plot re-sampled weighted logvars as histograms (second column)
        ax2 = plt.subplot2grid((num_dims, 7), (d, 1), colspan=1)
        for b in range(batch_size):
            hist, bin_edges = np.histogram(weighted_logvar[b, :, d], bins=n_bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax2.plot(bin_centers, hist, alpha=0.4, color=batch_colors[b])
        
        ax2.set_title(f'Re-sampled Weighted Logvar (Dim {d})')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Density')
        ax2.set_xlim(-5, 5)

        # Plot average MoG weights with 95% CI (columns 2–6, single subplot spanning 5 columns)
        if d == 0:  # Only create this subplot once
            ax3 = plt.subplot2grid((num_dims, 7), (d, 2), colspan=5, rowspan=num_dims)
            for b in range(batch_size):
                # Plot bars with error bars
                bar = ax3.bar(np.arange(K) + 0.1 * b, avg_mogpreds[b], width=0.1, alpha=0.4, 
                              color=batch_colors[b])
                ax3.errorbar(np.arange(K) + 0.1 * b, avg_mogpreds[b], yerr=ci_mogpreds[b], fmt='none', 
                             ecolor='darkgray', capsize=3, capthick=1, elinewidth=1)
            ax3.set_title(f'Average MoG Weights (Across Tokens) with 95% CI\nToken-Level MoG-Weighted Means & Logvars: Single Forward Passes')
            ax3.set_xlabel('MoG Component')
            ax3.set_ylabel('Average Token Weight')
            ax3.set_xticks(np.arange(K))  # Show all MoG component indices on the x-axis

            # Add the combined legend to the right plot
            ax3.legend(legend_handles, legend_labels, loc='upper right')

    # Adjust layout and save the plot
    plt.tight_layout()
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savename_jpg = f"{savedir}/RealtimeLatents_epoch{epoch}_iter{iter_curr}.jpg"
    plt.savefig(savename_jpg)
    plt.close(fig)

def print_recon_realtime(x, x_hat, savedir, epoch, iter_curr, file_name, num_realtime_channels_recon, num_recon_samples, **kwargs):

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
    random_ch_idxs = r[0:num_realtime_channels_recon]

    # Make new grid/fig
    if x_fused.shape[2] > num_recon_samples:
        gs = gridspec.GridSpec(batchsize, num_realtime_channels_recon * 2) # *2 because beginning and end of transformer sequence
    else:
        sqrt_num = int(np.ceil(np.sqrt(batchsize * num_realtime_channels_recon)))
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
    pl.savefig(savename_jpg)
    pl.close(fig)   

    pl.close('all') 

def print_classprobs_realtime(class_probs, class_labels, savedir, epoch, iter_curr, file_name, classifier_num_pats, **kwargs):
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
        pl.savefig(savename_jpg)
        pl.close(fig)    

        pl.close('all') 

def print_confusion_realtime(class_probs, class_labels, savedir, epoch, iter_curr, classifier_num_pats, **kwargs):
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
    pl.savefig(savename_jpg)
    pl.close(fig)    

    pl.close('all') 

def print_attention_realtime(epoch, iter_curr, pat_idxs, scores_byLayer_meanHeads, savedir, **kwargs):

    scores_byLayer_meanHeads = scores_byLayer_meanHeads.detach().cpu().numpy()

    batchsize, n_layers, rows, cols = scores_byLayer_meanHeads.shape

    # Make new grid/fig for every batch
    for b in range(0, batchsize):
        gs = gridspec.GridSpec(1, 2) 
        fig = pl.figure(figsize=(20, 14))

        # Only plotting First and Last layer
        for l in range(n_layers):

            ax_curr = fig.add_subplot(gs[0, l]) 
            plot_data = scores_byLayer_meanHeads[b, l, :, :] # Add small amount to avoid log error when scaling

            # Replace diagonal with NaN

            mask = np.eye(plot_data.shape[0], dtype=bool)
            assert np.where(~mask, 0, plot_data).sum() == 0  # Make sure diaganal sums to 0, or masking was not done correctly
            plot_data = np.where(mask, np.nan, plot_data)   

            # # Multiply each row of the data by its row index
            # for i in range(plot_data.shape[0]):
            #     plot_data[i, :] *= (i + 1)  # Multiply by row index (1-based)

            # Plot the heatmap
            sns.heatmap(plot_data, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax_curr, cbar_kws={'label': 'Row-Weighted Attention', 'orientation': 'horizontal'}) 
            if l == 0: ax_curr.set_title(f"First Layer - Mean of Heads")
            else: ax_curr.set_title(f"Last Layer - Mean of Heads")
            ax_curr.set_aspect('equal', adjustable='box')

        fig.suptitle(f"Attention Weights - Batch:{b}")
        if not os.path.exists(savedir): os.makedirs(savedir)
        savename_jpg = f"{savedir}/ByLayer_MeanHead_Attention_epoch{epoch}_iter{iter_curr}_batch{b}_patidx{pat_idxs[b].cpu().numpy()}.jpg"
        pl.savefig(savename_jpg)
        pl.close(fig)   

    pl.close('all') 

def plot_recon(x, x_hat, plot_dict, batch_file_names, epoch, savedir, gpu_id, pat_id, iter, FS, num_rand_recon_plots, recon_sec=4, **kwargs):

    num_loops = num_rand_recon_plots

    all_starts = plot_dict['start_dt']
    all_stops = plot_dict['stop_dt']

    if gpu_id == 0: time.sleep(1) # Avoid file collision 

    for i in range(num_loops):

        np.random.seed(seed=None) # should replace with Generator for newer code
        ch_idx = np.random.randint(0, x.shape[1])
        
        # Pick a random starting point within timeseries
        sample_duration = recon_sec * FS
        np.random.seed(seed=None) # should replace with Generator for newer code
        start_idx = np.random.randint(0, x.shape[2] - sample_duration - 1)

        # Pick a random starting point within batch
        np.random.seed(seed=None) # should replace with Generator for newer code
        batch_idx = np.random.randint(0, x.shape[0])

        gs = gridspec.GridSpec(1, 5)
        fig = pl.figure(figsize=(30, 14))
        
        # Plot X
        ax1 = pl.subplot(gs[0, :])
        ax1.plot(x[batch_idx, ch_idx, start_idx:(start_idx + sample_duration)])

        # Plot X_HAT
        ax1.plot(x_hat[batch_idx, ch_idx, start_idx:(start_idx + sample_duration)])

        ax1.legend(['Original', 'Reconstruction'])
        plot_dict['start_dt'] = all_starts[batch_idx]
        plot_dict['stop_dt'] = all_stops[batch_idx]
        fig.suptitle(batch_file_names[batch_idx] + create_metadata_subtitle(plot_dict))

        if not os.path.exists(savedir): os.makedirs(savedir)
        savename_jpg = savedir + f"/Recon_epoch{str(epoch)}_iter_{str(iter)}_{pat_id}_batchIdx{str(batch_idx)}_chIdx{str(ch_idx)}_duration{str(recon_sec)}sec_startIdx{str(start_idx)}_gpu{str(gpu_id)}.jpg"
        pl.savefig(savename_jpg)
        pl.close(fig) 

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
    pl.savefig(savename)
    pl.close('all')

def rgbtoint32(rgb):
    rgb = np.array(rgb*256, dtype=int)
    color = 0
    for c in rgb[::-1]:
        color = (color<<8) + c
        # Do not forget parenthesis.
        # color<< 8 + c is equivalent of color << (8+c)
    return color

def int32torgb(color):
    if color == np.nan: return np.nan
    rgb = []
    for i in range(3):
        rgb.append(color&0xff)
        color = color >> 8
    rgb = np.array(float(rgb)/256)
    return rgb

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap



# FILE I/O

def exec_kwargs(kwargs):
    # exec all kwargs in case there is python code
    for k in kwargs:
            if isinstance(kwargs[k], str):
                if kwargs[k][0:6] == 'kwargs':
                    exec(kwargs[k])

    return kwargs

def load_data_tensor(filename):
    file = open(filename,'rb')
    data = pickle.load(file) 
    file.close()
    # data_channel_subset = data[0:self.num_channels,:]   
    return torch.FloatTensor(data)

def filename_to_dateobj(f: str, start_or_end: int):
    # if start_or_end is '0' the beginning of file timestamp is used, if '1' the end

    splits = f.split("_")

    if start_or_end == 0:
        split_date_idx = 1
        split_time_idx = 2

    elif start_or_end == 1:
        split_date_idx = 4
        split_time_idx = 5

    year = int(splits[split_date_idx][4:8])
    month = int(splits[split_date_idx][0:2])
    day = int(splits[split_date_idx][2:4])
    hour = int(splits[split_time_idx][0:2])
    minute = int(splits[split_time_idx][2:4])
    second = int(splits[split_time_idx][4:6])
    microsecond = int((int(splits[2][6:8])/100)*1e6)
    # datetime(year, month, day, hour, minute, second, microsecond)
    return datetime.datetime(year, month, day, hour, minute, second, microsecond)

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

def delete_old_checkpoints(dir: str, curr_epoch: int, Reg_stall_epochs, Reg_epochs_AT_max, Reg_epochs_TO_max, **kwargs):

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

    # Reg Epoch cycle save - save the last epoch in a Reg annealing cycle
    mod_val = Reg_stall_epochs + Reg_epochs_AT_max + Reg_epochs_TO_max - 1
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

def get_start_stop_from_latent_path(latent_file):
    s = latent_file.split('_')[-5] + latent_file.split('_')[-4]
    abs_start_datetime = datetime.datetime(int(s[4:8]),int(s[0:2]),int(s[2:4]),int(s[8:10]),int(s[10:12]),int(s[12:14]),int(int(s[14:16])*1e4))
    s = latent_file.split('_')[-2] + latent_file.split('_')[-1].replace('.pkl','')
    abs_stop_datetime = datetime.datetime(int(s[4:8]),int(s[0:2]),int(s[2:4]),int(s[8:10]),int(s[10:12]),int(s[12:14]),int(int(s[14:16])*1e4))

    return abs_start_datetime, abs_stop_datetime

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

def datetimes_to_filename(start_dt, stop_dt):
    if len(start_dt) != 1: raise Exception("Expected only one datetime for start and one for stop")
    start_microsec_trunc_str = start_dt[0].strftime('%f')[0:2]
    start_str = f"{start_dt[0].strftime('%m%d%Y_%H%M%S')}{start_microsec_trunc_str}"
    stop_microsec_trunc_str = stop_dt[0].strftime('%f')[0:2]
    stop_str = f"{stop_dt[0].strftime('%m%d%Y_%H%M%S')}{stop_microsec_trunc_str}"

    out = f"{start_str}_to_{stop_str}"
    if out == None: raise Exception("ERROR: out string formatted to none")

    return out

def get_training_dir_name(train_val_pat_perc, **kwargs):
    
    train_str = str(train_val_pat_perc[0]*100)
    val_str = str(train_val_pat_perc[1]*100)
    run_params_dir_name = "dataset_train" + train_str + "_val" + val_str 

    return run_params_dir_name

def get_hours_inferred_str(intrapatient_dataset_style):

    if intrapatient_dataset_style[0] == 0: f = 'seizure_based_inference'
    elif intrapatient_dataset_style[0] == 1: f = 'inference_on_all_epochs_withoutSPES'
    elif intrapatient_dataset_style[0] == 2: f = 'inference_on_all_epochs_withSPES'
    return f

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

def get_emu_timestamps(atd_file, pat_id):
    
    atd_df = pd.read_csv(atd_file, sep='\t')
    file_bool = (atd_df['Pat ID'] == pat_id) & (atd_df['Type'] == "File")
    
    df_subset = atd_df.loc[file_bool, ['onset_datetime', 'offset_datetime', 'FileName']]
    
    emu_file_starts_str = df_subset.loc[:,'onset_datetime'].astype(str).values.tolist() 
    emu_file_stops_str = df_subset.loc[:,'offset_datetime'].astype(str).values.tolist()
    emu_filenames = df_subset.loc[:,'FileName'].astype(str).values.tolist()

    # Convert to datetime
    emu_file_starts_dt = [atd_str2datetime(s) for s in emu_file_starts_str]
    emu_file_stops_dt = [atd_str2datetime(s) for s in emu_file_stops_str]

    return emu_filenames, emu_file_starts_dt, emu_file_stops_dt
    
def digest_timestamps(atd_file, pat_id):

    # Seizures
    seiz_starts_dt, seiz_stops_dt, seiz_types = get_pat_seiz_datetimes(pat_id, 
                           atd_file=atd_file,
                           FBTC_bool=True, 
                           FIAS_bool=True, 
                           FAS_to_FIAS_bool=True,
                           FAS_bool=True, 
                           subclinical_bool=True, 
                           focal_unknown_bool=True,
                           unknown_bool=True, 
                           non_electro_bool=False)

    # File Timestamps
    emu_filenames, emu_file_starts_dt, emu_file_stops_dt = get_emu_timestamps(atd_file=atd_file, pat_id=pat_id)


    return seiz_starts_dt, seiz_stops_dt, seiz_types, emu_filenames, emu_file_starts_dt, emu_file_stops_dt

def get_files_spanning_datetimes(search_dir, start_dt, stop_dt):
    pkl_paths = glob.glob(f"{search_dir}/*.pkl")
    pkl_files = [p.split("/")[-1] for p in pkl_paths]
    pkl_starts_dt, pkl_stops_dt = filename_to_datetimes(pkl_files)

    files_in_range = None

    for i in range(len(pkl_starts_dt)):
        if ((pkl_stops_dt[i] > start_dt) & (pkl_stops_dt[i] < stop_dt)) or ((pkl_starts_dt[i] > start_dt) & (pkl_starts_dt[i] < stop_dt)) or ((pkl_starts_dt[i] < start_dt) & (pkl_stops_dt[i] > stop_dt)):
            if files_in_range == None:
                files_in_range = [pkl_paths[i]]
            else:
                files_in_range.append(pkl_paths[i])

    return files_in_range

def append_timestamp(filename):
    ts = time.asctime(time.localtime(time.time()))
    ts = ts.replace(" ", "_")
    ts = ts.replace(":", "_")
    return filename + ts

def get_sorted_datetimes_from_files(files):

    start_dt, stop_dt = filename_to_datetimes([files[i].split("/")[-1] for i in range(len(files))])
    sort_idxs = [i[0] for i in sorted(enumerate(start_dt), key=lambda x:x[1])]
    files_sorted = [files[sort_idxs[i]] for i in range(len(sort_idxs))]; 
    start_dt_sorted = [start_dt[sort_idxs[i]] for i in range(len(sort_idxs))]; 
    stop_dt_sorted = [stop_dt[sort_idxs[i]] for i in range(len(sort_idxs))]; 

    return files_sorted, start_dt_sorted, stop_dt_sorted


# SIGNAL PROCESSING

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



# INITIALIZATIONS

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

def assemble_model_save_path(base_path: str,
                        bipole_or_monopole: str,
                        channel_scale_style: str,
                        freq_bands_str: str,
                        scale_type: str,
                        scale_epoch_str: str,
                        duration_stride_str: str,
                        # pat_id: str,
                        **kwargs):
    
    # Check that strings fall within acceptable options
    if (bipole_or_monopole != 'Bipole_datasets') & (bipole_or_monopole != 'Monopole_datasets'):
        raise Exception("Assemble path error: 'bipole_or_monopole' must equal: 'Bipole_datasets' or 'Monopole_datasets' ")
    
    if (channel_scale_style != 'Same_Scale_For_All_Channels') & (channel_scale_style != 'By_Channel_Scale'):
        raise Exception("Assemble path error: 'channel_scale_style' must equal: 'Same_Scale_For_All_Channels' or 'By_Channel_Scale' ")
    
    if (scale_type != 'LinearScale') & (scale_type != 'HyperTanScaling') & (scale_type != 'CubeRootScale') & (scale_type != 'HistEqualScale'):
        raise Exception("Assemble path error: 'scale_type' must equal: 'LinearScale', 'HyperTanScaling', 'CubeRootScale' or 'HistEqualScale'")
    
    return base_path + '/' + bipole_or_monopole + '/' + channel_scale_style + '/' + scale_type + '/' + scale_epoch_str + f'/{freq_bands_str}/' + duration_stride_str # + '/' + pat_id

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
        kwargs['vae_state_dict_prev_path'] = check_dir + f'/Epoch_{str(max_epoch)}/core_checkpoints/checkpoint_epoch{str(max_epoch)}_vae.pt'
        kwargs['vae_opt_state_dict_prev_path'] = check_dir + f'/Epoch_{str(max_epoch)}/core_checkpoints/checkpoint_epoch{str(max_epoch)}_vae_opt.pt'

        # Proper names for running latents 
        kwargs['running_mean_path'] = check_dir + f'/Epoch_{str(max_epoch)}/core_checkpoints/checkpoint_epoch{str(max_epoch)}_running_means.pkl'
        kwargs['running_logvar_path'] = check_dir + f'/Epoch_{str(max_epoch)}/core_checkpoints/checkpoint_epoch{str(max_epoch)}_running_logvars.pkl'
        kwargs['running_zmeaned_path'] = check_dir + f'/Epoch_{str(max_epoch)}/core_checkpoints/checkpoint_epoch{str(max_epoch)}_running_zmeaned.pkl'
        kwargs['running_mogpreds_path'] = check_dir + f'/Epoch_{str(max_epoch)}/core_checkpoints/checkpoint_epoch{str(max_epoch)}_running_mogpreds.pkl' 
        kwargs['running_patidxs_path'] = check_dir + f'/Epoch_{str(max_epoch)}/core_checkpoints/checkpoint_epoch{str(max_epoch)}_running_patidxs.pkl' 

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
    kwargs['root_save_dir'] = assemble_model_save_path(**kwargs)
    # kwargs['data_dir'] = kwargs['root_save_dir'] + kwargs['data_dir_subfolder']
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

    return world_size, kwargs