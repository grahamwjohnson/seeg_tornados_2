# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 22:05:34 2023

@author: grahamwjohnson
"""

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
import joblib
import json
import os
import sys
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
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
import hdbscan
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import seaborn as sns
import multiprocessing as mp
import heapq
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import math
import pacmap
from scipy.stats import norm
import matplotlib.colors as colors
import auraloss
from tkinter import filedialog
import re
from functools import partial

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
        f'checkpoint_epoch{epoch}_cluster_reorder_indexes.pkl', # for cluster timeline
        f'checkpoint_epoch{epoch}_hdbscan.pkl',
        f'checkpoint_epoch{epoch}_cluster_reorder_indexes.pkl',
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
        f'checkpoint_epoch{epoch}_info_score_idxs.pkl'
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
    "\nKLD Multiplier: " + str(round(plot_dict["KL_multiplier"],4)) + 
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
        KL_max, KL_min, KL_epochs_TO_max, KL_epochs_AT_max, 
        LR_max_core, LR_min_core, 
        LR_max_transformer, LR_min_transformer, 
        LR_epochs_TO_max_core, LR_epochs_AT_max_core, 
        LR_epochs_TO_max_transformer, LR_epochs_AT_max_transformer, 
        manual_gamma_core, manual_step_size_core,
        manual_gamma_transformer, manual_step_size_transformer,
        KL_rise_first=True, LR_rise_first=True, **kwargs):
            
    
    # *** KL SCHEDULE ***
    
    KL_epoch_period = KL_epochs_TO_max + KL_epochs_AT_max
    KL_epoch_residual = epoch % KL_epoch_period

    KL_range = 10**KL_max - 10**KL_min
    # KL_range = KL_max - KL_min

    # START with rise
    # Logarithmic rise
    if KL_rise_first: 
        if KL_epoch_residual < KL_epochs_TO_max:
            # KL_state_length = KL_epochs_AT_max
            # KL_ceil = KL_max - ( KL_range * (KL_epoch_residual/KL_state_length) )
            # KL_floor = KL_ceil - ( KL_range * (KL_epoch_residual + 1) /KL_state_length)
            # KL_val = KL_ceil - iter_curr/iters_per_epoch * (KL_ceil - KL_floor) 

            KL_state_length = KL_epochs_TO_max 
            KL_floor = 10 ** KL_min + KL_range * (KL_epoch_residual/KL_state_length)
            KL_ceil = KL_floor + KL_range * (1) /KL_state_length
            KL_val = math.log10(KL_floor + iter_curr/iters_per_epoch * (KL_ceil - KL_floor))
        else:
            KL_val = KL_max

    else:
        raise Exception("ERROR: not coded up")
        # if KL_epoch_residual < KL_epochs_AT_max:
        #     KL_val = KL_max
        # else:
        #     KL_state_length = KL_epochs_AT_max
        #     KL_ceil = KL_max - ( KL_range * (KL_epoch_residual/KL_state_length) )
        #     KL_floor = KL_ceil - ( KL_range * (KL_epoch_residual + 1) /KL_state_length)
        #     KL_val = KL_ceil - iter/iters_per_epoch * (KL_ceil - KL_floor)   

 
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


    # TRANSFORMER
    LR_val_transformer = LR_subfunction(
        iter_curr=iter_curr,
        LR_min=LR_min_transformer,
        LR_max=LR_max_transformer,
        epoch=epoch, 
        manual_gamma=manual_gamma_transformer, 
        manual_step_size=manual_step_size_transformer, 
        LR_epochs_TO_max=LR_epochs_TO_max_transformer, 
        LR_epochs_AT_max=LR_epochs_AT_max_transformer, 
        iters_per_epoch=iters_per_epoch,
        LR_rise_first=LR_rise_first
    )

            
    return KL_val, LR_val_core, LR_val_transformer

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
                
def in_seizure(file_name, start_idx, end_idx, samp_freq):

    pat_id = file_name.split("_")[0]
    start_datetimes, stop_datetimes = filename_to_datetimes([file_name])
    start_datetime, stop_datetime = start_datetimes[0], stop_datetimes[0]
    seiz_start_dt, seiz_stop_dt, seiz_types = get_pat_seiz_datetimes(pat_id)
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

def get_all_is_ictal(file_name, data_samples):
    pat_id = file_name.split("_")[0]
    start_datetimes, stop_datetimes = filename_to_datetimes([file_name])
    start_datetime, stop_datetime = start_datetimes[0], stop_datetimes[0]
    seiz_start_dt, seiz_stop_dt, seiz_types = get_pat_seiz_datetimes(pat_id)

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
    if 2 in a:
        return 2
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

def hash_to_vector(input_string, num_channels, latent_dim, modifier):
    # Incorporate the modifier into the input string to vary the output
    modified_input = f"{input_string}_{modifier}"

    # Generate a SHA-256 hash from the modified input string
    hash_object = hashlib.sha256(modified_input.encode('utf-8'))
    hash_digest = hash_object.digest()  # 32 bytes (256 bits)

    # If latent_dim > 256, repeat the hash digest to ensure we have enough data
    extended_hash = (hash_digest * ((latent_dim // 32) + 1))[:latent_dim]  # Repeat and slice to exactly latent_dim bytes
    
    # Generate a vector of size latent_dim with values from -1 to 1
    hashed_vector = np.zeros(latent_dim)

    for i in range(latent_dim):
        # Use the i-th byte from the extended hash digest
        byte_value = extended_hash[i]
        
        # Normalize the byte value to the range [-1, 1]
        hashed_vector[i] = (byte_value / 127.5) - 1  # Normalize to [-1, 1]

    # Convert hashed_vector to a PyTorch tensor
    hashed_vector_tensor = torch.tensor(hashed_vector, dtype=torch.float32)

    # Generate a vector of shuffled numbers 0 to num_channels-1
    ordered_vector = list(range(num_channels))
    
    # Set the seed for deterministic shuffling based on the hash of the modified input string
    random.seed(int.from_bytes(hash_digest[:8], 'big'))  # Use first 8 bytes of hash as the seed
    random.shuffle(ordered_vector)  # Shuffle the list in place
    
    return hashed_vector_tensor, ordered_vector

# def un_pseudobatch_raw_data(x_batched, token_samples):
#     ''' 
#     x_batched [batch * token_lengths, channels, datapoints]
#     returns: x [batch, channels, datapoints]
#     '''
#     x = torch.stack(torch.split(x_batched, token_samples, dim=0), dim=1).transpose(2,3).transpose(1,2)
#     x = x.reshape(x.shape[0]* x.shape[1], x.shape[2], x.shape[3]).transpose(0,1).transpose(1,2)

#     return x


# PLOTTING

def print_latent_realtime(target_emb, predicted_emb, savedir, epoch, iter_curr, pat_id, num_realtime_dims, **kwargs):

    dims_to_plot = np.arange(0,num_realtime_dims)

    batchsize = target_emb.shape[0]

    for b in range(0, batchsize):
        
        # Make new grid/fig
        gs = gridspec.GridSpec(1, 2)
        fig = pl.figure(figsize=(20, 14))

        # Only print for one batch index at a time
        target_emb_plot = target_emb[b, :, 0:len(dims_to_plot)]
        predicted_emb_plot = predicted_emb[b, :, 0:len(dims_to_plot)]

        df = pd.DataFrame({
            'dimension': np.tile(dims_to_plot, target_emb_plot.shape[0]),
            'target_emb': target_emb_plot.flatten(),
            'predicted_emb': predicted_emb_plot.flatten() 
        })

        sns.jointplot(data=df, x="target_emb", y="predicted_emb", hue="dimension")
        fig.suptitle(f"{pat_id}, epoch: {epoch}, iter: {iter_curr}")

        if not os.path.exists(savedir + '/JPEGs'): os.makedirs(savedir + '/JPEGs')
        if not os.path.exists(savedir + '/SVGs'): os.makedirs(savedir + '/SVGs')
        savename_jpg = f"{savedir}/JPEGs/RealtimeLatent_epoch{epoch}_iter{iter_curr}_{pat_id}_batch{b}.jpg"
        savename_svg = f"{savedir}/SVGs/RealtimeLatent_epoch{epoch}_iter{iter_curr}_{pat_id}_batch_{b}.svg"
        pl.savefig(savename_jpg)
        pl.savefig(savename_svg)
        pl.close(fig)    

        pl.close('all') 
    
def print_recon_realtime(x_decode_shifted, x_hat, savedir, epoch, iter_curr, pat_id, num_realtime_channels_recon, num_recon_samples, **kwargs):

    x_hat = x_hat.detach().cpu().numpy()
    x_decode_shifted = x_decode_shifted.detach().cpu().numpy()

    # Fuse the sequential decodes/predictions together
    x_decode_shifted_fused = np.moveaxis(x_decode_shifted, 3, 2)
    x_decode_shifted_fused = x_decode_shifted_fused.reshape(x_decode_shifted_fused.shape[0], x_decode_shifted_fused.shape[1] * x_decode_shifted_fused.shape[2], x_decode_shifted_fused.shape[3])
    x_decode_shifted_fused = np.moveaxis(x_decode_shifted_fused, 1, 2)

    x_hat_fused = np.moveaxis(x_hat, 3, 2)
    x_hat_fused = x_hat_fused.reshape(x_hat_fused.shape[0], x_hat_fused.shape[1] * x_hat_fused.shape[2], x_hat_fused.shape[3])
    x_hat_fused = np.moveaxis(x_hat_fused, 1, 2)

    batchsize = x_hat.shape[0]

    np.random.seed(seed=None) 
    r = np.arange(0,x_hat_fused.shape[1])
    np.random.shuffle(r)
    random_ch_idxs = r[0:num_realtime_channels_recon]

    # Make new grid/fig
    gs = gridspec.GridSpec(batchsize, num_realtime_channels_recon * 2) # *2 because beginning and end of transformer sequence
    fig = pl.figure(figsize=(20, 14))
    palette = sns.cubehelix_palette(n_colors=2, start=3, rot=1) 
    for b in range(0, batchsize):
        for c in range(0,len(random_ch_idxs)):
            for seq in range(0,2):
                if seq == 0:
                    x_decode_plot = x_decode_shifted_fused[b, random_ch_idxs[c], :num_recon_samples]
                    x_hat_plot = x_hat_fused[b, random_ch_idxs[c], :num_recon_samples]
                    title_str = 'StartOfTransSeq'
                else:
                    x_decode_plot = x_decode_shifted_fused[b, random_ch_idxs[c], -num_recon_samples:]
                    x_hat_plot = x_hat_fused[b, random_ch_idxs[c], -num_recon_samples:]   
                    title_str = 'EndOfTransSeq'             

                df = pd.DataFrame({
                    "Target": x_decode_plot,
                    "Prediction": x_hat_plot
                })

                ax = fig.add_subplot(gs[b, c*2 + seq]) 
                sns.lineplot(data=df, palette=palette, linewidth=1.5, dashes=False, ax=ax)
                ax.set_title(f"B:{b}, Ch:{random_ch_idxs[c]}, {title_str}")
            
    fig.suptitle(f"Batches 0:{batchsize-1}, Ch:{random_ch_idxs}")
    if not os.path.exists(savedir + '/JPEGs'): os.makedirs(savedir + '/JPEGs')
    if not os.path.exists(savedir + '/SVGs'): os.makedirs(savedir + '/SVGs')
    savename_jpg = f"{savedir}/JPEGs/RealtimeRecon_epoch{epoch}_iter{iter_curr}_{pat_id}_allbatch.jpg"
    savename_svg = f"{savedir}/SVGs/RealtimeRecon_epoch{epoch}_iter{iter_curr}_{pat_id}_allbatch.svg"
    pl.savefig(savename_jpg)
    pl.savefig(savename_svg)
    pl.close(fig)   

    pl.close('all') 

def print_autoreg_latent_predictions(gpu_id, epoch, pat_id, rand_file_count, latent_context, latent_predictions, latent_target, savedir, num_realtime_dims, **kwargs):
    
    latent_target=latent_target.detach().cpu().numpy()
    latent_context=latent_context.detach().cpu().numpy() # NOT CURRENTLY USING
    latent_predictions=latent_predictions.detach().cpu().numpy()

    dims_to_plot = np.arange(0,num_realtime_dims)
    batchsize = latent_context.shape[0]

    for b in range(0, batchsize):
        
        # Make new grid/fig
        gs = gridspec.GridSpec(1, 2)
        fig = pl.figure(figsize=(20, 14))

        # Only print for one batch index at a time
        target_emb_plot = latent_target[b, :, 0:len(dims_to_plot)]
        predicted_emb_plot = latent_predictions[b, :, 0:len(dims_to_plot)]

        df = pd.DataFrame({
            'dimension': np.tile(dims_to_plot, target_emb_plot.shape[0]),
            'target_emb': target_emb_plot.flatten(),
            'predicted_emb': predicted_emb_plot.flatten() 
        })

        sns.jointplot(data=df, x="target_emb", y="predicted_emb", hue="dimension")
        fig.suptitle(f"{pat_id}, epoch: {epoch}, file: {rand_file_count}")
        if gpu_id == 0: time.sleep(1)
        if not os.path.exists(savedir + '/JPEGs'): os.makedirs(savedir + '/JPEGs')
        if not os.path.exists(savedir + '/SVGs'): os.makedirs(savedir + '/SVGs')
        savename_jpg = f"{savedir}/JPEGs/AutoregressiveLatent_epoch{epoch}_{pat_id}_batch{b}_randfile{rand_file_count}_gpu{gpu_id}.jpg"
        savename_svg = f"{savedir}/SVGs/AutoregressiveLatent_epoch{epoch}_{pat_id}_batch_{b}_randfile{rand_file_count}_gpu{gpu_id}.svg"
        pl.savefig(savename_jpg)
        pl.savefig(savename_svg)
        pl.close(fig)    

        pl.close('all') 

def print_autoreg_raw_predictions(gpu_id, epoch, pat_id, rand_file_count, raw_context, raw_pred, raw_target, autoreg_channels, savedir, num_realtime_dims, **kwargs):

    raw_context = raw_context.detach().cpu().numpy()
    raw_pred = raw_pred.detach().cpu().numpy()
    raw_target = raw_target.detach().cpu().numpy()

    batchsize = raw_context.shape[0]

    np.random.seed(seed=None) 
    r = np.arange(0,raw_context.shape[1])
    np.random.shuffle(r)
    random_ch_idxs = r[0:autoreg_channels]

    # Make new grid/fig for every batch
    for b in range(0, batchsize):
        gs = gridspec.GridSpec(autoreg_channels, 1) 
        fig = pl.figure(figsize=(20, 14))
        palette = sns.cubehelix_palette(n_colors=2, start=3, rot=1) 
        for c in range(0,len(random_ch_idxs)):
            context_plot = raw_context[b, random_ch_idxs[c], :]
            prediction_plot = raw_pred[b, random_ch_idxs[c], :]
            target_plot = raw_target[b, random_ch_idxs[c], :]

            contextTarget_plot = np.concatenate((context_plot, target_plot), axis=0)
            prediction_buffered_plot = np.zeros_like(contextTarget_plot)
            prediction_buffered_plot[-len(prediction_plot):] = prediction_plot
                
            df = pd.DataFrame({
                "Target": contextTarget_plot,
                "Prediction": prediction_buffered_plot
            })

            ax = fig.add_subplot(gs[c, 0]) 
            sns.lineplot(data=df, palette=palette, linewidth=1.5, dashes=False, ax=ax)
            ax.set_title(f"Ch:{random_ch_idxs[c]}")
            
        fig.suptitle(f"Ch:{random_ch_idxs}")
        if gpu_id == 0: time.sleep(1)
        if not os.path.exists(savedir + '/JPEGs'): os.makedirs(savedir + '/JPEGs')
        if not os.path.exists(savedir + '/SVGs'): os.makedirs(savedir + '/SVGs')
        savename_jpg = f"{savedir}/JPEGs/AutoregressiveRecon_epoch{epoch}_{pat_id}_batch{b}_gpu{gpu_id}.jpg"
        savename_svg = f"{savedir}/SVGs/AutoregressiveRecon_epoch{epoch}_{pat_id}_batch{b}_gpu{gpu_id}.svg"
        pl.savefig(savename_jpg)
        pl.savefig(savename_svg)
        pl.close(fig)   

    pl.close('all') 

def print_autoreg_AttentionScores_AlongSeq(gpu_id, epoch, pat_id, rand_file_count, scores_allSeq_firstLayer_meanHeads_lastRow, savedir, **kwargs):

    scores_allSeq_firstLayer_meanHeads_lastRow = scores_allSeq_firstLayer_meanHeads_lastRow.detach().cpu().numpy()

    batchsize = scores_allSeq_firstLayer_meanHeads_lastRow.shape[0]

    # Make new grid/fig for every batch
    for b in range(0, batchsize):
        gs = gridspec.GridSpec(1, 1) 
        fig = pl.figure(figsize=(20, 14))

        scores_row_plot = scores_allSeq_firstLayer_meanHeads_lastRow[b, :, :].swapaxes(0,1)
            
        # df = pd.DataFrame({
        #     "scores_row": scores_row_plot,
        #     "scores_col": scores_col_plot
        # }, columns=np.arange(0, scores_row_plot.shape[0]))

        ax1 = fig.add_subplot(gs[0]) 
        sns.heatmap(scores_row_plot, cmap=sns.cubehelix_palette(as_cmap=True))
        ax1.set_title(f"Score Rows, First Transformer Layer - Average of Heads, Batch:{b}")
        ax1.set_xlabel("Autoregression Step")
        ax1.set_ylabel("Attention Weight by Current Past Index")
            
        fig.suptitle(f"Attention Weights")
        if gpu_id == 0: time.sleep(1)
        if not os.path.exists(savedir + '/JPEGs'): os.makedirs(savedir + '/JPEGs')
        if not os.path.exists(savedir + '/SVGs'): os.makedirs(savedir + '/SVGs')
        savename_jpg = f"{savedir}/JPEGs/AutoregressiveAttention_epoch{epoch}_{pat_id}_batch{b}_gpu{gpu_id}.jpg"
        savename_svg = f"{savedir}/SVGs/AutoregressiveAttention_epoch{epoch}_{pat_id}_batch{b}_gpu{gpu_id}.svg"
        pl.savefig(savename_jpg)
        pl.savefig(savename_svg)
        pl.close(fig)   

    pl.close('all') 

def plot_MeanStd(plot_mean, plot_std, plot_dict, file_name, epoch, savedir, gpu_id, pat_id, iter): # plot_weights

    mean_mean = np.mean(plot_mean, axis=2)
    plot_std = np.mean(plot_std, axis=2)
    # weight_mean = np.mean(plot_weights, axis=2)
    num_dims = mean_mean.shape[1]
    x=np.linspace(0,num_dims-1,num_dims)

    df = pd.DataFrame({
        'mean': np.mean(mean_mean, axis=0),
        'std': np.mean(plot_std, axis=0),
    })

    gs = gridspec.GridSpec(2, 5)
    fig = pl.figure(figsize=(20, 14))
    
    # Plot Means
    ax1 = pl.subplot(gs[0, :])
    sns.barplot(df, ax=ax1, x=x, y="mean", native_scale=True, errorbar=None,)

    # Plot Logvar
    ax1 = pl.subplot(gs[1, :])
    sns.barplot(df, ax=ax1, x=x, y="std", native_scale=True, errorbar=None,)

    # Pull out toaken start times because it's plotting whole batch at once
    fig.suptitle(file_name + create_metadata_subtitle(plot_dict))

    if not os.path.exists(savedir + '/JPEGs'): os.makedirs(savedir + '/JPEGs')
    if not os.path.exists(savedir + '/SVGs'): os.makedirs(savedir + '/SVGs')
    savename_jpg = savedir + f"/JPEGs/MeanLogvarWeights_batch{str(plot_mean.shape[0])}_" + "epoch" + str(epoch) + "_iter" + str(iter) + "_" + pat_id + "_gpu" + str(gpu_id) + ".jpg"
    savename_svg = savedir + f"/SVGs/MeanLogvarWeights_batch{str(plot_mean.shape[0])}_" + "epoch" + str(epoch) + "_iter" + str(iter) + "_" + pat_id + "_gpu" + str(gpu_id) + ".svg"
    pl.savefig(savename_jpg)
    pl.savefig(savename_svg)
    pl.close(fig)    

    mean_of_mean_mean = np.mean(np.abs(mean_mean))
    std_of_mean_mean = np.std(np.abs(mean_mean))
    mean_zscores = np.mean((np.abs(mean_mean) - mean_of_mean_mean)/std_of_mean_mean , axis=0)

    mean_of_plot_std = np.mean(plot_std)
    std_of_plot_std = np.std(plot_std)
    std_zscores = np.mean((plot_std - mean_of_plot_std)/std_of_plot_std , axis=0)

    pl.close('all') 

    return mean_zscores, std_zscores

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

        if not os.path.exists(savedir + '/JPEGs'): os.makedirs(savedir + '/JPEGs')
        if not os.path.exists(savedir + '/SVGs'): os.makedirs(savedir + '/SVGs')
        savename_jpg = savedir + f"/JPEGs/Recon_epoch{str(epoch)}_iter_{str(iter)}_{pat_id}_batchIdx{str(batch_idx)}_chIdx{str(ch_idx)}_duration{str(recon_sec)}sec_startIdx{str(start_idx)}_gpu{str(gpu_id)}.jpg"
        savename_svg = savedir + f"/SVGs/Recon_epoch{str(epoch)}_iter_{str(iter)}_{pat_id}_batchIdx{str(batch_idx)}_chIdx{str(ch_idx)}_duration{str(recon_sec)}sec_startIdx{str(start_idx)}_gpu{str(gpu_id)}.svg"
        pl.savefig(savename_jpg)
        pl.savefig(savename_svg)
        pl.close(fig) 
        
def pacmap_latent(  
    FS, 
    latent_data_windowed,
    in_data_samples,
    decode_samples,
    start_datetimes_epoch,
    stop_datetimes_epoch,
    epoch, 
    iter, 
    gpu_id, 
    win_sec, 
    stride_sec, 
    absolute_latent,
    file_name,
    savedir,
    pat_ids_list, 
    plot_dict,
    premade_PaCMAP,
    premade_PaCMAP_MedDim,
    pacmap_MedDim_numdims,
    premade_PCA,
    premade_HDBSCAN,
    xy_lims,
    info_score_idxs,
    xy_lims_RAW_DIMS,
    xy_lims_PCA,
    cluster_reorder_indexes,
    pacmap_LR,
    pacmap_NumIters,
    pacmap_NN,
    pacmap_MN_ratio,
    pacmap_FP_ratio,
    pacmap_MN_ratio_MedDim,
    pacmap_FP_ratio_MedDim,
    HDBSCAN_min_cluster_size,
    HDBSCAN_min_samples,
    interictal_contour=False,
    verbose=True,
    **kwargs):

    # Goal of function:
    # Make 2D PaCMAP, make 10D PaCMAP, HDBSCAN cluster on 10D, visualize clusters on 2D

    WIN_STYLE='end'

    # Metadata
    latent_dim = latent_data_windowed[0][0].shape[0]
    num_timepoints_in_windowed_file = latent_data_windowed[0][0].shape[1]
    modified_FS = 1 / stride_sec

    # Flatten data into [points, dim] to feed into PaCMAP, original data is [pat, file, dim, points]
    latent_windowed_flat_perpat = [np.concatenate(latent_data_windowed[i], axis=1).swapaxes(0,1) for i in range(len(latent_data_windowed))]
    latent_PaCMAP_input = np.concatenate(latent_windowed_flat_perpat, axis=0)

    # PaCMAP 2-Dim
    # Make new PaCMAP
    if premade_PaCMAP == []:
        print("Making new 2-dim PaCMAP to use for visualization")
        # initializing the pacmap instance
        # Setting n_neighbors to "None" leads to a default choice shown below in "parameter" section
        reducer = pacmap.PaCMAP(
            distance='angular',
            lr=pacmap_LR,
            num_iters=pacmap_NumIters, # will default ~27 if left as None
            n_components=2, 
            n_neighbors=pacmap_NN, # default None, 
            MN_ratio=pacmap_MN_ratio, # default 0.5, 
            FP_ratio=pacmap_FP_ratio, # default 2.0,
            save_tree=True, # Save tree to enable 'transform" method
            apply_pca=True, 
            verbose=verbose) 

        # fit the data (The index of transformed data corresponds to the index of the original data)
        reducer.fit(latent_PaCMAP_input, init='pca')

    # Use premade PaCMAP
    else: 
        print("Using existing 2-dim PaCMAP for visualization")
        reducer = premade_PaCMAP

    # Project data through reducer (i.e. PaCMAP) one patient at a time
    latent_flat_postPaCMAP_perpat = [reducer.transform(latent_windowed_flat_perpat[i]) for i in range(len(latent_windowed_flat_perpat))]
    latent_embedding_allFiles_perpat = [latent_flat_postPaCMAP_perpat[i].reshape(num_timepoints_in_windowed_file, -1, 2).swapaxes(1, 2).swapaxes(0, 2) for i in range(len(latent_windowed_flat_perpat))]

    # **** PaCMAP (MedDim)--> HDBSCAN ***** i.e. NOTE This is the pacmap used for clustering
    if premade_PaCMAP_MedDim == []: 
        # Make new PaCMAP
        print("Making new medium dim PaCMAP to use for HDBSCAN clustering")
        
        # initializing the pacmap instance
        # Setting n_neighbors to "None" leads to a default choice shown below in "parameter" section
        reducer_MedDim = pacmap.PaCMAP(
            distance='angular',
            lr=pacmap_LR,
            num_iters=pacmap_NumIters, # will default ~27 if left as None
            n_components=pacmap_MedDim_numdims, # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            n_neighbors=pacmap_NN, # default None, 
            MN_ratio=pacmap_MN_ratio_MedDim, # default 0.5, 
            FP_ratio=pacmap_FP_ratio_MedDim, # default 2.0,
            save_tree=True, 
            apply_pca=True, 
            verbose=verbose) # Save tree to enable 'transform" method?

        # fit the data (The index of transformed data corresponds to the index of the original data)
        reducer_MedDim.fit(latent_PaCMAP_input, init='pca')

    # Use premade PaCMAP
    else: 
        print("Using existing medium dim PaCMAP to use for HDBSCAN clustering")
        reducer_MedDim = premade_PaCMAP_MedDim

    # Project data through reducer (i.e. PaCMAP) to get embeddings in shape [timepoint, med-dim, file]
    latent_flat_postPaCMAP_perpat_MEDdim = [reducer_MedDim.transform(latent_windowed_flat_perpat[i]) for i in range(len(latent_windowed_flat_perpat))]
    # latent_embedding_allFiles_MEDdim_perpat = [latent_flat_postPaCMAP_perpat_MEDdim[i].reshape(num_timepoints_in_windowed_file, -1, pacmap_MedDim_numdims).swapaxes(1, 2).swapaxes(0, 2) for i in range(len(latent_windowed_flat_perpat))]

    # Concatenate to feed into HDBSCAN
    hdbscan_input = np.concatenate(latent_flat_postPaCMAP_perpat_MEDdim, axis=0)

    # If training, create new cluster model, otherwise "approximate_predict()" if running on val data
    if premade_HDBSCAN == []:
        # Now do the clustering with HDBSCAN
        print("Building new HDBSCAN model")
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_min_cluster_size,
            min_samples=HDBSCAN_min_samples,
            max_cluster_size=0,
            metric='euclidean',
            # memory=Memory(None, verbose=1)
            algorithm='best',
            cluster_selection_method='eom',
            prediction_data=True
            )
        
        hdb.fit(hdbscan_input)

         #TODO Look into soft clustering
        # soft_cluster_vecs = np.array(hdbscan.all_points_membership_vectors(hdb))
        # soft_clusters = np.array([np.argmax(x) for x in soft_cluster_vecs], dtype=int)
        # hdb_color_palette = sns.color_palette('Paired', int(np.max(soft_clusters) + 3))

        hdb_labels_flat = hdb.labels_
        # hdb_labels_flat = soft_clusters
        hdb_probabilities_flat = hdb.probabilities_
        # hdb_probabilities_flat = np.array([np.max(x) for x in soft_cluster_vecs])
                
    # If HDBSCAN is already made/provided, then predict cluster with built in HDBSCAN method
    else:
        print("Using pre-built HDBSCAN model")
        hdb = premade_HDBSCAN
        

    #TODO Destaurate according to probability of being in cluster

    # Per patient, Run data through model & Reshape the labels and probabilities for plotting
    hdb_labels_flat_perpat = [-1] * len(latent_flat_postPaCMAP_perpat_MEDdim)
    hdb_probabilities_flat_perpat = [-1] * len(latent_flat_postPaCMAP_perpat_MEDdim)
    for i in range(len(latent_flat_postPaCMAP_perpat_MEDdim)):
        hdb_labels_flat_perpat[i], hdb_probabilities_flat_perpat[i] = hdbscan.prediction.approximate_predict(hdb, latent_flat_postPaCMAP_perpat_MEDdim[i])
        
    # Reshape to get [file/epoch, timepoint]
    hdb_labels_allFiles_perpat = [hdb_labels_flat_perpat[i].reshape(num_timepoints_in_windowed_file, -1).swapaxes(0,1) for i in range(len(latent_windowed_flat_perpat))]
    hdb_probabilities_allFiles_perpat = [hdb_probabilities_flat_perpat[i].reshape(num_timepoints_in_windowed_file, -1).swapaxes(0,1) for i in range(len(latent_windowed_flat_perpat))]


    ###### START OF PLOTTING #####

    # Get all of the seizure times and types
    seiz_start_dt_perpat = [-1] * len(latent_flat_postPaCMAP_perpat_MEDdim)
    seiz_stop_dt_perpat = [-1] * len(latent_flat_postPaCMAP_perpat_MEDdim)
    seiz_types_perpat = [-1] * len(latent_flat_postPaCMAP_perpat_MEDdim)
    for i in range(len(latent_flat_postPaCMAP_perpat_MEDdim)):
        seiz_start_dt_perpat[i], seiz_stop_dt_perpat[i], seiz_types_perpat[i] = get_pat_seiz_datetimes(pat_ids_list[i])

    # Stack the patients data together for plotting
    latent_plotting_allpats_filestacked = np.concatenate(latent_embedding_allFiles_perpat ,axis=0)
    hdb_labels_allpats_filestacked = np.expand_dims(np.concatenate(hdb_labels_allFiles_perpat ,axis=0), axis=1)
    hdb_probabilities_allpats_filestacked = np.expand_dims(np.concatenate(hdb_probabilities_allFiles_perpat ,axis=0), axis=1)
    start_datetimes_allpats_filestacked = [element for nestedlist in start_datetimes_epoch for element in nestedlist]
    stop_datetimes_allpats_filestacked = [element for nestedlist in stop_datetimes_epoch for element in nestedlist]
    file_patids_allpats_filestacked = [pat_ids_list[i] for i in range(len(stop_datetimes_epoch)) for element in stop_datetimes_epoch[i]]
    seiz_start_dt_allpats_stacked = [element for nestedlist in seiz_start_dt_perpat for element in nestedlist]
    seiz_stop_dt_allpats_stacked = [element for nestedlist in seiz_stop_dt_perpat for element in nestedlist]
    seiz_types_allpats_stacked = [element for nestedlist in seiz_types_perpat for element in nestedlist]
    seiz_patids_allpats_stacked = [pat_ids_list[i] for i in range(len(seiz_types_perpat)) for element in seiz_types_perpat[i]]
    
    # Intialize master figure 
    fig = pl.figure(figsize=(40, 25))
    gs = gridspec.GridSpec(3, 5, figure=fig)


    # **** PACMAP PLOTTING ****

    print(f"[GPU{str(gpu_id)}] PaCMAP Plotting")
    ax20 = fig.add_subplot(gs[2, 0]) 
    ax21 = fig.add_subplot(gs[2, 1]) 
    ax22 = fig.add_subplot(gs[2, 2]) 
    ax23 = fig.add_subplot(gs[2, 3]) 
    ax24 = fig.add_subplot(gs[2, 4]) 

    # Latent space plot
    # NOTE: datetimes are unsorted at this time, but will be sorted within plot_latent
    ax20, ax21, ax22, ax23, ax24, xy_lims = plot_latent(
        ax=ax20, 
        interCont_ax=ax21,
        seiztype_ax=ax22,
        time_ax=ax23,
        cluster_ax=ax24,
        latent_data=latent_plotting_allpats_filestacked, ## stacked all pats
        modified_samp_freq=modified_FS,  ############ update to be 'modified FS' to account for un-expanded data
        start_datetimes=start_datetimes_allpats_filestacked, 
        stop_datetimes=stop_datetimes_allpats_filestacked, 
        win_sec=win_sec,
        stride_sec=stride_sec, 
        seiz_start_dt=seiz_start_dt_allpats_stacked, 
        seiz_stop_dt=seiz_stop_dt_allpats_stacked, 
        seiz_types=seiz_types_allpats_stacked,
        preictal_dur=plot_dict["plot_preictal_color"],
        postictal_dur=plot_dict["plot_postictal_color"],
        plot_ictal=True,
        hdb_labels=hdb_labels_allpats_filestacked,
        hdb_probabilities=hdb_probabilities_allpats_filestacked,
        hdb=hdb,
        xy_lims=xy_lims,
        **kwargs)        

    ax20.title.set_text('PaCMAP Latent Space: ' + 
        'Window mean, dur/str=' + str(win_sec) + 
        '/' + str(stride_sec) +' seconds,' + 
        f'\nLR: {str(pacmap_LR)}, ' +
        f'NumIters: {str(pacmap_NumIters)}, ' +
        f'NN: {pacmap_NN}, MN_ratio: {str(pacmap_MN_ratio)}, FP_ratio: {str(pacmap_FP_ratio)}'
        )
    
    if interictal_contour:
        ax21.title.set_text('Interictal Contour (no peri-ictal data)')


    # ***** PCA PLOTTING *****
        
    if premade_PCA == []:
        print("Calculating new PCA")
        pca = PCA(n_components=2, svd_solver='full')
        latent_PCA_flat_transformed = pca.fit_transform(latent_PaCMAP_input)

    else:
        print("Using existing PCA")
        pca = premade_PCA
        
    # Project data through PCA one pat at a time
    latent_PCA_flat_transformed_perpat = [pca.transform(latent_windowed_flat_perpat[i]) for i in range(len(latent_windowed_flat_perpat))]
    latent_PCA_allFiles_perpat = [latent_PCA_flat_transformed_perpat[i].reshape(num_timepoints_in_windowed_file, -1, 2).swapaxes(1, 2).swapaxes(0, 2) for i in range(len(latent_windowed_flat_perpat))]

    # Stack the PCA data 
    latent_PCA_plotting_allpats_filestacked = np.concatenate(latent_PCA_allFiles_perpat ,axis=0)

    print(f"[GPU{str(gpu_id)}] PCA Plotting")
    ax10 = fig.add_subplot(gs[1, 0]) 
    ax11 = fig.add_subplot(gs[1, 1]) 
    ax12 = fig.add_subplot(gs[1, 2]) 
    ax13 = fig.add_subplot(gs[1, 3]) 
    ax14 = fig.add_subplot(gs[1, 4]) 

    # Latent space plot
    # NOTE: datetimes are unsorted at this time, but will be sorted within plot_latent
    ax10, ax11, ax12, ax13, ax14, xy_lims_PCA = plot_latent(
        ax=ax10, 
        interCont_ax=ax11,
        seiztype_ax=ax12,
        time_ax=ax13,
        cluster_ax=ax14,
        latent_data=latent_PCA_plotting_allpats_filestacked,   # latent_PCA_expanded_allFiles, 
        modified_samp_freq=modified_FS,
        start_datetimes=start_datetimes_allpats_filestacked, 
        stop_datetimes=stop_datetimes_allpats_filestacked, 
        win_sec=win_sec,
        stride_sec=stride_sec, 
        seiz_start_dt=seiz_start_dt_allpats_stacked, 
        seiz_stop_dt=seiz_stop_dt_allpats_stacked, 
        seiz_types=seiz_types_allpats_stacked,
        preictal_dur=plot_dict["plot_preictal_color"],
        postictal_dur=plot_dict["plot_postictal_color"],
        plot_ictal=True,
        hdb_labels=hdb_labels_allpats_filestacked,
        hdb_probabilities=hdb_probabilities_allpats_filestacked,
        hdb=hdb,
        xy_lims=xy_lims_PCA,
        **kwargs)        

    ax10.title.set_text("PCA Components 1,2")
    ax11.title.set_text('Interictal Contour (no peri-ictal data)')


    # **** INFO RAW DIM PLOTTING *****

    raw_dims_to_plot = info_score_idxs[-2:]

    # Pull out the raw dims of interest and stack the data by file
    latent_flat_RawDim_perpat = [latent_windowed_flat_perpat[i][:, raw_dims_to_plot] for i in range(len(latent_windowed_flat_perpat))]
    latent_RawDim_allFiles_perpat = [latent_flat_RawDim_perpat[i].reshape(num_timepoints_in_windowed_file, -1, 2).swapaxes(1, 2).swapaxes(0, 2) for i in range(len(latent_windowed_flat_perpat))]

    # Stack the raw data
    latent_RawDim_filestacked = np.concatenate(latent_RawDim_allFiles_perpat ,axis=0)

    print(f"[GPU{str(gpu_id)}] Raw Dims Plotting")
    ax00 = fig.add_subplot(gs[0, 0]) 
    ax01 = fig.add_subplot(gs[0, 1]) 
    ax02 = fig.add_subplot(gs[0, 2]) 
    ax03 = fig.add_subplot(gs[0, 3]) 
    ax04 = fig.add_subplot(gs[0, 4])

    # Latent space plot
    # NOTE: datetimes are unsorted at this time, but will be sorted within plot_latent
    ax00, ax01, ax02, ax03, ax04, xy_lims_RAW_DIMS = plot_latent(
        ax=ax00, 
        interCont_ax=ax01,
        seiztype_ax=ax02,
        time_ax=ax03,
        cluster_ax=ax04,
        latent_data=latent_RawDim_filestacked,
        modified_samp_freq=modified_FS,
        start_datetimes=start_datetimes_allpats_filestacked, 
        stop_datetimes=stop_datetimes_allpats_filestacked, 
        win_sec=win_sec,
        stride_sec=stride_sec, 
        seiz_start_dt=seiz_start_dt_allpats_stacked, 
        seiz_stop_dt=seiz_stop_dt_allpats_stacked, 
        seiz_types=seiz_types_allpats_stacked,
        preictal_dur=plot_dict["plot_preictal_color"],
        postictal_dur=plot_dict["plot_postictal_color"],
        plot_ictal=True,
        hdb_labels=hdb_labels_allpats_filestacked,
        hdb_probabilities=hdb_probabilities_allpats_filestacked,
        hdb=hdb,
        xy_lims=xy_lims_RAW_DIMS,
        **kwargs)        

    ax00.title.set_text(f'Dims [{raw_dims_to_plot[0]},{raw_dims_to_plot[1]}], Window mean, dur/str=' + str(win_sec) + '/' + str(stride_sec) +' seconds,' )
    ax01.title.set_text('Interictal Contour (no peri-ictal data)')

    # **** Save entire figure *****

    fig.suptitle(file_name + create_metadata_subtitle(plot_dict))
    if not os.path.exists(savedir + '/JPEGs'): os.makedirs(savedir + '/JPEGs')
    if not os.path.exists(savedir + '/SVGs'): os.makedirs(savedir + '/SVGs')
    savename_jpg = savedir + f"/JPEGs/{file_name}_latent_smoothsec" + str(win_sec) + "Stride" + str(stride_sec) + "_epoch" + str(epoch) + "_iter" + str(iter) + "_LR" + str(pacmap_LR) + "_NumIters" + str(pacmap_NumIters) + "_gpu" + str(gpu_id)  + ".jpg"
    savename_svg = savedir + f"/SVGs/{file_name}latent_smoothsec" + str(win_sec) + "Stride" + str(stride_sec) + "_epoch" + str(epoch) + "_iter" + str(iter) + "_LR" + str(pacmap_LR) + "_NumIters" + str(pacmap_NumIters) + "_gpu" + str(gpu_id)  + ".svg"
    pl.savefig(savename_jpg, dpi=300)
    pl.savefig(savename_svg)

    # TODO Upload to WandB

    pl.close(fig)

    # Bundle the save metrics together
    # save_tuple = (latent_data_windowed.swapaxes(1,2), latent_PCA_allFiles, latent_topPaCMAP_allFiles, latent_topPaCMAP_MedDim_allFiles, hdb_labels_allFiles, hdb_probabilities_allFiles)
    return reducer, reducer_MedDim, hdb, pca, xy_lims, xy_lims_PCA, xy_lims_RAW_DIMS, cluster_reorder_indexes # save_tuple

def print_dataset_bargraphs(pat_id, curr_file_list, curr_fpaths, dataset_pic_dir, pre_ictal_taper_sec=120, post_ictal_taper_sec=120):

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
    seiz_start_dt, seiz_stop_dt, seiz_types = get_pat_seiz_datetimes(pat_id)
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

def get_PaCMAP_model(model_common_prefix, pre_PaCMAP_window_sec_path, pre_PaCMAP_stride_sec_path):
    # with open(model_path, "rb") as f: PaCMAP = pickle.load(f)
    PaCMAP = pacmap.load(model_common_prefix)
    with open(pre_PaCMAP_window_sec_path, "rb") as f: pre_PaCMAP_window_sec = pickle.load(f)
    with open(pre_PaCMAP_stride_sec_path, "rb") as f: pre_PaCMAP_stride_sec = pickle.load(f)

    return PaCMAP, pre_PaCMAP_window_sec, pre_PaCMAP_stride_sec

def load_inferred_pkl(file):
    # Assumes the .pkl files have the following order for their save tuple:
    # save_tuple = (latent_data_windowed.swapaxes(1,2), latent_PCA_allFiles, latent_topPaCMAP_allFiles, latent_topPaCMAP_MedDim_allFiles, hdb_labels_allFiles, hdb_probabilities_allFiles)
   
    # Each array expected to have shape of [batch, latent dim/label dim, time elements]

    # Import the window/smooth seconds and stride seconds from filename
    splitties = file.split("/")[-1].split("_")
    str_of_interest = splitties[8]
    if ('window' not in str_of_interest) | ('stride' not in str_of_interest): raise Exception(f"Expected string to have 'window' and 'stride' parsed from filename, but got {str_of_interest}")
    str_of_interest = str_of_interest.split("seconds")[0]
    window_sec = float(str_of_interest.split('window')[-1].split('stride')[0])
    stride_sec = float(str_of_interest.split('window')[-1].split('stride')[1])

    expected_len = 6
    with open(file, "rb") as f: S = pickle.load(f)
    if len(S) != expected_len: raise Exception(f"ERROR: expected tuple to have {expected_len} elements, but it has {len(S)}")
    latent_data_windowed = S[0]                                         
    latent_PCA_allFiles = S[1]
    latent_topPaCMAP_allFiles = S[2]
    latent_topPaCMAP_MedDim_allFiles = S[3]
    hdb_labels_allFiles = S[4]
    hdb_probabilities_allFiles = S[5]

    return window_sec, stride_sec, latent_data_windowed, latent_PCA_allFiles, latent_topPaCMAP_allFiles, latent_topPaCMAP_MedDim_allFiles, hdb_labels_allFiles, hdb_probabilities_allFiles

def load_data_tensor(filename):
    file = open(filename,'rb')
    data = pickle.load(file) 
    file.close()
    # data_channel_subset = data[0:self.num_channels,:]   
    return torch.FloatTensor(data)

def collect_latent_tmp_files(path, keyword, expected_GPU_count, approx_file_count, win_sec, stride_sec, decode_samples, FS):
    # Determine how many GPUs have saved tmp files
    potential_paths = glob.glob(f"{path}/*{keyword}*")

    print("WARNING: expected file count check suspended")
    # file_buffer = 2 # files should be spread evenly over GPUs by DDP, so buffer of 1 is even probably sufficient
    # if (len(potential_paths) < (approx_file_count * expected_GPU_count - file_buffer)) or (len(potential_paths) > (approx_file_count * expected_GPU_count + file_buffer)):
    #     raise Exception (f"ERROR: expected approximately {str(approx_file_count)} files across {str(expected_GPU_count)} GPUs, but found {str(len(potential_paths))} in {path}")

    for f in range(len(potential_paths)):
        with open(potential_paths[f], 'rb') as file: 
            latent_tuple = pickle.load(file)
        
        latent_raw = latent_tuple[0].detach().numpy()
        # Average the data temporally according to smoothing seconds, before feeding into PaCMAP
        num_iters = int((latent_raw.shape[2]*decode_samples - win_sec * FS)/(stride_sec * FS)+ 1)
        if num_iters == 0: raise Exception("ERROR: num_iters = 0 somehow. It should be > 0")
        window_subsamps = int((win_sec * FS) / decode_samples)
        stride_subsamps = int((stride_sec * FS) / decode_samples)
        latent_windowed = [np.zeros([latent_raw.shape[1], num_iters], dtype=np.float16)] * latent_raw.shape[0]
        for j in range(0, latent_raw.shape[0]):
            latent_windowed[j] = np.array([np.mean(latent_raw[j, :, i*stride_subsamps: i*stride_subsamps + window_subsamps], axis=1) for i in range(0,num_iters)], dtype=np.float32).transpose()

        if f == 0:            
            latent_windowed_ALL = latent_windowed
            start_ALL = latent_tuple[1] # ensure it is a list 
            stop_ALL = latent_tuple[2] # ensure it is a list 

        else: 
            latent_windowed_ALL = latent_windowed_ALL + latent_windowed
            start_ALL = start_ALL + latent_tuple[1]
            stop_ALL = stop_ALL + latent_tuple[2]

    return latent_windowed_ALL, start_ALL, stop_ALL

def collate_latent_tmps(save_dir: str, samp_freq: int, patid: str, epoch_used: int, hours_inferred_str: str, save_dimension_style: str, stride: int):
    print("\nCollating tmp files")
    # Pull in all the tmp files across all tmp directories (assumes directory is named 'tmp<#>')
    dirs = glob.glob(save_dir + '/tmp*')
    file_count = len(glob.glob(save_dir + "/tmp*/*.pkl"))
    all_filenames = ["NaN"]*(file_count)

    # Pull in one latent file to get the sample size in latent variable
    f1 = glob.glob(dirs[0] + "/*.pkl")[0]
    with open(f1, 'rb') as file: 
        latent_sample_data = pickle.load(file)
    latent_dims = latent_sample_data[1].shape[1]
    latent_samples_in_epoch = latent_sample_data[1].shape[2]
    del latent_sample_data

    all_latent = np.zeros([file_count, latent_dims, latent_samples_in_epoch], dtype=np.float16)
    d_count = 0
    ff_count = 0
    individual_count = 0
    for d in dirs:
        d_count= d_count + 1
        files = glob.glob(d + "/*.pkl")
        for ff in files:
            ff_count = ff_count + 1
            if ff_count%10 == 0: print("GPUDir " + str(d_count) + "/" + str(len(dirs)) + ": File " + str(ff_count) + "/" + str(file_count))
            with open(ff, 'rb') as file:
                file_data = pickle.load(file)

                # Check to see the actual count of data within the batch
                batch_count = len(file_data[0])
                if batch_count != 1: raise Exception("Batch size not equal to one, batch size must be one")
                
                all_filenames[individual_count : individual_count + batch_count] = file_data[0]

                # Get the start and end datetime objects. IMPORTANT: base the start datetime on the end datetime working backwards,
                # because there may have been time shrinkage due to NO PADDING in the time dimension

                # Append the data together batchwise (mandated batch of 1)
                all_latent[individual_count : individual_count + batch_count, :, :] = file_data[1][0, :, :].astype(np.float16)
               
                individual_count = individual_count + batch_count

    # sort the filenames to find the last file and the first file to get a total number of samples to initialize in final variable
    sort_idxs = np.argsort(all_filenames)

    # Get all of end objects to prepare for iterating through batches and placing 
    file_end_objects = [filename_to_dateobj(f, start_or_end=1) for f in all_filenames]
    first_end_datetime = file_end_objects[sort_idxs[0]]

    # Get the length of the latent variable (will be shorter than input data if NO PADDING)
    # TODO Get the total EMU time utilized in seconds and sample
    # VERY IMPORTANT, define the start time based off the end time and samples in latent space  
    dur_latentVar_seconds = all_latent.shape[2] / samp_freq
    file_start_objects = [fend - datetime.timedelta(seconds=dur_latentVar_seconds) for fend in file_end_objects] # All of the start times for the files (NOT the same as filename start times if there is latent time shrinking)
    
    first_start_dateobj = first_end_datetime - datetime.timedelta(seconds=dur_latentVar_seconds) # Only the first file
    last_end_dateobj = file_end_objects[sort_idxs[-1]]

    # Total samples of latent space (may not all get filled)
    master_latent_samples = round(((last_end_dateobj - first_start_dateobj).total_seconds() * samp_freq)) 

    # Initialize the final output variable
    master_latent = np.zeros([latent_dims, master_latent_samples], dtype=np.float16)
    num_files_avgd_at_sample = np.zeros([master_latent.shape[1]], dtype=np.uint8) # WIll use this for weighting the new data as it comes in

    # Fill the master latent variables using the filiename timestamps
    # Average latent variables for any time overlap due to sliding window of training data (weighted average as new overlaps are discovered)
    for i in range(0,len(sort_idxs)):
        curr_idx = sort_idxs[i]
        latent_data = all_latent[curr_idx, :, :]
        dt = file_start_objects[sort_idxs[i]]
        ai = round((dt - first_start_dateobj).total_seconds() * samp_freq) # insert start sample index
        bi = ai + latent_data.shape[1] # insert end sample index

        # Insert each latent channel as a weighted average of what is already there
        master_latent[:, ai:bi] = master_latent[:, ai:bi] * (num_files_avgd_at_sample[ai:bi]/(num_files_avgd_at_sample[ai:bi] + 1)) + latent_data * (1/(num_files_avgd_at_sample[ai:bi] + 1))

        # Increment the number of files used at these sample points                                                              
        num_files_avgd_at_sample[ai:bi] = num_files_avgd_at_sample[ai:bi] + 1                                                                           

    # Change wherever there are zero files contributing to latent data into np.nan
    zero_files_used_idxs = np.where(num_files_avgd_at_sample == 0)[0]
    master_latent[:,zero_files_used_idxs] = np.nan

    # Pickle the master_latent variable
    s_start = all_filenames[sort_idxs[0]].split("_")
    s_end = all_filenames[sort_idxs[-1]].split("_")
    master_filename = save_dir + "/" + patid + "_" + save_dimension_style + "_master_latent_" + hours_inferred_str + "_trainedepoch" + str(epoch_used) + "_" + s_start[1] + "_" + s_start[2] + "_to_" + s_end[4] + "_" + s_end[5] + ".pkl"
    with open(master_filename, 'wb') as file: pickle.dump(master_latent, file)

    # Delete tmp directories
    for d in dirs:
        shutil.rmtree(d)

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

def delete_old_checkpoints(dir: str, curr_epoch: int):

    SAVE_KEYWORDS = ["hdbscan", "pacmap"]

    all_dir_names = glob.glob(f"{dir}/Epoch*")

    epoch_nums = [int(f.split("/")[-1].replace("Epoch_","")) for f in all_dir_names]

    # Add current epoch to save files
    save_epochs = [curr_epoch]
    for i in range(len(all_dir_names)):
        subepoch_dirs = [f.split("/")[-1] for f in glob.glob(all_dir_names[i] + "/*")]
        for f in subepoch_dirs:
            if any(substr in f for substr in SAVE_KEYWORDS):
                save_epochs.append(epoch_nums[i])
                break

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

def import_all_val_files(file_list):
    
    # Import pacmap variables
    win_key = '_pre_PaCMAP_window_sec.pkl'
    win_path = file_list[[idx for idx, s in enumerate(file_list) if win_key in s][0]]
    stride_key = '_pre_PaCMAP_stride_sec.pkl'
    stride_path = file_list[[idx for idx, s in enumerate(file_list) if stride_key in s][0]]
    pac2d_key = 'PaCMAP.pkl'
    pac2d_common_path = file_list[[idx for idx, s in enumerate(file_list) if pac2d_key in s][0]].split('.pkl')[0]
    PaCMAP, pre_PaCMAP_window_sec, pre_PaCMAP_stride_sec = get_PaCMAP_model(model_common_prefix=pac2d_common_path, pre_PaCMAP_window_sec_path=win_path, pre_PaCMAP_stride_sec_path=stride_path)
    pac_MEDDIM_key = 'PaCMAP_MedDim.pkl'
    pac_MEDDIM_common_path = file_list[[idx for idx, s in enumerate(file_list) if pac_MEDDIM_key in s][0]].split('.pkl')[0]
    PaCMAP_MedDim, _, _ = get_PaCMAP_model(model_common_prefix=pac_MEDDIM_common_path, pre_PaCMAP_window_sec_path=win_path, pre_PaCMAP_stride_sec_path=stride_path)

    # Import HDBSCAN model
    hdbscan_key = '_hdbscan.pkl'
    hdbscan_path = file_list[[idx for idx, s in enumerate(file_list) if hdbscan_key in s][0]]
    with open(hdbscan_path, "rb") as f: HDBSCAN = pickle.load(f)

    # Import PCA model
    pca_key = '_PCA.pkl'
    pca_path = file_list[[idx for idx, s in enumerate(file_list) if pca_key in s][0]]
    with open(pca_path, "rb") as f: pca = pickle.load(f)

    # Cluster reorder indexes
    cluster_key = '_cluster_reorder_indexes.pkl'
    cluster_path = file_list[[idx for idx, s in enumerate(file_list) if cluster_key in s][0]]
    with open(cluster_path, "rb") as f: cluster_reorder_indexes = pickle.load(f)

    # Import xy_lims
    key = '_xy_lims.pkl'
    path = file_list[[idx for idx, s in enumerate(file_list) if key in s][0]]
    with open(path, "rb") as f: xy_lims = pickle.load(f)

    # Import xy_lims_PCA
    key = '_xy_lims_PCA.pkl'
    path = file_list[[idx for idx, s in enumerate(file_list) if key in s][0]]
    with open(path, "rb") as f: xy_lims_PCA = pickle.load(f)

    # Import xy_lims_RAW_DIMS
    key = '_xy_lims_RAW_DIMS.pkl'
    path = file_list[[idx for idx, s in enumerate(file_list) if key in s][0]]
    with open(path, "rb") as f: xy_lims_RAW_DIMS = pickle.load(f)

    # Import info_score_idxs
    key = '_info_score_idxs.pkl'
    path = file_list[[idx for idx, s in enumerate(file_list) if key in s][0]]
    with open(path, "rb") as f: info_score_idxs = pickle.load(f)

    return pre_PaCMAP_window_sec, pre_PaCMAP_stride_sec, PaCMAP, PaCMAP_MedDim, pca, HDBSCAN, cluster_reorder_indexes, xy_lims, xy_lims_PCA, xy_lims_RAW_DIMS, info_score_idxs

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
    atd_file='/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/data/all_time_data_01092023_112957.csv',
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


    # Debugging
    print(pat_id)

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

    if gpu_id == 0:
        print_dataset_bargraphs(pat_id, curr_fnames, curr_fnames, dataset_pic_dir)

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

def exec_kwargs(kwargs):
    # exec all kwargs in case there is python code
    for k in kwargs:
            if isinstance(kwargs[k], str):
                if kwargs[k][0:6] == 'kwargs':
                    exec(kwargs[k])

    return kwargs

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

def prepare_dataloader(dataset: Dataset, batch_size: int, droplast=True, num_workers=0):

    if num_workers > 0:
        print("WARNING: num workers >0, have experienced odd errors...")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,    
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        drop_last=droplast
    )

def initialize_directories(
        run_notes,
        cont_train_model_dir,
        pic_sub_dirs,
        pic_types,
        **kwargs):

    # *** CONTINUE EXISTING RUN initialization ***

    if kwargs['continue_existing_training']:

        kwargs['model_dir'] = cont_train_model_dir
        kwargs['pic_dataset_dir'] = kwargs['model_dir'] + '/dataset_bargraphs'

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
        kwargs['transformer_state_dict_prev_path'] = check_dir + f'/Epoch_{str(max_epoch)}/transformer_checkpoints/checkpoint_epoch{str(max_epoch)}_transformer.pt'
        kwargs['opt_transformer_state_dict_prev_path'] = check_dir + f'/Epoch_{str(max_epoch)}/transformer_checkpoints/checkpoint_epoch{str(max_epoch)}_opt_transformer.pt'

        # Set the start epoch 1 greater than max trained
        kwargs['start_epoch'] = (max_epoch + 1) 
        

    # *** NEW RUN initialization ***  

    else:
        # Make run directories
        kwargs['model_dir'] = append_timestamp(kwargs['root_save_dir'] + '/trained_models/' + kwargs['run_params_dir_name'] + '/' + run_notes + '_')
        os.makedirs(kwargs['model_dir'])
        kwargs['pic_dataset_dir'] = kwargs['model_dir'] + '/dataset_bargraphs'
        os.makedirs(kwargs['pic_dataset_dir'])

        # Fresh run 
        kwargs['start_epoch'] = 0

    return kwargs

def run_setup(**kwargs):
    # Print to console
    print("\n\n***************** MAIN START " + datetime.datetime.now().strftime("%I:%M%p-%B/%d/%Y") + "******************\n\n")        
    os.environ['KMP_DUPLICATE_LIB_OK']='True'    
    mp.set_start_method('spawn', force=True)
    mp_lock = mp.Lock()

    # All Time Data file to get event timestamps
    kwargs['root_save_dir'] = assemble_model_save_path(**kwargs)
    # kwargs['data_dir'] = kwargs['root_save_dir'] + kwargs['data_dir_subfolder']
    kwargs['tmp_model_dir'] = os.getcwd() + kwargs['tmp_file_dir']                             
    kwargs['run_params_dir_name'] = get_training_dir_name(**kwargs)

    # Set world size to number of GPUs in system available to CUDA
    world_size = torch.cuda.device_count()
        
    # Random animal name
    run_notes = random_animal(**kwargs) 

    # Call the initialization script to start new run or continue existing run
    kwargs = initialize_directories(run_notes=run_notes, **kwargs)

    # Print the model forward pass sizes
    fake_data = torch.rand(kwargs['wdecode_batch_size'], 199, kwargs['autoencode_samples']) # 199 is just an example of number of patient channels
    print_models_flow(x=fake_data, **kwargs)

    # Get the timestamp ID for this run (will be used to resume wandb logging if this is a restarted training)
    s = kwargs['model_dir'].split("/")[-1]
    kwargs['timestamp_id'] = ''.join(map(str, s))
    kwargs['run_name'] = '_'.join(map(str,s.split('_')[0:2]))

    return world_size, kwargs