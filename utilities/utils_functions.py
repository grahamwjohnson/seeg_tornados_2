# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 22:05:34 2023

@author: grahamwjohnson
"""

import time
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

# exec all kwargs in case there is python code
def exec_kwargs(kwargs):

    for k in kwargs:
            if isinstance(kwargs[k], str):
                if kwargs[k][0:6] == 'kwargs':
                    exec(kwargs[k])

    return kwargs

def rgbtoint32(rgb):
    rgb = np.array(rgb*256, dtype=int)
    color = 0
    for c in rgb[::-1]:
        color = (color<<8) + c
        # Do not forget parenthesis.
        # color<< 8 + c is equivalent of color << (8+c)
    return color

def int32torgb(color):

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

def pacmap_latent_LBM(  
    FS, 
    latent_data_cpu,
    in_data_samples,
    decode_samples,
    start_datetimes_epoch,
    stop_datetimes_epoch,
    epoch, 
    iter, 
    b_iter,
    gpu_id, 
    win_sec, 
    stride_sec, 
    absolute_latent,
    file_name,
    savedir,
    targ_vs_out_str,
    pat_id,
    plot_dict,
    premade_PaCMAP,
    premade_PaCMAP_MedDim,
    premade_PCA,
    premade_HDBSCAN,
    xy_lims,
    info_score_idxs,
    xy_lims_RAW_DIMS,
    xy_lims_PCA,
    interictal_contour=False,
    **kwargs):

    WIN_STYLE='end'

    # Use premade PaCMAPs and HDBSCAN
    reducer = premade_PaCMAP
    reducer_MedDim = premade_PaCMAP_MedDim
    hdb = premade_HDBSCAN

    # Set PaCMAP verbose false
    reducer.verbose = False
    reducer_MedDim.verbose = False

    # Project data through reducer (i.e. PaCMAP)
    latent_postPaCMAP = reducer.transform(latent_data_cpu).swapaxes(0,1)
    latent_embedding_expanded = latent_expander(latent_postPaCMAP, in_data_samples, interp=True)
    
    # Project data through reducer (i.e. PaCMAP)
    latent_postPaCMAP_MedDim = reducer_MedDim.transform(latent_data_cpu) # not swapped

    # Get HDBSCAN cluster nearest labels
    hdb_labels, hdb_probabilities = hdbscan.prediction.approximate_predict(hdb, latent_postPaCMAP_MedDim)
    

    # # Destaurate according to probability of being in cluster
    # hdb_c_prob_flat = np.array([sns.desaturate(hdb_color_palette[hdb_labels_flat[i]], hdb_probabilities_flat[i])
    #         for i in range(len(hdb_labels_flat))])

    # Reshape the labels and probabilities for plotting
    hdb_labels_doubled = np.stack([hdb_labels, hdb_labels], axis=1) # Double it to work with latent expander
    hdb_probabilities_doubled = np.stack([hdb_probabilities, hdb_probabilities], axis=1) # Double it to work with latent expander
    hdb_labels_doubled_expanded = latent_expander(hdb_labels_doubled, in_data_samples, interp=False) # Do not interp cluster classes
    hdb_probabilities_doubled_expanded = latent_expander(hdb_probabilities_doubled, in_data_samples, interp=False) # Do not interp cluster classes
    # Strip second dimension (was passed as dummy to use latent expander)
    hdb_labels_expanded = hdb_labels_doubled_expanded[0:1, :]     
    hdb_probabilities_expanded = hdb_probabilities_doubled_expanded[0:1, :]     

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
    seiz_start_dt, seiz_stop_dt, seiz_types = get_pat_seiz_datetimes(pat_id)
    ax20, ax21, ax22, ax23, ax24, xy_lims = plot_latent(
        ax=ax20, 
        interCont_ax=ax21,
        seiztype_ax=ax22,
        time_ax=ax23,
        cluster_ax=ax24,
        latent_data=np.expand_dims(latent_embedding_expanded, axis=0), 
        win_style=WIN_STYLE,
        samp_freq=FS,
        start_datetimes=[start_datetimes_epoch], 
        stop_datetimes=[stop_datetimes_epoch], 
        abs_start_datetime=start_datetimes_epoch,
        abs_stop_datetime=stop_datetimes_epoch,
        win_sec=win_sec,
        stride_sec=stride_sec, 
        seiz_start_dt=seiz_start_dt, 
        seiz_stop_dt=seiz_stop_dt, 
        seiz_types=seiz_types,
        preictal_dur=plot_dict["plot_preictal_color"],
        postictal_dur=plot_dict["plot_postictal_color"],
        absolute_latent=absolute_latent,
        max_latent=False,
        plot_ictal=True,
        hdb_labels_allFiles_expanded=np.expand_dims(hdb_labels_expanded, axis=0),
        hdb_probabilities_allFiles_expanded=np.expand_dims(hdb_probabilities_expanded, axis=0),
        hdb=hdb,
        # hdb_c_prob=hdb_c_prob,  #TODO HERE &**************************************************
        # hdb_colormap=hdb_color_palette,
        xy_lims=xy_lims,
        **kwargs)        

    ax20.title.set_text('PaCMAP Latent Space: ' + str(latent_embedding_expanded.shape[1]/FS) + 
        ' second epoch, Window mean, dur/str=' + str(win_sec) + 
        '/' + str(stride_sec) +' seconds,')
    if interictal_contour:
        ax21.title.set_text('Interictal Contour (no peri-ictal data)')


    # ***** PCA PLOTTING *****
        
    print("Calculating PCA")
        
    pca = premade_PCA
    latent_PCA = pca.transform(latent_data_cpu).swapaxes(0,1)
    latent_PCA_expanded = latent_expander(latent_PCA, in_data_samples, interp=True)

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
        latent_data=np.expand_dims(latent_PCA_expanded, axis=0), 
        win_style=WIN_STYLE,
        samp_freq=FS,
        start_datetimes=[start_datetimes_epoch], 
        stop_datetimes=[stop_datetimes_epoch], 
        abs_start_datetime=start_datetimes_epoch,
        abs_stop_datetime=stop_datetimes_epoch,
        win_sec=win_sec,
        stride_sec=stride_sec, 
        seiz_start_dt=seiz_start_dt, 
        seiz_stop_dt=seiz_stop_dt, 
        seiz_types=seiz_types,
        preictal_dur=plot_dict["plot_preictal_color"],
        postictal_dur=plot_dict["plot_postictal_color"],
        absolute_latent=absolute_latent,
        max_latent=False,
        plot_ictal=True,
        hdb_labels_allFiles_expanded=np.expand_dims(hdb_labels_expanded, axis=0),
        hdb_probabilities_allFiles_expanded=np.expand_dims(hdb_probabilities_expanded, axis=0),
        hdb=hdb,
        xy_lims=xy_lims_PCA,
        **kwargs)        

    ax10.title.set_text("PCA Components 1,2")
    ax11.title.set_text('Interictal Contour (no peri-ictal data)')


    # **** INFO RAW DIM PLOTTING *****

    raw_dims_to_plot = info_score_idxs[-2:]

    # Check for dims greater than length of possible dims (WHY DOES THIS HAPPEN??)
    for dim in raw_dims_to_plot:
        if dim > latent_data_cpu.shape[1] - 1:
            raw_dims_to_plot = np.array([1, 2])
            break

    raw_dims_latent_expanded = latent_expander(latent_data_cpu[:, raw_dims_to_plot].swapaxes(0,1), in_data_samples, interp=True)

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
        latent_data=np.expand_dims(raw_dims_latent_expanded, axis=0), 
        win_style=WIN_STYLE,
        samp_freq=FS,
        start_datetimes=[start_datetimes_epoch], 
        stop_datetimes=[stop_datetimes_epoch], 
        abs_start_datetime=start_datetimes_epoch,
        abs_stop_datetime=stop_datetimes_epoch,
        win_sec=win_sec,
        stride_sec=stride_sec, 
        seiz_start_dt=seiz_start_dt, 
        seiz_stop_dt=seiz_stop_dt, 
        seiz_types=seiz_types,
        preictal_dur=plot_dict["plot_preictal_color"],
        postictal_dur=plot_dict["plot_postictal_color"],
        absolute_latent=absolute_latent,
        max_latent=False,
        plot_ictal=True,
        hdb_labels_allFiles_expanded=np.expand_dims(hdb_labels_expanded, axis=0),
        hdb_probabilities_allFiles_expanded=np.expand_dims(hdb_probabilities_expanded, axis=0),
        hdb=hdb,
        xy_lims=xy_lims_RAW_DIMS,
        **kwargs)        

    ax00.title.set_text(f'Dims [{raw_dims_to_plot[0]},{raw_dims_to_plot[1]}] Latent Space: ' + str(latent_embedding_expanded.shape[1]/FS) + 
        ' second epoch, Window mean, dur/str=' + str(win_sec) + 
        '/' + str(stride_sec) +' seconds,' 
        )
    
    ax01.title.set_text('Interictal Contour (no peri-ictal data)')


    # **** Save entire figure *****

    fig.suptitle(file_name + create_metadata_subtitle(plot_dict))
    if not os.path.exists(savedir + '/JPEGs'): os.makedirs(savedir + '/JPEGs')
    if not os.path.exists(savedir + '/SVGs'): os.makedirs(savedir + '/SVGs')

    save_str = f"{pat_id}_latent_smoothsec{win_sec}Stride{stride_sec}_epoch{epoch}_iter{iter}_batchiter{b_iter}_{targ_vs_out_str}_gpu{gpu_id}"
    savename_jpg = savedir + f"/JPEGs/{save_str}.jpg"
    savename_svg = savedir + f"/SVGs/{save_str}.svg"
    pl.savefig(savename_jpg, dpi=300)
    pl.savefig(savename_svg)


    # Upload fig to WandB


    pl.close(fig)

    # Bundle the save metrics together
    # save_tuple = (latent_data_windowed.swapaxes(1,2), latent_PCA_allFiles, latent_topPaCMAP_allFiles, latent_topPaCMAP_MedDim_allFiles, hdb_labels_allFiles, hdb_probabilities_allFiles)

    return # save_tuple

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


    check_dir = model_dir + "/checkpoints_BSE"

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

def print_model_summary(model):
    print("Calculating model summary")
    summary(model, num_classes=1)
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = (mem_params + mem_bufs) / 1e9  # in bytes
    print("Expected GPU memory requirement (parameters + buffers): " + str(mem) +" GB")

def prepare_dataloader(dataset: Dataset, batch_size: int, droplast=True, num_workers=6):

    if num_workers > 0:
        print("WARNING: num workers >0, have experienced odd errors...")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,    #6 works
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        drop_last=droplast
    )

def get_start_stop_from_latent_path(latent_file):
    s = latent_file.split('_')[-5] + latent_file.split('_')[-4]
    abs_start_datetime = datetime.datetime(int(s[4:8]),int(s[0:2]),int(s[2:4]),int(s[8:10]),int(s[10:12]),int(s[12:14]),int(int(s[14:16])*1e4))
    s = latent_file.split('_')[-2] + latent_file.split('_')[-1].replace('.pkl','')
    abs_stop_datetime = datetime.datetime(int(s[4:8]),int(s[0:2]),int(s[2:4]),int(s[8:10]),int(s[10:12]),int(s[12:14]),int(int(s[14:16])*1e4))

    return abs_start_datetime, abs_stop_datetime

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

# if start_or_end is '0' the beginning of file timestamp is used, if '1' the end
def filename_to_dateobj(f: str, start_or_end: int):
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
                #  "\nMax Weight Dimension, Pre: " + str(np.argmax(np.abs(weight_mean[0,:]))) + ", Post: " + str(np.argmax(np.abs(weight_mean[1,:]))))

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

    return mean_zscores, std_zscores

def plot_recon(x, x_hat, plot_dict, batch_file_names, epoch, savedir, gpu_id, pat_id, iter, FS, num_rand_recon_plots, recon_sec=4, **kwargs):

    num_loops = num_rand_recon_plots

    all_starts = plot_dict['start_dt']
    all_stops = plot_dict['stop_dt']

    if gpu_id == 0: time.sleep(2) # Avoid file collision 

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
        
def get_PaCMAP_model(model_common_prefix, pre_PaCMAP_window_sec_path, pre_PaCMAP_stride_sec_path):
    # with open(model_path, "rb") as f: PaCMAP = pickle.load(f)
    PaCMAP = pacmap.load(model_common_prefix)
    with open(pre_PaCMAP_window_sec_path, "rb") as f: pre_PaCMAP_window_sec = pickle.load(f)
    with open(pre_PaCMAP_stride_sec_path, "rb") as f: pre_PaCMAP_stride_sec = pickle.load(f)

    return PaCMAP, pre_PaCMAP_window_sec, pre_PaCMAP_stride_sec

def LBM_LR_schedule(eon_wide_epoch, iter, iters_per_epoch, LR_max, LR_min, LR_epochs_TO_max, LR_epochs_AT_max, manual_gamma, manual_step_size, LR_rise_first=True, **kwargs):

    # Adjust max and min based on gamma value
    LR_gamma_iter = np.floor(eon_wide_epoch / manual_step_size)
    gamma_curr = manual_gamma ** LR_gamma_iter 
    LR_max_curr = LR_max * gamma_curr
    LR_min_curr = LR_min * gamma_curr
    LR_range = LR_max_curr - LR_min_curr

    # Get current residual
    LR_epoch_period = LR_epochs_TO_max + LR_epochs_AT_max
    LR_epoch_residual = eon_wide_epoch % LR_epoch_period

    # START with rise
    if LR_rise_first:    

        if LR_epoch_residual < LR_epochs_TO_max:
            LR_floor = LR_min_curr + ( LR_range * (LR_epoch_residual/LR_epochs_TO_max) )
            LR_ceil = LR_floor + ( LR_range * (LR_epoch_residual + 1) /LR_epochs_TO_max)
            LR_val = LR_floor + iter/iters_per_epoch * (LR_ceil - LR_floor) 

        else: 
            LR_val = LR_max_curr
        
    else:
        # LR_ceil = LR_max_curr - ( LR_range * (LR_epoch_residual/LR_epochs_cycle) )
        # LR_floor = LR_ceil - ( LR_range * (LR_epoch_residual + 1) /LR_epochs_cycle)
        # LR_val = LR_ceil - iter/iters_per_epoch * (LR_ceil - LR_floor)        
        raise Exception("ERROR: not coded up")


    return LR_val

def LR_subfunction(iter_curr, LR_min, LR_max, eon_wide_epoch, manual_gamma, manual_step_size, LR_epochs_TO_max, LR_epochs_AT_max, iters_per_epoch, LR_rise_first=True):

    # Adjust max and min based on gamma value
    LR_gamma_iter = np.floor(eon_wide_epoch / manual_step_size)
    gamma_curr = manual_gamma ** LR_gamma_iter 
    LR_max_curr = LR_max * gamma_curr
    LR_min_curr = LR_min * gamma_curr
    LR_range = LR_max_curr - LR_min_curr

    # Get current residual
    LR_epoch_period = LR_epochs_TO_max + LR_epochs_AT_max
    LR_epoch_residual = eon_wide_epoch % LR_epoch_period

    # START with rise
    if LR_rise_first:    

        if LR_epoch_residual < LR_epochs_TO_max:
            LR_floor = LR_min_curr + ( LR_range * (LR_epoch_residual/LR_epochs_TO_max) )
            LR_ceil = LR_floor + ( LR_range * (LR_epoch_residual + 1) /LR_epochs_TO_max)
            LR_val = LR_floor + iter_curr/iters_per_epoch * (LR_ceil - LR_floor) 

        else: 
            LR_val = LR_max_curr
        
    else:
        # LR_ceil = LR_max_curr - ( LR_range * (LR_epoch_residual/LR_epochs_cycle) )
        # LR_floor = LR_ceil - ( LR_range * (LR_epoch_residual + 1) /LR_epochs_cycle)
        # LR_val = LR_ceil - iter/iters_per_epoch * (LR_ceil - LR_floor)        
        raise Exception("ERROR: not coded up")


    return LR_val

def BSE_KL_LR_schedule(
        eon_wide_epoch, iter_curr, iters_per_epoch, 
        KL_max, KL_min, KL_epochs_TO_max, KL_epochs_AT_max, 
        LR_max_heads, LR_min_heads, 
        LR_max_core, LR_min_core, 
        LR_epochs_TO_max_core, LR_epochs_AT_max_core, 
        LR_epochs_TO_max_heads, LR_epochs_AT_max_heads, 
        manual_gamma_core, manual_step_size_core,
        manual_gamma_heads, manual_step_size_heads,
        KL_rise_first=True, LR_rise_first=True, **kwargs):
            
    
    # *** KL SCHEDULE ***
    
    KL_epoch_period = KL_epochs_TO_max + KL_epochs_AT_max
    KL_epoch_residual = eon_wide_epoch % KL_epoch_period

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
        eon_wide_epoch=eon_wide_epoch, 
        manual_gamma=manual_gamma_core, 
        manual_step_size=manual_step_size_core, 
        LR_epochs_TO_max=LR_epochs_TO_max_core, 
        LR_epochs_AT_max=LR_epochs_AT_max_core, 
        iters_per_epoch=iters_per_epoch,
        LR_rise_first=LR_rise_first 
    )

    # HEADS
    LR_val_heads = LR_subfunction(
        iter_curr=iter_curr,
        LR_min=LR_min_heads,
        LR_max=LR_max_heads,
        eon_wide_epoch=eon_wide_epoch, 
        manual_gamma=manual_gamma_heads, 
        manual_step_size=manual_step_size_heads, 
        LR_epochs_TO_max=LR_epochs_TO_max_heads, 
        LR_epochs_AT_max=LR_epochs_AT_max_heads, 
        iters_per_epoch=iters_per_epoch,
        LR_rise_first=LR_rise_first
    )

            
    return KL_val, LR_val_core, LR_val_heads

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
        
def load_data_tensor(filename):
    file = open(filename,'rb')
    data = pickle.load(file) 
    file.close()
    # data_channel_subset = data[0:self.num_channels,:]   
    return torch.FloatTensor(data)

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

def calc_cluster_metrics(files, start_dt, stop_dt, state_labels, stability_sec2eval, timebased_epoch_sec, perc_timebased_window_overlap, skip_noise, random_windows, overwrite_file_end_labels=True):

    # Assumes files are sorted ascending by datetime

    # Load in all files and stack by batch, seed with first file
    # Initialize variables for fasxter loading
    window_sec, stride_sec, latent_data_windowed, latent_PCA_allFiles, latent_topPaCMAP_allFiles, latent_topPaCMAP_MedDim_allFiles, hdb_labels_allFiles, hdb_probabilities_allFiles = load_inferred_pkl(files[0])
    latent_data_windowed_ALL = np.ones((len(files), latent_data_windowed.shape[1], latent_data_windowed.shape[2]))
    latent_PCA_allFiles_ALL = np.ones((len(files), latent_PCA_allFiles.shape[1], latent_PCA_allFiles.shape[2]))
    latent_topPaCMAP_allFiles_ALL = np.ones((len(files), latent_topPaCMAP_allFiles.shape[1], latent_topPaCMAP_allFiles.shape[2]))
    latent_topPaCMAP_MedDim_allFiles_ALL = np.ones((len(files), latent_topPaCMAP_MedDim_allFiles.shape[1], latent_topPaCMAP_MedDim_allFiles.shape[2]))
    hdb_labels_allFiles_ALL = np.ones((len(files), hdb_labels_allFiles.shape[1], hdb_labels_allFiles.shape[2]))
    hdb_probabilities_allFiles_ALL = np.ones((len(files), hdb_probabilities_allFiles.shape[1], hdb_probabilities_allFiles.shape[2]))

    if stability_sec2eval < stride_sec: raise Exception(f"ERROR: 'stability_sec2eval' (got {stability_sec2eval}) cannot got below stride ({stride_sec})")
    for i in range(0, len(files)):
        sys.stdout.write(f"\rLoading inferred latent spaces: File {i}/{len(files)-1}")
        sys.stdout.flush() 
        w, s, latent_data_windowed, latent_PCA_allFiles, latent_topPaCMAP_allFiles, latent_topPaCMAP_MedDim_allFiles, hdb_labels_allFiles, hdb_probabilities_allFiles = load_inferred_pkl(files[i])
        if (w != window_sec) | (s !=stride_sec): raise Exception(f"ERROR: window/stride mismatch in file, got window/stride seconds {w}/{s}, but expected {window_sec}/{stride_sec}")
        latent_data_windowed_ALL[i, :, :] = latent_data_windowed
        latent_PCA_allFiles_ALL[i, :, :] = latent_PCA_allFiles
        latent_topPaCMAP_allFiles_ALL[i, :, :] = latent_topPaCMAP_allFiles
        latent_topPaCMAP_MedDim_allFiles_ALL[i, :, :] = latent_topPaCMAP_MedDim_allFiles
        hdb_labels_allFiles_ALL[i, :, :] = hdb_labels_allFiles
        hdb_probabilities_allFiles_ALL[i, :, :] = hdb_probabilities_allFiles
        
    # Check that states found make sense with given state_labels and calculate number of states
    max_hdb_label = np.max(hdb_labels_allFiles_ALL)
    min_hdb_label = np.min(hdb_labels_allFiles_ALL)
    if (max_hdb_label > max(state_labels)) | (min_hdb_label < min(state_labels)): raise Exception(f"Found hdb labels outside of expected range. Got min/max of {min_hdb_label}/{max_hdb_label}, but expected {min(state_labels)}/{max(state_labels)}")
    num_states = len(state_labels) # label of -1 is always noise as per HDBSCAN construction
    
    # Fill all MASTER variables
    # Construct a master state one-hot matrix with timesteps of stridce that spans all time possioble for this epoch
    # Missing timepoints will be labels as -99
    # Values align with the end of a window, thus the beginning will be null values
    # WARNING: assumes files are ordered to save compute time
    if overwrite_file_end_labels:
        first_dt = start_dt[0]; last_dt = stop_dt[-1]; num_seconds = int(round((last_dt - first_dt).total_seconds(), 2))
        num_timepoints = int((num_seconds - window_sec) / stride_sec) + 1
        master_state_dt_skeleton = [first_dt + datetime.timedelta(seconds=stride_sec * i) for i in range(num_timepoints)]
        master_state_dt = [-99] * num_timepoints
        
        master_state_onehot = np.ones(shape=(num_states, num_timepoints))* -99
        master_full_latent = np.ones(shape=(latent_data_windowed_ALL.shape[1], num_timepoints)) * -99
        master_PCA_latent = np.ones(shape=(latent_PCA_allFiles_ALL.shape[1], num_timepoints)) * -99
        master_pacmap2_latent = np.ones(shape=(latent_topPaCMAP_allFiles_ALL.shape[1], num_timepoints)) * -99
        master_pacmapMedDim_latent = np.ones(shape=(latent_topPaCMAP_MedDim_allFiles_ALL.shape[1], num_timepoints)) * -99
        master_HDBSCAN_labels = np.ones(shape=(hdb_labels_allFiles_ALL.shape[1], num_timepoints)) * -99
        master_HDBSCAN_probs = np.ones(shape=(hdb_probabilities_allFiles_ALL.shape[1], num_timepoints)) * -99
        
        last_skeleton_idx = 0
        print('')
        for i in range(len(files)):
            skeleton_dt_found = False # flag to indicate that slot for file was found 
            curr_start_dt = start_dt[i]
            sys.stdout.write(f"\rFilling Master One-Hot Matrix: File {i}/{len(files)-1}              ")
            sys.stdout.flush() 

            # find the closest datetime index in skeleton
            for j in range(last_skeleton_idx, num_timepoints):
                
                # Find first skeleton index where current dt exceeds skeleton
                if master_state_dt_skeleton[j] >= (curr_start_dt - datetime.timedelta(seconds=stride_sec/2)):
                    skeleton_discrepency_sec = (curr_start_dt - master_state_dt_skeleton[j]).total_seconds()
                    if abs(skeleton_discrepency_sec) > stride_sec/2: raise Exception(
                        f"ERROR in datetime skelton search, discrepency of {skeleton_discrepency_sec}, expected strides of {stride_sec} with max discpency being {stride_sec/2}")
                    skeleton_dt_found = True
                    skeleton_start = j
                    for k in range(hdb_labels_allFiles_ALL.shape[2]):
                        master_idx_curr = skeleton_start + k
                        label_curr = hdb_labels_allFiles_ALL[i, 0, k]
                        master_state_dt[master_idx_curr] = curr_start_dt + datetime.timedelta(seconds=stride_sec * k)
                        
                        # Fill Master Onehot
                        master_state_onehot[:, master_idx_curr] = 0 # initialize this timepoint to all zeros (away from -99) to indicate that data is present
                        label_curr_idx = np.where(state_labels == label_curr)
                        master_state_onehot[label_curr_idx, master_idx_curr] = 1 

                        # Fill the rest of the master variables
                        master_full_latent[:, master_idx_curr] = latent_data_windowed_ALL[i, :, k]
                        master_PCA_latent[:, master_idx_curr] = latent_PCA_allFiles_ALL[i, :, k]
                        master_pacmap2_latent[:, master_idx_curr] = latent_topPaCMAP_allFiles_ALL[i, :, k]
                        master_pacmapMedDim_latent[:, master_idx_curr] = latent_topPaCMAP_MedDim_allFiles_ALL[i, :, k]
                        master_HDBSCAN_labels[:, master_idx_curr] = hdb_labels_allFiles_ALL[i, :, k]
                        master_HDBSCAN_probs[:, master_idx_curr] = hdb_probabilities_allFiles_ALL[i, :, k]

                    # GO BACK A WHOLE FILE'S worth of indexes due to presumed file overlap. 
                    #Assign next idx as next loops first idx to search, assumes time ordered files
                    last_skeleton_idx = master_idx_curr - k - 1
                    if last_skeleton_idx < 0: last_skeleton_idx = 0
                    break

            if not skeleton_dt_found: raise Exception(f"ERROR: skeleton dt index never found for file: {files[i]}")
                        
    else: raise Exception("Only coded up to handle overlapping files by overwriting the ends of previous file")

    fraction_unfilled = np.count_nonzero(master_state_onehot == -99)/master_state_onehot.size
    if fraction_unfilled > 0.1: raise Exception(f"Percent of master_onehot_matrix == -99 is {fraction_unfilled*100}%")

    
    # *** Find Cluster Visit Events for Each State ***

    print('')
    visit_tupels = [calc_cluster_visit_tuple(
        i, 
        onehot_array_curr=master_state_onehot[i, :], 
        master_state_dt=master_state_dt, 
        master_HDBSCAN_labels=master_HDBSCAN_labels, 
        num_states=num_states, 
        num_timepoints=num_timepoints,
        state_labels=state_labels
        ) for i in range(num_states)]
    
    allstate_cluster_visits_start_dt = [visit_tupels[i][0] for i in range(num_states)]
    allstate_cluster_visits_stop_dt = [visit_tupels[i][1] for i in range(num_states)]
    allstate_cluster_visits_start_idx = [visit_tupels[i][2] for i in range(num_states)]
    allstate_cluster_visits_stop_idx = [visit_tupels[i][3] for i in range(num_states)]


    # *** State-Specific Metrics ***

    # Now that all cluster visit events have been found, can calculate metrics for each event
    # Calculate the state stability for the evaluation period desired
    # I.e. how often in an X second period does this state change state (can decide to to ignore the noise state, -1, with 'skip noise'=True)
    print('')
    state_tuples = [calc_state_metric_tuple(
        i,
        state_labels=state_labels[i],
        num_states=num_states, 
        master_HDBSCAN_labels=master_HDBSCAN_labels,
        allstate_cluster_visits_start_dt=allstate_cluster_visits_start_dt[i], 
        allstate_cluster_visits_stop_dt=allstate_cluster_visits_stop_dt[i],
        allstate_cluster_visits_start_idx=allstate_cluster_visits_start_idx[i],
        allstate_cluster_visits_stop_idx=allstate_cluster_visits_stop_idx[i],
        num_timepoints=num_timepoints,
        stability_sec2eval=stability_sec2eval
    ) for i in range(num_states)]

    # Visit-level
    allstate_cluster_visit_timeincluster = [state_tuples[i][0] for i in range(num_states)]
    allstate_cluster_visit_numvisits = [state_tuples[i][1] for i in range(num_states)]
    allstate_cluster_visit_prev_clusters = [state_tuples[i][2] for i in range(num_states)]
    allstate_cluster_visit_next_clusters = [state_tuples[i][3] for i in range(num_states)]

    # State-level
    total_time_in_state = [state_tuples[i][4] for i in range(num_states)]
    mean_time_in_state = [state_tuples[i][5] for i in range(num_states)]
    std_time_in_state = [state_tuples[i][6] for i in range(num_states)]
    mode_prev_cluster = [state_tuples[i][7] for i in range(num_states)]
    mode_next_cluster = [state_tuples[i][8] for i in range(num_states)]
    mode_prev_cluster_fraction = [state_tuples[i][9] for i in range(num_states)]
    mode_next_cluster_fraction = [state_tuples[i][10] for i in range(num_states)]
    state_stability = [state_tuples[i][11] for i in range(num_states)]


    # Time-based metrics
    print(f"\nCalculating time-based metrics for sliding epoch duratioin of {timebased_epoch_sec}")
    timebased_tuple = calc_timebased_cluster_metric(
        master_state_onehot=master_state_onehot,
        timebased_epoch_sec=timebased_epoch_sec,
        original_stride=stride_sec,
        perc_timebased_window_overlap=perc_timebased_window_overlap,
        state_labels=state_labels,
        random_windows=random_windows)
    
    num_windows_analyzed = timebased_tuple[0]

    total_num_state_transitions = timebased_tuple[1]
    mean_state_transitions = timebased_tuple[2]
    std_state_transitions = timebased_tuple[3]
    ci95_state_transitions = timebased_tuple[4]
    
    total_num_different_clusters = timebased_tuple[5]
    mean_num_different_clusters = timebased_tuple[6]
    std_different_clusters = timebased_tuple[7]
    ci95_different_clusters = timebased_tuple[8]
    
    
    # Assemble the output disct
    cluster_metric_dicts = {
        'master_data_arrays':{
            'num_timepoints': num_timepoints,
            'master_state_dt': master_state_dt,
            'master_state_onehot': master_state_onehot,
            'master_full_latent': master_full_latent,
            'master_PCA_latent': master_PCA_latent,
            'master_pacmap2_latent': master_pacmap2_latent,
            'master_pacmapMedDim_latent': master_pacmapMedDim_latent,
            'master_HDBSCAN_labels': master_HDBSCAN_labels,
            'master_HDBSCAN_probs': master_HDBSCAN_probs
        },
        'state_level':{
            'state_stability': state_stability, 
            'total_time_in_state': total_time_in_state,
            'mean_time_in_state': mean_time_in_state,
            'std_time_in_state': std_time_in_state,
            'mode_prev_cluster': mode_prev_cluster,
            'mode_prev_cluster_fraction': mode_prev_cluster_fraction,
            'mode_next_cluster': mode_next_cluster,
            'mode_next_cluster_fraction': mode_next_cluster_fraction

        },
        'visit_level':{
            'visit_numvisits': allstate_cluster_visit_numvisits,
            'visit_time_in_cluster': allstate_cluster_visit_timeincluster,
            'visit_prev_cluster_label': allstate_cluster_visit_prev_clusters,
            'visit_next_cluster_label': allstate_cluster_visit_next_clusters,
        },
        'timebased':{
            'num_windows_analyzed' : num_windows_analyzed,

            'total_num_state_transitions': total_num_state_transitions,
            'mean_state_transitions': mean_state_transitions,
            'std_state_transitions' : std_state_transitions,
            'ci95_state_transitions' : ci95_state_transitions,

            'total_num_different_clusters' : total_num_different_clusters,
            'mean_num_different_clusters' : mean_num_different_clusters,
            'std_different_clusters' : std_different_clusters,
            'ci95_different_clusters' : ci95_different_clusters
        }
    } 

    return cluster_metric_dicts

def calc_cluster_visit_tuple(i, onehot_array_curr, master_state_dt, master_HDBSCAN_labels, num_states, num_timepoints, state_labels):

    sys.stdout.write(f"\rFinding Cluster Visit Events: State {i}/{num_states - 1}              ")
    sys.stdout.flush() 

    cluster_visits_start_dt = []
    cluster_visits_stop_dt = []
    cluster_visits_start_idx = []
    cluster_visits_stop_idx = []
    done = False; 
    if state_labels[i] == -1: done = True # skip the noise label
    time_idx_curr = 0
    while not done:
        # If start of a cluster visit found
        if onehot_array_curr[time_idx_curr] == 1:
            cluster_visits_start_dt = cluster_visits_start_dt + [master_state_dt[time_idx_curr]]
            cluster_visits_start_idx = cluster_visits_start_idx + [time_idx_curr]

            # Determine how long the cluster visit is
            still_in_cluster = True
            time_idx_curr = time_idx_curr + 1
            while still_in_cluster:
                # If cluster is exited and not because of just going into noise label, then close the cluster visit
                if ((onehot_array_curr[time_idx_curr] == 0) & (master_HDBSCAN_labels[0, time_idx_curr] != -1)) | (time_idx_curr == num_timepoints - 1):
                    # Backtrack to last NON noise label
                    backtrack_finished = False
                    back_idx = 0
                    while not backtrack_finished:
                        if (onehot_array_curr[time_idx_curr - back_idx] == 1):
                            backtrack_finished = True
                            cluster_visits_stop_dt = cluster_visits_stop_dt + [master_state_dt[time_idx_curr - back_idx]]
                            cluster_visits_stop_idx = cluster_visits_stop_idx + [time_idx_curr - back_idx]

                        if not backtrack_finished: back_idx = back_idx + 1

                    still_in_cluster = False

                if still_in_cluster: time_idx_curr = time_idx_curr + 1

        # Check if at end of time epoch
        if time_idx_curr == num_timepoints - 1:
            done = True

        # Iterate for next while loop
        time_idx_curr = time_idx_curr + 1

    visit_tuple = (cluster_visits_start_dt, cluster_visits_stop_dt, cluster_visits_start_idx, cluster_visits_stop_idx)
    return visit_tuple

def calc_state_metric_tuple( # Already indexed in function call, so no 'i' in this function
        i,
        state_labels, 
        num_states, 
        master_HDBSCAN_labels, 
        allstate_cluster_visits_start_dt, 
        allstate_cluster_visits_stop_dt,
        allstate_cluster_visits_start_idx,
        allstate_cluster_visits_stop_idx,
        num_timepoints,
        stability_sec2eval
        ):
        
    sys.stdout.write(f"\rCalculating State Metrics: State {i}/{num_states - 1}              ")
    sys.stdout.flush() 

    # initialize variables incase calc is skipped
    visit_times = [] # 0 output tuple index
    num_visits = -99 # 1
    visit_prev_cluster_label = []  # 2
    visit_next_cluster_label = [] # 3

    total_time_in_state = -99 # 4
    mean_time_in_state = -99 # 5 
    std_time_in_state = -99 # 6
    mode_prev_cluster = -99 # 7
    mode_next_cluster = -99 # 8
    mode_prev_cluster_fraction = -99 # 9
    mode_next_cluster_fraction = -99 # 10
    state_stability = -99 # 11


    if state_labels != -1: # skip the noise

        # Get the time in the cluster for each cluster visit
        num_visits = len(allstate_cluster_visits_start_dt)
        visit_times = np.ones(shape=num_visits) * -99
        visit_prev_cluster_label =  np.ones(shape=num_visits, dtype=int) * -99
        visit_next_cluster_label =  np.ones(shape=num_visits, dtype=int) * -99
        for j in range(num_visits):
            # How long were we in cluster
            visit_times[j] = (allstate_cluster_visits_stop_dt[j] - allstate_cluster_visits_start_dt[j]).total_seconds()
            
            # Determine the previous cluster and next cluster for each visit
            visit_entry_idx = allstate_cluster_visits_start_idx[j]
            prev_found = False
            back_idx = 1
            while not prev_found:
                if visit_entry_idx - back_idx < 0:
                    visit_prev_cluster_label[j] = -99
                    prev_found = True
                elif master_HDBSCAN_labels[0, visit_entry_idx - back_idx] > -1: # not noise (-1) or missing data (-99)
                    prev_found = True
                    visit_prev_cluster_label[j] = int(master_HDBSCAN_labels[0, visit_entry_idx - back_idx])
                back_idx = back_idx + 1

            visit_exit_idx = allstate_cluster_visits_stop_idx[j]
            next_found = False
            forw_idx = 1
            while not next_found:
                if visit_exit_idx + forw_idx >= num_timepoints:
                    visit_next_cluster_label[j] = -99
                    next_found = True
                elif master_HDBSCAN_labels[0, visit_exit_idx + forw_idx] > -1: # not noise (-1) or missing data (-99)
                    next_found = True
                    visit_next_cluster_label[j] = int(master_HDBSCAN_labels[0, visit_exit_idx + forw_idx])
                forw_idx = forw_idx + 1

        # State-level summary metrics
        total_time_in_state = np.sum(visit_times)
        mean_time_in_state = np.mean(visit_times)
        std_time_in_state = np.std(visit_times)

        if visit_prev_cluster_label.size == 0:
            mode_prev_cluster = -99; mode_prev_cluster_fraction = np.nan
            mode_next_cluster = -99; mode_next_cluster_fraction = np.nan
        else:
            u, c = np.unique(visit_prev_cluster_label, return_counts=True); mode_prev_cluster = u[c.argmax()]; mode_prev_cluster_fraction = np.max(c)/np.sum(c)
            u, c = np.unique(visit_next_cluster_label, return_counts=True); mode_next_cluster = u[c.argmax()]; mode_next_cluster_fraction = np.max(c)/np.sum(c)
        

        # *** State stability: how likely is this state to change in X seconds ***

        if total_time_in_state == 0: # cannot evaluate stability for these extremely rare/unstable clusters
            state_stability = np.nan
        
        else:
            # Randomly sample the times spent in this state
            sample_iters = 1000
            transition_boolean = np.ones(sample_iters) * -99
            for j in range(sample_iters):
                # Find a random time that state was in this cluster
                random_seconds = np.random.uniform(0, total_time_in_state)
                
                # Find timepoint in cluster visit that this random sampling represents
                cum_visit_seconds = 0
                for k in range(len(visit_times)):
                    cum_visit_seconds = cum_visit_seconds + visit_times[k]
                    if cum_visit_seconds > random_seconds: # sampled into this cluster
                        if (random_seconds + stability_sec2eval ) > cum_visit_seconds: # exited sampled cluster
                            transition_boolean[j] = 1
                        else:
                            transition_boolean[j] = 0
            
            state_stability = 1 - np.sum(transition_boolean) / sample_iters
    
    state_tuple = (
        visit_times, 
        num_visits,
        visit_prev_cluster_label, 
        visit_next_cluster_label, 
        total_time_in_state, 
        mean_time_in_state, 
        std_time_in_state,
        mode_prev_cluster,
        mode_next_cluster,
        mode_prev_cluster_fraction,
        mode_next_cluster_fraction,
        state_stability)
                   
    return state_tuple

def calc_timebased_cluster_metric(master_state_onehot, timebased_epoch_sec, original_stride, perc_timebased_window_overlap, state_labels, random_windows):

    """

    Args:
        random_windows: if > 0, then will only return this number of random windows from epoch

    """

    
    # Find the noise idx
    noise_idx = int(np.where(state_labels == -1)[0])

    idx_range = int(np.ceil(timebased_epoch_sec / original_stride))

    # For total metrics
    noise_time_idxs_all = np.where(master_state_onehot[noise_idx])[0]
    null_idxs = np.where(master_state_onehot[noise_idx] == -99)[0]
    all_void_idxs = np.concatenate([noise_time_idxs_all, null_idxs])
            
    # Initialize the output metrics
    total_num_state_transitions = find_specified_number_motif_2D(array=np.delete(master_state_onehot, all_void_idxs, axis=1), motif=[1,0], verbose=True)
    mean_state_transitions = -99
    std_state_transitions = -99
    ci95_state_transitions = -99
    total_num_different_clusters = int(np.sum(np.max(np.delete(master_state_onehot, noise_idx, axis=0), axis=1)))
    mean_num_different_clusters = -99
    std_different_clusters = -99
    ci95_different_clusters = -99

    # Tmp vars
    state_transitions_list = []
    numdiff_clusters_list = []

    done = False
    curr_stride_idx = 0
    num_windows_analyzed = 0
    print('')
    while not done:
        
        if curr_stride_idx%100 == 0:
            sys.stdout.write(f"\rTimebase metric: Current stride idx {curr_stride_idx}/{master_state_onehot.shape[1] - idx_range - 1}              ")
            sys.stdout.flush() 

        onehot_subsection = master_state_onehot[:, curr_stride_idx: curr_stride_idx + idx_range]

        if -99 not in onehot_subsection: # skip if data is missing
            

            # For this purpose can collapse/delete any timepoints that cluster identity is noise - makes calculations easier
            noise_time_idxs = np.where(onehot_subsection[noise_idx])[0]

            if len(noise_time_idxs) < idx_range: # skip if all noise

                num_windows_analyzed = num_windows_analyzed + 1
                onehot_subsection = np.delete(onehot_subsection, noise_time_idxs, axis=1)

                # Update the mean number of different clusters present in this epoch
                curr_num_diff_clusters = [np.sum(np.max(onehot_subsection, axis=1))]
                # mean_num_different_clusters = mean_num_different_clusters * ((num_windows_analyzed-1)/(num_windows_analyzed)) + curr_num_diff_clusters * (1/num_windows_analyzed)
                numdiff_clusters_list = numdiff_clusters_list + curr_num_diff_clusters

                # Update the mean number of transitions
                curr_num_state_transitions = [find_specified_number_motif_2D(array=onehot_subsection, motif=[1,0], verbose=False)]
                # mean_state_transitions = mean_state_transitions * ((num_windows_analyzed-1)/(num_windows_analyzed)) + curr_num_state_transitions * (1/num_windows_analyzed)
                state_transitions_list = state_transitions_list + curr_num_state_transitions

        # Advance the loop
        if random_windows > 0:
            if num_windows_analyzed == random_windows: done = True
            else:
                curr_stride_idx = int(np.random.randint(low=0, high=(master_state_onehot.shape[1] - 1 - idx_range), size=1))
        else:
            curr_stride_idx = curr_stride_idx + int(idx_range * (1-perc_timebased_window_overlap))

        # Check for end of master_onehot duration
        if (curr_stride_idx + idx_range) > master_state_onehot.shape[1] - 1: done = True


    # Calculate the mean and standard deviation
    mean_num_different_clusters = np.mean(numdiff_clusters_list)
    std_different_clusters = np.std(numdiff_clusters_list)
    ci95_different_clusters = 1.96 * std_different_clusters / np.sqrt(len(numdiff_clusters_list))
    
    mean_state_transitions = np.mean(state_transitions_list)
    std_state_transitions = np.std(state_transitions_list)
    ci95_state_transitions = 1.96 * std_state_transitions / np.sqrt(len(state_transitions_list))

    timebased_tuple = (
        num_windows_analyzed, # 0

        total_num_state_transitions, # 1
        mean_state_transitions,  # 2
        std_state_transitions, # 3
        ci95_state_transitions, # 4

        total_num_different_clusters, # 5
        mean_num_different_clusters,  # 6
        std_different_clusters, # 7
        ci95_different_clusters # 8
        ) 
    
    return timebased_tuple

def get_sorted_datetimes_from_files(files):

    start_dt, stop_dt = filename_to_datetimes([files[i].split("/")[-1] for i in range(len(files))])
    sort_idxs = [i[0] for i in sorted(enumerate(start_dt), key=lambda x:x[1])]
    files_sorted = [files[sort_idxs[i]] for i in range(len(sort_idxs))]; 
    start_dt_sorted = [start_dt[sort_idxs[i]] for i in range(len(sort_idxs))]; 
    stop_dt_sorted = [stop_dt[sort_idxs[i]] for i in range(len(sort_idxs))]; 

    return files_sorted, start_dt_sorted, stop_dt_sorted

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

def flatten(list_of_lists):
  return [item for sublist in list_of_lists for item in sublist]

def delete_old_checkpoints(dir: str, curr_epoch: int):

    SAVE_KEYWORDS = ["hdbscan", "PaCMAP"]

    all_files = glob.glob(f"{dir}/*")

    # Get the epoch number for each file
    epoch = flatten([[int(f.split("/")[-1].split("_")[i].replace("epoch", "").split(".")[0]) for i in range(len(f.split("/")[-1].split("_"))) if "epoch" in f.split("/")[-1].split("_")[i]] for f in all_files])
    if len(epoch) != len(all_files): raise Exception(f"ERROR: epoch list length {len(epoch)} does not match length of all_files {len(all_files)} in checkpoint folder, make sure keyword 'epoch<num>' is in filename")

    # Add current epoch to save files
    save_epochs = [curr_epoch]
    unique_epochs = np.unique(epoch)
    for i in range(len(unique_epochs)):
        epoch_files = [f.split("/")[-1] for f in glob.glob(f"{dir}/*_epoch{unique_epochs[i]}_*")]
        for f in epoch_files:
            if any(substr in f for substr in SAVE_KEYWORDS):
                save_epochs.append(unique_epochs[i])
                break

    [os.remove(all_files[i]) if epoch[i] not in save_epochs else print(f"saved: {all_files[i].split('/')[-1]}") for i in range(len(all_files))]

    return

def delete_old_checkpoints_BSE(dir: str, curr_epoch: int):

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

def initialize_run(
        run_notes,
        cont_train_model_dir,
        pic_sub_dirs,
        pic_types,
        **kwargs):
    
    # *** ONLY INFERENCE initialization ***

    if kwargs['run_inference_now']:
        kwargs['model_dir'] = cont_train_model_dir
        kwargs['pic_save_dir'] = kwargs['model_dir'] + '/latent_snapshots_BSE'
        kwargs['pic_dataset_dir'] = kwargs['model_dir'] + '/dataset_bargraphs_BSE'        
        start_eon = int((kwargs['epoch_used_for_inference'])/kwargs['epochs_per_eon'])
        start_epoch = kwargs['epoch_used_for_inference'] % kwargs['epochs_per_eon']

        # Check that designated eon has proper dim reduction and clustering models
        if not epoch_contains_pacmap_hdbscan(model_dir=kwargs['model_dir'], epoch=kwargs['epoch_used_for_inference']):
            raise Exception(f"Eon {start_eon} does NOT contain all files for inference with: 2D PacMap, MedDim PaCMAP, HDBSCAN ... etc. Ensure validation has been run on this eon.")
        

    # *** CONTINUE EXISTING RUN initialization ***

    elif kwargs['continue_existing_training']:

        raise Exception("ERROR: NEED TO CODE UP LBM weight load")

        kwargs['model_dir'] = cont_train_model_dir
        kwargs['pic_save_dir'] = kwargs['model_dir'] + '/latent_snapshots_BSE'
        kwargs['pic_dataset_dir'] = kwargs['model_dir'] + '/dataset_bargraphs_BSE'

        # Find the eon to start training
        check_dir = kwargs['model_dir'] + "/checkpoints_BSE"
        epoch_dirs = glob.glob(check_dir + '/Epoch*')
        epoch_nums = [int(f.split("/")[-1].replace("Epoch_","")) for f in epoch_dirs]

        # Find the highest epoch already trained
        max_epoch = max(epoch_nums)
        print(f"Resuming training after saved epoch: {str(max_epoch)}")
        
        # Construct the proper file names to get CORE state dicts
        kwargs['core_state_dict_prev_path'] = check_dir + f'/Epoch_{str(max_epoch)}/core_checkpoints/checkpoint_epoch{str(max_epoch)}_vaecore.pt'
        kwargs['core_opt_state_dict_prev_path'] = check_dir + f'/Epoch_{str(max_epoch)}/core_checkpoints/checkpoint_epoch{str(max_epoch)}_vaecore_opt.pt'
        kwargs['heads_prev_dir'] = check_dir + f'/Epoch_{str(max_epoch)}/heads_checkpoints'
        
        # Set the start epoch 1 greater than max trained, i.e. next eon
        start_eon = int((max_epoch + 1)/kwargs['epochs_per_eon'])
        start_epoch = (max_epoch + 1) % kwargs['epochs_per_eon'] 
        

    # *** NEW RUN initialization ***  

    else:
        # Make run directories
        kwargs['model_dir'] = append_timestamp(kwargs['root_save_dir'] + '/trained_models/' + kwargs['run_params_dir_name'] + '/' + run_notes + '_')
        os.makedirs(kwargs['model_dir'])
        kwargs['pic_save_dir'] = kwargs['model_dir'] + '/latent_snapshots_BSE'
        os.makedirs(kwargs['pic_save_dir'])
        kwargs['pic_dataset_dir'] = kwargs['model_dir'] + '/dataset_bargraphs_BSE'
        os.makedirs(kwargs['pic_dataset_dir'])
        [os.makedirs(f"{kwargs['pic_save_dir']}/{pic_sub_dirs[i]}/{ptype}") for i in range(0, len(pic_sub_dirs)) for ptype in pic_types]

        # # Copy the code directory to the root save dir, but ignore TB run folder
        # print("Copying code folder")
        # shutil.copytree(os.getcwd(), f"{kwargs['model_dir']}/code_used", ignore = shutil.ignore_patterns('runs/', 'old_runs/', '*pycache*', 'tmp_files/', 'tmp_files_LSE', '*.pt', '*.conda', '*.vscode'))
        shutil.copyfile(f"{os.getcwd()}/config_train_BSE.yml", f"{kwargs['model_dir']}/config_train_BSE.yml")

        # Hyperparameter checks
        if kwargs['mini_batch_stride'] > kwargs['mini_batch_window_size']: raise Exception("Stride must be same or smaller than window size")

        # Fresh run starts at eon 0
        start_eon = 0
        start_epoch = 0

    return start_eon, start_epoch, kwargs


