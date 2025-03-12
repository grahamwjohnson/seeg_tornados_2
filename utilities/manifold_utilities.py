import os
# import umap
import pacmap
import parampacmap
import hdbscan
import matplotlib.pylab as pl
pl.switch_backend('agg')
# import phate
import datetime
import csv
import matplotlib.gridspec as gridspec
from .latent_plotting import plot_latent
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from minisom import MiniSom
from SOM import SOM
import sys
import torch
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

'''
@author: grahamwjohnson

Seperate utilities repository from utils_functions.py to enable a slimmer/simpler conda env for manifolds.py

'''

# def phate_subfunction(  
#     atd_file,
#     pat_ids_list,
#     latent_data_windowed, 
#     start_datetimes_epoch,  
#     stop_datetimes_epoch,
#     epoch, 
#     FS, 
#     win_sec, 
#     stride_sec, 
#     savedir,
#     HDBSCAN_min_cluster_size,
#     HDBSCAN_min_samples,
#     plot_preictal_color_sec,
#     plot_postictal_color_sec,
#     interictal_contour=False,
#     verbose=True,
#     knn=5,
#     decay=15,  
#     phate_metric='cosine',
#     phate_solver='smacof',
#     apply_pca_phate=True,
#     pca_comp_phate = 100,
#     xy_lims = [],
#     custom_nn_bool = False,
#     phate_annoy_tree_size = 20,
#     knn_indices = [],
#     knn_distances = [],
#     premade_PHATE = [],
#     premade_HDBSCAN = [],
#     plot_pat_ids = [],
#     store_nn = True,
#     **kwargs):

#     '''
#     Goal of function:
#     Run PHATE and plot

#     '''

#     # Metadata
#     latent_dim = latent_data_windowed[0].shape[1]
#     num_timepoints_in_windowed_file = latent_data_windowed[0].shape[0]
#     modified_FS = 1 / stride_sec

#     # Check for NaNs in files
#     delete_file_idxs = []
#     for i in range(len(latent_data_windowed)):
#         if np.sum(np.isnan(latent_data_windowed[i])) > 0:
#             delete_file_idxs = delete_file_idxs + [i]
#             print(f"WARNING: Deleted file {start_datetimes_epoch[i]} that had NaNs")

#     # Delete entries/files in lists where there is NaN in latent space for that file
#     latent_data_windowed = [item for i, item in enumerate(latent_data_windowed) if i not in delete_file_idxs]
#     start_datetimes_epoch = [item for i, item in enumerate(start_datetimes_epoch) if i not in delete_file_idxs]  
#     stop_datetimes_epoch = [item for i, item in enumerate(stop_datetimes_epoch) if i not in delete_file_idxs]
#     pat_ids_list = [item for i, item in enumerate(pat_ids_list) if i not in delete_file_idxs]

#     # Generate numerical IDs for each unique patient, and give each datapoint an ID
#     unique_ids = list(set(pat_ids_list))
#     id_to_index = {id: idx for idx, id in enumerate(unique_ids)}  # Create mapping dictionary
#     pat_idxs = [id_to_index[id] for id in pat_ids_list]
#     pat_idxs_expanded = [item for item in pat_idxs for _ in range(latent_data_windowed[0].shape[0])]

    
#     ### PHATE ###

#     # Flatten data into [miniepoch, dim] to feed into PHATE, original data is [file, seq_miniepoch_in_file, latent_dim]
#     latent_PHATE_input = np.concatenate(latent_data_windowed, axis=0)

#     # No PHATE object passed in, make new one
#     if premade_PHATE == []:

#         # Custom NN search
#         if custom_nn_bool:
#             print("*** Custom *** PHATE NN Search")

#             # Custom NN to exclude intra-patient points
#             # Compute inter-patient k-NN graph using Annoy
#             if len(knn_indices) == 0 or len(knn_distances) == 0: # if either is epmty, then need to compute
#                 print("Computing new NN, no precomputed values given")

#                 # PCA
#                 if apply_pca_phate:
#                     print(f"Applying PCA, new dimension of data is {pca_comp_phate}")
#                     pca = PCA(n_components=pca_comp_phate, svd_solver='full') # Different than PCA used for PaCMAP
#                     latent_PHATE_input = pca.fit_transform(latent_PHATE_input) # Overwrite the data with PCA version

#                 # Now find NN indices and distances 
#                 knn_indices, knn_distances = inter_patient_knn_annoy(latent_PHATE_input, phate_annoy_tree_size, pat_idxs_expanded, knn, metric=phate_metric)

#                 # Save the NN indices and dists  
#                 if store_nn:
#                     if not os.path.exists(savedir + '/nn_pickles'): os.makedirs(savedir + '/nn_pickles')
#                     savename_root = savedir + f"/nn_pickles/Window{win_sec}_Stride{stride_sec}_epoch{epoch}_{phate_metric}_knn{knn}"
#                     savename_ind = savename_root + "_KNN_INDICES.pkl"
#                     with open(savename_ind, 'wb') as handle: pickle.dump(knn_indices, handle)
#                     savename_dist = savename_root + "_KNN_DISTANCES.pkl"
#                     with open(savename_dist, 'wb') as handle: pickle.dump(knn_distances, handle)
#                     print("\nSaved ANNOY outputs")

#             else: print("Using precomputed NN")

#             # Create a sparse matrix for the KNN distances (only store nearest neighbors)
#             # Convert the KNN distances to a sparse CSR matrix (Compressed Sparse Row format)
#             rows = np.repeat(np.arange(latent_PHATE_input.shape[0]), knn)
#             cols = knn_indices.flatten()
#             values = knn_distances.flatten()
#             sparse_knn_distances = csr_matrix((values, (rows, cols)), shape=(latent_PHATE_input.shape[0], latent_PHATE_input.shape[0]))

#             # Build and fit PHATE object
#             phate_op = phate.PHATE(
#                 # pat_idxs=pat_idxs_expanded, # must now match input data
#                 knn_dist='precomputed_distance', # override the default method of detecting 
#                 knn=knn,
#                 decay=decay,
#                 mds_solver=phate_solver,
#                 n_jobs= -2
#             )

#             # Fit and transform the data using distance matrix
#             phate_output = phate_op.fit_transform(sparse_knn_distances) # pass distance matrix

#         # Default PHATE NN search
#         else:
#             print("DEFAULT PHATE NN Search")
#             # Build and fit default PHATE object
#             phate_op = phate.PHATE(
#                 knn=knn,
#                 decay=decay,
#                 mds_solver=phate_solver,
#                 n_jobs= -2)
#             phate_output = phate_op.fit_transform(latent_PHATE_input) # Pass raw data

#     # Premade PHATE object has been passed in 
#     else:
#         phate_op = premade_PHATE
#         phate_output = phate_op.transform(latent_PHATE_input)

#     # Split reduced output back into files
#     latent_postPHATE_perfile = np.stack(np.split(phate_output, len(latent_data_windowed), axis=0),axis=0)


#     ### HDBSCAN ###
#     # If training, create new cluster model, otherwise "approximate_predict()" if running on val data
#     if premade_HDBSCAN == []:
#         # Now do the clustering with HDBSCAN
#         print("Building new HDBSCAN model")
#         hdb = hdbscan.HDBSCAN(
#             min_cluster_size=HDBSCAN_min_cluster_size,
#             min_samples=HDBSCAN_min_samples,
#             max_cluster_size=0,
#             metric='euclidean',  # cosine, manhattan
#             # memory=Memory(None, verbose=1)
#             algorithm='best',
#             cluster_selection_method='eom',
#             prediction_data=True
#             )
        
#         hdb.fit(latent_postPHATE_perfile.reshape(latent_postPHATE_perfile.shape[0]*latent_postPHATE_perfile.shape[1], latent_postPHATE_perfile.shape[2]))  # []

#          #TODO Look into soft clustering
#         # soft_cluster_vecs = np.array(hdbscan.all_points_membership_vectors(hdb))
#         # soft_clusters = np.array([np.argmax(x) for x in soft_cluster_vecs], dtype=int)
#         # hdb_color_palette = sns.color_palette('Paired', int(np.max(soft_clusters) + 3))

#         hdb_labels_flat = hdb.labels_
#         # hdb_labels_flat = soft_clusters
#         hdb_probabilities_flat = hdb.probabilities_
#         # hdb_probabilities_flat = np.array([np.max(x) for x in soft_cluster_vecs])
                
#     # If HDBSCAN is already made/provided, then predict cluster with built in HDBSCAN method
#     else:
#         print("Using pre-built HDBSCAN model")
#         hdb = premade_HDBSCAN
        
#     #TODO Destaurate according to probability of being in cluster

#     # Per patient, Run data through model & Reshape the labels and probabilities for plotting
#     hdb_labels_flat_perfile = [-1] * latent_postPHATE_perfile.shape[0]
#     hdb_probabilities_flat_perfile = [-1] * latent_postPHATE_perfile.shape[0]
#     for i in range(len(latent_postPHATE_perfile)):
#         hdb_labels_flat_perfile[i], hdb_probabilities_flat_perfile[i] = hdbscan.prediction.approximate_predict(hdb, latent_postPHATE_perfile[i, :, :])


#     ###### START OF PLOTTING #####

#     # Get all of the seizure times and types
#     seiz_start_dt_perfile = [-1] * len(pat_ids_list)
#     seiz_stop_dt_perfile = [-1] * len(pat_ids_list)
#     seiz_types_perfile = [-1] * len(pat_ids_list)
#     for i in range(len(pat_ids_list)):
#         seiz_start_dt_perfile[i], seiz_stop_dt_perfile[i], seiz_types_perfile[i] = get_pat_seiz_datetimes(pat_ids_list[i], atd_file=atd_file)

#     # Intialize master figure 
#     fig_height = 15 + 5 * len(plot_pat_ids)
#     fig = pl.figure(figsize=(40, fig_height))
#     gs = gridspec.GridSpec(1 + len(plot_pat_ids), 5, figure=fig)

#     # **** PHATE PLOTTING ****

#     print(f"PHATE Plotting")
#     ax00 = fig.add_subplot(gs[0, 0]) 
#     ax01 = fig.add_subplot(gs[0, 1]) 
#     ax02 = fig.add_subplot(gs[0, 2]) 
#     ax03 = fig.add_subplot(gs[0, 3]) 
#     ax04 = fig.add_subplot(gs[0, 4]) 
#     ax00, ax01, ax02, ax03, ax04, xy_lims = plot_latent(
#         ax=ax00, 
#         interCont_ax=ax01,
#         seiztype_ax=ax02,
#         time_ax=ax03,
#         cluster_ax=ax04,
#         latent_data=latent_postPHATE_perfile.swapaxes(1,2), # [epoch, 2, timesample]
#         modified_samp_freq=modified_FS,  # accounts for windowing/stride
#         start_datetimes=start_datetimes_epoch, 
#         stop_datetimes=stop_datetimes_epoch, 
#         win_sec=win_sec,
#         stride_sec=stride_sec, 
#         seiz_start_dt=seiz_start_dt_perfile, 
#         seiz_stop_dt=seiz_stop_dt_perfile, 
#         seiz_types=seiz_types_perfile,
#         preictal_dur=plot_preictal_color_sec,
#         postictal_dur=plot_postictal_color_sec,
#         plot_ictal=True,
#         hdb_labels=np.expand_dims(np.stack(hdb_labels_flat_perfile, axis=0),axis=1),
#         hdb_probabilities=np.expand_dims(np.stack(hdb_probabilities_flat_perfile, axis=0),axis=1),
#         hdb=hdb,
#         xy_lims=xy_lims,
#         **kwargs)        

#     ax00.title.set_text('PHATE Latent Space: ' + 
#         'Window mean, dur/str=' + str(win_sec) + 
#         '/' + str(stride_sec) +' seconds' )
    
#     if interictal_contour:
#         ax01.title.set_text('Interictal Contour (no peri-ictal data)')


#     #### Plot the individual patient IDs defined in 'plot_pat_ids'
#     for i in range(len(plot_pat_ids)):
#         print(f"Patient Specific PHATE Plotting")
#         pat_id_curr = plot_pat_ids[i]
        
#         # Subindex the patient's data
#         found_file_ids = [index for index, value in enumerate(pat_ids_list) if value == pat_id_curr]
#         pat_data = latent_postPHATE_perfile[found_file_ids] 
#         pat_start_datetimes_epoch = [start_datetimes_epoch[x] for x in found_file_ids]
#         pat_stop_datetimes_epoch = [stop_datetimes_epoch[x] for x in found_file_ids]
#         pat_seiz_start_dt_perfile = [seiz_start_dt_perfile[x] for x in found_file_ids]
#         pat_seiz_stop_dt_perfile = [seiz_stop_dt_perfile[x] for x in found_file_ids]
#         pat_seiz_types_perfile = [seiz_types_perfile[x] for x in found_file_ids]
#         pat_hdb_labels_flat_perfile = [hdb_labels_flat_perfile[x] for x in found_file_ids]
#         pat_hdb_probabilities_flat_perfile = [hdb_probabilities_flat_perfile[x] for x in found_file_ids]

#         # Make the patient's subplots
#         axi0 = fig.add_subplot(gs[1 + i, 0]) 
#         axi1 = fig.add_subplot(gs[1 + i, 1]) 
#         axi2 = fig.add_subplot(gs[1 + i, 2]) 
#         axi3 = fig.add_subplot(gs[1 + i, 3]) 
#         axi4 = fig.add_subplot(gs[1 + i, 4]) 

#         plot_latent(
#             ax=axi0, 
#             interCont_ax=axi1,
#             seiztype_ax=axi2,
#             time_ax=axi3,
#             cluster_ax=axi4,
#             latent_data=pat_data.swapaxes(1,2), # [epoch, 2, timesample]
#             modified_samp_freq=modified_FS,  # accounts for windowing/stride
#             start_datetimes=pat_start_datetimes_epoch, 
#             stop_datetimes=pat_stop_datetimes_epoch, 
#             win_sec=win_sec,
#             stride_sec=stride_sec, 
#             seiz_start_dt=pat_seiz_start_dt_perfile, 
#             seiz_stop_dt=pat_seiz_stop_dt_perfile, 
#             seiz_types=pat_seiz_types_perfile,
#             preictal_dur=plot_preictal_color_sec,
#             postictal_dur=plot_postictal_color_sec,
#             plot_ictal=True,
#             hdb_labels=np.expand_dims(np.stack(pat_hdb_labels_flat_perfile, axis=0),axis=1),
#             hdb_probabilities=np.expand_dims(np.stack(pat_hdb_probabilities_flat_perfile, axis=0),axis=1),
#             hdb=hdb,
#             xy_lims=xy_lims, # passed from above
#             **kwargs)     

#         axi0.title.set_text(f"Only {pat_id_curr}")


#     # **** Save entire figure *****
#     if not os.path.exists(savedir + '/JPEGs'): os.makedirs(savedir + '/JPEGs')
#     if not os.path.exists(savedir + '/PDFs'): os.makedirs(savedir + '/PDFs')
#     savename_jpg = savedir + f"/JPEGs/PHATE_latent_smoothsec{win_sec}Stride{stride_sec}_epoch{epoch}_{phate_metric}_knn{knn}_decay{decay}.jpg"
#     savename_pdf = savedir + f"/PDFs/PHATE_latent_smoothsec{win_sec}Stride{stride_sec}_epoch{epoch}_{phate_metric}_knn{knn}_decay{decay}.pdf"
#     pl.savefig(savename_jpg, dpi=600)
#     pl.savefig(savename_pdf, dpi=600)

#     # TODO Upload to WandB

#     pl.close(fig)

#     # Bundle the save metrics together
#     # save_tuple = (latent_data_windowed.swapaxes(1,2), latent_PCA_allFiles, latent_topPaCMAP_allFiles, latent_topPaCMAP_MedDim_allFiles, hdb_labels_allFiles, hdb_probabilities_allFiles)
#     return ax00, phate_op, hdb, xy_lims # save_tuple

# def umap_subfunction(  
#     atd_file,
#     pat_ids_list,
#     latent_data_windowed, 
#     start_datetimes_epoch,  
#     stop_datetimes_epoch,
#     epoch, 
#     FS, 
#     win_sec, 
#     stride_sec, 
#     output_metric,
#     n_neighbors,
#     metric,
#     min_dist,
#     densmap,
#     dens_lambda,
#     init,
#     spread,
#     local_connectivity, 
#     repulsion_strength,
#     savedir,
#     HDBSCAN_min_cluster_size,
#     HDBSCAN_min_samples,
#     plot_preictal_color_sec,
#     plot_postictal_color_sec,
#     apply_pca=True,
#     pca_comp=100,
#     exclude_self_pat=False,
#     interictal_contour=False,
#     verbose=True,
#     xy_lims = [],
#     premade_UMAP = [],
#     premade_HDBSCAN = [],
#     **kwargs):

#     '''
#     Goal of function:

#     '''

#     # Metadata
#     latent_dim = latent_data_windowed[0].shape[1]
#     num_timepoints_in_windowed_file = latent_data_windowed[0].shape[0]
#     modified_FS = 1 / stride_sec

#     # Check for NaNs in files
#     delete_file_idxs = []
#     for i in range(len(latent_data_windowed)):
#         if np.sum(np.isnan(latent_data_windowed[i])) > 0:
#             delete_file_idxs = delete_file_idxs + [i]
#             print(f"WARNING: Deleted file {start_datetimes_epoch[i]} that had NaNs")

#     # Delete entries/files in lists where there is NaN in latent space for that file
#     latent_data_windowed = [item for i, item in enumerate(latent_data_windowed) if i not in delete_file_idxs]
#     start_datetimes_epoch = [item for i, item in enumerate(start_datetimes_epoch) if i not in delete_file_idxs]  
#     stop_datetimes_epoch = [item for i, item in enumerate(stop_datetimes_epoch) if i not in delete_file_idxs]
#     pat_ids_list = [item for i, item in enumerate(pat_ids_list) if i not in delete_file_idxs]

#     # Flatten data into [miniepoch, dim] to feed into PaCMAP, original data is [file, seq_miniepoch_in_file, latent_dim]
#     latent_UMAP_input = np.concatenate(latent_data_windowed, axis=0)

#     # Generate numerical IDs for each unique patient, and give each datapoint an ID
#     unique_ids = list(set(pat_ids_list))
#     id_to_index = {id: idx for idx, id in enumerate(unique_ids)}  # Create mapping dictionary
#     pat_idxs = [id_to_index[id] for id in pat_ids_list]
#     pat_idxs_expanded = [item for item in pat_idxs for _ in range(latent_data_windowed[0].shape[0])]

#     ### UMAP 2-Dim ###

#     # Make new UMAP
#     if premade_UMAP == []:
#         print("Making new 2-dim UMAP to use for visualization")
#         # initializing the pacmap instance
#         # Setting n_neighbors to "None" leads to a default choice shown below in "parameter" section
#         reducer = umap.UMAP(
#             output_metric=output_metric,
#             n_neighbors=n_neighbors,
#             metric=metric,
#             min_dist=min_dist,
#             verbose=True,
#             densmap=densmap,
#             dens_lambda=dens_lambda,
#             init=init,
#             spread=spread,
#             local_connectivity=local_connectivity, 
#             repulsion_strength=repulsion_strength,
#         ) 

#         # fit the data (The index of transformed data corresponds to the index of the original data)
#         reducer.fit(latent_UMAP_input)

#     # Use premade PaCMAP
#     else: 
#         print("Using existing 2-dim UMAP for visualization")
#         reducer = premade_PaCMAP

#     # Project data through reducer (i.e. PaCMAP) and split back into files
#     latent_postUMAP_perfile = reducer.transform(latent_UMAP_input)
#     latent_postUMAP_perfile = np.stack(np.split(latent_postUMAP_perfile, len(latent_data_windowed), axis=0),axis=0)

#     ### HDBSCAN ###
#     # If training, create new cluster model, otherwise "approximate_predict()" if running on val data
#     if premade_HDBSCAN == []:
#         # Now do the clustering with HDBSCAN
#         print("Building new HDBSCAN model")
#         hdb = hdbscan.HDBSCAN(
#             min_cluster_size=HDBSCAN_min_cluster_size,
#             min_samples=HDBSCAN_min_samples,
#             max_cluster_size=0,
#             metric='euclidean',  # cosine, manhattan
#             # memory=Memory(None, verbose=1)
#             algorithm='best',
#             cluster_selection_method='eom',
#             prediction_data=True
#             )
        
#         hdb.fit(latent_postUMAP_perfile.reshape(latent_postUMAP_perfile.shape[0]*latent_postUMAP_perfile.shape[1], latent_postUMAP_perfile.shape[2]))  # []

#          #TODO Look into soft clustering
#         # soft_cluster_vecs = np.array(hdbscan.all_points_membership_vectors(hdb))
#         # soft_clusters = np.array([np.argmax(x) for x in soft_cluster_vecs], dtype=int)
#         # hdb_color_palette = sns.color_palette('Paired', int(np.max(soft_clusters) + 3))

#         hdb_labels_flat = hdb.labels_
#         # hdb_labels_flat = soft_clusters
#         hdb_probabilities_flat = hdb.probabilities_
#         # hdb_probabilities_flat = np.array([np.max(x) for x in soft_cluster_vecs])
                
#     # If HDBSCAN is already made/provided, then predict cluster with built in HDBSCAN method
#     else:
#         print("Using pre-built HDBSCAN model")
#         hdb = premade_HDBSCAN
        
#     #TODO Destaurate according to probability of being in cluster

#     # Per patient, Run data through model & Reshape the labels and probabilities for plotting
#     hdb_labels_flat_perfile = [-1] * latent_postUMAP_perfile.shape[0]
#     hdb_probabilities_flat_perfile = [-1] * latent_postUMAP_perfile.shape[0]
#     for i in range(len(latent_postUMAP_perfile)):
#         hdb_labels_flat_perfile[i], hdb_probabilities_flat_perfile[i] = hdbscan.prediction.approximate_predict(hdb, latent_postUMAP_perfile[i, :, :])


#     ###### START OF PLOTTING #####

#     # Get all of the seizure times and types
#     seiz_start_dt_perfile = [-1] * len(pat_ids_list)
#     seiz_stop_dt_perfile = [-1] * len(pat_ids_list)
#     seiz_types_perfile = [-1] * len(pat_ids_list)
#     for i in range(len(pat_ids_list)):
#         seiz_start_dt_perfile[i], seiz_stop_dt_perfile[i], seiz_types_perfile[i] = get_pat_seiz_datetimes(pat_ids_list[i], atd_file=atd_file)

#     # Intialize master figure 
#     fig = pl.figure(figsize=(40, 15))
#     gs = gridspec.GridSpec(1, 5, figure=fig)
#     # fig_height = 15 + 5 * len(plot_pat_ids)
#     # fig = pl.figure(figsize=(40, fig_height))
#     # gs = gridspec.GridSpec(1 + len(plot_pat_ids), 5, figure=fig)

#     # **** UMAP PLOTTING ****

#     print(f"UMAP Plotting")
#     ax00 = fig.add_subplot(gs[0, 0]) 
#     ax01 = fig.add_subplot(gs[0, 1]) 
#     ax02 = fig.add_subplot(gs[0, 2]) 
#     ax03 = fig.add_subplot(gs[0, 3]) 
#     ax04 = fig.add_subplot(gs[0, 4]) 
#     ax00, ax01, ax02, ax03, ax04, xy_lims = plot_latent(
#         ax=ax00, 
#         interCont_ax=ax01,
#         seiztype_ax=ax02,
#         time_ax=ax03,
#         cluster_ax=ax04,
#         latent_data=latent_postUMAP_perfile.swapaxes(1,2), # [epoch, 2, timesample]
#         modified_samp_freq=modified_FS,  # accounts for windowing/stride
#         start_datetimes=start_datetimes_epoch, 
#         stop_datetimes=stop_datetimes_epoch, 
#         win_sec=win_sec,
#         stride_sec=stride_sec, 
#         seiz_start_dt=seiz_start_dt_perfile, 
#         seiz_stop_dt=seiz_stop_dt_perfile, 
#         seiz_types=seiz_types_perfile,
#         preictal_dur=plot_preictal_color_sec,
#         postictal_dur=plot_postictal_color_sec,
#         plot_ictal=True,
#         hdb_labels=np.expand_dims(np.stack(hdb_labels_flat_perfile, axis=0),axis=1),
#         hdb_probabilities=np.expand_dims(np.stack(hdb_probabilities_flat_perfile, axis=0),axis=1),
#         hdb=hdb,
#         xy_lims=xy_lims,
#         **kwargs)        

#     ax00.title.set_text('UMAP Latent Space: ' + 
#         'Window mean, dur/str=' + str(win_sec) + 
#         '/' + str(stride_sec) +' seconds' )
    
#     if interictal_contour:
#         ax01.title.set_text('Interictal Contour (no peri-ictal data)')
    
    
#     # **** Save entire figure *****
#     if not os.path.exists(savedir + '/JPEGs'): os.makedirs(savedir + '/JPEGs')
#     if not os.path.exists(savedir + '/PDFs'): os.makedirs(savedir + '/PDFs')
#     savename_jpg = savedir + f"/JPEGs/UMAP_latent_smoothsec{win_sec}Stride{stride_sec}_epoch{epoch}_{output_metric}_nn{n_neighbors}_metric{metric}_mindist{min_dist}_densmap{densmap}_denslambda{dens_lambda}_init{init}.jpg"
#     savename_pdf = savedir + f"/PDFs/UMAP_latent_smoothsec{win_sec}Stride{stride_sec}_epoch{epoch}_{output_metric}_nn{n_neighbors}_metric{metric}_mindist{min_dist}_densmap{densmap}_denslambda{dens_lambda}_init{init}.pdf"
#     pl.savefig(savename_jpg, dpi=600)
#     pl.savefig(savename_pdf, dpi=600)

#     # TODO Upload to WandB

#     pl.close(fig)


#     # Bundle the save metrics together
#     # save_tuple = (latent_data_windowed.swapaxes(1,2), latent_PCA_allFiles, latent_topPaCMAP_allFiles, latent_topPaCMAP_MedDim_allFiles, hdb_labels_allFiles, hdb_probabilities_allFiles)
#     return ax00, reducer, hdb, xy_lims # save_tuple

def pacmap_subfunction(  
    atd_file,
    pat_ids_list,
    latent_data_windowed, 
    start_datetimes_epoch,  
    stop_datetimes_epoch,
    epoch, 
    FS, 
    win_sec, 
    stride_sec, 
    savedir,
    pacmap_LR,
    pacmap_NumIters,
    pacmap_NN,
    pacmap_MN_ratio,
    pacmap_FP_ratio,
    HDBSCAN_min_cluster_size,
    HDBSCAN_min_samples,
    plot_preictal_color_sec,
    plot_postictal_color_sec,
    apply_pca=True,
    interictal_contour=False,
    verbose=True,
    xy_lims = [],
    xy_lims_RAW_DIMS = [],
    xy_lims_PCA = [],
    premade_PaCMAP = [],
    premade_PaCMAP_MedDim = [],
    premade_PCA = [],
    premade_HDBSCAN = [],
    **kwargs):

    '''
    Goal of function:
    Make 2D PaCMAP, make HigherDim PaCMAP, HDBSCAN cluster on HigherDim, visualize clusters on 2D

    '''

    # Metadata
    latent_dim = latent_data_windowed[0].shape[1]
    num_timepoints_in_windowed_file = latent_data_windowed[0].shape[0]
    modified_FS = 1 / stride_sec

    # Check for NaNs in files
    delete_file_idxs = []
    for i in range(len(latent_data_windowed)):
        if np.sum(np.isnan(latent_data_windowed[i])) > 0:
            delete_file_idxs = delete_file_idxs + [i]
            print(f"WARNING: Deleted file {start_datetimes_epoch[i]} that had NaNs")

    # Delete entries/files in lists where there is NaN in latent space for that file
    latent_data_windowed = [item for i, item in enumerate(latent_data_windowed) if i not in delete_file_idxs]
    start_datetimes_epoch = [item for i, item in enumerate(start_datetimes_epoch) if i not in delete_file_idxs]  
    stop_datetimes_epoch = [item for i, item in enumerate(stop_datetimes_epoch) if i not in delete_file_idxs]
    pat_ids_list = [item for i, item in enumerate(pat_ids_list) if i not in delete_file_idxs]

    # Flatten data into [miniepoch, dim] to feed into PaCMAP, original data is [file, seq_miniepoch_in_file, latent_dim]
    latent_PaCMAP_input = np.concatenate(latent_data_windowed, axis=0)

    # Generate numerical IDs for each unique patient, and give each datapoint an ID
    unique_ids = list(set(pat_ids_list))
    id_to_index = {id: idx for idx, id in enumerate(unique_ids)}  # Create mapping dictionary
    pat_idxs = [id_to_index[id] for id in pat_ids_list]
    pat_idxs_expanded = [item for item in pat_idxs for _ in range(latent_data_windowed[0].shape[0])]

    ### PaCMAP 2-Dim ###

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
            apply_pca=apply_pca, 
            verbose=verbose)  

        # fit the data (The index of transformed data corresponds to the index of the original data)
        latent_postPaCMAP_perfile = reducer.fit_transform(latent_PaCMAP_input, init='pca')
        latent_postPaCMAP_perfile = np.stack(np.split(latent_postPaCMAP_perfile, len(latent_data_windowed), axis=0),axis=0)

    # Use premade PaCMAP
    else: 
        print("Using existing 2-dim PaCMAP for visualization")
        reducer = premade_PaCMAP  # Project data through reducer (i.e. PaCMAP) and split back into file
        latent_postPaCMAP_perfile = reducer.transform(latent_PaCMAP_input)
        latent_postPaCMAP_perfile = np.stack(np.split(latent_postPaCMAP_perfile, len(latent_data_windowed), axis=0),axis=0)

    ### HDBSCAN ###
    # If training, create new cluster model, otherwise "approximate_predict()" if running on val data
    if premade_HDBSCAN == []:
        # Now do the clustering with HDBSCAN
        print("Building new HDBSCAN model")
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_min_cluster_size,
            min_samples=HDBSCAN_min_samples,
            max_cluster_size=0,
            metric='euclidean',  # cosine, manhattan
            # memory=Memory(None, verbose=1)
            algorithm='best',
            cluster_selection_method='eom',
            prediction_data=True
            )
        
        hdb.fit(latent_postPaCMAP_perfile.reshape(latent_postPaCMAP_perfile.shape[0]*latent_postPaCMAP_perfile.shape[1], latent_postPaCMAP_perfile.shape[2]))  # []

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
    hdb_labels_flat_perfile = [-1] * latent_postPaCMAP_perfile.shape[0]
    hdb_probabilities_flat_perfile = [-1] * latent_postPaCMAP_perfile.shape[0]
    for i in range(len(latent_postPaCMAP_perfile)):
        hdb_labels_flat_perfile[i], hdb_probabilities_flat_perfile[i] = hdbscan.prediction.approximate_predict(hdb, latent_postPaCMAP_perfile[i, :, :])


    ###### START OF PLOTTING #####

    # Get all of the seizure times and types
    seiz_start_dt_perfile = [-1] * len(latent_postPaCMAP_perfile)
    seiz_stop_dt_perfile = [-1] * len(latent_postPaCMAP_perfile)
    seiz_types_perfile = [-1] * len(latent_postPaCMAP_perfile)
    for i in range(len(latent_postPaCMAP_perfile)):
        seiz_start_dt_perfile[i], seiz_stop_dt_perfile[i], seiz_types_perfile[i] = get_pat_seiz_datetimes(pat_ids_list[i], atd_file=atd_file)

    # Intialize master figure 
    fig = pl.figure(figsize=(40, 15))
    gs = gridspec.GridSpec(1, 5, figure=fig)


    # **** PACMAP PLOTTING ****

    print(f"PaCMAP Plotting")
    ax20 = fig.add_subplot(gs[0, 0]) 
    ax21 = fig.add_subplot(gs[0, 1]) 
    ax22 = fig.add_subplot(gs[0, 2]) 
    ax23 = fig.add_subplot(gs[0, 3]) 
    ax24 = fig.add_subplot(gs[0, 4]) 
    ax20, ax21, ax22, ax23, ax24, xy_lims = plot_latent(
        ax=ax20, 
        interCont_ax=ax21,
        seiztype_ax=ax22,
        time_ax=ax23,
        cluster_ax=ax24,
        latent_data=latent_postPaCMAP_perfile.swapaxes(1,2), # [epoch, 2, timesample]
        modified_samp_freq=modified_FS,  # accounts for windowing/stride
        start_datetimes=start_datetimes_epoch, 
        stop_datetimes=stop_datetimes_epoch, 
        win_sec=win_sec,
        stride_sec=stride_sec, 
        seiz_start_dt=seiz_start_dt_perfile, 
        seiz_stop_dt=seiz_stop_dt_perfile, 
        seiz_types=seiz_types_perfile,
        preictal_dur=plot_preictal_color_sec,
        postictal_dur=plot_postictal_color_sec,
        plot_ictal=True,
        hdb_labels=np.expand_dims(np.stack(hdb_labels_flat_perfile, axis=0),axis=1),
        hdb_probabilities=np.expand_dims(np.stack(hdb_probabilities_flat_perfile, axis=0),axis=1),
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


    # # ***** PCA PLOTTING *****
        
    # if premade_PCA == []:
    #     print("Calculating new PCA")
    #     pca = PCA(n_components=2, svd_solver='full') # Different than PCA used for PaCMAP
    #     latent_PCA_flat_transformed = pca.fit_transform(latent_PaCMAP_input)

    # else:
    #     print("Using existing PCA")
    #     pca = premade_PCA
        
    # # Project data through PCA and split into files
    # print("Projecting data through built PCA")
    # latent_PCA_flat_transformed_perfile = pca.transform(latent_PaCMAP_input)
    # latent_PCA_flat_transformed_perfile = np.stack(np.split(latent_PCA_flat_transformed_perfile, len(latent_data_windowed), axis=0),axis=0)

    # print(f"PCA Plotting")
    # ax10 = fig.add_subplot(gs[1, 0]) 
    # ax11 = fig.add_subplot(gs[1, 1]) 
    # ax12 = fig.add_subplot(gs[1, 2]) 
    # ax13 = fig.add_subplot(gs[1, 3]) 
    # ax14 = fig.add_subplot(gs[1, 4]) 
    # ax10, ax11, ax12, ax13, ax14, xy_lims_PCA = plot_latent(
    #     ax=ax10, 
    #     interCont_ax=ax11,
    #     seiztype_ax=ax12,
    #     time_ax=ax13,
    #     cluster_ax=ax14,
    #     latent_data=latent_PCA_flat_transformed_perfile.swapaxes(1,2),   # [epoch, 2, timesample]
    #     modified_samp_freq=modified_FS,
    #     start_datetimes=start_datetimes_epoch, 
    #     stop_datetimes=stop_datetimes_epoch, 
    #     win_sec=win_sec,
    #     stride_sec=stride_sec, 
    #     seiz_start_dt=seiz_start_dt_perfile, 
    #     seiz_stop_dt=seiz_stop_dt_perfile, 
    #     seiz_types=seiz_types_perfile,
    #     preictal_dur=plot_preictal_color_sec,
    #     postictal_dur=plot_postictal_color_sec,
    #     plot_ictal=True,
    #     hdb_labels=np.expand_dims(np.stack(hdb_labels_flat_perfile, axis=0),axis=1),
    #     hdb_probabilities=np.expand_dims(np.stack(hdb_probabilities_flat_perfile, axis=0),axis=1),
    #     hdb=hdb,
    #     xy_lims=xy_lims_PCA,
    #     **kwargs)        

    # ax10.title.set_text("PCA Components 1,2")
    # ax11.title.set_text('Interictal Contour (no peri-ictal data)')


    # # **** INFO RAW DIM PLOTTING *****

    # raw_dims_to_plot = [0,1]

    # # Pull out the raw dims of interest and stack the data by file
    # latent_flat_RawDim_perfile = [latent_data_windowed[i][:, raw_dims_to_plot] for i in range(len(latent_data_windowed))]

    # print(f"Raw Dims Plotting")
    # ax00 = fig.add_subplot(gs[0, 0]) 
    # ax01 = fig.add_subplot(gs[0, 1]) 
    # ax02 = fig.add_subplot(gs[0, 2]) 
    # ax03 = fig.add_subplot(gs[0, 3]) 
    # ax04 = fig.add_subplot(gs[0, 4])
    # ax00, ax01, ax02, ax03, ax04, xy_lims_RAW_DIMS = plot_latent(
    #     ax=ax00, 
    #     interCont_ax=ax01,
    #     seiztype_ax=ax02,
    #     time_ax=ax03,
    #     cluster_ax=ax04,
    #     latent_data=np.stack(latent_flat_RawDim_perfile,axis=0).swapaxes(1,2), # [epoch, 2, timesample]
    #     modified_samp_freq=modified_FS,
    #     start_datetimes=start_datetimes_epoch, 
    #     stop_datetimes=stop_datetimes_epoch, 
    #     win_sec=win_sec,
    #     stride_sec=stride_sec, 
    #     seiz_start_dt=seiz_start_dt_perfile, 
    #     seiz_stop_dt=seiz_stop_dt_perfile, 
    #     seiz_types=seiz_types_perfile,
    #     preictal_dur=plot_preictal_color_sec,
    #     postictal_dur=plot_postictal_color_sec,
    #     plot_ictal=True,
    #     hdb_labels=np.expand_dims(np.stack(hdb_labels_flat_perfile, axis=0),axis=1),
    #     hdb_probabilities=np.expand_dims(np.stack(hdb_probabilities_flat_perfile, axis=0),axis=1),
    #     hdb=hdb,
    #     xy_lims=xy_lims_RAW_DIMS,
    #     **kwargs)        

    # ax00.title.set_text(f'Dims [{raw_dims_to_plot[0]},{raw_dims_to_plot[1]}], Window mean, dur/str=' + str(win_sec) + '/' + str(stride_sec) +' seconds,' )
    # ax01.title.set_text('Interictal Contour (no peri-ictal data)')

    # **** Save entire figure *****
    if not os.path.exists(savedir): os.makedirs(savedir)
    savename_jpg = savedir + f"/pacmap_latent_smoothsec" + str(win_sec) + "Stride" + str(stride_sec) + "_epoch" + str(epoch) + "_LR" + str(pacmap_LR) + "_NumIters" + str(pacmap_NumIters) + f"PCA{apply_pca}_NN{pacmap_NN}_MNratio{pacmap_MN_ratio}_FPratio{pacmap_FP_ratio}.jpg"
    pl.savefig(savename_jpg, dpi=600)

    # TODO Upload to WandB

    pl.close(fig)

    # Bundle the save metrics together
    # save_tuple = (latent_data_windowed.swapaxes(1,2), latent_PCA_allFiles, latent_topPaCMAP_allFiles, latent_topPaCMAP_MedDim_allFiles, hdb_labels_allFiles, hdb_probabilities_allFiles)
    return ax21, reducer, hdb, xy_lims # save_tuple

def prpacmap_subfunction(  
    atd_file,
    pat_ids_list,
    latent_data_windowed, 
    start_datetimes_epoch,  
    stop_datetimes_epoch,
    epoch, 
    FS, 
    win_sec, 
    stride_sec, 
    savedir,
    prpacmap_metric,
    prpacmap_batchsize,
    prpacmap_LR,
    prpacmap_NumEpochs,
    prpacmap_weight_schedule,
    prpacmap_NN,
    prpacmap_n_MN,
    prpacmap_n_FP,
    HDBSCAN_min_cluster_size,
    HDBSCAN_min_samples,
    plot_preictal_color_sec,
    plot_postictal_color_sec,
    prpacmap_num_workers,
    apply_pca=True,
    interictal_contour=False,
    verbose=True,
    xy_lims = [],
    premade_prPaCMAP = [],
    premade_HDBSCAN = [],
    **kwargs):

    '''
    Goal of function:
    Make 2D PR-PaCMAP, make 10D PR-PaCMAP, HDBSCAN cluster on 10D, visualize clusters on 2D

    '''

    # Metadata
    latent_dim = latent_data_windowed[0].shape[1]
    num_timepoints_in_windowed_file = latent_data_windowed[0].shape[0]
    modified_FS = 1 / stride_sec

    # Check for NaNs in files
    delete_file_idxs = []
    for i in range(len(latent_data_windowed)):
        if np.sum(np.isnan(latent_data_windowed[i])) > 0:
            delete_file_idxs = delete_file_idxs + [i]
            print(f"WARNING: Deleted file {start_datetimes_epoch[i]} that had NaNs")

    # Delete entries/files in lists where there is NaN in latent space for that file
    latent_data_windowed = [item for i, item in enumerate(latent_data_windowed) if i not in delete_file_idxs]
    start_datetimes_epoch = [item for i, item in enumerate(start_datetimes_epoch) if i not in delete_file_idxs]  
    stop_datetimes_epoch = [item for i, item in enumerate(stop_datetimes_epoch) if i not in delete_file_idxs]
    pat_ids_list = [item for i, item in enumerate(pat_ids_list) if i not in delete_file_idxs]

    # Flatten data into [miniepoch, dim] to feed into PaCMAP, original data is [file, seq_miniepoch_in_file, latent_dim]
    latent_PaCMAP_input = np.concatenate(latent_data_windowed, axis=0)

    # Generate numerical IDs for each unique patient, and give each datapoint an ID
    unique_ids = list(set(pat_ids_list))
    id_to_index = {id: idx for idx, id in enumerate(unique_ids)}  # Create mapping dictionary
    pat_idxs = [id_to_index[id] for id in pat_ids_list]
    pat_idxs_expanded = [item for item in pat_idxs for _ in range(latent_data_windowed[0].shape[0])]

    ### PR-PaCMAP 2-Dim ###

    # Make new PR-PaCMAP
    if premade_prPaCMAP == []:
        print(f"Making new 2-dim PR-PaCMAP, PCA:{apply_pca}, Metric:{prpacmap_metric}, NN:{prpacmap_NN}, MN:{prpacmap_n_MN}, FP:{prpacmap_n_FP}, LR:{prpacmap_LR}, Batch:{prpacmap_batchsize}" )
        # initializing the pacmap instance
        # Setting n_neighbors to "None" leads to a default choice shown below in "parameter" section
        reducer = parampacmap.ParamPaCMAP( 
            distance=prpacmap_metric,
            lr=prpacmap_LR,
            n_components=2, 
            n_neighbors=prpacmap_NN, 
            n_MN=prpacmap_n_MN, 
            n_FP=prpacmap_n_FP, 
            apply_pca=apply_pca, 
            batch_size=prpacmap_batchsize,
            num_epochs=prpacmap_NumEpochs,
            verbose=verbose,
            weight_schedule=prpacmap_weight_schedule,
            num_workers=prpacmap_num_workers) 

            # Defaults
            # n_components: int = 2,
            # n_neighbors: int = 10,
            # n_FP: int = 20,
            # n_MN: int = 5,
            # distance: str = "euclidean",
            # optim_type: str = "Adam",
            # lr: float = 1e-3,
            # lr_schedule: Optional[bool] = None,
            # apply_pca: bool = True,
            # apply_scale: Optional[str] = None,
            # model_dict: Optional[dict] = utils.DEFAULT_MODEL_DICT,
            # intermediate_snapshots: Optional[list] = [],
            # loss_weight: Optional[list] = [1, 1, 1],
            # batch_size: int = 1024,
            # data_reshape: Optional[list] = None,
            # num_epochs: int = 450,
            # verbose: bool = False,
            # weight_schedule: Callable = paramrep_weight_schedule,
            # const_schedule: Optional[Callable] = paramrep_const_schedule,
            # num_workers: int = 1,
            # dtype: torch.dtype = torch.float32,
            # embedding_init: str = "pca",
            # seed: Optional[int] = None,
            # save_pairs: bool = False,
            

        # fit the data (The index of transformed data corresponds to the index of the original data)
        latent_postPaCMAP_perfile = reducer.fit_transform(latent_PaCMAP_input)
        latent_postPaCMAP_perfile = np.stack(np.split(latent_postPaCMAP_perfile, len(latent_data_windowed), axis=0),axis=0)

    else:  # Use premade PR-PaCMAP
        print("Using existing 2-dim PR-PaCMAP")
        reducer = premade_prPaCMAP
        latent_postPaCMAP_perfile = reducer.transform(latent_PaCMAP_input)
        latent_postPaCMAP_perfile = np.stack(np.split(latent_postPaCMAP_perfile, len(latent_data_windowed), axis=0),axis=0)

    ### HDBSCAN ###
    # If training, create new cluster model, otherwise "approximate_predict()" if running on val data
    if premade_HDBSCAN == []:
        # Now do the clustering with HDBSCAN
        print("Building new HDBSCAN model")
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_min_cluster_size,
            min_samples=HDBSCAN_min_samples,
            max_cluster_size=0,
            metric='euclidean',  # cosine, manhattan
            # memory=Memory(None, verbose=1)
            algorithm='best',
            cluster_selection_method='eom',
            prediction_data=True
            )
        
        hdb.fit(latent_postPaCMAP_perfile.reshape(latent_postPaCMAP_perfile.shape[0]*latent_postPaCMAP_perfile.shape[1], latent_postPaCMAP_perfile.shape[2]))  # []

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
    hdb_labels_flat_perfile = [-1] * latent_postPaCMAP_perfile.shape[0]
    hdb_probabilities_flat_perfile = [-1] * latent_postPaCMAP_perfile.shape[0]
    for i in range(len(latent_postPaCMAP_perfile)):
        hdb_labels_flat_perfile[i], hdb_probabilities_flat_perfile[i] = hdbscan.prediction.approximate_predict(hdb, latent_postPaCMAP_perfile[i, :, :])


    ###### START OF PLOTTING #####

    # Get all of the seizure times and types
    seiz_start_dt_perfile = [-1] * len(latent_postPaCMAP_perfile)
    seiz_stop_dt_perfile = [-1] * len(latent_postPaCMAP_perfile)
    seiz_types_perfile = [-1] * len(latent_postPaCMAP_perfile)
    for i in range(len(latent_postPaCMAP_perfile)):
        seiz_start_dt_perfile[i], seiz_stop_dt_perfile[i], seiz_types_perfile[i] = get_pat_seiz_datetimes(pat_ids_list[i], atd_file=atd_file)

    # Intialize master figure 
    fig = pl.figure(figsize=(40, 25))
    gs = gridspec.GridSpec(1, 5, figure=fig)

    # **** PR-PACMAP PLOTTING ****

    print(f"PR-PaCMAP Plotting")
    ax20 = fig.add_subplot(gs[0, 0]) 
    ax21 = fig.add_subplot(gs[0, 1]) 
    ax22 = fig.add_subplot(gs[0, 2]) 
    ax23 = fig.add_subplot(gs[0, 3]) 
    ax24 = fig.add_subplot(gs[0, 4]) 
    ax20, ax21, ax22, ax23, ax24, xy_lims = plot_latent(
        ax=ax20, 
        interCont_ax=ax21,
        seiztype_ax=ax22,
        time_ax=ax23,
        cluster_ax=ax24,
        latent_data=latent_postPaCMAP_perfile.swapaxes(1,2), # [epoch, 2, timesample]
        modified_samp_freq=modified_FS,  # accounts for windowing/stride
        start_datetimes=start_datetimes_epoch, 
        stop_datetimes=stop_datetimes_epoch, 
        win_sec=win_sec,
        stride_sec=stride_sec, 
        seiz_start_dt=seiz_start_dt_perfile, 
        seiz_stop_dt=seiz_stop_dt_perfile, 
        seiz_types=seiz_types_perfile,
        preictal_dur=plot_preictal_color_sec,
        postictal_dur=plot_postictal_color_sec,
        plot_ictal=True,
        hdb_labels=np.expand_dims(np.stack(hdb_labels_flat_perfile, axis=0),axis=1),
        hdb_probabilities=np.expand_dims(np.stack(hdb_probabilities_flat_perfile, axis=0),axis=1),
        hdb=hdb,
        xy_lims=xy_lims,
        **kwargs)        

    ax20.title.set_text('PR-PaCMAP Latent Space: ' + 
        'Window mean, dur/str=' + str(win_sec) + 
        '/' + str(stride_sec) +' seconds,' + 
        f'\nLR: {str(prpacmap_LR)}, ' +
        f'NumIters: {str(prpacmap_NumEpochs)}, ' +
        f'numberNN: {prpacmap_NN}, numberMN: {str(prpacmap_n_MN)}, numberFP: {str(prpacmap_n_FP)}'
        )

    if interictal_contour:
        ax21.title.set_text('Interictal Contour (no peri-ictal data)')

    # **** Save entire figure *****
    if not os.path.exists(savedir): os.makedirs(savedir)
    savename_jpg = savedir + f"/pacmap_latent_smoothsec" + str(win_sec) + "Stride" + str(stride_sec) + "_epoch" + str(epoch) + "_LR" + str(prpacmap_LR) + f"_batch{prpacmap_batchsize}" + "_NumIters" + str(prpacmap_NumEpochs) + f"_PCA{apply_pca}_{prpacmap_metric}_NN{prpacmap_NN}_MN{prpacmap_n_MN}_FP{prpacmap_n_FP}.jpg"
    pl.savefig(savename_jpg, dpi=600)

    # TODO Upload to WandB

    pl.close(fig)

    # Bundle the save metrics together
    # save_tuple = (latent_data_windowed.swapaxes(1,2), latent_PCA_allFiles, latent_topPaCMAP_allFiles, latent_topPaCMAP_MedDim_allFiles, hdb_labels_allFiles, hdb_probabilities_allFiles)
    return ax20, reducer, hdb, xy_lims # save_tuple

# def save_phate_objects(savedir, epoch, axis, phate, hdb, xy_lims):
    
#     if not os.path.exists(savedir): os.makedirs(savedir) 

#     # Save the PHATE model for use in inference
#     phate_path = savedir + "/epoch" +str(epoch) + "_phate.pkl"
#     output_obj = open(phate_path, 'wb')
#     pickle.dump(phate, output_obj)
#     output_obj.close()
#     print("Saved PHATE object")

#     hdbscan_path = savedir + "/epoch" +str(epoch) + "_hdbscan.pkl"
#     output_obj = open(hdbscan_path, 'wb')
#     pickle.dump(hdb, output_obj)
#     output_obj.close()
#     print("Saved PHATE HDBSCAN")

#     xylim_path = savedir + "/epoch" +str(epoch) + "_xy_lims.pkl"
#     output_obj = open(xylim_path, 'wb')
#     pickle.dump(xy_lims, output_obj)
#     output_obj.close()
#     print("Saved PHATE xy_lims for PaCMAP")

#     axis_path = savedir + "/epoch" +str(epoch) + "_plotaxis.pkl"
#     output_obj = open(axis_path, 'wb')
#     pickle.dump(axis, output_obj)
#     output_obj.close()
#     print("Saved PHATE plot axis object")

def save_pacmap_objects(pacmap_dir, epoch, axis, reducer, hdb, xy_lims):

    if not os.path.exists(pacmap_dir): os.makedirs(pacmap_dir) 

    # Save the PaCMAP model for use in inference
    PaCMAP_common_prefix = pacmap_dir + "/epoch" +str(epoch) + "_PaCMAP"
    pacmap.pacmap.save(reducer, PaCMAP_common_prefix)
    print("Saved PaCMAP 2-dim model")

    hdbscan_path = pacmap_dir + "/epoch" +str(epoch) + "_hdbscan.pkl"
    output_obj4 = open(hdbscan_path, 'wb')
    pickle.dump(hdb, output_obj4)
    output_obj4.close()
    print("Saved HDBSCAN model")

    xylim_path = pacmap_dir + "/epoch" +str(epoch) + "_xy_lims.pkl"
    output_obj7 = open(xylim_path, 'wb')
    pickle.dump(xy_lims, output_obj7)
    output_obj7.close()
    print("Saved xy_lims for PaCMAP")

    axis_path = pacmap_dir + "/epoch" +str(epoch) + "_plotaxis.pkl"
    output_obj10 = open(axis_path, 'wb')
    pickle.dump(axis, output_obj10)
    output_obj10.close()
    print("Saved plot axis")

def save_som_model(som, grid_size, input_dim, lr, sigma, lr_epoch_decay, sigma_epoch_decay, sigma_min, epoch, batch_size, save_path):
    # Save the state_dict (model weights), weights, and relevant hyperparameters
    torch.save({
        'model_state_dict': som.state_dict(),
        'weights': som.weights,  # Save the weights explicitly
        'grid_size': grid_size,
        'input_dim': input_dim,
        'lr': lr,
        'sigma': sigma,
        'lr_epoch_decay': lr_epoch_decay,
        'sigma_epoch_decay': sigma_epoch_decay,
        'sigma_min': sigma_min,
        'epoch': epoch,
        'batch_size': batch_size
    }, save_path)
    print(f"SOM model saved at {save_path}")

def load_som_model(load_path, gpu_id):
    checkpoint = torch.load(load_path)
    
    # Retrieve the hyperparameters, decay parameters, and weights
    grid_size = checkpoint['grid_size']
    input_dim = checkpoint['input_dim']
    lr = checkpoint['lr']
    sigma = checkpoint['sigma']
    lr_epoch_decay = checkpoint['lr_epoch_decay']
    sigma_epoch_decay = checkpoint['sigma_epoch_decay']
    sigma_min = checkpoint['sigma_min']
    epoch = checkpoint['epoch']
    batch_size = checkpoint['batch_size']
    
    # Create a new SOM instance with the same parameters
    som = SOM(grid_size=(grid_size, grid_size), input_dim=input_dim, batch_size=batch_size, 
              lr=lr, lr_epoch_decay=lr_epoch_decay, sigma=sigma, 
              sigma_epoch_decay=sigma_epoch_decay, sigma_min=sigma_min, device=gpu_id)
    
    # Load the saved state_dict into the model (for training parameters)
    som.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore the weights explicitly
    som.weights = checkpoint['weights']

    # Reset the device
    som.reset_device(gpu_id)
    
    print(f"SOM model loaded from {load_path}")
    
    return som

def kohonen_subfunction_pytorch(  
    atd_file,
    pat_ids_list,
    latent_data_windowed, 
    start_datetimes_epoch,  
    stop_datetimes_epoch,
    epoch, 
    FS, 
    win_sec, 
    stride_sec, 
    savedir,
    som_batch_size,
    som_lr,
    som_lr_epoch_decay,
    som_sigma,
    som_sigma_epoch_decay,
    som_epochs,
    som_gridsize,
    som_sigma_min,
    HDBSCAN_min_cluster_size,
    HDBSCAN_min_samples,
    plot_preictal_color_sec,
    plot_postictal_color_sec,
    interictal_contour=False,
    verbose=True,
    som_precomputed_path = None,
    som_object = None,
    premade_HDBSCAN = [],
    som_device = 0,
    **kwargs):

    if not os.path.exists(savedir): os.makedirs(savedir)

    # DATA PREPARATION

    # Metadata
    latent_dim = latent_data_windowed[0].shape[1]
    num_timepoints_in_windowed_file = latent_data_windowed[0].shape[0]
    modified_FS = 1 / stride_sec

    # Check for NaNs in files
    delete_file_idxs = []
    for i in range(len(latent_data_windowed)):
        if np.sum(np.isnan(latent_data_windowed[i])) > 0:
            delete_file_idxs = delete_file_idxs + [i]
            print(f"WARNING: Deleted file {start_datetimes_epoch[i]} that had NaNs")

    # Delete entries/files in lists where there is NaN in latent space for that file
    latent_data_windowed = [item for i, item in enumerate(latent_data_windowed) if i not in delete_file_idxs]
    start_datetimes_epoch = [item for i, item in enumerate(start_datetimes_epoch) if i not in delete_file_idxs]  
    stop_datetimes_epoch = [item for i, item in enumerate(stop_datetimes_epoch) if i not in delete_file_idxs]
    pat_ids_list = [item for i, item in enumerate(pat_ids_list) if i not in delete_file_idxs]

    # Flatten data into [miniepoch, dim] to feed into PaCMAP, original data is [file, seq_miniepoch_in_file, latent_dim]
    latent_input = np.concatenate(latent_data_windowed, axis=0)
    pat_ids_input = [item for item in pat_ids_list for _ in range(latent_data_windowed[0].shape[0])]
    start_datetimes_input = [item + datetime.timedelta(seconds=stride_sec * i) for item in start_datetimes_epoch for i in range(latent_data_windowed[0].shape[0])]
    stop_datetimes_input = [item + datetime.timedelta(seconds=stride_sec * i) + datetime.timedelta(seconds=win_sec) for item in start_datetimes_epoch for i in range(latent_data_windowed[0].shape[0])]  # ITERATE FROM SHIFTED START using WINDOW_SEC, not from STOP datetimes


    # TRAINING

    if som_object != None: # Object passed directly
        print("SOM object passed directly into subfunction, using that as pretrained SOM")
        som = som_object
        som.reset_device(som_device) # Ensure on proper GPU

    elif som_precomputed_path != None: # Load existing model weights from FILE
        print(f"Loading SOM pretrained weights from FILE: {som_precomputed_path}")
        som = load_som_model(som_precomputed_path, som_device)
    
    else: # Make new Kohonen SOM object and train it
        print(f"Training brand new SOM: gridsize:{som_gridsize}, lr:{som_lr} w/ {som_lr_epoch_decay} decay per epoch, sigma:{som_sigma} w/ {som_sigma_epoch_decay} decay per epoch")     
        som = SOM(grid_size=grid_size, input_dim=latent_input.shape[1], batch_size=som_batch_size, lr=som_lr, 
                    lr_epoch_decay=som_lr_epoch_decay, sigma=som_sigma, sigma_epoch_decay=som_sigma_epoch_decay, sigma_min=som_sigma_min, device=som_device)
        # Train SOM
        som.train(latent_input, num_epochs=som_epochs)
        savepath = savedir + "/som_state_dict.pt"
        save_som_model(som, grid_size=som_gridsize, input_dim=latent_input.shape[1], 
            lr=som_lr, sigma=som_sigma, lr_epoch_decay=som_lr_epoch_decay, sigma_epoch_decay=som_sigma_epoch_decay, sigma_min=som_sigma_min,
            epoch=som_epochs, batch_size=som_batch_size, save_path=savepath)


    # PLOT PREPARATION

    # Pre-Ictal Float values by data point
    preictal_float_input = preictal_weight(atd_file, plot_preictal_color_sec, pat_ids_input, start_datetimes_input, stop_datetimes_input)

    weights = som.get_weights()

    # Initialize maps
    grid_size = (som_gridsize, som_gridsize)
    preictal_sums = np.zeros(grid_size)  # To accumulate float values (class sums)
    hit_map = np.zeros(grid_size)
    neuron_patient_dict = {}  # Dictionary to track unique patients per neuron

    # Inference on all data in batches
    for i in range(0, len(latent_input), som_batch_size):
        batch = latent_input[i:i + som_batch_size]
        batch_patients = pat_ids_input[i:i + som_batch_size]
        batch_labels = preictal_float_input[i:i + som_batch_size]  # Using float values

        batch = torch.tensor(batch, dtype=torch.float32, device=som_device)
        bmu_rows, bmu_cols = som.find_bmu(batch)
        bmu_rows, bmu_cols = bmu_rows.cpu().numpy(), bmu_cols.cpu().numpy()

        # Efficiently accumulate hit counts using NumPy's `np.add.at`
        np.add.at(hit_map, (bmu_rows, bmu_cols), 1)

        # Accumulate the float values (preictal scores) for each node (bmu)
        for j, (bmu_row, bmu_col) in enumerate(zip(bmu_rows, bmu_cols)):
            preictal_sums[bmu_row, bmu_col] += batch_labels[j]  # Add float class values

        # Track unique patients for each node (BMU)
        for j, (bmu_row, bmu_col) in enumerate(zip(bmu_rows, bmu_cols)):
            if (bmu_row, bmu_col) not in neuron_patient_dict:
                neuron_patient_dict[(bmu_row, bmu_col)] = set()
            neuron_patient_dict[(bmu_row, bmu_col)].add(batch_patients[j])  # Track unique patients

    # Normalize preictal_sums to ensure values are between 0 and 1
    min_val = np.min(preictal_sums)
    max_val = np.max(preictal_sums)
    preictal_sums = (preictal_sums - min_val) / (max_val - min_val) 

    # Normalize patient diversity map
    patient_map = np.zeros(grid_size)
    max_unique_patients = max(len(pats) for pats in neuron_patient_dict.values()) if neuron_patient_dict else 1
    for (bmu_row, bmu_col), patients in neuron_patient_dict.items():
        patient_map[bmu_row, bmu_col] = len(patients) / max_unique_patients  # Normalize

    # Compute U-Matrix
    u_matrix = np.zeros(grid_size)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            neuron = weights[i, j]
            distances = [np.linalg.norm(neuron - weights[x, y]) for x in range(max(0, i-1), min(grid_size[0], i+2))
                        for y in range(max(0, j-1), min(grid_size[1], j+2)) if (x, y) != (i, j)]
            u_matrix[i, j] = np.mean(distances)

    
    # PLOTTING

    # Create a 2x3 plot layout
    fig, axes = pl.subplots(2, 4, figsize=(28, 12))

    # 1. U-Matrix
    u_plot = axes[0, 0].pcolor(u_matrix.T, cmap='bone_r')
    axes[0, 0].set_title("U-Matrix")
    pl.colorbar(u_plot, ax=axes[0, 0])

    # 2. Hit Map
    hit_plot = axes[0, 1].pcolor(hit_map.T, cmap='Blues')
    axes[0, 1].set_title("Hit Map")
    pl.colorbar(hit_plot, ax=axes[0, 1])

    # 3. Component Plane (Feature 0)
    comp_plot = axes[1, 0].pcolor(weights[:, :, 0].T, cmap='coolwarm')
    axes[1, 0].set_title("Component Plane (Feature 0)")
    pl.colorbar(comp_plot, ax=axes[1, 0])

    # 4. Patient Diversity Map (New)
    patient_plot = axes[1, 1].pcolor(patient_map.T, cmap='viridis', vmin=0, vmax=1)
    axes[1, 1].set_title("Patient Diversity Map (1.0 = Most Blended)")
    pl.colorbar(patient_plot, ax=axes[1, 1])

    # 5. Pre-Ictal Sums
    blended_plot = axes[0, 2].pcolor(preictal_sums.T, cmap='flare', vmin=0, vmax=1)
    axes[0, 2].set_title("Pre-Ictal Density")
    pl.colorbar(blended_plot, ax=axes[0, 2])

    # 6. Pre-Ictal Sums * Patient Diversity
    rescale_preictal = preictal_sums * patient_map
    rescale_preictal = rescale_preictal / np.max(rescale_preictal)
    if np.any(np.isnan(rescale_preictal)) or np.all(rescale_preictal == 0):  # Ensure the rescaling doesn't lead to values too small or NaN
        print("Warning: rescale_preictal has NaN or zero values.")
        rescale_preictal = np.zeros_like(rescale_preictal)
    blended_plot_recaled = axes[1, 2].pcolor((rescale_preictal).T, cmap='flare', vmin=0, vmax=1)
    axes[1, 2].set_title("Pre-Ictal * Patient Diversity")
    pl.colorbar(blended_plot_recaled, ax=axes[1, 2])

    # 7. Pre-Ictal Sums & SMOOTHED
    sigma_plot = 1  # Adjust this value for more or less smoothing
    preictal_sums_smoothed = gaussian_filter(preictal_sums, sigma=sigma_plot)  # Apply Gaussian filter to smooth the preictal_sums data
    min_val = np.min(preictal_sums_smoothed)
    max_val = np.max(preictal_sums_smoothed)
    preictal_sums_smoothed = (preictal_sums_smoothed - min_val) / (max_val - min_val) 
    blended_plot = axes[0, 3].pcolor(preictal_sums_smoothed.T, cmap='flare', vmin=0, vmax=1)
    axes[0, 3].set_title(f"Pre-Ictal Density - Smoothed (S:{sigma_plot})")
    pl.colorbar(blended_plot, ax=axes[0, 3])

    # 8. Pre-Ictal Sums * Patient Diversity & SMOOTHED
    sigma_plot = 1  # Adjust this value for more or less smoothing
    rescale_preictal_smoothed = gaussian_filter(rescale_preictal, sigma=sigma_plot)  # Apply Gaussian filter to smooth the preictal_sums data
    min_val = np.min(rescale_preictal_smoothed)
    max_val = np.max(rescale_preictal_smoothed)
    rescale_preictal_smoothed = (rescale_preictal_smoothed - min_val) / (max_val - min_val) 
    blended_plot_recaled = axes[1, 3].pcolor((rescale_preictal_smoothed).T, cmap='flare', vmin=0, vmax=1)
    axes[1, 3].set_title(f"Pre-Ictal * Patient Diversity - Smoothed (S:{sigma_plot})")
    pl.colorbar(blended_plot_recaled, ax=axes[1, 3])

    pl.tight_layout()

    # EXPORT FIGURE
    print("Exporting Kohonen to JPG")
    savename_jpg = savedir + f"/SOM_latent_smoothsec{win_sec}_Stride{stride_sec}_epoch{epoch}_preictalSec{plot_preictal_color_sec}_gridsize{som_gridsize}_lr{som_lr}with{som_lr_epoch_decay}decay_sigma{som_sigma}with{som_sigma_epoch_decay}decay_numfeatures{latent_input.shape[0]}_dims{latent_input.shape[1]}_batchsize{som_batch_size}_epochs{som_epochs}.jpg"
    pl.savefig(savename_jpg, dpi=600)

    # TODO Upload to WandB

    pl.close(fig)


    # 3D surface plot of Pre-Ictal Density with multiple views
    fig_3d = pl.figure(figsize=(20, 12)) # Create a figure with multiple 3D subplots
    ax_3d_2 = fig_3d.add_subplot(121, projection='3d')
    ax_3d_3 = fig_3d.add_subplot(122, projection='3d')
    x = np.linspace(0, grid_size[0] - 1, grid_size[0])
    y = np.linspace(0, grid_size[1] - 1, grid_size[1])
    X, Y = np.meshgrid(x, y) 
    Z = preictal_sums_smoothed.T
    surf_2 = ax_3d_2.plot_surface(X, Y, Z, cmap='flare', vmin=0, vmax=1, edgecolor='none')
    ax_3d_2.set_title("Preictal Density: Isometric View -45")
    ax_3d_2.set_xlabel('Grid X')
    ax_3d_2.set_ylabel('Grid Y')
    ax_3d_2.set_zlabel('Density')
    ax_3d_2.view_init(elev=30, azim=-45)  # Isometric view -45
    surf_3 = ax_3d_3.plot_surface(X, Y, Z, cmap='flare', vmin=0, vmax=1, edgecolor='none')
    ax_3d_3.set_title("Preictal Density: Isometric View +45")
    ax_3d_3.set_xlabel('Grid X')
    ax_3d_3.set_ylabel('Grid Y')
    ax_3d_3.set_zlabel('Density')
    ax_3d_3.view_init(elev=30, azim=45)  # Isometric view +45
    print("Exporting 3D Kohonen to JPG")
    savename_jpg = savedir + f"/3Dpreictal_SOM_latent_smoothsec{win_sec}_Stride{stride_sec}_epoch{epoch}_preictalSec{plot_preictal_color_sec}_gridsize{som_gridsize}_lr{som_lr}with{som_lr_epoch_decay}decay_sigma{som_sigma}with{som_sigma_epoch_decay}decay_numfeatures{latent_input.shape[0]}_dims{latent_input.shape[1]}_batchsize{som_batch_size}_epochs{som_epochs}.jpg"
    pl.savefig(savename_jpg, dpi=600)
    pl.close(fig_3d)


    # 3D surface plot of Rescaled Pre-Ictal Density with multiple views
    fig_3d = pl.figure(figsize=(20, 12)) # Create a figure with multiple 3D subplots
    ax_3d_2 = fig_3d.add_subplot(121, projection='3d')
    ax_3d_3 = fig_3d.add_subplot(122, projection='3d')
    x = np.linspace(0, grid_size[0] - 1, grid_size[0])
    y = np.linspace(0, grid_size[1] - 1, grid_size[1])
    X, Y = np.meshgrid(x, y) 
    Z = rescale_preictal_smoothed.T
    surf_2 = ax_3d_2.plot_surface(X, Y, Z, cmap='flare', vmin=0, vmax=1, edgecolor='none')
    ax_3d_2.set_title("Preictal Density * Patient Diversity: Isometric View -45")
    ax_3d_2.set_xlabel('Grid X')
    ax_3d_2.set_ylabel('Grid Y')
    ax_3d_2.set_zlabel('Density')
    ax_3d_2.view_init(elev=30, azim=-45)  # Isometric view -45
    surf_3 = ax_3d_3.plot_surface(X, Y, Z, cmap='flare', vmin=0, vmax=1, edgecolor='none')
    ax_3d_3.set_title("Preictal Density * Patient Diversity: Isometric View +45")
    ax_3d_3.set_xlabel('Grid X')
    ax_3d_3.set_ylabel('Grid Y')
    ax_3d_3.set_zlabel('Density')
    ax_3d_3.view_init(elev=30, azim=45)  # Isometric view +45
    print("Exporting 3D Rescaled Kohonen to JPG")
    savename_jpg = savedir + f"/3DpreictalRescaled_SOM_latent_smoothsec{win_sec}_Stride{stride_sec}_epoch{epoch}_preictalSec{plot_preictal_color_sec}_gridsize{som_gridsize}_lr{som_lr}with{som_lr_epoch_decay}decay_sigma{som_sigma}with{som_sigma_epoch_decay}decay_numfeatures{latent_input.shape[0]}_dims{latent_input.shape[1]}_batchsize{som_batch_size}_epochs{som_epochs}.jpg"
    pl.savefig(savename_jpg, dpi=600)
    pl.close(fig_3d)

    return axes, som



def compute_histograms(data, min_val, max_val, B):
    """
    Compute histogram bin counts for each dimension of a 2D array.

    Parameters:
        data (np.ndarray): Input 2D array of shape (N, M).
        min_val (float): Minimum value for the histogram range.
        max_val (float): Maximum value for the histogram range.
        B (int): Number of bins for the histogram.

    Returns:
        np.ndarray: A 2D array of shape (M, B) containing bin counts for each dimension.
    """
    # Initialize an array to store the histogram bin counts for each dimension
    histograms = np.zeros((data.shape[1], B), dtype=int)

    # Compute the bin edges
    bin_edges = np.linspace(min_val, max_val, B + 1)

    # Iterate over each dimension (column) of the input data
    for i in range(data.shape[1]):
        # Compute the histogram for the current dimension
        hist, _ = np.histogram(data[:, i], bins=bin_edges)
        histograms[i, :] = hist

    return histograms

def histogram_latent(
    pat_ids_list,
    latent_data_windowed, 
    start_datetimes_epoch,  
    stop_datetimes_epoch,
    epoch, 
    FS, 
    win_sec, 
    stride_sec, 
    savedir,
    bincount_perdim=200,
    perc_max=99.99,
    perc_min=0.01):

    # Establish edge of histo bins
    thresh_max = 0
    thresh_min = 0
    for i in range(len(latent_data_windowed)):
        if np.percentile(latent_data_windowed[i], perc_max) > thresh_max: thresh_max = np.percentile(latent_data_windowed[i],perc_max)
        if np.percentile(latent_data_windowed[i], perc_min) < thresh_min: thresh_min = np.percentile(latent_data_windowed[i], perc_min)

    # Range stats
    thresh_range = thresh_max - thresh_min
    thresh_step = thresh_range / bincount_perdim
    all_bin_values = [np.round(thresh_min + i*thresh_step, 2) for i in range(bincount_perdim)]
    zero_bin = np.argmax(np.array(all_bin_values) > 0)

    num_ticks = 10 # Manually set 10 x-ticks for bins
    xtick_positions = np.linspace(0, bincount_perdim - 1, num_ticks).astype(int)  # Create 10 evenly spaced positions
    xtick_labels = [np.round(thresh_min + i*thresh_step, 2) for i in xtick_positions]  # Create labels for these positions
    
    # Count up histo for each dim
    histo_counts = np.zeros([latent_data_windowed[0].shape[1], bincount_perdim], dtype=int)
    for i in range(len(latent_data_windowed)):
        out = compute_histograms(latent_data_windowed[i], thresh_min, thresh_max, bincount_perdim)
        histo_counts = histo_counts + out

    # Log hist data
    log_hist_data = np.log1p(histo_counts) 

    fig = pl.figure(figsize=(10, 25))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Create a custom colormap from pink to purple
    white_to_darkpurple_cmap = LinearSegmentedColormap.from_list(
        "white_to_darkpurple", ["white", "#2E004F"]  # White to dark purple (#2E004F)
    )

    sns.heatmap(log_hist_data, cmap=white_to_darkpurple_cmap, cbar=True, yticklabels=False, xticklabels=False)
    pl.axvline(x=zero_bin, color='gray', linestyle='-', linewidth=2)  # Gray solid line at x = 0

    pl.xticks(xtick_positions, xtick_labels, rotation=45)

    # Customize the plot
    pl.title('Heatmap of Histograms for all Dimensions', fontsize=16)
    pl.ylabel('Dimensions', fontsize=14)
    pl.xlabel('Bins', fontsize=14)

    # **** Save entire figure *****
    if not os.path.exists(savedir + '/JPEGs'): os.makedirs(savedir + '/JPEGs')
    savename_jpg = savedir + f"/JPEGs/pacmap_latent_smoothsec" + str(win_sec) + "Stride" + str(stride_sec) + "_epoch" + str(epoch) + f"_bincount{bincount_perdim}.jpg"
    pl.savefig(savename_jpg, dpi=300)


    pl.close(fig)

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
    artifact_bool=False,
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
    seiz_type_list = ['FBTC', 'FIAS', 'FAS_to_FIAS', 'FAS', 'Subclinical', 'Focal unknown awareness', 'Unknown', 'Non-electrographic', 'Artifact']
    delete_seiz_type_bool_list = [FBTC_bool, FIAS_bool, FAS_to_FIAS_bool, FAS_bool, subclinical_bool, focal_unknown_bool, unknown_bool, non_electro_bool, artifact_bool]
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
    delete_list_A = [i for i, val in enumerate(pat_seiz_starttime_str) if (val=='nan' or val=='Unknown' or val=='None')]
    delete_list_B = [i for i, val in enumerate(pat_seiz_stoptime_str) if (val=='nan' or val=='Unknown' or val=='None')]
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

def preictal_weight(atd_file, plot_preictal_color_sec, pat_ids_input, start_datetimes_input, stop_datetimes_input):
    
    data_window_preictal_score = np.zeros_like(pat_ids_input, dtype=float)

    # Generate numerical IDs for each unique patient, and give each datapoint an ID
    unique_ids = list(set(pat_ids_input))
    id_to_index = {id: idx for idx, id in enumerate(unique_ids)}  # Create mapping dictionary
    pat_idxs = [id_to_index[id] for id in pat_ids_input]

    pat_seiz_start_datetimes, pat_seiz_stop_datetimes, pat_seiz_types_str = [-1] * len(unique_ids), [-1] * len(unique_ids), [-1] * len(unique_ids)
    for i in range(len(unique_ids)):
        id_curr = unique_ids[i]
        idx_curr = id_to_index[id_curr]
        pat_seiz_start_datetimes[i], pat_seiz_stop_datetimes[i], pat_seiz_types_str[i] = get_pat_seiz_datetimes(id_curr, atd_file) 

    for i in range(len(pat_idxs)):
        data_window_start = start_datetimes_input[i]
        data_window_stop = stop_datetimes_input[i]
        
        seiz_starts_curr, seiz_stops_curr, seiz_types_curr = pat_seiz_start_datetimes[pat_idxs[i]], pat_seiz_stop_datetimes[pat_idxs[i]], pat_seiz_types_str[pat_idxs[i]]

        for j in range(len(seiz_starts_curr)):
            seiz_start = seiz_starts_curr[j]
            buffered_preictal_start = seiz_start - datetime.timedelta(seconds=plot_preictal_color_sec)

            # Compute time difference from seizure start (0 at start of preictal buffer, increasing as closer to seizure)
            # Assign value of 1 for in seizure

            # Case where end of data window is in pre-ictal buffer (ignore start of preictal window)
            if data_window_stop < seiz_start and data_window_stop > buffered_preictal_start:
                dist_to_seizure = (seiz_start - data_window_stop).total_seconds()
                new_score = 1.0 - (dist_to_seizure / plot_preictal_color_sec)
                data_window_preictal_score[i] = max(data_window_preictal_score[i], new_score)  
                # Keep max score, and do NOT break because could find higher value

            # Case where end of the window is in the seizure
            elif data_window_stop > seiz_start and data_window_stop < seiz_stop:
                data_window_preictal_score[i] = 1 
                break # Cannot get higher value

            # Case where start of the window overlaps the preictal/ictal buffer, but end is past seizure end
            elif data_window_start > buffered_preictal_start and data_window_start < seiz_stop:
                data_window_preictal_score[i] = 1 
                break # Cannot get higher value

    # Ensure values remain between 0 and 1
    return np.clip(data_window_preictal_score, 0, 1)

