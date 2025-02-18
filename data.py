from __future__ import print_function, division
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import pickle
import glob
from datetime import datetime, timedelta
import sys
from utilities import utils_functions
import random
 
pd.set_option('display.max_rows', None)

# Seizure based datset curation
class SEEG_Tornado_Dataset(Dataset):
    def __init__(
            self,
            gpu_id,
            pat_list,
            pat_dirs,
            FS,
            atd_file, 
            pat_num_channels_LUT, 
            data_dir_subfolder,
            intrapatient_dataset_style, 
            hour_dataset_range,
            dataset_pic_dir, 
            num_samples,
            transformer_seq_length,
            autoencode_samples,
            periictal_augmentation_perc,
            preictal_augmentation_seconds,
            random_pats_in_batch,
            random_files_per_pat,
            random_epochs_per_file,
            num_rand_hashes, 
            padded_channels,
            latent_dim,
            num_forward_passes,
            **kwargs):

        self.gpu_id = gpu_id
        self.num_samples = num_samples
        self.autoencode_samples = autoencode_samples
        self.FS = FS
        self.kwargs = kwargs

        self.periictal_augmentation_perc = periictal_augmentation_perc
        self.preictal_augmentation_seconds = preictal_augmentation_seconds

        self.num_windows = int((self.num_samples - self.autoencode_samples)/self.autoencode_samples) - 2
        
        self.random_pats_in_batch = random_pats_in_batch
        self.random_files_per_pat = random_files_per_pat
        self.random_epochs_per_file = random_epochs_per_file

        self.num_rand_hashes = num_rand_hashes
        self.latent_dim = latent_dim
        self.padded_channels = padded_channels
        
        self.transformer_seq_length = transformer_seq_length

        self.num_forward_passes = num_forward_passes
    
        # Get ONLY the .pkl file names in the subdirectories of choice
        # self.data_dir = data_dir

        # Get the channel count (will be used to truncate if smaller than actual patient's channel count)
        self.pat_num_channels = np.ones(len(pat_list), dtype=np.int32)*-1
        self.pat_ids = pat_list
        self.pat_dirs = pat_dirs

        # Initilaize the lists of lists for data filenames
        self.pat_fnames = [[] for i in range(len(self.pat_ids))]

        # ### Now determine which files in the data directory shoud be given to the dataset
        # Dataset splits is a tuple of 3 floats for train/val/test and must add up to 1
        # NOTE: the ratio referes to seizures had, no longer is based on time in EMU
        # dataset_style 2 int list:
            # First int:
            # 0: Seizures with pre and post ictal taper (no SPES)
            # 1: All data (no SPES)
            # 2: All data (with SPES)
            # 3: Only SPES
            # Second int:
            # 0 return the train dataset
            # 1 return the val dataset
            # 2 return the test dataset
            # 3 return ALL files
        
        for i in range(0, len(pat_list)):
            
            self.pat_num_channels[i] = utils_functions.get_num_channels(pat_list[i], pat_num_channels_LUT)

            self.pat_fnames[i] = utils_functions.get_desired_fnames(
                gpu_id = self.gpu_id,
                pat_id = pat_list[i], 
                atd_file = atd_file, 
                data_dir = f"{pat_dirs[i]}{data_dir_subfolder}", 
                intrapatient_dataset_style = intrapatient_dataset_style, 
                hour_dataset_range = hour_dataset_range,
                dataset_pic_dir = dataset_pic_dir,
                **kwargs)

            # Sort the filenames
            self.pat_fnames[i] = utils_functions.sort_filenames(self.pat_fnames[i])

    def get_script_filename(self):
        return __file__

    def get_pat_count(self):
        return len(self.pat_ids)

    def set_pat_curr(self, idx):
        self.pat_curr = idx
        
        if idx < 0: self.single_pat_seq = False # resets back to all pats with random generation
        else: self.single_pat_seq = True # mandated if setting current pat

    def get_pat_curr(self):
        return self.pat_curr, self.pat_ids[self.pat_curr], self.pat_dirs[self.pat_curr], self.pat_fnames[self.pat_curr]

    def rand_start_idxs(self):

        last_possible_start_idx = self.num_samples - (self.transformer_seq_length + 1) * self.autoencode_samples 

        start_idxs = np.zeros(self.random_epochs_per_file, dtype=int)
        for i in range(self.random_epochs_per_file):
            np.random.seed(seed=None) 
            start_idxs[i] = np.random.randint(0, last_possible_start_idx+1)
        
        return start_idxs

    def __len__(self):

        if self.single_pat_seq:
            return len(self.pat_fnames[self.pat_curr])

        else:
            return self.num_forward_passes
    
    def __getitem__(self, idx): 
        
        if self.single_pat_seq:
            
            file = open(self.pat_fnames[self.pat_curr][idx],'rb')
            data = pickle.load(file) 
            file.close()
            data_tensor = torch.FloatTensor(data)

            file_name = self.pat_fnames[self.pat_curr][idx].split("/")[-1].split(".")[0]      

            file_class_label = torch.tensor(self.pat_curr) 

            return data_tensor, file_name, file_class_label

        else:
            # Calculate batchsize
            self.batchsize = self.random_pats_in_batch * self.random_files_per_pat * self.random_epochs_per_file 
            data_tensor_np = np.zeros((self.batchsize, self.transformer_seq_length, self.padded_channels, self.autoencode_samples), dtype=np.float32)
            file_name = [-1]*self.batchsize
            file_class = torch.empty(self.batchsize, dtype=torch.long)
            hash_channel_order = [-1]*self.batchsize
            hash_pat_embedding = torch.empty((self.batchsize, self.latent_dim), dtype=torch.float32)

            # Random samplings of pat/file/epoch
            np.random.seed(seed=None)
            pat_idxs = np.random.choice(self.get_pat_count(), self.random_pats_in_batch, replace=True)
            file_idxs = [np.random.choice(len(self.pat_fnames[pi]), self.random_files_per_pat, replace=True) for pi in pat_idxs]
            start_idxs = self.rand_start_idxs()

            idx_output = -1
            for i in range(len(pat_idxs)):
                p_curr = pat_idxs[i]
                for j in range(len(file_idxs[i])):
                    f_curr = file_idxs[i][j]

                    # Load the file's pickle
                    file = open(self.pat_fnames[p_curr][f_curr],'rb')
                    data = pickle.load(file) 
                    file.close()

                    # Pull out epoch's from this file
                    for k in range(len(start_idxs)):
                        idx_output = idx_output + 1

                        start_idx_curr = start_idxs[k]

                        file_name[idx_output] = self.pat_fnames[p_curr][f_curr].split("/")[-1].split(".")[0]
                        file_class[idx_output] = p_curr

                        # print(f"p_curr: {p_curr}, f_curr: {f_curr}, e_curr: {e_curr}, idx_output: {idx_output}")

                        # e_curr = start_idxs[k]

                        # Generate hashes for feedforward conditioning
                        np.random.seed(seed=None) 
                        rand_modifer = int(random.uniform(0, self.num_rand_hashes -1))
                        hash_pat_embedding[idx_output], hash_channel_order[idx_output] = utils_functions.hash_to_vector(
                            input_string=self.pat_ids[p_curr], 
                            num_channels=data.shape[0], 
                            latent_dim=self.latent_dim, 
                            modifier=rand_modifer)

                        # Collect sequential embeddings for transformer by running sequential raw data windows through BSE N times 
                        for embedding_idx in range(0, self.transformer_seq_length):
                            # Pull out data for this window
                            end_idx = start_idx_curr + self.autoencode_samples * embedding_idx + self.autoencode_samples 
                            data_tensor_np[idx_output, embedding_idx, :len(hash_channel_order[idx_output]), :] = data[hash_channel_order[idx_output], end_idx-self.autoencode_samples : end_idx] # Padding implicit in zeros intitilization

            # Convert to Torch Tensor
            data_tensor = torch.Tensor(data_tensor_np)

            return data_tensor, file_name, file_class, hash_channel_order, hash_pat_embedding
        

