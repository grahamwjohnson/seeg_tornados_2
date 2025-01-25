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
            decode_samples,
            periictal_augmentation_perc,
            preictal_augmentation_seconds,
            **kwargs
            ):

        self.gpu_id = gpu_id
        self.num_samples = num_samples
        self.decode_samples = decode_samples
        self.FS = FS
        self.kwargs = kwargs

        self.periictal_augmentation_perc = periictal_augmentation_perc
        self.preictal_augmentation_seconds = preictal_augmentation_seconds

        self.num_windows = int((self.num_samples - self.decode_samples)/self.decode_samples) - 2
    
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
                dataset_pic_dir = dataset_pic_dir)

            # Sort the filenames
            self.pat_fnames[i] = utils_functions.sort_filenames(self.pat_fnames[i])

    def get_script_filename(self):
        return __file__

    def get_pat_count(self):
        return len(self.pat_ids)

    def set_pat_curr(self, idx):
        self.pat_curr = idx
        
        if idx < 0: self.single_pat_seq = False # resets back to all pats
        else: self.single_pat_seq = True # mandated if setting current pat

    def get_pat_curr(self):
        return self.pat_curr, self.pat_ids[self.pat_curr], self.pat_dirs[self.pat_curr], self.pat_fnames[self.pat_curr]

    def __len__(self):

        if self.single_pat_seq:
            return len(self.pat_fnames[self.pat_curr])

        else:
            # Return file_count for the patient with min number of files
            file_counts = [-1] * len(self.pat_fnames)
            for i in range(0, len(self.pat_fnames)):
                file_counts[i] = len(self.pat_fnames[i])

            return min(file_counts)
    
    def __getitem__(self, idx): 
        

        if self.single_pat_seq:
            
            file = open(self.pat_fnames[self.pat_curr][idx],'rb')
            data = pickle.load(file) 
            file.close()
            data_tensor = torch.FloatTensor(data)

            file_name = self.pat_fnames[self.pat_curr][idx].split("/")[-1].split(".")[0]         

            return data_tensor, file_name


        else:
            file_name_by_pat = [-1]*len(self.pat_ids)
            data_tensor_by_pat = [-1]*len(self.pat_ids)
            for pat_idx in range(0, len(self.pat_ids)):

                # Regenerate random file idxs each call
                np.random.seed(seed=None)
                rand_idx = int(random.uniform(0, len(self.pat_fnames[pat_idx]) -1))


                # TODO this is probably the better place to implement random peri-ictal augmentation



                # Load the epoch's pickle
                file = open(self.pat_fnames[pat_idx][rand_idx],'rb')
                data = pickle.load(file) 
                file.close()
                data_tensor_by_pat[pat_idx] = torch.FloatTensor(data)
                
                file_name_by_pat[pat_idx] = self.pat_fnames[pat_idx][rand_idx].split("/")[-1].split(".")[0]

            return data_tensor_by_pat, file_name_by_pat
        


