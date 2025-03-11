# Standard Python Libraries
from __future__ import print_function, division
import os, sys, time, pickle, glob, random, csv, json, gzip, shutil
from datetime import datetime, timedelta
from functools import partial
from concurrent.futures import ThreadPoolExecutor

# Third-Party Libraries
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

# Local Imports
from utilities import utils_functions
 
'''
@author: grahamwjohnson
2023-2025

'''

pd.set_option('display.max_rows', None)

class JSONLinesLogger:
    def __init__(self, filename):
        self.filename = filename
        self.initialize_file()

    def initialize_file(self):
        """Initialize the file if it doesn't exist."""
        if not os.path.isfile(self.filename):
            header = ['file_class', 'random_hash_modifier', 'start_idx', 'encode_token_samples', 'end_idx', 'file_name']
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = {"timestamp": timestamp, "data": header}
            with gzip.open(self.filename, 'wt', encoding='UTF-8') as f:
                f.write(json.dumps(log_entry) + '\n')

    def log(self, data):
        """Log data with a timestamp, appending to the file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {"timestamp": timestamp, "data": data}

        # Append the log entry as a new line
        with gzip.open(self.filename, 'at', encoding='UTF-8') as f:
            f.write(json.dumps(log_entry) + '\n')

# Seizure based datset curation
class SEEG_Tornado_Dataset(Dataset):
    def __init__(
        self,
        gpu_id,
        pat_list,
        pat_dirs,
        model_dir,
        FS,
        atd_file, 
        pat_num_channels_LUT, 
        data_dir_subfolder,
        intrapatient_dataset_style, 
        hour_dataset_range,
        dataset_pic_dir, 
        num_samples,
        transformer_seq_length,
        encode_token_samples,
        random_pulls_in_batch, 
        num_rand_hashes, 
        padded_channels,
        latent_dim,
        num_forward_passes,
        random_gen_script_path,
        env_python_path,
        data_logger_enabled,
        data_logger_file,
        **kwargs):

        self.gpu_id = gpu_id
        self.num_samples = num_samples
        self.encode_token_samples = encode_token_samples
        self.FS = FS
        self.num_windows = int((self.num_samples - self.encode_token_samples)/self.encode_token_samples) - 2
        self.random_pulls_in_batch = random_pulls_in_batch
        self.num_rand_hashes = num_rand_hashes
        self.latent_dim = latent_dim
        self.padded_channels = padded_channels
        self.transformer_seq_length = transformer_seq_length
        self.num_forward_passes = num_forward_passes
        self.data_logger_enabled = data_logger_enabled
        self.data_logger_file = data_logger_file
        self.kwargs = kwargs
        self.random_gen_script_path = random_gen_script_path
        self.env_python_path = env_python_path
    
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

        if self.data_logger_enabled: self.data_logger = JSONLinesLogger(self.data_logger_file) # Initiate

    def initiate_generator(self):
        # Launches a seperate thread that will randomly generate forward pass ready data from little pickles
        # Actually a bit slower than pulling in little pickles directly, but can always be running in background
        self.tmp_dir = f"/dev/shm/tornado_tmp_{utils_functions.random_filename_string()}"
        self.fname_csv = f"{self.tmp_dir}/fnames.csv"
        if not os.path.exists(self.tmp_dir): os.makedirs(self.tmp_dir)
        with open(self.fname_csv, 'w', newline='') as file:
            writer = csv.writer(file, delimiter = ',')
            writer.writerows(self.pat_fnames)
        
        # Start the generator
        self.rand_generator_process = utils_functions.run_script_from_shell(self.env_python_path, self.random_gen_script_path, self.tmp_dir, 'fnames.csv', f"{self.num_rand_hashes}")

    def kill_generator(self):
        # To kill the generator, just delete the tmp dir path
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def get_script_filename(self):
        return __file__

    def get_pat_count(self):
        return len(self.pat_ids)

    def set_pat_curr(self, idx):
        self.pat_curr = idx
        
        if idx < 0: 
            self.single_pat_seq = False # resets back to all pats with random generation
            self.initiate_generator() # Starts the random generator

        else: 
            self.single_pat_seq = True # mandated if setting current pat
            self.kill_generator()

    def get_pat_curr(self):
        return self.pat_curr, self.pat_ids[self.pat_curr], self.pat_dirs[self.pat_curr], self.pat_fnames[self.pat_curr]

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

            # Label for classifier
            file_class_label = torch.tensor(self.pat_curr) 

            return data_tensor, file_name, file_class_label

        else:
            # Sometimes must wait for the subprocess to generate data
            while True:

                # Merely need to pull from the top of the random generator tmp_dir
                pkls_curr = glob.glob(f"{self.tmp_dir}/*.pkl") 

                if len(pkls_curr) > 2:

                    pkl_idxs = [int(pkls_curr[i].split("/")[-1].split(".")[0].split("_")[-1]) for i in range(len(pkls_curr))]
                    min_idx = min(pkl_idxs)
                    the_one = glob.glob(f"{self.tmp_dir}/T*_{min_idx}.pkl")[0] # From any thread

                    with open(the_one, "rb") as f: data = pickle.load(f)
                    data_tensor = data['data_tensor']
                    file_name = data['file_name']
                    file_class = data['file_class']
                    hash_channel_order = data['hash_channel_order']
                    hash_pat_embedding = data['hash_pat_embedding']
                    random_hash_modifier = data['random_hash_modifier']
                    start_idx = data['start_idx']
                    autoencode_samps = data['autoencode_samps']
                    end_idx = data['end_idx']

                    # Delete the used batch pickle
                    os.remove(the_one)

                    # Convert data_tensor to float32
                    data_tensor = data_tensor.to(torch.float32)

                    # If logger is enabled, save pat_idx & what file was pulled 
                    if self.data_logger_enabled: 
                        for i in range(len(file_name)):
                            self.data_logger.log([int(file_class[i]), random_hash_modifier[i], start_idx[i], autoencode_samps[i], end_idx[i], file_name[i]])

                    return data_tensor, file_name, file_class, hash_channel_order, hash_pat_embedding
                
                else:
                    # print("Random Generator not fast enough")
                    time.sleep(0.5)

