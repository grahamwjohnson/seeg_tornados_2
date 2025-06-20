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
    """
    A logger class that writes structured log entries to a gzipped JSONL (JSON Lines) file.

    This class provides functionality to log structured data entries to a gzipped file, 
    where each log entry is written as a JSON object on a separate line. The log file 
    is initialized with a header row if it doesn't already exist, and every log entry 
    is timestamped for tracking when it was logged.

    Attributes:
    -----------
    filename : str
        The name of the gzipped log file where log entries will be stored.

    Methods:
    --------
    __init__(self, filename):
        Initializes the logger and sets the log file name. If the log file doesn't exist, 
        it is created with a header entry.

    initialize_file(self):
        Initializes the log file if it doesn't already exist. The file is created with a 
        header row containing predefined columns and a timestamp indicating when the file 
        was initialized.

    log(self, data):
        Appends a new log entry to the log file. Each entry contains a timestamp and the 
        provided data, which is written as a JSON object on a new line.

    Usage Example:
    --------------
    # Create a logger for a file called 'log.jsonl'
    logger = JSONLinesLogger('log.jsonl')

    # Log some data
    logger.log({
        "file_class": "example_class",
        "random_hash_modifier": "123abc",
        "start_idx": 0,
        "encode_token_samples": 10,
        "end_idx": 10,
        "file_name": "example_file.txt"
    })
    
    The log entries are appended to 'log.jsonl' with a timestamp.
    """
    
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
    """
    A PyTorch Dataset class for handling SEEG (stereoelectroencephalography) data in the context of the Tornado project.
    This dataset is designed to manage and process SEEG recordings from multiple patients, enabling efficient loading,
    transformation, and batching of data for training and evaluation of machine learning models.

    The dataset supports two modes of operation:
    1. **Single-Patient Sequential Mode**: Loads data sequentially from a single patient.
    2. **Multi-Patient Random Generation Mode**: Randomly generates batches of data from multiple patients using a
       background process for efficient data loading.

    Key Features:
    - Supports loading SEEG data from `.pkl` files.
    - Handles patient-specific data splits (train/val/test) based on seizure annotations or time ranges.
    - Manages padded channels and latent dimensions for consistent input shapes.
    - Provides options for random data generation and logging for reproducibility.

    Args:
        gpu_id (int): The GPU ID used for logging and data processing.
        pat_list (list): List of patient IDs.
        pat_dirs (list): List of directories containing patient-specific data.
        FS (int): Sampling frequency of the SEEG data.
        atd_file (str): Path to the annotation file containing seizure and SPES (Single Pulse Electrical Stimulation) information.
        pat_num_channels_LUT (dict): Lookup table mapping patient IDs to their respective number of channels.
        data_dir_subfolder (str): Subfolder within each patient directory containing the SEEG data files.
        intrapatient_dataset_style (list): A 2-element list specifying the dataset style and split:
            - First element (int):
                - 0: Seizures with pre- and post-ictal taper (no SPES).
                - 1: All data (no SPES).
                - 2: All data (with SPES).
                - 3: Only SPES.
            - Second element (int):
                - 0: Train dataset.
                - 1: Validation dataset.
                - 2: Test dataset.
                - 3: All files.
        hour_dataset_range (list): Time range (in hours) for filtering data files.
        dataset_pic_dir (str): Directory for saving dataset visualization or metadata.
        num_samples (int): Total number of samples in the dataset.
        transformer_seq_length (int): Sequence length for transformer-based models.
        encode_token_samples (int): Number of samples used for encoding tokens.
        random_pulls_in_batch (bool): Whether to randomly pull data within a batch.
        num_rand_hashes (int): Number of random hashes for data generation.
        padded_channels (int): Number of padded channels for consistent input shapes.
        latent_dim (int): Latent dimension of the data.
        num_forward_passes (int): Number of forward passes for random data generation.
        random_gen_script_path (str): Path to the script for random data generation.
        env_python_path (str): Path to the Python executable for running the random generator script.
        data_logger_enabled (bool): Whether to enable data logging.
        data_logger_file (str): File path for saving data logs.
        **kwargs: Additional keyword arguments for flexibility.

    Attributes:
        gpu_id (int): The GPU ID used for logging and data processing.
        num_samples (int): Total number of samples in the dataset.
        encode_token_samples (int): Number of samples used for encoding tokens.
        FS (int): Sampling frequency of the SEEG data.
        num_windows (int): Number of windows derived from the dataset.
        random_pulls_in_batch (bool): Whether to randomly pull data within a batch.
        num_rand_hashes (int): Number of random hashes for data generation.
        latent_dim (int): Latent dimension of the data.
        padded_channels (int): Number of padded channels for consistent input shapes.
        transformer_seq_length (int): Sequence length for transformer-based models.
        num_forward_passes (int): Number of forward passes for random data generation.
        data_logger_enabled (bool): Whether data logging is enabled.
        data_logger_file (str): File path for saving data logs.
        kwargs (dict): Additional keyword arguments.
        random_gen_script_path (str): Path to the script for random data generation.
        env_python_path (str): Path to the Python executable for running the random generator script.
        pat_num_channels (np.ndarray): Array storing the number of channels for each patient.
        pat_ids (list): List of patient IDs.
        pat_dirs (list): List of directories containing patient-specific data.
        pat_fnames (list): List of lists containing filenames for each patient's data.
        single_pat_seq (bool): Whether the dataset is in single-patient sequential mode.
        pat_curr (int): Index of the current patient in single-patient mode.
        tmp_dir (str): Temporary directory for random data generation.
        fname_csv (str): CSV file containing filenames for random data generation.
        rand_generator_process (subprocess.Popen): Process handle for the random data generator.

    Methods:
        __init__: Initializes the dataset and sets up patient-specific data.
        initiate_generator: Launches a background process for random data generation.
        kill_generator: Stops the random data generation process.
        get_script_filename: Returns the filename of the current script.
        get_pat_count: Returns the number of patients in the dataset.
        set_pat_curr: Sets the current patient for single-patient sequential mode.
        get_pat_curr: Returns information about the current patient.
        __len__: Returns the length of the dataset.
        __getitem__: Retrieves a data sample from the dataset.
    """

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

        # Incase the random generator is never initialized 
        self.tmp_dir = None

        # ### Now determine which files in the data directory shoud be given to the dataset
        # Dataset splits is a tuple of 3 floats for train/val/test and must add up to 1
        # NOTE: the ratio referes to seizures had, no longer is based on time in EMU        
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
        if self.tmp_dir != None:
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

    def update_pat_inference_status(self, inference_save_dir, inference_window_sec_list, inference_stride_sec_list):
        if self.single_pat_seq == False: raise Exception("Must be in single pat seq to call this function")
        first_save_dir = f"{inference_save_dir}/latent_files/{inference_window_sec_list[0]}SecondWindow_{inference_stride_sec_list[0]}SecondStride"
        pat_completed_files = glob.glob(f'{first_save_dir}/{self.pat_ids[self.pat_curr]}*.pkl')
        pat_all_files = self.pat_fnames[self.pat_curr]

        pat_completed_roots = ['_'.join(x.split("/")[-1].split("_")[0:6]) for x in pat_completed_files]
        pat_all_roots = ['_'.join(x.split("/")[-1].split("_")[0:6]) for x in pat_all_files]

        missing_indices = []
        completed_set = set(pat_completed_roots)
        for index, root in enumerate(pat_all_roots):
            if root not in completed_set:
                missing_indices.append(index)

        # Create a new list containing only the files at the missing indices
        updated_pat_fnames = [pat_all_files[i] for i in missing_indices]

        # Update self.pat_fnames for the current patient
        self.pat_fnames[self.pat_curr] = updated_pat_fnames

        # Optional: Print the updated list to verify
        print(f"[{self.pat_ids[self.pat_curr]}] Updated self.pat_fnames: There are {len(missing_indices)} files left to process out of {len(pat_all_roots)}")

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

class SEEG_BSP_Dataset(Dataset):

    def __init__(
        self,
        gpu_id,

        # BSP Info
        bsp_source_dir,
        bsp_transformer_seq_length,
        bsp_epoch_dataset_size,
        bsp_latent_dim,

        # BSE Info (needed tp pull proper data sizes for feeding through pretrained BSE)
        transformer_seq_length, # for BSE
        encode_token_samples,
        padded_channels,

        **kwargs):

        self.gpu_id = gpu_id

        # BSP Info
        self.bsp_source_dir = bsp_source_dir
        self.bsp_transformer_seq_length = bsp_transformer_seq_length
        self.bsp_epoch_dataset_size = bsp_epoch_dataset_size
        self.bsp_latent_dim = bsp_latent_dim
        self.kwargs = kwargs

        # BSE info
        self.bse_samples = transformer_seq_length * encode_token_samples
        self.padded_channels = padded_channels

        file_names = glob.glob(f'{bsp_source_dir}/*.pkl')
        pat_ids_all = [x.split("/")[-1].split("_")[0] for x in file_names]
        self.pat_ids_unique = list(set(pat_ids_all))
        self.pat_ids_unique.sort()
        self.num_pats = len(self.pat_ids_unique)
        self.files_bypat = [''] * self.num_pats
        self.numfiles_bypat = [-1] * self.num_pats
        for i in range(self.num_pats):
            self.files_bypat[i] = glob.glob(f'{self.bsp_source_dir.replace('*', self.pat_ids_unique[i])}/*.pkl')
            self.numfiles_bypat[i] = len(self.files_bypat[i])

        print(f"Num Pats in Dataset: {self.num_pats}, Num Total Little Pickles in Dataset: {sum(self.numfiles_bypat)} (range per pat: {min(self.numfiles_bypat)}-{max(self.numfiles_bypat)})")


    def make_ch_index_vec(self, padded_positions, shuffled_channel_indices):
        """
        Returns a vector of length self.padded_channels where:
        - Entries at padded_positions are filled with shuffled_channel_indices.
        - All other entries are -1.
        """
        ch_index_vec = torch.ones(self.padded_channels, dtype=torch.long, device=padded_positions.device) * -1
        ch_index_vec[padded_positions] = shuffled_channel_indices
        return ch_index_vec

    def __len__(self):
        return self.bsp_epoch_dataset_size
    
    def __getitem__(self, idx): 
        rand_pat_idx = int(random.uniform(0, self.num_pats))
        rand_file_idx = int(random.uniform(0, self.numfiles_bypat[rand_pat_idx]))
        rand_filename = self.files_bypat[rand_pat_idx][rand_file_idx]

        # Load the file's pickle
        with open(rand_filename, 'rb') as file: 
            file_data = pickle.load(file)  # shape: [channels, time]

        samples_in_file = file_data.shape[1]
        samples_needed = self.bsp_transformer_seq_length * self.bse_samples
        rand_start_idx = int(random.uniform(0, samples_in_file - samples_needed - 1))
        x = file_data[:, rand_start_idx:rand_start_idx + samples_needed]
        x = torch.tensor(x, dtype=torch.float32)

        # Reshape into [channels, seq_len, samples_per_step]
        x = x.view(x.shape[0], self.bsp_transformer_seq_length, self.bse_samples)

        actual_channels = x.shape[0]
        assert actual_channels <= self.padded_channels, "More channels than padded_channels!"

        # Prepare output tensor [sequence, padded_channels, latent_dim]
        padded = torch.zeros(self.bsp_transformer_seq_length, self.padded_channels, self.bse_samples, dtype=torch.float32)

        # Randomize channel mapping for each time step independently
        rand_ch_orders = torch.zeros(self.bsp_transformer_seq_length, self.padded_channels)

        # Shuffle ONCE per entire token sequence, so each BSP token has same channel order
        shuffled_channel_indices = torch.randperm(actual_channels)
        padded_positions = torch.randperm(self.padded_channels)[:actual_channels]

        # Assign tokens
        for t in range(self.bsp_transformer_seq_length):

            # Assign channel orders to variable (done within FOR loop to ensure it is clear that they all have the same channel order)
            rand_ch_orders[t, :] = self.make_ch_index_vec(padded_positions, shuffled_channel_indices)

            # Assign data to variable
            for src_idx, dst_pos in zip(shuffled_channel_indices, padded_positions):
                padded[t, dst_pos, :] = x[src_idx, t, :]  # use the corresponding time step

        # Unsqueeze a dimension and transpose to be ready for BSE
        # [seq, padded_channels, FS] --> [seq, FS, padded_channel, 1]
        out = padded.permute(0, 2, 1).unsqueeze(3)

        return out, rand_filename, rand_pat_idx, rand_start_idx, rand_ch_orders






