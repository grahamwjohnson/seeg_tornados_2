import sys
import yaml
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from utilities import utils_functions
import time
import os
import csv
import glob
import numpy as np
import torch
import pickle
import random
import threading 

'''
@author: grahamwjohnson
March 2025

To be run continously as a subprocess to generate adequate random data to keep the Trainer DataLoader happy

'''

def rand_start_idx(num_samples, transformer_seq_length, autoencode_samples):

    last_possible_start_idx = num_samples - (transformer_seq_length + 1) * autoencode_samples 

    np.random.seed(seed=None) 
    start_idx = np.random.randint(0, last_possible_start_idx+1)
    
    return start_idx

# Modified function to load a single sample (pat/file/epoch)
def load_data_sample(pat_idx, file_idx, start_idx, pat_fnames, pat_ids, latent_dim, transformer_seq_length, padded_channels, autoencode_samples, num_rand_hashes, hash_output_range):

    # Load the file's pickle
    with open(pat_fnames[pat_idx][file_idx], 'rb') as file:
        data = pickle.load(file)

    # Generate hashes for feedforward conditioning
    rand_modifier = int(random.uniform(0, num_rand_hashes - 1))
    hash_pat_embedding, hash_channel_order = utils_functions.hash_to_vector(
        input_string=pat_ids[pat_idx], 
        num_channels=data.shape[0], 
        latent_dim=latent_dim, 
        modifier=rand_modifier,
        hash_output_range=hash_output_range)

    # data_tensor_np = np.zeros((transformer_seq_length, padded_channels, autoencode_samples ), dtype=np.float16)
    # # Collect sequential embeddings for transformer by running sequential raw data windows through BSE N times
    # for embedding_idx in range(0, transformer_seq_length):
    #     end_idx = start_idx + autoencode_samples * embedding_idx + autoencode_samples
    #     data_tensor_np[embedding_idx, :len(hash_channel_order), :] = data[hash_channel_order, end_idx - autoencode_samples : end_idx]  # Padding implicit in zeros initialization

    # initialize & fill the data tensor
    end_idx = start_idx +  autoencode_samples * transformer_seq_length
    data_tensor_np = np.zeros((padded_channels, transformer_seq_length * autoencode_samples ), dtype=np.float16)
    data_tensor_np[ :len(hash_channel_order), :] = data[hash_channel_order, start_idx : end_idx]  # [Seq, padded_channels, autoencode_sample]
    data_tensor_np = np.swapaxes(data_tensor_np.reshape(data_tensor_np.shape[0], transformer_seq_length, autoencode_samples), 0,1)

    # Add file info
    file_name = pat_fnames[pat_idx][file_idx].split("/")[-1].split(".")[0]
    file_class = pat_idx

    return data_tensor_np, file_name, file_class, hash_channel_order, hash_pat_embedding, rand_modifier, start_idx, end_idx

def thread_task(thread_num, nested_max_workers, tmp_dir, pat_fnames, num_buffer_batches, pat_ids, latent_dim, batchsize, num_samples, transformer_seq_length, padded_channels, autoencode_samples, num_rand_hashes, hash_output_range):
    
    file_idx_next = 0
    while True:

        start_time = time.time()

        pkls_curr = glob.glob(f"{tmp_dir}/*.pkl") 
        B = start_time - time.time()

        if len(pkls_curr) < num_buffer_batches:

            # Initialize parallel pull variables
            data_tensor_np = np.zeros((batchsize, transformer_seq_length, padded_channels, autoencode_samples), dtype=np.float16)
            file_name = [-1]*batchsize
            file_class = torch.empty(batchsize, dtype=torch.long)
            hash_channel_order = [-1]*batchsize
            hash_pat_embedding = torch.empty((batchsize, latent_dim), dtype=torch.float16)
            random_hash_modifier = [-1]*batchsize
            start_idx = [-1]*batchsize
            autoencode_samps = [-1]*batchsize
            end_idx = [-1]*batchsize

            C = start_time - time.time()

            # Random samplings of pat/file/epoch
            np.random.seed(seed=None)
            pat_idxs = np.random.choice(num_pats, batchsize, replace=True)
            file_idxs = [int(np.random.choice(len(pat_fnames[pi]), 1, replace=True)) for pi in pat_idxs]
            start_idxs = [rand_start_idx(num_samples, transformer_seq_length, autoencode_samples) for pi in pat_idxs]

            idx_output = -1
            with ThreadPoolExecutor(max_workers=nested_max_workers) as executor:
                # Create a partial function that passes the shared arguments
                load_data_partial = partial(
                    load_data_sample, 
                    pat_fnames=pat_fnames, pat_ids=pat_ids, latent_dim=latent_dim, 
                    transformer_seq_length=transformer_seq_length, padded_channels=padded_channels, autoencode_samples=autoencode_samples, 
                    num_rand_hashes=num_rand_hashes, hash_output_range=hash_output_range)

                futures = []
                for i in range(len(pat_idxs)):
                    # Submit tasks for parallel loading of data samples
                    futures.append(executor.submit(load_data_partial, pat_idxs[i], file_idxs[i], start_idxs[i]))

                # Collect results from all futures
                for future in futures:
                    data_tensor_np_i, file_name_i, file_class_i, hash_channel_order_i, hash_pat_embedding_i, random_hash_modifier_i, start_idx_i, end_idx_i = future.result()

                    # Append results to the final variables
                    idx_output += 1
                    data_tensor_np[idx_output] = data_tensor_np_i
                    file_name[idx_output] = file_name_i
                    file_class[idx_output] = file_class_i
                    hash_channel_order[idx_output] = hash_channel_order_i
                    hash_pat_embedding[idx_output] = hash_pat_embedding_i
                    random_hash_modifier[idx_output] = random_hash_modifier_i
                    start_idx[idx_output] = start_idx_i
                    autoencode_samps[idx_output] = autoencode_samples # Do not need to return dynamically
                    end_idx[idx_output] = end_idx_i

            D = start_time - time.time()

            # Convert to Torch Tensor
            data_tensor = torch.Tensor(data_tensor_np).to(torch.float16)

            batch_dict = {
                "data_tensor": data_tensor, 
                "file_name": file_name,
                "file_class": file_class,
                "hash_channel_order": hash_channel_order,
                "hash_pat_embedding": hash_pat_embedding,
                "random_hash_modifier": random_hash_modifier,
                "start_idx": start_idx,
                "autoencode_samps": autoencode_samps,
                "end_idx": end_idx}

            # # Save the batch as one batch pickle
            batch_path = f"{tmp_dir}/T{thread_num}_{file_idx_next}.pkl"
            output_obj = open(batch_path, 'wb')
            pickle.dump(batch_dict, output_obj)
            output_obj.close()
            
            E = start_time - time.time()

            # if file_idx_next == 0: print("Random data generator first cycle complete")
            file_idx_next = file_idx_next + 1

        else:
            time.sleep(1) # To prevent unecessary looping while buffer is full

if __name__ == "__main__":

    try: # Put in try clause to avoid debugging truggers when killing script purposefully by deleteing tmp directory 

        # Read in configuration file & setup the run
        config_f = 'train_config.yml'
        with open(config_f, "r") as f: kwargs = yaml.load(f,Loader=yaml.FullLoader)
        kwargs = utils_functions.exec_kwargs(kwargs) # Execute the arithmatic build into kwargs and reassign kwargs

        transformer_seq_length = kwargs['transformer_seq_length']
        num_samples = kwargs['num_samples'] # in one file
        batchsize = kwargs['random_pulls_in_batch'] # Use as same value for how many files to pull at once
        padded_channels = kwargs['padded_channels']
        autoencode_samples = kwargs['autoencode_samples']
        latent_dim = kwargs['latent_dim']
        num_buffer_batches = kwargs['num_buffer_batches']
        num_data_threads = kwargs['num_data_threads']
        nested_max_workers = kwargs['nested_max_workers']
        hash_output_range = kwargs['hash_output_range']

        # Passed as args
        tmp_dir = sys.argv[1]
        fnames_csv = f"{tmp_dir}/{sys.argv[2]}"
        num_rand_hashes = int(sys.argv[3])

        # Load the fnames
        with open(fnames_csv, mode='r') as file:
            csv_reader = csv.reader(file)
            pat_fnames = [row for row in csv_reader]

        # Metadata
        num_pats = len(pat_fnames)
        pat_ids = [pat_fnames[i][0].split("/")[-1].split("_")[0] for i in range(len(pat_fnames))]

        if num_data_threads > 1:
            # Start threads
            threads = []
            for i in range(num_data_threads):
                thread = threading.Thread(target=thread_task, args=(i, nested_max_workers, tmp_dir, pat_fnames, num_buffer_batches, pat_ids, latent_dim, batchsize, num_samples, transformer_seq_length, padded_channels, autoencode_samples, num_rand_hashes, hash_output_range))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

        else: # Only run one thread
            thread_task(0, nested_max_workers, tmp_dir, pat_fnames, num_buffer_batches, pat_ids, latent_dim, batchsize, num_samples, transformer_seq_length, padded_channels, autoencode_samples, num_rand_hashes, hash_output_range)

    finally:
        print("End of script")


