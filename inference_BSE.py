'''
@author: grahamwjohnson
April 2025

This script is intended to be used with a fully trained BSE pulled from GitHub using torch.hub in a tagged release.
It will run all of your preprocessed data through the BSE to get embeddings.
This is intended to be used for retrospective data anylyses on your dataset. 

IMPORTANT: Data must be preprocessed using the provided preprocessing pipeline. 
Specifically, histogram equalization is used and the BSE will not function properly without proper input distribution of data. 

'''
# Package imports 
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier
import yaml, sys, os, pickle, numpy as np, datetime, glob, random

# Local imports
from utilities import utils_functions
from data import SEEG_Tornado_Dataset

def ddp_setup(gpu_id, world_size):
    """
    Args:
        gpu_id: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = 1  # Can only do this OR "NCCL_BLOCKING_WAIT" = 1
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"
    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    init_process_group(backend="nccl", rank=gpu_id, world_size=world_size, timeout=datetime.timedelta(minutes=999999))

def load_dataset(
    gpu_id,
    inference_pat_dir,
    inference_intrapatient_dataset_style,
    inference_hour_dataset_range,
    inference_num_rand_hashes,
    inference_forward_passes,
    **kwargs
    ):

    # Delete the conflicting kwargs entries
    kwargs.pop('intrapatient_dataset_style')
    kwargs.pop('train_hour_dataset_range')
    kwargs.pop('train_num_rand_hashes')
    kwargs.pop('train_forward_passes')
    
    # Split pats into train and test
    all_pats_dirs = glob.glob(f"{inference_pat_dir}/*pat*")
    all_pats_list = [x.split('/')[-1] for x in all_pats_dirs]
    kwargs['dataset_pic_dir'] = kwargs['inference_save_dir'] + '/dataset_bargraphs'
    inference_dataset = SEEG_Tornado_Dataset(
        gpu_id=gpu_id, 
        pat_list=all_pats_list,
        pat_dirs=all_pats_dirs,
        intrapatient_dataset_style=inference_intrapatient_dataset_style, 
        hour_dataset_range=inference_hour_dataset_range,
        num_rand_hashes=inference_num_rand_hashes,
        num_forward_passes=inference_forward_passes,
        data_logger_enabled=False,
        data_logger_file=None,
        **kwargs)
    
    return inference_dataset

def get_bse(bse_codename, **kwargs):
    torch.hub.set_dir('./.torch_hub_cache') # Set a local cache directory for testing

    # Load the BSE model with pretrained weights from GitHub
    bse = torch.hub.load(
        'grahamwjohnson/seeg_tornados_2',
        'load',
        codename=bse_codename,
        pretrained=True,
        load_bse=True, 
        load_bsp=False,
        trust_repo='check',
        # force_reload=True
    )

    return bse

def bse_export_embeddings(
    world_size,
    gpu_id,
    bse,
    dataset_curr, 
    dataset_string,
    num_dataloader_workers,
    max_batch_size,
    padded_channels,
    transformer_seq_length,
    encode_token_samples,
    latent_dim,
    prior_mog_components,
    inference_num_rand_hashes,
    hash_output_range,
    inference_window_sec_list,
    inference_stride_sec_list,
    inference_save_dir,
    FS,
    **kwargs):

    """
    This function runs inference with the BSE encoder (only the encoder) and saves 
    the resulting latent space embeddings for each file, in the form of pickle files. 
    Unlike the `_run_train_epoch` function, which pulls random data, this function 
    iterates sequentially through the entire dataset (one patient at a time). 

    The function processes the dataset and saves latent embeddings using the BSE encoder 
    for each patient, file, and window-stride configuration. The embeddings are smoothed 
    according to specified window and stride sizes.

    ### Key Features:
    - **Sequential Inference:** This function sequentially processes data for each patient 
    and file, as opposed to random data pulls.
    - **Latent Embedding Calculation:** It calculates latent embeddings for each file's 
    data and saves them as pickle files.
    - **Window-Stride Smoothing:** The latent embeddings are smoothed according to the 
    window and stride sizes defined by `inference_window_sec_list` and 
    `inference_stride_sec_list`.
    - **Conditioning:** Hashes for each patient and channel are generated to condition 
    the feedforward process.
    - **GPU Support:** The function processes the data on GPU for efficient computations.

    ### Parameters:
    - `dataset_curr` (Dataset): The dataset object that holds the current patient data 
    to be processed.
    - `dataset_string` (str): A string identifier for the dataset (for tracking or naming).
    - `attention_dropout` (float): Dropout rate for attention layers (though not directly 
    used in this function).
    - `num_dataloader_workers` (int): Number of worker threads for loading data batches.
    - `max_batch_size` (int): The maximum batch size used for inference.
    - `inference_batch_mult` (float): A multiplier for adjusting batch size during inference.
    - `padded_channels` (int): The number of padded channels in the data tensor (to 
    accommodate variable input sizes).
    - `**kwargs` (dict): Additional arguments that may be passed (not used directly here).

    ### Outputs:
    - For each patient, the function processes their data and saves the computed latent 
    embeddings into pickle files. 
    - The files are organized according to the window and stride configurations, with 
    directories and filenames reflecting these settings (e.g., `latent_60secWindow_30secStride.pkl`).
    - Each file’s latent space embeddings are saved after the BSE encoder processes the 
    raw data windows with the specified stride and window sizes.

    ### Notes:
    - **Data Processing:** The function processes files one by one per patient. Data for 
    each file is passed through the BSE encoder to extract the latent space.
    - **Windowing and Striding:** After processing each file, the latent space is divided 
    into overlapping windows based on the stride and window settings, smoothing the 
    embeddings as necessary.
    - **File Structure:** Latent embeddings are saved in directories named after the 
    window and stride sizes, and each file's latent embeddings are saved separately.

    """

    print(f"[GPU{gpu_id}] INFERENCE")

    # Go through every subject in this dataset
    for pat_idx in range(0,len(dataset_curr.pat_ids)):
        dataset_curr.set_pat_curr(pat_idx)
        _, pat_id_curr, _, _ = dataset_curr.get_pat_curr()
        dataloader_curr =  utils_functions.prepare_dataloader(dataset_curr, batch_size=max_batch_size, num_workers=num_dataloader_workers)

        # Go through every file in dataset
        file_count = 0
        for data_tensor, file_name, _ in dataloader_curr: # Hash done outside data.py for single pat inference

            file_count = file_count + len(file_name)

            num_channels_curr = data_tensor.shape[1]

            # Create the sequential latent sequence array for the file 
            num_samples_in_forward = transformer_seq_length * encode_token_samples
            num_windows_in_file = data_tensor.shape[2] / num_samples_in_forward
            assert (num_windows_in_file % 1) == 0
            num_windows_in_file = int(num_windows_in_file)
            num_samples_in_forward = int(num_samples_in_forward)

            # Prep the output tensor and put on GPU
            files_means = torch.zeros([data_tensor.shape[0], num_windows_in_file, latent_dim]).to(gpu_id)
            files_logvars = torch.zeros([data_tensor.shape[0], num_windows_in_file, latent_dim]).to(gpu_id)
            files_mogpreds = torch.zeros([data_tensor.shape[0], num_windows_in_file, prior_mog_components]).to(gpu_id)

            # Put whole file on GPU now for speed of iterating over all windows
            data_tensor = data_tensor.to(gpu_id)

            for w in range(num_windows_in_file):
                
                # Print Status
                print_interval = 100
                if (gpu_id == 0) & (w % print_interval == 0):
                    sys.stdout.write(f"\r{dataset_string}: Pat {pat_idx}/{len(dataset_curr.pat_ids)-1}, File {file_count - len(file_name)}:{file_count}/{len(dataset_curr)/world_size - 1}  * GPUs (DDP), Intrafile Iter {w}/{num_windows_in_file}          ") 
                    sys.stdout.flush() 
                                
                # Generate random channel order
                rand_modifier = int(random.uniform(0, inference_num_rand_hashes - 1))
                _, hash_channel_order = utils_functions.hash_to_vector(
                    input_string=pat_id_curr, 
                    num_channels=num_channels_curr, 
                    padded_channels=padded_channels,
                    latent_dim=latent_dim, 
                    modifier=rand_modifier,
                    hash_output_range=hash_output_range)

                # Now fill the input tensor with random channel order and random zero channels
                start_idx = w * num_samples_in_forward
                end_idx = start_idx + encode_token_samples * transformer_seq_length
                x = torch.zeros((data_tensor.shape[0], padded_channels, transformer_seq_length * encode_token_samples), device=gpu_id)
                for i, channel_idx in enumerate(hash_channel_order):
                    if channel_idx != -1:
                        x[:, i, :] = data_tensor[:, channel_idx, start_idx:end_idx]
                x = x.reshape(x.shape[0], x.shape[1], transformer_seq_length, encode_token_samples).permute(0, 2, 1, 3)
                # x should now be shaped [B, seq, channel, encode_sample]

                ### BSE ENCODER
                # Forward pass in stacked batch through BSE encoder
                # latent, _, _ = self.bse(x[:, :-1, :, :], reverse=False, alpha=self.classifier_alpha)   # 1 shifted just to be aligned with training style
                _, mean_pseudobatch, logvar_pseudobatch, mogpreds_pseudobatch, _ = bse(x, reverse=False) # No shift if not causal masking
                
                # Theoretical levels of detail to save (Token or Token-Meaned level):
                # 1) Save mogpreds [CURRENTLY SAVED at Token-Mean level]
                # 2) Save the weighted means from each component [CURRENTLY SAVED at Token-Mean level]
                # 3) Save weighted means and uncertainty with weighted logvars [CURRENTLY SAVED at Token-Mean level]
                # 4) Or just save one component of interest [NOT currently saved in this way]

                # Reshape back to token level 
                mogpreds = mogpreds_pseudobatch.split(transformer_seq_length, dim=0)
                mogpreds = torch.stack(mogpreds, dim=0)
                mean = mean_pseudobatch.split(transformer_seq_length, dim=0)
                mean = torch.stack(mean, dim=0)
                logvar = logvar_pseudobatch.split(transformer_seq_length, dim=0)
                logvar = torch.stack(logvar, dim=0)

                # Weight the means using mogpreds
                # mogpreds shape: [batch_size, seq_length, num_components]
                # mean shape: [batch_size, seq_length, num_components, latent_dim]
                # Expand mogpreds to match the latent_dim for broadcasting
                mogpreds_expanded = mogpreds.unsqueeze(-1)  # Shape: [batch_size, seq_length, num_components, 1]

                # Weight the means & logvars
                weighted_means = mean * mogpreds_expanded  # Shape: [batch_size, seq_length, num_components, latent_dim]
                weighted_logvars = logvar * mogpreds_expanded 

                # Sum over the components to get the final weighted mean for each token
                weighted_means_summed = torch.sum(weighted_means, dim=2)  # Shape: [batch_size, seq_length, latent_dim]
                weighted_logvars_summed = torch.sum(weighted_logvars, dim=2)  # Shape: [batch_size, seq_length, latent_dim]

                # Take the mean across the token dimension (optional, if you want to reduce further)
                mean_tokenmeaned = torch.mean(weighted_means_summed, dim=1)  # Shape: [batch_size, latent_dim]
                logvar_tokenmeaned = torch.mean(weighted_logvars_summed, dim=1)  # Shape: [batch_size, latent_dim]

                # Save the results
                files_means[:, w, :] = mean_tokenmeaned
                files_logvars[:, w, :] = logvar_tokenmeaned
                files_mogpreds[:, w, :] = torch.mean(mogpreds, dim=1)  # Save the mean mogpreds across tokens

            # After file complete, pacmap_window/stride the file and save each file from batch seperately
            # Seperate directory for each win/stride combination
            # First pull off GPU and convert to numpy
            files_means = files_means.cpu().numpy()
            files_logvars = files_logvars.cpu().numpy()
            files_mogpreds = files_mogpreds.cpu().numpy()
            for i in range(len(inference_window_sec_list)):

                win_sec_curr = inference_window_sec_list[i]
                stride_sec_curr = inference_stride_sec_list[i]
                sec_in_forward = num_samples_in_forward/FS

                if (win_sec_curr < sec_in_forward) or (stride_sec_curr < sec_in_forward):
                    raise Exception("Window or stride is too small compared to input sequence to encoder")
                
                num_latents_in_win = win_sec_curr / sec_in_forward
                assert (num_latents_in_win % 1) == 0
                num_latents_in_win = int(num_latents_in_win)

                num_latents_in_stride = stride_sec_curr / sec_in_forward
                assert (num_latents_in_stride % 1) == 0 
                num_latents_in_stride = int(num_latents_in_stride)

                # May not go in evenly, that is ok
                num_strides_in_file = int((files_means.shape[1] - num_latents_in_win) / num_latents_in_stride) 
                windowed_file_means = np.zeros([data_tensor.shape[0], num_strides_in_file, latent_dim])
                windowed_file_logvars = np.zeros([data_tensor.shape[0], num_strides_in_file, latent_dim])
                windowed_file_mogpreds = np.zeros([data_tensor.shape[0], num_strides_in_file, prior_mog_components])
                for s in range(num_strides_in_file):
                    windowed_file_means[:, s, :] = np.mean(files_means[:, s*num_latents_in_stride: s*num_latents_in_stride + num_latents_in_win, :], axis=1)
                    windowed_file_logvars[:, s, :] = np.mean(files_logvars[:, s*num_latents_in_stride: s*num_latents_in_stride + num_latents_in_win, :], axis=1)
                    windowed_file_mogpreds[:, s, :] = np.mean(files_mogpreds[:, s*num_latents_in_stride: s*num_latents_in_stride + num_latents_in_win, :], axis=1)

                # Save each windowed latent in a pickle for each file
                for b in range(data_tensor.shape[0]):
                    filename_curr = file_name[b]
                    save_dir = f"{inference_save_dir}/latent_files/{win_sec_curr}SecondWindow_{stride_sec_curr}SecondStride"
                    if not os.path.exists(save_dir): os.makedirs(save_dir)
                    output_obj = open(f"{save_dir}/{filename_curr}_latent_{win_sec_curr}secWindow_{stride_sec_curr}secStride.pkl", 'wb')
                    save_dict = {
                        'windowed_weighted_means': windowed_file_means[b, :, :],
                        'windowed_weighted_logvars': windowed_file_logvars[b, :, :],
                        'windowed_mogpreds': windowed_file_mogpreds[b, :, :]}
                    pickle.dump(save_dict, output_obj)
                    output_obj.close()

def main(
    gpu_id,
    world_size,
    config, # aka kwargs

    **kwargs):

    # Set the number of threads for this MP subprocess
    torch.set_num_threads(kwargs['subprocess_num_threads'])

    # Initialize DDP 
    ddp_setup(gpu_id, world_size)

    # Load the inference patients
    inference_dataset = load_dataset(gpu_id, **kwargs)

    # Load the pretrained Brain-State Embedder (BSE) from GitHub and put on GPU, and initialize DDP
    bse = get_bse(**kwargs)
    bse = bse.to(gpu_id) 
    bse.gpu_id = gpu_id
    bse.transformer_encoder.freqs_cis = bse.transformer_encoder.freqs_cis.to(gpu_id)
    DDP(bse, device_ids=[gpu_id])

    # Run preprocessed data through BSE and save according to export settings in config.yaml
    with torch.inference_mode():
        bse_export_embeddings(
            world_size=world_size,
            gpu_id=gpu_id,
            bse=bse,
            dataset_curr=inference_dataset,
            dataset_string='bse_inference',
            **kwargs)

if __name__ == "__main__":
    
    # Read in configuration file 
    config_f = 'config.yml'
    with open(config_f, "r") as f: kwargs = yaml.load(f,Loader=yaml.FullLoader)
    kwargs = utils_functions.exec_kwargs(kwargs) # Execute the arithmatic build into kwargs and reassign kwargs

    # Spawn subprocesses with start/join (mp.spawn causes memory sigdev errors??)
    ctx = mp.get_context('spawn') # necessary to use context if have set_start_method anove?
    children = []
    for i in range(kwargs['num_gpus']):
        subproc = ctx.Process(target=main, args=(i, kwargs['num_gpus'], kwargs), kwargs=kwargs)
        children.append(subproc)
        subproc.start()

    for i in range(kwargs['num_gpus']):
        children[i].join()

    print("End of script")
