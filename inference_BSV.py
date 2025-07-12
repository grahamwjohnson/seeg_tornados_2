"""
@author: grahamwjohnson
April 2025

This script is intended to pull a fully trained LBM(i.e. BSE-BSP-BSV) from GitHub using torch.hub in a tagged release. 
You can then run the pipeline as follows: data --> preprocessed data --> BSE --> BSP --> BSP. 
This script is intended for retrospective visualizations of prediction potential in the data. 
You can adapt this technique to design a realtime implementation for your live data stream. 

"""

# Package imports 
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier
import yaml, sys, os, pickle, numpy as np, datetime, glob, random

# Local imports
from utilities import utils_functions
from data import SEEG_Tornado_Dataset
from utilities import loss_functions

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
    kwargs['dataset_pic_dir'] = kwargs['bsv_inference_save_dir'] + '/dataset_bargraphs'
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

def get_models(models_codename, gpu_id, bsp_transformer_seq_length, bsp_batchsize, **kwargs):
    torch.hub.set_dir('./.torch_hub_cache') # Set a local cache directory for testing

    # Load the BSE model with pretrained weights from GitHub
    bse, _, bsp, bsv, _ = torch.hub.load(
        'grahamwjohnson/seeg_tornados_2',
        'load_lbm', # entry function in hubconfig.py
        codename=models_codename,
        gpu_id=gpu_id,
        pretrained=True,
        load_bse=True, 
        load_discriminator=False,
        load_bsp=True,
        load_bsv=True,
        load_pacmap=False,
        trust_repo='check',
        max_batch_size=bsp_transformer_seq_length*bsp_batchsize, # update for pseudobatching
        # force_reload=True
    )

    return bse, bsp, bsv

def remove_padded_channels_samePat(x: torch.Tensor, hash_channel_order):
    """
    Removes padded channels (-1) from x for each sample in the batch.
    
    Args:
        x: Input tensor of shape [batch, tokens, max_channels, seq_len].
        hash_channel_order: List of lists where -1 indicates padded channels.
    
    Returns:
        List of tensors with padded channels removed (one per batch sample).
    """
    # Convert hash_channel_order to a tensor
    channel_order_tensor = torch.tensor(hash_channel_order, dtype=torch.long).to(x)  # [max_channels]
    valid_mask = (channel_order_tensor != -1)  # [batch, max_channels]

    # Filter out padded channels per sample
    x_nopad = []
    for i in range(x.shape[0]):  # Loop over batch
        x_nopad.append(x[i, :, valid_mask, :])  # [tokens, valid_channels, seq_len]

    return x_nopad

def bsv_export_embeddings(
    world_size,
    gpu_id,
    bse,
    bsp,
    bsv,
    dataset_curr, 
    dataset_string,
    num_dataloader_workers,
    bsv_infer_max_batch_size,
    padded_channels,
    transformer_seq_length,
    encode_token_samples,
    latent_dim,
    bsv_dims,
    prior_mog_components,
    inference_num_rand_hashes,
    hash_output_range,
    inference_window_sec_list,
    inference_stride_sec_list,
    bsv_inference_save_dir,
    FS,
    **kwargs):

    print(f"[GPU{gpu_id}] INFERENCE")

    bsv_latent_dim = bsv_dims[-1]
    pat_list = range(0,len(dataset_curr.pat_ids)) # Go through every subject in this dataset

    for pat_idx in pat_list:
        dataset_curr.set_pat_curr(pat_idx)
        _, pat_id_curr, _, _ = dataset_curr.get_pat_curr()

        # Check which files have already been processed and update file list accordingly before building dataloader
        dataset_curr.update_pat_inference_status(bsv_inference_save_dir, inference_window_sec_list, inference_stride_sec_list)
        dataloader_curr, _ =  utils_functions.prepare_ddp_dataloader(dataset_curr, batch_size=bsv_infer_max_batch_size, num_workers=num_dataloader_workers)

        file_count = 0
        batch_count = 0
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
            files_means = torch.zeros([data_tensor.shape[0], num_windows_in_file, bsv_latent_dim]).to(gpu_id)
            files_logvars = torch.zeros([data_tensor.shape[0], num_windows_in_file, bsv_latent_dim]).to(gpu_id)
            files_z = torch.zeros([data_tensor.shape[0], num_windows_in_file, bsv_latent_dim]).to(gpu_id)

            # Put whole file on GPU now for speed of iterating over all windows
            data_tensor = data_tensor.to(gpu_id)

            for w in range(0, num_windows_in_file):
                
                # Print Status
                print_interval = 10
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

                # # Now fill the input tensor with random channel order and random zero channels
                start_idx = w * num_samples_in_forward
                end_idx = start_idx + encode_token_samples * transformer_seq_length
                # x = torch.zeros((data_tensor.shape[0], padded_channels, transformer_seq_length * encode_token_samples), device=gpu_id)
                # for i, channel_idx in enumerate(hash_channel_order):
                #     if channel_idx != -1:
                #         x[:, i, :] = data_tensor[:, channel_idx, start_idx:end_idx]                
                x = torch.zeros((data_tensor.shape[0], padded_channels, transformer_seq_length * encode_token_samples), device=gpu_id) # Create a zero tensor on GPU
                channel_indices = torch.tensor(hash_channel_order, device=gpu_id) # Convert to tensor for advanced indexing
                valid_mask = channel_indices != -1. # Find valid (non -1) channels
                valid_channels = channel_indices[valid_mask]
                target_positions = torch.arange(padded_channels, device=gpu_id)[valid_mask]
                x[:, target_positions, :] = data_tensor[:, valid_channels, start_idx:end_idx] # Select data in bulk and assign it
                x = x.reshape(x.shape[0], x.shape[1], transformer_seq_length, encode_token_samples).permute(0, 2, 1, 3)
                # x should now be shaped [B, seq, channel, encode_sample]

                ### BSE Encoder
                # Forward pass in stacked batch through BSE encoder
                z_pseudobatch, _, _, _, _ = bse(x, reverse=False) # No shift if not causal masking

                ### BSP2E
                post_bse_z = z_pseudobatch.reshape(-1, transformer_seq_length * encode_token_samples, latent_dim).unsqueeze(1)
                _, _, post_bse2p_z = bsp.bse2p(post_bse_z)

                ### BSV Encoder
                _, bsv_mu, bsv_logvar, bsv_z = bsv(post_bse2p_z)

                # Save the results
                files_means[:, w, :] = bsv_mu.squeeze() # Must change if not doing 1 window at a time
                files_logvars[:, w, :] = bsv_logvar.squeeze()
                files_z[:, w, :] = bsv_z.squeeze()

            # After file complete, window/stride the file and save each file from batch seperately
            # Seperate directory for each win/stride combination
            # First pull off GPU and convert to numpy
            files_means = files_means.cpu().numpy()
            files_logvars = files_logvars.cpu().numpy()
            files_z = files_z.cpu().numpy()
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
                windowed_file_means = np.zeros([data_tensor.shape[0], num_strides_in_file, bsv_latent_dim])
                windowed_file_logvars = np.zeros([data_tensor.shape[0], num_strides_in_file, bsv_latent_dim])
                windowed_file_z = np.zeros([data_tensor.shape[0], num_strides_in_file, prior_mog_components])
                for s in range(num_strides_in_file):
                    windowed_file_means[:, s, :] = np.mean(files_means[:, s*num_latents_in_stride: s*num_latents_in_stride + num_latents_in_win, :], axis=1)
                    windowed_file_logvars[:, s, :] = np.mean(files_logvars[:, s*num_latents_in_stride: s*num_latents_in_stride + num_latents_in_win, :], axis=1)
                    windowed_file_z[:, s, :] = np.mean(files_z[:, s*num_latents_in_stride: s*num_latents_in_stride + num_latents_in_win, :], axis=1)

                # Save each windowed latent in a pickle for each file
                for b in range(data_tensor.shape[0]):
                    filename_curr = file_name[b]
                    save_dir = f"{bsv_inference_save_dir}/latent_files/{win_sec_curr}SecondWindow_{stride_sec_curr}SecondStride"
                    if not os.path.exists(save_dir): os.makedirs(save_dir)
                    output_obj = open(f"{save_dir}/{filename_curr}_latent_{win_sec_curr}secWindow_{stride_sec_curr}secStride.pkl", 'wb')
                    save_dict = {
                        'windowed_means': windowed_file_means[b, :, :],
                        'windowed_logvars': windowed_file_logvars[b, :, :],
                        'windowed_z': windowed_file_z[b, :, :]}
                    pickle.dump(save_dict, output_obj)
                    output_obj.close()

            batch_count += 1

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

    # Load the pretrained models from GitHub and put on GPU, and initialize DDP
    bse, bsp, bsv = get_models(gpu_id=gpu_id, **kwargs)
    bse = bse.to(gpu_id) 
    bsp = bsp.to(gpu_id) 
    bsv = bsv.to(gpu_id) 
    DDP(bse, device_ids=[gpu_id])
    DDP(bsp, device_ids=[gpu_id])
    DDP(bsv, device_ids=[gpu_id])

    # Run preprocessed data through BSE and save according to export settings in config.yaml
    with torch.inference_mode():
        # Preprocessed --> BSE --> BSE2P --> BSV --> latent
        bsv_export_embeddings(
            world_size=world_size,
            gpu_id=gpu_id,
            bse=bse,
            bsp=bsp,
            bsv=bsv,
            dataset_curr=inference_dataset,
            dataset_string='bsv_inference',
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
