import torch
from torch.utils.data import DataLoader
import os
import wandb
import numpy as np
import datetime
import pickle
import glob
import yaml
import wandb 

# Local Imports
from utilities import utils_functions
from utilities import manifold_utilities
from utilities import loss_functions
from data import SEEG_BSP_Dataset
from models.BSP import BSP

def get_models(models_codename, gpu_id, bsp_transformer_seq_length, bsp_batchsize, **kwargs):
    torch.hub.set_dir('./.torch_hub_cache') # Set a local cache directory for testing

    # Load the BSE model with pretrained weights from GitHub
    bse, _, bsp, bsv, som, hub_kwargs = torch.hub.load(
        'grahamwjohnson/seeg_tornados_2',
        'load_lbm', # entry function in hubconfig.py
        codename=models_codename,
        gpu_id=gpu_id,
        pretrained=True,
        load_bse=True, 
        load_discriminator=False,
        load_bsp=True,
        load_bsv=True,
        load_som=True,
        trust_repo='check',
        max_batch_size=bsp_transformer_seq_length*bsp_batchsize, # update for pseudobatching
        force_reload=True
    )

    return bse, bsp, bsv, som, hub_kwargs

import numpy as np
from typing import Literal

def rewindow_array(
    data: np.ndarray,
    file_windowsecs: int,
    file_stridesecs: int,
    rewin_windowsecs: int,
    rewin_strideseconds: int,
    reduction: Literal['mean', 'sum', 'cat'] = 'mean',
) -> np.ndarray:
    """
    Rewindows sequential data from an original windowing scheme to a new windowing scheme.

    Args:
        data: Array of windowed data, shape [original_windows, feature_dim].
        file_windowsecs: Duration in seconds of each original window.
        file_stridesecs: Stride in seconds of the original windows.
        rewin_windowsecs: Desired new window duration in seconds.
        rewin_strideseconds: Desired new stride in seconds.
        reduction: Method for aggregating windows ('mean', 'sum', or 'cat').

    Returns:
        Rewindowed data of shape:
        - [new_windows, feature_dim] if reduction is 'mean' or 'sum'
        - [new_windows, feature_dim * num_concat] if reduction is 'cat'
    """

    original_windows, feature_dim = data.shape

    if rewin_windowsecs % file_windowsecs != 0:
        raise ValueError("rewin_windowsecs must be a multiple of file_windowsecs.")
    if rewin_strideseconds % file_stridesecs != 0:
        raise ValueError("rewin_strideseconds must be a multiple of file_stridesecs.")
    if rewin_windowsecs < file_stridesecs and rewin_strideseconds > file_stridesecs:
        raise ValueError("rewin_windowsecs cannot be less than file_stridesecs if stride is increased.")

    orig_stride = file_stridesecs
    orig_win = file_windowsecs
    new_stride = rewin_strideseconds
    new_win = rewin_windowsecs

    new_samples_per_window = new_win // orig_stride
    new_window_starts = np.arange(0, original_windows * orig_stride, new_stride) // orig_stride

    # Compute valid window count
    max_start = original_windows - new_samples_per_window + 1
    new_window_starts = new_window_starts[new_window_starts < max_start]
    new_windows = len(new_window_starts)

    # Handle output shape
    if reduction == 'cat':
        output_dim = feature_dim * new_samples_per_window
    else:
        output_dim = feature_dim

    rewin_data = np.zeros((new_windows, output_dim), dtype=data.dtype)

    for i, start_idx in enumerate(new_window_starts):
        end_idx = start_idx + new_samples_per_window
        segment = data[start_idx:end_idx]

        if segment.shape[0] < new_samples_per_window:
            continue  # skip incomplete windows

        if reduction == 'mean':
            rewin_data[i] = np.mean(segment, axis=0)
        elif reduction == 'sum':
            rewin_data[i] = np.sum(segment, axis=0)
        elif reduction == 'cat':
            rewin_data[i] = segment.flatten()
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

    return rewin_data


def autoregress_plot(
    gpu_id,
    bse,
    bsp,
    bsv,
    som,
    dataloader_curr,
    num_rand_plots,
    bse_transformer_seq_length,
    bsp_history_tokens,
    bsp_prediction_tokens,
    bsp_context_tokens,
    save_root,
    hub_kwargs,
    bsp_autogressive_subbatch_to_plot = 1,
    **kwargs):
    
    autoregress_batch_idx = 0
    for x, _, pat_idxs, _, _ in dataloader_curr: 
        x_history = x[:, 0:bsp_history_tokens, :].to(gpu_id)
        x_context = x[:, bsp_history_tokens:bsp_history_tokens + bsp_context_tokens, :].to(gpu_id)
        x_truth = x[:, bsp_history_tokens + bsp_context_tokens:, :].to(gpu_id)

        # Embed the X through BSE
        with torch.no_grad():
            bse.eval()
            post_bse_z_history = torch.zeros(x_history.shape[0], bsp_history_tokens, x_history.shape[2], bse.latent_dim, dtype=torch.float32).to(gpu_id)
            post_bse_z_context = torch.zeros(x_context.shape[0], bsp_context_tokens, x_context.shape[2], bse.latent_dim, dtype=torch.float32).to(gpu_id)
            post_bse_z_truth = torch.zeros(x_truth.shape[0], bsp_prediction_tokens, x_truth.shape[2], bse.latent_dim, dtype=torch.float32).to(gpu_id)
            for b in range(x.shape[0]): # One batch index at a time to not have to double pseudobatch
                # History
                x_in_history = x_history[b, :, :, :, :]
                z_pseudobatch, _, _, mogpreds_pseudobatch_softmax, _ = bse(x_in_history, reverse=False) 
                z_split = z_pseudobatch.split(bse_transformer_seq_length, dim=0) # Reshape variables back to token level 
                post_bse_z_history[b, :, :, :] = torch.stack(z_split, dim=0)

                # Context
                x_in_context = x_context[b, :, :, :, :]
                z_pseudobatch, _, _, mogpreds_pseudobatch_softmax, _ = bse(x_in_context, reverse=False) 
                z_split = z_pseudobatch.split(bse_transformer_seq_length, dim=0) # Reshape variables back to token level 
                post_bse_z_context[b, :, :, :] = torch.stack(z_split, dim=0)

                # Future truth
                x_in_truth = x_truth[b, :, :, :, :]
                z_pseudobatch, _, _, mogpreds_pseudobatch_softmax, _ = bse(x_in_truth, reverse=False) 
                z_split = z_pseudobatch.split(bse_transformer_seq_length, dim=0) # Reshape variables back to token level 
                post_bse_z_truth[b, :, :, :] = torch.stack(z_split, dim=0)
        
        # Embed the BSE outputs into pre-BSP space
        with torch.no_grad():
            bsp.eval()
            _, _, post_bse2p_z_history, _, _, _ = bsp(post_bse_z_history)
            _, _, post_bse2p_z_context, _, _, _ = bsp(post_bse_z_context)
            _, _, post_bse2p_z_truth, _, _, _ = bsp(post_bse_z_truth)

        # Prepare outputs of autoregressive BSP runs (BSP operates in the post_bse2p_z latent space)
        full_output = torch.zeros([x.shape[0], x_context.shape[1] + bsp_prediction_tokens, bsp.bsp_latent_dim])
        full_output[:, 0:bsp_context_tokens, :] = post_bse2p_z_context.clone().detach().to(gpu_id) # fill the running vector with initial context
        
        # Run BSP autoregressively (ONLY need to run the transformer from within the BSP)
        # Just runs from the 'context' size forward 
        auto_step = 0
        for i in range(bsp_prediction_tokens):
            context_curr = full_output[:,auto_step:auto_step+bsp_context_tokens,:].clone().detach().to(gpu_id)
            post_bsp = bsp.transformer(context_curr, start_pos=0, return_attW=False, causal_mask_bool=True, self_mask=False)
            full_output[:,auto_step+bsp_context_tokens,:] = post_bsp[:, -1, :].clone().detach()
            auto_step += 1

        # Run the BSP space through BSV (can be done after autoregression because BSV is just a tangeant off BSP space, not required to run BSP)
        post_bse2p_z_pred = full_output[:, bsp_context_tokens:, :].to(gpu_id)
        with torch.no_grad():
            bsv.eval()
            _, _, _, bsv_z_history = bsv(post_bse2p_z_history)
            _, _, _, bsv_z_context = bsv(post_bse2p_z_context)
            _, _, _, bsv_z_pred = bsv(post_bse2p_z_pred)
            _, _, _, bsv_z_truth = bsv(post_bse2p_z_truth)

        # How many from batch to plot?
        num_to_plot = min(bsp_autogressive_subbatch_to_plot, x.shape[0])


        # Plot the autoregressed predictions
        # Include 1 point overlap in predictions and ground truth for plotting purposes 
        if not os.path.exists(save_root): os.makedirs(save_root)
        for b in range(num_to_plot):
            
            # Window the data for Kohonen (a.k.a. SOM)
            som_full_pred = rewindow_array(
                torch.cat((bsv_z_history[b,:,:], bsv_z_context[b,:,:], bsv_z_pred[b,:,:])),
                file_stridesecs=hub_kwargs['']
                
                )
            som_full_truth = rewindow_array(torch.cat((bsv_z_history[b,:,:], bsv_z_context[b,:,:], bsv_z_truth[b,:,:])))
            
            # Now pull out relevant rewindowed sizes
            # Need to calculate based on window and stride
    #         som_context = som_full_pred[-]
                
    #             data: np.ndarray,
    # file_windowsecs: int,
    # file_stridesecs: int,
    # rewin_windowsecs: int,
    # rewin_strideseconds: int,)
    #         som_pred
    #         som_truth





            # HUBCONFIG NEEDS TO HAVE SOM WINDOW AND STRIDE in it


            try:
                pred_plot_axis = manifold_utilities.plot_kohonen_prediction(
                    gpu_id=gpu_id,
                    save_dir = save_root, 
                    som = som, 
                    batch_idx = b,
                    pat_id = pat_idxs[b],
                    context = som_context,
                    ground_truth_future=bsv_z_truth[b, :, :], 
                    predictions=bsv_z_pred[b, :, :], 
                    undo_log=True, 
                    smoothing_factor=10)  
            except:
                print(f"Plotting batch {b} failed")

        # Kill after desired number of batches
        autoregress_batch_idx += 1
        if autoregress_batch_idx >= num_rand_plots: break

if __name__ == "__main__":

    # Set the hash seed 
    os.environ['PYTHONHASHSEED'] = '1234'  

    # Read in configuration file & setup the run
    config_f = 'config.yml'
    with open(config_f, "r") as f: kwargs = yaml.load(f,Loader=yaml.FullLoader)
    kwargs = utils_functions.exec_kwargs(kwargs) # Execute the arithmatic build into kwargs and reassign kwargs

    ### RUN PARAMETERS ###
    gpu_id = 0
    bsp_history_tokens = 32 # Need to get to 64 for SOM window size
    bsp_context_tokens = 32 # Currently limited by model mismatch if devaited from 32 (also would need to change chache size in BSE if wanted to go bigger anyway)
    bsp_prediction_tokens = 16
    full_tokens = bsp_context_tokens + bsp_prediction_tokens
    num_rand_plots = 2
    
    # rand_file_directory = 
    bsp_source_dir = '/media/glommy1/data/vanderbilt_seeg/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/DurStr_1024s896s_epoched_datasets/val13/*/scaled_data_epochs/all_epochs_woSPES'

    # Save destination
    save_root = '/media/glommy1/tornados/bsv_inference/commongonolek_epoch296_sheldrake_epoch1138_val13/bsp_autoregression'

    # Build dataset and dataloader for random sampling
    dataset = SEEG_BSP_Dataset(
        gpu_id=0,
        bsp_source_dir=bsp_source_dir,
        bsp_transformer_seq_length = bsp_history_tokens + bsp_context_tokens + bsp_prediction_tokens, # Need context + prediction to also get ground truth
        bsp_epoch_dataset_size=kwargs['bsp_epoch_dataset_size'],
        bsp_latent_dim=kwargs['bsp_latent_dim'],
        transformer_seq_length=kwargs['transformer_seq_length'],
        encode_token_samples=kwargs['encode_token_samples'],
        padded_channels=kwargs['padded_channels'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Get the pretrained models from Torch Hub
    bse, bsp, bsv, som, hub_kwargs = get_models(gpu_id=gpu_id, **kwargs)
    
    # Load models on GPU
    bse = bse.to(gpu_id)
    bsp = bsp.to(gpu_id)
    bsv = bsv.to(gpu_id)

    # Run through autoregressive function and plot
    autoregress_plot(
        gpu_id=gpu_id,
        bse=bse,
        bsp=bsp,
        bsv=bsv,
        som=som,
        dataloader_curr=dataloader,
        num_rand_plots=num_rand_plots,
        bse_transformer_seq_length = kwargs['transformer_seq_length'],
        bsp_history_tokens = bsp_history_tokens,
        bsp_prediction_tokens=bsp_prediction_tokens,
        bsp_context_tokens = bsp_context_tokens,
        save_root=save_root,
        hub_kwargs=hub_kwargs,
        **kwargs)

