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
    bse, _, bsp, bsv, som = torch.hub.load(
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

    return bse, bsp, bsv, som

def autoregress_plot(
    self,
    bse,
    bsp,
    bsv,
    som,
    dataloader_curr,
    num_rand_plots,
    bsp_autoregressive_plot_steps,
    svae_root,
    bsp_autogressive_subbatch_to_plot,
    **kwargs):
    
    autoregress_batch_idx = 0
    for x, _, pat_idxs in dataloader_curr: 
        x = x.to(self.gpu_id)
        full_output = torch.zeros_like(x)
        full_output[:, 0:self.bsp_transformer_seq_length, :] = x[:, 0:self.bsp_transformer_seq_length, :].clone().detach() # fill the running vector with initial context

        auto_step = 0
        for i in range(bsp_autoregressive_plot_steps):
            context_curr = full_output[:,auto_step:auto_step+self.bsp_transformer_seq_length,:].clone().detach()
            pred, _ = self.bsp(context_curr)
            full_output[:,auto_step+self.bsp_transformer_seq_length,:] = pred[:, -1, :].clone().detach()
            auto_step += 1

        # How many from batch to plot?
        num_to_plot = min(bsp_autogressive_subbatch_to_plot, x.shape[0])

        # Plot the autoregressed predictions
        # Include 1 point overlap in predictions and ground truth for plotting purposes 
        if not os.path.exists(save_root): os.makedirs(save_root)
        for b in range(num_to_plot):
            try:
                pred_plot_axis = manifold_utilities.plot_kohonen_prediction(
                    gpu_id=self.gpu_id,
                    save_dir = save_root, 
                    som = som, 
                    batch_idx = b,
                    pat_id = pat_idxs[b],
                    context = x[b, 0:self.bsp_transformer_seq_length, :], 
                    ground_truth_future=x[b, self.bsp_transformer_seq_length-1:, :], 
                    predictions=full_output[b, self.bsp_transformer_seq_length-1:, :], 
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
    context_tokens = 64
    prediction_tokens = 16
    num_rand_plots = 2
    
    # rand_file_directory = 
    bsp_source_dir = '/media/glommy1/data/vanderbilt_seeg/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/DurStr_1024s896s_epoched_datasets/val13/*/scaled_data_epochs/all_epochs_woSPES'

    # Save destination
    save_root = '/media/glommy1/tornados/bsv_inference/commongonolek_epoch296_sheldrake_epoch1138_val13/bsp_autoregression'

    # Build dataset and dataloader for random sampling
    dataset = SEEG_BSP_Dataset(
        gpu_id=0,
        bsp_source_dir=bsp_source_dir,
        bsp_transformer_seq_length=context_tokens,
        bsp_epoch_dataset_size=kwargs['bsp_epoch_dataset_size'],
        bsp_latent_dim=kwargs['bsp_latent_dim'],
        transformer_seq_length=kwargs['transformer_seq_length'],
        encode_token_samples=kwargs['encode_token_samples'],
        padded_channels=kwargs['padded_channels'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Get the pretrained models from Torch Hub
    bse, bsp, bsv, som = get_models(gpu_id=gpu_id, **kwargs)
    
    # Load models on GPU
    bse = bse.to(gpu_id)
    bsp = bsp.to(gpu_id)
    bsv = bsv.to(gpu_id)

    # Run through autoregressive function and plot
    autoregress_plot(
        bse=bse,
        bsp=bsp,
        bsv=bsv,
        som=som,
        dataloader_curr=dataloader,
        num_rand_plots=num_rand_plots,
        bsp_autoregressive_plot_steps=prediction_tokens,
        save_root=save_root


    )

