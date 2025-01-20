import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.utils.data import Dataset, DataLoader
import sys  
import os
import shutil
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import os
import wandb
import copy
import pandas as pd
import random
import numpy as np
import time
import datetime
import pickle
import joblib
import gc
import heapq
import traceback
import glob
import pacmap
import yaml
import auraloss
import wandb
import math

# Local Imports
from utilities import latent_plotting
from utilities import utils_functions
from utilities import loss_functions
from data import SEEG_Tornado_Dataset
from models.Transformer import ModelArgs, Transformer
from models.VAE import VAE


######
torch.autograd.set_detect_anomaly(False)


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

def load_train_objs(
    gpu_id,  
    pat_dir, 
    train_val_pat_perc, 
    intrapatient_dataset_style, 
    train_hour_dataset_range, 
    val_finetune_hour_dataset_range, 
    val_unseen_hour_dataset_range, 
    inference_selection, 
    
    # Optimizers
    core_weight_decay, 
    transformer_weight_decay,
    adamW_beta1,
    adamW_beta2,
    **kwargs):

    # Split pats into train and test
    all_pats_dirs = glob.glob(f"{pat_dir}/*pat*")
    all_pats_list = [x.split('/')[-1] for x in all_pats_dirs]

    val_pats_count = int(np.ceil(train_val_pat_perc[1] * len(all_pats_list)))
    val_pats_dirs = all_pats_dirs[-val_pats_count:]
    val_pats_list = all_pats_list[-val_pats_count:]

    train_pats_dirs = all_pats_dirs[:-val_pats_count]
    train_pats_list = all_pats_list[:-val_pats_count]

    # Sequential dataset used to run inference on train data and build PaCMAP projection
    print(f"[GPU{str(gpu_id)}] Generating TRAIN dataset")
    kwargs['dataset_pic_dir'] = kwargs['model_dir'] + '/dataset_bargraphs/Train_Grouped'
    train_dataset = SEEG_Tornado_Dataset(
        gpu_id=gpu_id, 
        pat_list=train_pats_list,
        pat_dirs=train_pats_dirs,
        intrapatient_dataset_style=intrapatient_dataset_style, 
        hour_dataset_range=train_hour_dataset_range,
        **kwargs)
    
    print(f"[GPU{str(gpu_id)}] Generating VALIDATION FINETUNE dataset")
    kwargs['dataset_pic_dir'] = kwargs['model_dir'] + '/dataset_bargraphs/Val_Grouped_Finetune'
    valfinetune_dataset = SEEG_Tornado_Dataset(
        gpu_id=gpu_id, 
        pat_list=val_pats_list,
        pat_dirs=val_pats_dirs,
        intrapatient_dataset_style=intrapatient_dataset_style, 
        hour_dataset_range=val_finetune_hour_dataset_range,
        **kwargs)

    print(f"[GPU{str(gpu_id)}] Generating VALIDATION UNSEEN dataset")
    kwargs['dataset_pic_dir'] = kwargs['model_dir'] + '/dataset_bargraphs/Val_Grouped_Unseen'
    valunseen_dataset = SEEG_Tornado_Dataset(
        gpu_id=gpu_id, 
        pat_list=val_pats_list,
        pat_dirs=val_pats_dirs,
        intrapatient_dataset_style=intrapatient_dataset_style, 
        hour_dataset_range=val_unseen_hour_dataset_range,
        **kwargs)
     
    ### VAE ###
    vae = VAE(gpu_id=gpu_id, **kwargs) 
    vae = vae.to(gpu_id) 
    opt_vae = torch.optim.AdamW(vae.parameters(), weight_decay=core_weight_decay, betas=(adamW_beta1, adamW_beta2), lr=kwargs['LR_min_core'])
    
    ### Transformer ###
    transformer = Transformer(ModelArgs(device=gpu_id, **kwargs))
    transformer = transformer.to(gpu_id)
    opt_transformer = torch.optim.AdamW(transformer.parameters(), weight_decay=transformer_weight_decay, betas=(adamW_beta1, adamW_beta2), lr=kwargs['LR_min_transformer'])
    print(f"[GPU{gpu_id}] transformer loaded")

    return train_dataset, valfinetune_dataset, valunseen_dataset, transformer, vae, opt_vae, opt_transformer 

def main(  
    # Ordered variables
    gpu_id: int, 
    world_size: int, 
    config, # aka kwargs
    
    # Passed by kwargs
    run_name: str,
    timestamp_id: int,
    start_epoch: int,
    wdecode_batch_size: int,
    onlylatent_batch_size: int,
    train_subsample_file_factor: int,
    valfinetune_subsample_file_factor: int,
    valunseen_subsample_file_factor: int,
    train_num_rand_hashes: int,
    val_num_rand_hashes: int,
    LR_val_vae: float,
    LR_val_transformer: float,
    finetune_pacmap: bool, 
    PaCMAP_model_to_infer = [],
    vae_state_dict_prev_path = [],
    vae_opt_state_dict_prev_path = [],
    transformer_state_dict_prev_path = [],
    opt_transformer_state_dict_prev_path = [],
    epochs_to_train: int = -1,
    **kwargs):

    '''
    Highest level loop to run train epochs, validate, pacmap... etc. 

    '''

    # Initialize new WandB here aand group GPUs together with DDP
    wandb.require("service")
    wandb_run = wandb.init(
        resume="allow",
        id=f"{timestamp_id}_GPU{gpu_id}",
        name=f"{run_name}_GPU{gpu_id}",
        project="Tornados",
        dir=kwargs['model_dir'], 
        group='DDP', 
        config=config)

    # Set the number of threads for this MP subprocess
    torch.set_num_threads(kwargs['subprocess_num_threads'])

    # Initialize DDP 
    ddp_setup(gpu_id, world_size)

    print(f"[GPU{str(gpu_id)}] Loading training objects (datasets, models, optimizers)")
    train_dataset, valfinetune_dataset, valunseen_dataset, transformer, vae, opt_vae, opt_transformer = load_train_objs(gpu_id=gpu_id, **kwargs) 
    
    # Load the model/opt/sch states if not first epoch & if in training mode
    if (start_epoch > 0):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}

        # Load in VAE Core weights and opts
        vae_state_dict_prev = torch.load(vae_state_dict_prev_path, map_location=map_location)
        vae.load_state_dict(vae_state_dict_prev)
        vae_opt_state_dict_prev = torch.load(vae_opt_state_dict_prev_path, map_location=map_location)
        opt_vae.load_state_dict(vae_opt_state_dict_prev)

        # Load in Transformer model weights and opt
        transformer_state_dict_prev = torch.load(transformer_state_dict_prev_path, map_location=map_location)
        transformer.load_state_dict(transformer_state_dict_prev)
        opt_transformer_state_dict_prev = torch.load(opt_transformer_state_dict_prev_path, map_location=map_location)
        opt_transformer.load_state_dict(opt_transformer_state_dict_prev)

        print("Weights and Opts loaded from checkpoints")

    # Create the training object
    trainer = Trainer(
        world_size=world_size,
        gpu_id=gpu_id, 
        transformer=transformer,
        opt_transformer=opt_transformer,
        vae=vae, 
        start_epoch=start_epoch,
        train_dataset=train_dataset, 
        valfinetune_dataset=valfinetune_dataset,
        valunseen_dataset=valunseen_dataset,
        opt_vae=opt_vae,
        wdecode_batch_size=wdecode_batch_size,
        onlylatent_batch_size=onlylatent_batch_size,
        PaCMAP_model_to_infer=PaCMAP_model_to_infer,
        wandb_run=wandb_run,
        finetune_pacmap=finetune_pacmap,
        **kwargs)
    
    # Run through all epochs
    for epoch in range(start_epoch, epochs_to_train):
        trainer.epoch = epoch

        # PACMAP
        if (epoch > 0) & ((trainer.epoch + 1) % trainer.pacmap_every == 0):

            # Save pre-finetune model/opt weights
            if finetune_pacmap:
                vae_dict = trainer.vae.module.state_dict()
                vae_opt_dict = trainer.opt_vae.state_dict()
                transformer_dict = trainer.transformer.module.state_dict()
                transformer_opt_dict = trainer.opt_transformer.state_dict()

                # FINETUNE on beginning of validation patients (currently only one epoch)
                # Set to train and change LR to validate settings
                trainer._set_to_train()
                trainer.opt_vae.param_groups[0]['lr'] = LR_val_vae
                trainer.opt_transformer.param_groups[0]['lr'] = LR_val_transformer
                trainer._run_epoch(
                    dataset_curr = trainer.valfinetune_dataset, 
                    dataset_string = "valfinetune",
                    batchsize=trainer.wdecode_batch_size,
                    random_bool = True, # will subsample and randomize
                    subsample_file_factor_curr = valfinetune_subsample_file_factor, # only valid if 'random_bool' is True
                    all_files_latent_only = False, # this will run every file for every patient instead of subsampling (changes how dataloaders are made)
                    val_finetune = True,
                    val_unseen = False,
                    backprop = True,
                    num_rand_hashes = val_num_rand_hashes,
                    **kwargs)

            # INFERENCE on all datasets
            dataset_list = [trainer.train_dataset, trainer.valfinetune_dataset, trainer.valunseen_dataset]
            dataset_strs = ["train", "valfinetune", "valunseen"]
            trainer._set_to_eval()
            with torch.no_grad():
                for d in range(0, len(dataset_list)):
                    trainer._run_epoch(
                        dataset_curr = dataset_list[d], 
                        dataset_string = dataset_strs[d],
                        batchsize=trainer.onlylatent_batch_size,
                        random_bool = False, # will subsample and randomize
                        subsample_file_factor_curr = -1, # only valid if 'random_bool' is True
                        all_files_latent_only = True, # this will run every file for every patient instead of subsampling (changes how dataloaders are made)
                        val_finetune = False,
                        val_unseen = False,
                        backprop = False,
                        num_rand_hashes = -1,
                        **kwargs)

            # PACMAP
            # Only initiate for one GPU process - but calculations are done on CPU
            if (gpu_id == 0): 
                # New pacmap run for every win/stride combination
                for i in range(len(trainer.pre_PaCMAP_window_sec_list)):
                    utils_functions.run_pacmap(
                        dataset_strs=dataset_strs, 
                        epoch=epoch, 
                        win_sec=trainer.pre_PaCMAP_window_sec_list[i], 
                        stride_sec=trainer.pre_PaCMAP_stride_sec_list[i], 
                        latent_subdir=f"/latent_files/Epoch{epoch}", 
                        **kwargs)

            # Restore model/opt weights to pre-finetune
            if finetune_pacmap:
                trainer.vae.module.load_state_dict(vae_dict)
                trainer.opt_vae.load_state_dict(vae_opt_dict)
                trainer.transformer.module.load_state_dict(transformer_dict)
                trainer.opt_transformer.load_state_dict(transformer_opt_dict)
        
        # AUTOREGRESSIVE INFERENCE 
        # Training data
        if (trainer.epoch + 1) % trainer.autoreg_every == 0:
            trainer._set_to_eval()
            print("RUNNING AUTOREGRESSION on random pats/files")
            with torch.no_grad():
                trainer._random_autoreg_plots(
                    dataset_curr = trainer.train_dataset, 
                    dataset_string = "train",
                    num_rand_hashes = train_num_rand_hashes,
                    **kwargs)

        # TRAIN
        trainer._set_to_train()
        trainer._run_epoch(
            dataset_curr = trainer.train_dataset, 
            dataset_string = "train",
            batchsize=trainer.wdecode_batch_size,
            random_bool = True, # will subsample and randomize
            subsample_file_factor_curr = train_subsample_file_factor, # only valid if 'random_bool' is True
            all_files_latent_only = False, # this will run every file for every patient instead of subsampling (changes how dataloaders are made)
            val_finetune = False,
            val_unseen = False,
            backprop = True,
            num_rand_hashes = train_num_rand_hashes,
            **kwargs)
        
        # CHECKPOINT
        # After every train epoch, optionally delete old checkpoints
        print(f"GPU{str(trainer.gpu_id)} at pre checkpoint save barrier")
        barrier()
        if trainer.gpu_id == 0: trainer._save_checkpoint(trainer.epoch, **kwargs)

        # VALIDATE 
        # (skip if it's a pacmap epoch because it will already have been done)
        if ((trainer.epoch + 1) % trainer.val_every == 0) & ((trainer.epoch + 1) % trainer.pacmap_every != 0):
            # Save pre-finetune model/opt weights
            vae_dict = trainer.vae.module.state_dict()
            vae_opt_dict = trainer.opt_vae.state_dict()
            transformer_dict = trainer.transformer.module.state_dict()
            transformer_opt_dict = trainer.opt_transformer.state_dict()

            # FINETUNE on beginning of validation patients (currently only one epoch)
            # Set to train and change LR to validate settings
            trainer._set_to_train()
            trainer.opt_vae.param_groups[0]['lr'] = LR_val_vae
            trainer.opt_transformer.param_groups[0]['lr'] = LR_val_transformer
            trainer._run_epoch(
                dataset_curr = trainer.valfinetune_dataset, 
                dataset_string = "valfinetune",
                batchsize=trainer.wdecode_batch_size,
                random_bool = True, # will subsample and randomize
                subsample_file_factor_curr = valfinetune_subsample_file_factor, # only valid if 'random_bool' is True
                all_files_latent_only = False, # this will run every file for every patient instead of subsampling (changes how dataloaders are made)
                val_finetune = True,
                val_unseen = False,
                backprop = True,
                num_rand_hashes = val_num_rand_hashes,
                **kwargs)

            # Inference on UNSEEN portion of validation patients 
            trainer._set_to_eval()
            with torch.no_grad():
                trainer._run_epoch(
                    dataset_curr = trainer.valunseen_dataset, 
                    dataset_string = "valunseen",
                    batchsize=trainer.wdecode_batch_size,
                    random_bool = True, # will subsample and randomize
                    subsample_file_factor_curr = valunseen_subsample_file_factor, # only valid if 'random_bool' is True
                    all_files_latent_only = False, # this will run every file for every patient instead of subsampling (changes how dataloaders are made)
                    val_finetune = False,
                    val_unseen = True,
                    backprop = False,
                    num_rand_hashes = val_num_rand_hashes,
                    **kwargs)

            # Restore model/opt weights to pre-finetune
            trainer.vae.module.load_state_dict(vae_dict)
            trainer.opt_vae.load_state_dict(vae_opt_dict)
            trainer.transformer.module.load_state_dict(transformer_dict)
            trainer.opt_transformer.load_state_dict(transformer_opt_dict)

    # Kill the process after training loop completes
    print(f"[GPU{gpu_id}]: End of train loop, killing subprocess")
    wandb.finish()
    destroy_process_group() 

class Trainer:
    def __init__(
        self,
        world_size: int,
        gpu_id: int,
        transformer: torch.nn.Module,
        opt_transformer: torch.optim.Optimizer,
        vae: torch.nn.Module,
        start_epoch: int,
        train_dataset: SEEG_Tornado_Dataset,
        valfinetune_dataset: SEEG_Tornado_Dataset,
        valunseen_dataset: SEEG_Tornado_Dataset,
        opt_vae: torch.optim.Optimizer,
        wdecode_batch_size: int,
        onlylatent_batch_size: int,
        wandb_run,
        model_dir: str,
        autoreg_every: int,
        val_every: int,
        pacmap_every: int,
        finetune_pacmap: bool,
        latent_dim: int,
        autoencode_samples: int,
        num_samples: int,
        transformer_seq_length: int,
        FS: int,
        intrapatient_dataset_style: list,
        transformer_weight: float,
        recon_weight: float,
        atd_file: str,
        PaCMAP_model_to_infer,
        pre_PaCMAP_window_sec_list: list,
        pre_PaCMAP_stride_sec_list: list,
        recent_display_iters: int,
        **kwargs
    ) -> None:
        self.world_size = world_size
        self.gpu_id = gpu_id
        self.transformer = transformer
        self.opt_transformer = opt_transformer
        self.vae = vae
        self.start_epoch = start_epoch
        self.train_dataset = train_dataset
        self.valfinetune_dataset = valfinetune_dataset
        self.valunseen_dataset = valunseen_dataset
        self.opt_vae = opt_vae
        self.wdecode_batch_size = wdecode_batch_size
        self.onlylatent_batch_size = onlylatent_batch_size
        self.model_dir = model_dir
        self.autoreg_every = autoreg_every
        self.val_every = val_every
        self.pacmap_every = pacmap_every
        self.finetune_pacmap = finetune_pacmap
        self.latent_dim = latent_dim
        self.autoencode_samples = autoencode_samples
        self.num_samples = num_samples
        self.transformer_seq_length = transformer_seq_length
        self.FS = FS
        self.intrapatient_dataset_style = intrapatient_dataset_style
        self.curr_LR_core = -1
        self.transformer_weight = transformer_weight
        self.recon_weight = recon_weight
        self.atd_file = atd_file
        self.PaCMAP_model_to_infer = PaCMAP_model_to_infer
        self.pre_PaCMAP_window_sec_list = pre_PaCMAP_window_sec_list
        self.pre_PaCMAP_stride_sec_list = pre_PaCMAP_stride_sec_list
        self.recent_display_iters = recent_display_iters
        self.wandb_run = wandb_run
        self.kwargs = kwargs

        assert len(self.pre_PaCMAP_window_sec_list) == len(self.pre_PaCMAP_stride_sec_list)

        self.KL_multiplier = -1 # dummy variable, only needed when debugging and training is skipped

        # Number of iterations per file
        self.num_windows = int((self.num_samples - self.transformer_seq_length * self.autoencode_samples - self.autoencode_samples)/self.autoencode_samples) - 2

        # Set up vae & transformer with DDP
        self.vae = DDP(vae, device_ids=[gpu_id])   # find_unused_parameters=True
        self.transformer = DDP(transformer, device_ids=[gpu_id])   # find_unused_parameters=True
                    
        # Watch with WandB
        wandb.watch(self.vae)
        wandb.watch(self.transformer)
        
    def _set_to_train(self):
        self.vae.train()
        self.transformer.train()

    def _set_to_eval(self):
        self.vae.eval()
        self.transformer.eval()

    def _zero_all_grads(self):
        self.opt_vae.zero_grad()
        self.opt_transformer.zero_grad()

    def _save_checkpoint(self, epoch, delete_old_checkpoints, **kwargs):
            
        print("CHECKPOINT SAVE")

        # Create new directory for this epoch
        base_checkpoint_dir = self.model_dir + f"/checkpoints"
        check_epoch_dir = base_checkpoint_dir + f"/Epoch_{str(epoch)}"

        print("Saving vae model weights")

        ### VAE CHECKPOINT 
        check_core_dir = check_epoch_dir + "/core_checkpoints"
        if not os.path.exists(check_core_dir): os.makedirs(check_core_dir)

        # Save vae model
        ckp = self.vae.module.state_dict()
        check_path = check_core_dir + "/checkpoint_epoch" +str(epoch) + "_vae.pt"
        torch.save(ckp, check_path)

        # Save vae core
        opt_ckp = self.opt_vae.state_dict()
        opt_path = check_core_dir + "/checkpoint_epoch" +str(epoch) + "_vae_opt.pt"
        torch.save(opt_ckp, opt_path)

        ### TRANSFORMER CHECKPOINT ###

        print("Saving Transformer model weights")

        # Save transformer model
        check_transformer_dir = check_epoch_dir + "/transformer_checkpoints"
        if not os.path.exists(check_transformer_dir): os.makedirs(check_transformer_dir)
        ckp = self.transformer.module.state_dict()
        check_path = check_transformer_dir + "/checkpoint_epoch" +str(epoch) + "_transformer.pt"
        torch.save(ckp, check_path)
        
        # Save transformer optimizer
        opt_ckp = self.opt_transformer.state_dict()
        opt_path = check_transformer_dir + "/checkpoint_epoch" +str(epoch) + "_opt_transformer.pt"
        torch.save(opt_ckp, opt_path)

        print(f"Epoch {epoch} | Training checkpoint saved at {check_epoch_dir}")

        if delete_old_checkpoints:
            utils_functions.delete_old_checkpoints(dir = base_checkpoint_dir, curr_epoch = epoch)
            print("Deleted old checkpoints, except epochs with PaCMAP/HDBSCAN models")

    def _train_start_idxs(self, subsample_file_factor, random_bool):

        np.random.seed(seed=None) # should replace with Generator for newer code
        
        if random_bool: frame_shift = int(random.uniform(0, self.autoencode_samples -1))
        else: frame_shift = 0
        
        start_idxs = np.arange(0,self.num_windows - self.transformer_seq_length - 1) * self.autoencode_samples + frame_shift
        if random_bool: np.random.shuffle(start_idxs)

        if random_bool: start_idxs = start_idxs[0::subsample_file_factor]
        
        return start_idxs

    def _autoreg(self, context, target, autoreg_tokens_to_gen, n_layers, hash_pat_embedding, **kwargs):

        real_batchsize = context.shape[0]
        out_channels = context.shape[1]
        context_token_len = int(context.shape[2]/self.autoencode_samples)
        target_token_len = int(target.shape[2]/self.autoencode_samples)

        # Pseudo-batch the context
        context_batched = utils_functions.pseudobatch_raw_data(context, self.autoencode_samples)
        target_batched = utils_functions.pseudobatch_raw_data(target, self.autoencode_samples)

        ### VAE ENCODER
        # Forward pass in stacked batch VAE encoder
        _, _, autoreg_latent_context = self.vae(context_batched, reverse=False)
        _, _, autoreg_latent_target = self.vae(target_batched, reverse=False)

        # Split the batched dimension and stack into sequence dimension [batch, seq, latent_dims]
        autoreg_latent_context = torch.split(autoreg_latent_context, context_token_len, dim=0)
        autoreg_latent_context = torch.stack(autoreg_latent_context, dim=0)
        autoreg_latent_target = torch.split(autoreg_latent_target, target_token_len, dim=0)
        autoreg_latent_target = torch.stack(autoreg_latent_target, dim=0)

        ### TRANSFORMER
        # Greedy decoder
        autoreg_latent_pred = torch.zeros(context.shape[0], context_token_len + autoreg_tokens_to_gen, self.latent_dim).to(self.gpu_id)
        autoreg_latent_pred[:,:context_token_len, :] = autoreg_latent_context
        scores_allSeq_firstLayer_meanHeads_lastRow = torch.zeros(context.shape[0], autoreg_tokens_to_gen, context_token_len)
        for i in range(0, autoreg_tokens_to_gen):
            # Run sequence through transformer and get transformer loss
            predicted_embeddings, scores_allSeq_firstLayer_meanHeads_lastRow[:, i, :] = self.transformer(autoreg_latent_pred[:, i:i + context_token_len, :], attention_dropout= 0.0, return_attW=True)
            autoreg_latent_pred[:, context_token_len + i, :] = predicted_embeddings[:, -1, :]

        # Prune the latent predictions to just the generated sequence
        autoreg_latent_pred = autoreg_latent_pred[:, -autoreg_tokens_to_gen:, :]
            
        ### VAE DECODER
        # Run the predicted embeddings through decoder
        autoreg_latent_pred_batched = autoreg_latent_pred.reshape(autoreg_latent_pred.shape[0]*autoreg_latent_pred.shape[1], autoreg_latent_pred.shape[2])                        
        x_hat_batched = self.vae(autoreg_latent_pred_batched, reverse=True, hash_pat_embedding=hash_pat_embedding, out_channels=out_channels)  
        x_hat = torch.split(x_hat_batched, autoreg_tokens_to_gen, dim=0)
        x_hat = torch.stack(x_hat, dim=0)
        x_hat = x_hat.transpose(3, 2)
        x_hat = x_hat.reshape(x_hat.shape[0], x_hat.shape[1] * x_hat.shape[2], x_hat.shape[3])
        x_hat = x_hat.transpose(1, 2)

        return autoreg_latent_context, autoreg_latent_pred, autoreg_latent_target, x_hat, scores_allSeq_firstLayer_meanHeads_lastRow

    def _random_autoreg_plots(self, dataset_curr, dataset_string, autoreg_num_rand_pats, autoreg_context_tokens, autoreg_tokens_to_gen, autoreg_batchsize, autoreg_num_rand_files, num_rand_hashes, num_dataloader_workers_SEQUENTIAL, **kwargs):

        # Set up which pats will be selected for random autoreg
        np.random.seed(seed=None) # should replace with Generator for newer code        
        rand_pats_idxs = np.arange(0,len(dataset_curr.pat_ids))
        np.random.shuffle(rand_pats_idxs)
        rand_pats_idxs = rand_pats_idxs[0:autoreg_num_rand_pats]
        for pat_idx in rand_pats_idxs:
            dataset_curr.set_pat_curr(pat_idx)
            _, pat_id, _, _ = dataset_curr.get_pat_curr()
            dataloader_curr = utils_functions.prepare_dataloader(dataset_curr, batch_size=autoreg_batchsize, num_workers=num_dataloader_workers_SEQUENTIAL)
            dataloader_curr.sampler.set_epoch(self.epoch)

            # Iterate through 
            rand_file_count = 0
            for data_tensor, file_name in dataloader_curr:

                # Get the random start idx in file
                random_start_idx = np.arange(0, self.num_samples - (autoreg_context_tokens + autoreg_tokens_to_gen)*self.autoencode_samples - 1)
                np.random.shuffle(random_start_idx)
                random_start_idx = random_start_idx[0]

                # HASHING: Get the patient channel order and patid_embedding for this channel order
                np.random.seed(seed=None) 
                rand_modifer = int(random.uniform(0, num_rand_hashes -1))
                hash_pat_embedding, hash_channel_order = utils_functions.hash_to_vector(
                    input_string=pat_id, 
                    num_channels=data_tensor.shape[1], 
                    latent_dim=self.latent_dim, 
                    modifier=rand_modifer)

                # Pull out context and target raw data and move to GPU
                context_raw = data_tensor[:, hash_channel_order, random_start_idx: random_start_idx + autoreg_context_tokens * self.autoencode_samples]
                target_raw = data_tensor[:, hash_channel_order, random_start_idx + autoreg_context_tokens * self.autoencode_samples : random_start_idx + autoreg_context_tokens * self.autoencode_samples + autoreg_tokens_to_gen * self.autoencode_samples]
                context_raw = context_raw.to(self.gpu_id)
                target_raw = target_raw.to(self.gpu_id)

                # Conduct the autoregressive decoding
                autoreg_latent_context, autoreg_latent_pred, autoreg_latent_target, autoreg_raw_pred, scores_allSeq_firstLayer_meanHeads_lastRow = self._autoreg(
                    context=context_raw, 
                    target=target_raw,
                    autoreg_tokens_to_gen=autoreg_tokens_to_gen,
                    hash_pat_embedding=hash_pat_embedding,
                    **kwargs)

                # Plot the latent predictions
                utils_functions.print_autoreg_latent_predictions(
                    gpu_id=self.gpu_id,
                    epoch=self.epoch,
                    pat_id=pat_id,
                    rand_file_count=rand_file_count,
                    latent_context=autoreg_latent_context,
                    latent_predictions=autoreg_latent_pred, 
                    latent_target=autoreg_latent_target,
                    savedir = self.model_dir + f"/autoreg_plots/{dataset_string}/autoreg_latents",
                    **kwargs)

                # Plot the raw predictions
                utils_functions.print_autoreg_raw_predictions(
                    gpu_id=self.gpu_id,
                    epoch=self.epoch,
                    pat_id=pat_id,
                    rand_file_count=rand_file_count,
                    raw_context=context_raw,
                    raw_pred=autoreg_raw_pred,
                    raw_target=target_raw,
                    savedir = self.model_dir + f"/autoreg_plots/{dataset_string}/autoreg_raw",
                    **kwargs)

                # Plot the Attention Scores along the generated sequence in First Transformer Layer (mean of heads)
                utils_functions.print_autoreg_AttentionScores_AlongSeq(
                    gpu_id=self.gpu_id,
                    epoch=self.epoch,
                    pat_id=pat_id,
                    rand_file_count=rand_file_count,
                    scores_allSeq_firstLayer_meanHeads_lastRow=scores_allSeq_firstLayer_meanHeads_lastRow, 
                    savedir = self.model_dir + f"/autoreg_plots/{dataset_string}/autoreg_attention")

                # Kill after number of random files is complete
                rand_file_count = rand_file_count + 1
                if rand_file_count >= autoreg_num_rand_files: break

    def _run_epoch(
        self, 
        dataset_curr, 
        dataset_string,
        batchsize,
        attention_dropout,
        random_bool, # will subsample and randomize
        subsample_file_factor_curr, # only valid if 'random' is True
        all_files_latent_only,
        num_dataloader_workers_SEQUENTIAL,
        val_finetune,
        val_unseen,
        backprop,
        realtime_latent_printing,
        realtime_printing_interval,
        num_rand_hashes,
        pseudobatch_onlylatent,
        **kwargs):

        print(f"autoencode_samples: {self.autoencode_samples}")

        ### ALL/FULL FILES - LATENT ONLY ### 
        # This setting is used for inference
        # If wanting all files from every patient, need to run patients serially
        if all_files_latent_only: 

            # Check for erroneous configs
            if backprop or random_bool or val_finetune or val_unseen:
                raise Exception("ERROR: innapropriate config: if running all files then backprop/random_bool/val_finetune/val_unseen must all be False")

            # Go through every subject in this dataset
            for pat_idx in range(0,len(dataset_curr.pat_ids)):
                dataset_curr.set_pat_curr(pat_idx)
                dataloader_curr =  utils_functions.prepare_dataloader(dataset_curr, batch_size=batchsize, num_workers=num_dataloader_workers_SEQUENTIAL)
                
                # Go through every file in dataset
                file_count = 0
                for data_tensor, file_name in dataloader_curr:

                    # Print status
                    file_count = file_count + len(file_name)
                    if (self.gpu_id == 0):
                        sys.stdout.write(f"\rPat {pat_idx}/{len(dataset_curr.pat_ids)-1}, File {file_count - len(file_name)}:{file_count}/{len(dataset_curr)/self.world_size - 1}  * GPUs (DDP)               ") 
                        sys.stdout.flush() 

                    # Pseudo batch the file to speed up processing - up to pseudobatch_onlylatent size each encode pass
                    # Determine how many pseudo windows are in file based on pseudobatch_onlylatent
                    num_pseudo_windows = data_tensor.shape[2] / self.autoencode_samples / pseudobatch_onlylatent
                    assert num_pseudo_windows % 1 == 0
                    num_pseudo_windows = int(num_pseudo_windows)
                    
                    # Split the data by number of pseudo windows
                    data_tensor_split = torch.stack(torch.split(data_tensor, self.autoencode_samples, dim=2), dim=1)

                    # Create the sequential latent sequence array for the file
                    num_windows_in_file = data_tensor.shape[2] / self.autoencode_samples
                    assert (num_windows_in_file % 1) == 0
                    num_windows_in_file = int(num_windows_in_file)
                    file_latents = np.zeros([data_tensor.shape[0], num_windows_in_file, self.latent_dim])

                    for w in range(num_pseudo_windows):
                        # Pseudobatch and encode
                        x = data_tensor_split[:, w * pseudobatch_onlylatent:  w * pseudobatch_onlylatent + pseudobatch_onlylatent, :, :]
                        x_batched = x.reshape([x.shape[0]*x.shape[1], x.shape[2], x.shape[3]])
                        x_batched = x_batched.to(self.gpu_id)

                         ### VAE ENCODER
                        # Forward pass in stacked batch through VAE encoder
                        _, _, latent_batched = self.vae(x_batched, reverse=False)
                        
                        # Split the batched dimension and stack into sequence dimension [batch, seq, latent_dims]
                        latent_seq = torch.split(latent_batched, pseudobatch_onlylatent, dim=0)
                        file_latents[:, w * pseudobatch_onlylatent:  w * pseudobatch_onlylatent + pseudobatch_onlylatent, :] = torch.stack(latent_seq, dim=0).cpu().numpy()

                    # After file complete, pacmap_window/stride the file and save each file from batch seperately
                    # Seperate directory for each win/stride combination
                    for i in range(len(self.pre_PaCMAP_window_sec_list)):
                        win_sec_curr = self.pre_PaCMAP_window_sec_list[i]
                        stride_sec_curr = self.pre_PaCMAP_stride_sec_list[i]
                        
                        num_latents_in_win = win_sec_curr / (self.autoencode_samples / self.FS) 
                        assert (num_latents_in_win % 1) == 0
                        num_latents_in_win = int(num_latents_in_win)

                        num_latents_in_stride = stride_sec_curr / (self.autoencode_samples / self.FS) 
                        assert (num_latents_in_stride % 1) == 0
                        num_latents_in_stride = int(num_latents_in_stride)

                        # May not go in evenly, that is ok
                        num_strides_in_file = int((file_latents.shape[1] - num_latents_in_win) / num_latents_in_stride) 
                        windowed_file_latent = np.zeros([data_tensor.shape[0], num_strides_in_file, self.latent_dim])
                        for s in range(num_strides_in_file):
                            windowed_file_latent[:, s, :] = np.mean(file_latents[:, s*num_latents_in_stride: s*num_latents_in_stride + num_latents_in_win], axis=1)

                        # Save each windowed latent in a pickle for each file
                        for b in range(data_tensor.shape[0]):
                            filename_curr = file_name[b]
                            save_dir = f"{self.model_dir}/latent_files/Epoch{self.epoch}/{win_sec_curr}SecondWindow_{stride_sec_curr}SecondStride/{dataset_string}"
                            if not os.path.exists(save_dir): os.makedirs(save_dir)
                            output_obj = open(f"{save_dir}/{filename_curr}_latent_{self.win_sec_curr}secWindow_{self.stride_sec_curr}secStride.pkl", 'wb')
                            pickle.dump(windowed_file_latent[b, :, :], output_obj)
                            output_obj.close()


        ### RANDOM SUBSET OF FILES ###
        # This setting is used for regular training 
        # Can run the patients in parallel with this setting
        else: 
            dataset_curr.set_pat_curr(-1) # -1 enables all pat mode

            # Build dataloader from dataset
            dataloader_curr =  utils_functions.prepare_dataloader(dataset_curr, batch_size=batchsize, num_workers=num_dataloader_workers_SEQUENTIAL)
            dataloader_curr.sampler.set_epoch(self.epoch) 
            num_pats_curr = dataloader_curr.dataset.get_pat_count()
            
            # Get miniepoch start indexes. Same random indexes will be used for all patients' files.
            start_idxs = self._train_start_idxs(random_bool=random_bool, subsample_file_factor=subsample_file_factor_curr)
            total_train_iters = int(len(dataloader_curr) * len(start_idxs) * num_pats_curr) # 
            iter_curr = 0 # Iteration across all sequential trian files

            for data_tensor_by_pat, file_name_by_pat in dataloader_curr: # Paralell random file pull accross patients
                for start_idx in start_idxs: # Same start_idx for all patients (has no biological meaning)

                    # For Training: Update the KL multiplier (BETA), and Learning Rate according for Heads Models and Core Model
                    self.KL_multiplier, self.curr_LR_core, self.transformer_LR = utils_functions.LR_and_weight_schedules(
                        epoch=self.epoch, iter_curr=iter_curr, iters_per_epoch=int(total_train_iters), **self.kwargs)
                    
                    # Update LR to schedule
                    if (not val_finetune) & (not val_unseen):
                        self.opt_vae.param_groups[0]['lr'] = self.curr_LR_core
                        self.opt_transformer.param_groups[0]['lr'] = self.transformer_LR

                    # Reset the cumulative losses & zero gradients
                    kld_loss = 0
                    recon_loss = 0
                    transformer_loss = 0
                    self._zero_all_grads()

                    # # Save mean/logvar across all patients' transformer sequences so we can calculate KLD across patient cohort instead on individual transformer sequences
                    # # [pat, batch, transformer seq idx, vals]
                    # mean_allpats = torch.zeros(num_pats_curr, batchsize, self.transformer_seq_length, self.latent_dim).to(self.gpu_id)
                    # logvar_allpats = torch.zeros(num_pats_curr, batchsize, self.transformer_seq_length, self.latent_dim).to(self.gpu_id)

                    # Iterate through all patients and accumulate losses before stepping optimizers
                    for pat_idx in np.arange(0,num_pats_curr): 

                        # Pull patient's data
                        data_tensor = data_tensor_by_pat[pat_idx]

                        # Reset the data vars for Transformer Sequence and put on GPU
                        x = torch.zeros(data_tensor.shape[0], self.transformer_seq_length, data_tensor.shape[1], self.autoencode_samples).to(self.gpu_id)

                        # HASHING: Get the patient channel order and patid_embedding for this channel order
                        np.random.seed(seed=None) 
                        rand_modifer = int(random.uniform(0, num_rand_hashes -1))
                        hash_pat_embedding, hash_channel_order = utils_functions.hash_to_vector(
                            input_string=dataset_curr.pat_ids[pat_idx], 
                            num_channels=data_tensor.shape[1], 
                            latent_dim=self.latent_dim, 
                            modifier=rand_modifer)

                        # Collect sequential embeddings for transformer by running sequential raw data windows through BSE N times 
                        for embedding_idx in range(0, self.transformer_seq_length):

                            # Pull out data for this window
                            end_idx = start_idx + self.autoencode_samples * embedding_idx + self.autoencode_samples 
                            x[:, embedding_idx, :, :] = data_tensor[:, hash_channel_order, end_idx-self.autoencode_samples : end_idx]

                        # Stack the vars into batch dimension 
                        x_batched = x.reshape([x.shape[0]*x.shape[1], x.shape[2], x.shape[3]])

                        # To remember how to split back to original:
                        # a = torch.split(x_decode_shifted_batched, self.transformer_seq_length-1, dim=0)
                        # b = torch.stack(a, dim=0)

                        ### VAE ENCODER
                        # Forward pass in stacked batch through VAE encoder
                        mean_batched, logvar_batched, latent_batched = self.vae(x_batched, reverse=False)

                        # Split the batched dimension and stack into sequence dimension [batch, seq, latent_dims]
                        latent_seq = torch.split(latent_batched, self.transformer_seq_length, dim=0)
                        latent_seq = torch.stack(latent_seq, dim=0)
  
                        ### TRANSFORMER
                        # Run sequence through transformer and get transformer loss
                        predicted_embeddings = self.transformer(latent_seq[:, :-1, :], attention_dropout=attention_dropout)
                        
                        ### VAE DECODER
                        # Run the predicted embeddings through decoder
                        predicted_embeddings_batched = predicted_embeddings.reshape(predicted_embeddings.shape[0]*predicted_embeddings.shape[1], predicted_embeddings.shape[2])                        
                        x_hat_batched = self.vae(predicted_embeddings_batched, reverse=True, hash_pat_embedding=hash_pat_embedding, out_channels=x_batched.shape[1])  
                        x_hat = torch.split(x_hat_batched, self.transformer_seq_length-1, dim=0)
                        x_hat = torch.stack(x_hat, dim=0)
 
                        # LOSSES: Intra-Patient 
                        transformer_loss = loss_functions.transformer_loss_function( 
                            latent_seq[:, 1:, :],  
                            predicted_embeddings, 
                            transformer_weight=self.transformer_weight) 

                        recon_loss = loss_functions.recon_loss_function(
                            x=x[:, 1:, :, :], # Shifted by 1 due to predictions having gone through transformer
                            x_hat=x_hat,
                            recon_weight=self.recon_weight)

                        kld_loss = loss_functions.kld_loss_function(
                            mean=mean_batched, 
                            logvar=logvar_batched,  
                            KL_multiplier=self.KL_multiplier)

                        mean_loss = loss_functions.simple_mean_latent_loss(latent_seq, **kwargs)

                        # Intrapatient backprop
                        loss = recon_loss # + kld_loss + transformer_loss # + mean_loss # + kld_loss + transformer_loss             ################ direct TRANSFORMER LOSS INCLUDED ?????????? ##############
                        if backprop: loss.backward()

                        # Realtime info as epoch is running
                        if (iter_curr%self.recent_display_iters==0):
                            if val_finetune: state_str = "VAL FINETUNE"
                            elif val_unseen: state_str = "VAL UNSEEN"
                            else: state_str = "TRAIN"
                            now_str = datetime.datetime.now().strftime("%I:%M%p-%B/%d/%Y")
                            if (self.gpu_id == 1):
                                sys.stdout.write(
                                    f"\r{now_str} [GPU{str(self.gpu_id)}]: {state_str}, EPOCH {self.epoch}, Iter [BatchSize:{batchsize}]: " + 
                                    str(iter_curr) + "/" + str(total_train_iters) + 
                                    ", MeanLoss: " + str(round(loss.detach().item(), 2)) + ", [" + 
                                        "Rec: " + str(round(recon_loss.detach().item(), 2)) + " + " + 
                                        "KLD: {:0.3e}".format(kld_loss.detach().item(), 2) + " + " +
                                        "Trnsfr {:0.3e}".format(transformer_loss.detach().item(), 2) + "], ") 
                                sys.stdout.flush() 

                            # Log to WandB
                            wandb.define_metric('Steps')
                            wandb.define_metric("*", step_metric="Steps")
                            train_step = self.epoch * int(total_train_iters) + iter_curr
                            if (not val_finetune) & (not val_unseen):
                                metrics = dict(
                                    train_loss=loss,
                                    train_transformer_loss=transformer_loss,
                                    train_recon_loss=recon_loss, 
                                    train_mean_loss=mean_loss,
                                    train_kld_loss=kld_loss, 
                                    train_LR_encoder=self.opt_vae.param_groups[0]['lr'], 
                                    train_LR_transformer=self.opt_transformer.param_groups[0]['lr'],
                                    train_KL_Beta=self.KL_multiplier, 
                                    train_ReconWeight=self.recon_weight,
                                    train_Transformer_weight=self.transformer_weight,
                                    train_epoch=self.epoch)

                            elif val_finetune:
                                metrics = dict(
                                    val_finetune_loss=loss, 
                                    val_finetune_transformer_loss=transformer_loss,
                                    val_finetune_recon_loss=recon_loss, 
                                    val_finetune_mean_loss=mean_loss,
                                    val_finetune_kld_loss=kld_loss, 
                                    val_finetune_LR_encoder=self.opt_vae.param_groups[0]['lr'], 
                                    val_finetune_LR_transformer=self.opt_transformer.param_groups[0]['lr'],
                                    val_finetune_KL_Beta=self.KL_multiplier, 
                                    val_finetune_ReconWeight=self.recon_weight,
                                    val_finetune_Transformer_weight=self.transformer_weight,
                                    val_finetune_epoch=self.epoch)

                            elif val_unseen:
                                metrics = dict(
                                    val_unseen_loss=loss, 
                                    val_unseen_transformer_loss=transformer_loss,
                                    val_unseen_recon_loss=recon_loss, 
                                    val_unseen_mean_loss=mean_loss,
                                    val_unseen_kld_loss=kld_loss, 
                                    val_unseen_LR_encoder=self.opt_vae.param_groups[0]['lr'], 
                                    val_unseen_LR_transformer=self.opt_transformer.param_groups[0]['lr'],
                                    val_unseen_KL_Beta=self.KL_multiplier, 
                                    val_unseen_ReconWeight=self.recon_weight,
                                    val_unseen_Transformer_weight=self.transformer_weight,
                                    val_unseen_epoch=self.epoch)

                            wandb.log({**metrics, 'Steps': train_step})

                        # Advance the iteration counter (one iter per complete patient loop - i.e. one backward pass)
                        iter_curr = iter_curr + 1

                        # Realtime latent visualizations
                        if realtime_latent_printing & ((iter_curr + 1) % realtime_printing_interval == 0):
                            # # Regenerate random GPU idx each call
                            # np.random.seed(seed=None)
                            # rand_gpu = int(random.uniform(0, torch.cuda.device_count()))
                            if self.gpu_id == 0:
                                utils_functions.print_latent_realtime(
                                    target_emb = latent_seq[:, 1:, :].cpu().detach().numpy(), 
                                    predicted_emb = predicted_embeddings.cpu().detach().numpy(),
                                    savedir = self.model_dir + f"/realtime_plots/{dataset_string}/realtime_latents",
                                    epoch = self.epoch,
                                    iter_curr = iter_curr,
                                    pat_id = dataset_curr.pat_ids[pat_idx],
                                    **kwargs)
                                utils_functions.print_recon_realtime(
                                    x_decode_shifted=x[:, 1:, :, :], 
                                    x_hat=x_hat, 
                                    savedir = self.model_dir + f"/realtime_plots/{dataset_string}/realtime_recon",
                                    epoch = self.epoch,
                                    iter_curr = iter_curr,
                                    pat_id = dataset_curr.pat_ids[pat_idx],
                                    **kwargs
                                )
            
                    ### AFTER PATIENT LOOP ###
                    # Step optimizers after all patients have been backpropgated
                    self.opt_transformer.step()
                    self.opt_vae.step()
                        
        
if __name__ == "__main__":

    # Set the hash seed 
    os.environ['PYTHONHASHSEED'] = '1234'  

    # Read in configuration file & setup the run
    config_f = 'train_config.yml'
    with open(config_f, "r") as f: kwargs = yaml.load(f,Loader=yaml.FullLoader)
    kwargs = utils_functions.exec_kwargs(kwargs) # Execute the arithmatic build into kwargs and reassign kwargs
    world_size, kwargs = utils_functions.run_setup(**kwargs)

    # Spawn subprocesses with start/join (mp.spawn causes memory sigdev errors??)
    ctx = mp.get_context('spawn') # necessary to use context if have set_start_method anove?
    children = []
    for i in range(world_size):
        subproc = ctx.Process(target=main, args=(i, world_size, kwargs), kwargs=kwargs)
        children.append(subproc)
        subproc.start()

    for i in range(world_size):
        children[i].join()

    print("End of script")