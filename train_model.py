import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.utils.data import Dataset, DataLoader
from torch.distributions.gamma import Gamma
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
import ot
# from ot.lp import wasserstein_1d
# from ot.utils import proj_simplex
from geomloss import SamplesLoss
# from torch.utils.data._utils import shared_memory_cleanup

# Local Imports
from utilities import latent_plotting
from utilities import utils_functions
from utilities import loss_functions
from data import SEEG_Tornado_Dataset
from models.WAE import WAE

'''
@author: grahamwjohnson
Developed between 2023-2025

Main script to train the Brain State Embedder (BSE)
'''

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
    num_dataloader_workers,

    # WAE Optimizers
    core_weight_decay, 
    adamW_beta1,
    adamW_beta2,

    # Classifier Optimizer
    classifier_weight_decay,
    classifier_adamW_beta1,
    classifier_adamW_beta2,

    train_num_rand_hashes,
    val_num_rand_hashes,

    train_forward_passes,
    valfinetune_forward_passes,
    valunseen_forward_passes,

    **kwargs):

    # Split pats into train and test
    all_pats_dirs = glob.glob(f"{pat_dir}/*pat*")
    all_pats_list = [x.split('/')[-1] for x in all_pats_dirs]

    # If validation patients indicated
    if train_val_pat_perc[1] > 0:
        val_pats_count = int(np.ceil(train_val_pat_perc[1] * len(all_pats_list)))
        val_pats_dirs = all_pats_dirs[-val_pats_count:]
        val_pats_list = all_pats_list[-val_pats_count:]

        train_pats_dirs = all_pats_dirs[:-val_pats_count]
        train_pats_list = all_pats_list[:-val_pats_count]

        print(f"[GPU{str(gpu_id)}] Generating VALIDATION FINETUNE dataset")
        kwargs['dataset_pic_dir'] = kwargs['model_dir'] + '/dataset_bargraphs/Val_Grouped_Finetune'
        valfinetune_dataset = SEEG_Tornado_Dataset(
            gpu_id=gpu_id, 
            pat_list=val_pats_list,
            pat_dirs=val_pats_dirs,
            intrapatient_dataset_style=intrapatient_dataset_style, 
            hour_dataset_range=val_finetune_hour_dataset_range,
            num_rand_hashes=val_num_rand_hashes,
            num_forward_passes=valfinetune_forward_passes,
            data_logger_enabled=True,
            data_logger_file=f"{kwargs['log_dir']}/data_forward_pass_log_Valfinetune_GPU{gpu_id}.jsonl.gz",
            **kwargs)

        print(f"[GPU{str(gpu_id)}] Generating VALIDATION UNSEEN dataset")
        kwargs['dataset_pic_dir'] = kwargs['model_dir'] + '/dataset_bargraphs/Val_Grouped_Unseen'
        valunseen_dataset = SEEG_Tornado_Dataset(
            gpu_id=gpu_id, 
            pat_list=val_pats_list,
            pat_dirs=val_pats_dirs,
            intrapatient_dataset_style=intrapatient_dataset_style, 
            hour_dataset_range=val_unseen_hour_dataset_range,
            num_rand_hashes=val_num_rand_hashes,
            num_forward_passes=valunseen_forward_passes,
            data_logger_enabled=True,
            data_logger_file=f"{kwargs['log_dir']}/data_forward_pass_log_Valunseen_GPU{gpu_id}.jsonl.gz",
            **kwargs)

        # Random Dataloaders for Validation #
        valfinetune_dataset.set_pat_curr(-1) # -1 sets to random generation and starts data generation subpocess
        valfinetune_dataloader = utils_functions.prepare_dataloader(valfinetune_dataset, batch_size=None, num_workers=num_dataloader_workers)

        valunseen_dataset.set_pat_curr(-1) # -1 sets to random generation and starts data generation subpocess
        valunseen_dataloader = utils_functions.prepare_dataloader(valunseen_dataset, batch_size=None, num_workers=num_dataloader_workers)

    else: # If no val, just make a train dataset
        train_pats_dirs = all_pats_dirs
        train_pats_list = all_pats_list

        # Dummy
        valfinetune_dataset = None
        valunseen_dataset = None
        valfinetune_dataloader = None
        valunseen_dataloader = None

    # Sequential dataset used to run inference on train data and build PaCMAP projection
    print(f"[GPU{str(gpu_id)}] Generating TRAIN dataset")
    kwargs['dataset_pic_dir'] = kwargs['model_dir'] + '/dataset_bargraphs/Train_Grouped'
    train_dataset = SEEG_Tornado_Dataset(
        gpu_id=gpu_id, 
        pat_list=train_pats_list,
        pat_dirs=train_pats_dirs,
        intrapatient_dataset_style=intrapatient_dataset_style, 
        hour_dataset_range=train_hour_dataset_range,
        num_rand_hashes=train_num_rand_hashes,
        num_forward_passes=train_forward_passes,
        data_logger_enabled=True,
        data_logger_file=f"{kwargs['log_dir']}/data_forward_pass_log_Train_GPU{gpu_id}.jsonl.gz",
        **kwargs)

    ### Random DataLoaders ###
    train_dataset.set_pat_curr(-1) # -1 sets to random generation and starts data generation subpocess
    train_dataloader = utils_functions.prepare_dataloader(train_dataset, batch_size=None, num_workers=num_dataloader_workers)
         
    ### WAE ###
    wae = WAE(gpu_id=gpu_id, **kwargs) 
    wae = wae.to(gpu_id) 

    # Separate the parameters into two groups
    classifier_params = []
    wae_params = []

    # Iterate through the model parameters
    for name, param in wae.named_parameters():
        # Check if the parameter is part of the encoder submodule
        if 'adversarial_classifier' in name:
            classifier_params.append(param)
        else:
            wae_params.append(param)
    
    # opt_wae = torch.optim.AdamW(wae.parameters(), weight_decay=core_weight_decay, betas=(adamW_beta1, adamW_beta2), lr=kwargs['LR_min_core'])
    # opt_cls = torch.optim.AdamW(wae.classifier.parameters(), weight_decay=classifier_weight_decay, betas=(classifier_adamW_beta1, classifier_adamW_beta2), lr=kwargs['LR_min_classifier'])
    
    opt_wae = torch.optim.AdamW(wae_params, weight_decay=core_weight_decay, betas=(adamW_beta1, adamW_beta2), lr=kwargs['LR_min_core'])
    opt_cls = torch.optim.AdamW(classifier_params, weight_decay=classifier_weight_decay, betas=(classifier_adamW_beta1, classifier_adamW_beta2), lr=kwargs['LR_min_classifier'])

    return train_dataset, train_dataloader, valfinetune_dataset, valfinetune_dataloader, valunseen_dataset, valunseen_dataloader, wae, opt_wae, opt_cls

def main(  
    # Ordered variables
    gpu_id: int, 
    world_size: int, 
    config, # aka kwargs
    
    # Passed by kwargs
    run_name: str,
    timestamp_id: int,
    start_epoch: int,
    LR_val_wae: float,
    LR_val_cls: float,
    val_finetine_bool: bool,
    finetune_inference: bool, 
    finetune_inference_epochs: int,
    realtime_printing_interval_val,
    realtime_printing_interval_train,
    wae_state_dict_prev_path = [],
    wae_opt_state_dict_prev_path = [],
    cls_opt_state_dict_prev_path = [],
    running_latent_path = [],
    running_prior_path = [],
    epochs_to_train: int = -1,

    **kwargs):

    '''
    Highest level loop to build training objects, run train epochs, validate, inference on whole files to save embeddings... etc. 

    '''

    # Initialize new WandB here and group GPUs together with DDP
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
    train_dataset, train_dataloader, valfinetune_dataset, valfinetune_dataloader, valunseen_dataset, valunseen_dataloader, wae, opt_wae, opt_cls = load_train_objs(gpu_id=gpu_id, **kwargs) 
    
    # Load the model/opt states if not first epoch & if in training mode
    if (start_epoch > 0):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}

        # Load in WAE weights and opts
        wae_state_dict_prev = torch.load(wae_state_dict_prev_path, map_location=map_location)
        wae.load_state_dict(wae_state_dict_prev)
        wae_opt_state_dict_prev = torch.load(wae_opt_state_dict_prev_path, map_location=map_location)
        opt_wae.load_state_dict(wae_opt_state_dict_prev)
        cls_opt_state_dict_prev = torch.load(cls_opt_state_dict_prev_path, map_location=map_location)
        opt_cls.load_state_dict(cls_opt_state_dict_prev)
        print(f"[GPU{gpu_id}] Model and Opt weights loaded from checkpoints")

        # Load running latents observed
        with open(running_latent_path, "rb") as f: accumulated_z = pickle.load(f)
        print(f"[GPU{gpu_id}] Running Latents loaded from checkpoints")

        # Load running prior
        with open(running_prior_path, "rb") as f: accumulated_prior = pickle.load(f)
        print(f"[GPU{gpu_id}] Running Prior loaded from checkpoints")
    
    else:
        accumulated_z = []
        accumulated_prior = []

    # Create the training object
    trainer = Trainer(
        world_size=world_size,
        gpu_id=gpu_id, 
        wae=wae, 
        opt_wae=opt_wae,
        opt_cls=opt_cls,
        start_epoch=start_epoch,
        train_dataset=train_dataset, 
        train_dataloader=train_dataloader,
        valfinetune_dataset=valfinetune_dataset,
        valfinetune_dataloader=valfinetune_dataloader,
        valunseen_dataset=valunseen_dataset,
        valunseen_dataloader=valunseen_dataloader,
        wandb_run=wandb_run,
        finetune_inference=finetune_inference,
        accumulated_z=accumulated_z,
        accumulated_prior=accumulated_prior,
        **kwargs)

    # Kill the val data generators if val_every is set artificially high
    if (kwargs['val_every'] > 999) & (kwargs['inference_every'] > 999) & (trainer.valfinetune_dataset != None) & (trainer.valunseen_dataset != None):
        print(f"[GPU{str(trainer.gpu_id)}] WARNING: val_every is {kwargs['val_every']}, which is over arbitrary limit of 999, thus going to kill val generators")
        trainer.valfinetune_dataset.kill_generator()
        trainer.valunseen_dataset.kill_generator()
    
    # Main loop through all epochs
    for epoch in range(start_epoch, epochs_to_train):
        trainer.epoch = epoch

        # Full INFERENCE on all data
        if (epoch > 0) & ((trainer.epoch + 1) % trainer.inference_every == 0):

            trainer.valunseen_dataset.kill_generator()
            trainer.train_dataset.kill_generator()
            print(f"[GPU{trainer.gpu_id}] Killed Train/ValUnseen random generators for inference mode")

            # Save pre-finetune model/opt weights
            if finetune_inference:
                wae_dict = trainer.wae.module.state_dict()
                wae_opt_dict = trainer.opt_wae.state_dict()
                cls_opt_dict = trainer.opt_cls.state_dict()
                accumulated_z = trainer.accumulated_z
                accumulated_prior = trainer.accumulated_prior
                accumulated_labels = trainer.accumulated_labels
                accumulated_class_probs = trainer.accumulated_class_probs

                # FINETUNE on beginning of validation patients (currently only one epoch)
                # Set to train and change LR to validate settings
                trainer._set_to_train()
                trainer.opt_wae.param_groups[0]['lr'] = LR_val_wae
                trainer.opt_cls.param_groups[0]['lr'] = LR_val_cls
                for finetune_epoch in range(finetune_inference_epochs):
                    trainer.epoch = epoch + finetune_epoch
                    trainer._run_train_epoch(
                        dataloader_curr = trainer.valfinetune_dataloader, 
                        dataset_string = "valfinetune",
                        val_finetune = True,
                        val_unseen = False,
                        realtime_printing_interval = realtime_printing_interval_val,
                        **kwargs)

                # Finetuned, so setup inference for all datasets
                dataset_list = [trainer.train_dataset, trainer.valfinetune_dataset, trainer.valunseen_dataset]
                dataset_strs = ["train", "valfinetune", "valunseen"]

            else: # No finetune, only run data for Train and ValUnseen datasets (running on val_finetune would just create confusion later when using embeddings)
                dataset_list = [trainer.train_dataset, trainer.valunseen_dataset]
                dataset_strs = ["train", "valunseen"]
            
            # INFERENCE on all selected datasets, kill finetune now that we also do not need it
            trainer.valfinetune_dataset.kill_generator()
            print(f"[GPU{trainer.gpu_id}] Killed val_finetune for inference mode")

            trainer._set_to_eval()
            with torch.inference_mode():
                for d in range(0, len(dataset_list)):
                    trainer._run_export_embeddings(
                        dataset_curr = dataset_list[d],  # Takes in a Dataset, NOT a DataLoader
                        dataset_string = dataset_strs[d],
                        **kwargs)

            # Restore model/opt weights to pre-finetune, and restart data generators
            if finetune_inference:
                trainer.wae.module.load_state_dict(wae_dict)
                trainer.opt_wae.load_state_dict(wae_opt_dict)
                trainer.opt_cls.load_state_dict(cls_opt_dict)
                trainer.accumulated_z = accumulated_z
                trainer.accumulated_prior = accumulated_prior
                trainer.accumulated_labels = accumulated_labels
                trainer.accumulated_class_probs = accumulated_class_probs
            
            trainer.train_dataset.initiate_generator()
            trainer.valfinetune_dataset.initiate_generator()
            trainer.valunseen_dataset.initiate_generator()
            print(f"GPU{str(trainer.gpu_id)} at post inference barrier - restarted random data generators")
            barrier()
        
        # TRAIN
        trainer._set_to_train()
        trainer._run_train_epoch(
            dataloader_curr = trainer.train_dataloader, 
            dataset_string = "train",
            val_finetune = False,
            val_unseen = False,
            realtime_printing_interval = realtime_printing_interval_train,
            **kwargs)
        
        # CHECKPOINT
        # After every train epoch, optionally delete old checkpoints
        if trainer.gpu_id == 0: trainer._save_checkpoint(trainer.epoch, **kwargs)
        print(f"GPU{str(trainer.gpu_id)} at post checkpoint save barrier")
        barrier()

        # VALIDATE 
        if ((trainer.epoch + 1) % trainer.val_every == 0):
            
            if val_finetine_bool:
                # Save pre-finetune model/opt weights
                wae_dict = trainer.wae.module.state_dict()
                wae_opt_dict = trainer.opt_wae.state_dict()
                cls_opt_dict = trainer.opt_cls.state_dict()
                accumulated_z = trainer.accumulated_z
                accumulated_prior = trainer.accumulated_prior
                accumulated_labels = trainer.accumulated_labels
                accumulated_class_probs = trainer.accumulated_class_probs

                # FINETUNE on beginning of validation patients (currently only one epoch)
                # Set to train and change LR to validate settings
                trainer._set_to_train()
                trainer.opt_wae.param_groups[0]['lr'] = LR_val_wae
                trainer.opt_cls.param_groups[0]['lr'] = LR_val_cls
                trainer._run_train_epoch(
                    dataloader_curr = trainer.valfinetune_dataloader, 
                    dataset_string = "valfinetune",
                    val_finetune = True,
                    val_unseen = False,
                    realtime_printing_interval = realtime_printing_interval_val,
                    **kwargs)

            # UNSEEN portion of validation patients 
            trainer._set_to_eval()
            with torch.inference_mode():
                trainer._run_train_epoch(
                    dataloader_curr = trainer.valunseen_dataloader, 
                    dataset_string = "valunseen",
                    val_finetune = False,
                    val_unseen = True,
                    realtime_printing_interval = realtime_printing_interval_val,
                    **kwargs)

            # Restore model/opt weights to pre-finetune
            if val_finetine_bool:
                trainer.wae.module.load_state_dict(wae_dict)
                trainer.opt_wae.load_state_dict(wae_opt_dict)
                trainer.opt_cls.load_state_dict(cls_opt_dict)
                trainer.accumulated_z = accumulated_z
                trainer.accumulated_prior = accumulated_prior
                trainer.accumulated_labels = accumulated_labels
                trainer.accumulated_class_probs = accumulated_class_probs

    # Kill the process after training loop completes
    print(f"[GPU{gpu_id}]: End of train loop, killing 'main' subprocess")
    wandb.finish()
    destroy_process_group() 

class Trainer:
    def __init__(
        self,
        world_size: int,
        gpu_id: int,
        wae: torch.nn.Module,
        start_epoch: int,
        train_dataset: SEEG_Tornado_Dataset,
        valfinetune_dataset: SEEG_Tornado_Dataset,
        valunseen_dataset: SEEG_Tornado_Dataset,
        train_dataloader: DataLoader,
        valfinetune_dataloader: DataLoader,
        valunseen_dataloader: DataLoader,
        opt_wae: torch.optim.Optimizer,
        opt_cls: torch.optim.Optimizer,
        wandb_run,
        model_dir: str,
        val_every: int,
        inference_every: int,
        finetune_inference: bool,
        latent_dim: int,
        encode_token_samples: int,
        num_samples: int,
        transformer_seq_length: int,
        # num_encode_concat_transformer_tokens: int,
        transformer_start_pos: int,
        FS: int,
        hash_output_range: tuple,
        intrapatient_dataset_style: list,
        # transformer_weight: float,
        recon_weight: float,
        atd_file: str,
        inference_window_sec_list: list,
        inference_stride_sec_list: list,
        recent_display_iters: int,
        total_collected_latents: int,
        running_reg_passes: int,
        classifier_num_pats: int,
        multimodal_shapes: list,
        multimodal_scales: list,
        multimodal_weights: list,
        sinkhorn_blur: int,
        wasserstein_order: float,
        accumulated_z,
        accumulated_prior,
        **kwargs
    ) -> None:
        self.world_size = world_size
        self.gpu_id = gpu_id
        self.wae = wae
        self.start_epoch = start_epoch
        self.train_dataset = train_dataset
        self.valfinetune_dataset = valfinetune_dataset
        self.valunseen_dataset = valunseen_dataset
        self.train_dataloader = train_dataloader
        self.valfinetune_dataloader = valfinetune_dataloader
        self.valunseen_dataloader = valunseen_dataloader
        self.opt_wae = opt_wae
        self.opt_cls = opt_cls
        self.model_dir = model_dir
        self.val_every = val_every
        self.inference_every = inference_every
        self.finetune_inference = finetune_inference
        self.latent_dim = latent_dim
        self.encode_token_samples = encode_token_samples
        self.num_samples = num_samples
        self.transformer_seq_length = transformer_seq_length
        # self.num_encode_concat_transformer_tokens = num_encode_concat_transformer_tokens
        self.transformer_start_pos = transformer_start_pos
        self.FS = FS
        self.hash_output_range = hash_output_range
        self.intrapatient_dataset_style = intrapatient_dataset_style
        self.curr_LR_core = -1
        self.recon_weight = recon_weight
        self.atd_file = atd_file
        self.inference_window_sec_list = inference_window_sec_list
        self.inference_stride_sec_list = inference_stride_sec_list
        self.recent_display_iters = recent_display_iters
        self.total_collected_latents = total_collected_latents
        self.running_reg_passes = running_reg_passes
        self.classifier_num_pats = classifier_num_pats
        self.multimodal_shapes = multimodal_shapes
        self.multimodal_scales = multimodal_scales
        self.multimodal_weights = multimodal_weights
        self.sinkhorn_blur = sinkhorn_blur
        self.wasserstein_order = wasserstein_order
        self.accumulated_z = accumulated_z
        self.accumulated_prior = accumulated_prior
        self.wandb_run = wandb_run
        self.kwargs = kwargs

        # Check variables that will cause delayed crashes 
        assert len(self.inference_window_sec_list) == len(self.inference_stride_sec_list)

        self.reg_weight = -1 # dummy variable, only needed when debugging and training is skipped

        # Number of iterations per file
        self.num_windows = int((self.num_samples - self.transformer_seq_length * self.encode_token_samples - self.encode_token_samples)/self.encode_token_samples) - 2

        # Set up wae & transformer with DDP
        self.wae = DDP(wae, device_ids=[gpu_id])   # find_unused_parameters=True
                    
        # Running Regulizer window for latent data
        if self.accumulated_z == []:
            # If first initialziing, then start off with observed ideal distributions from pure prior
            self.accumulated_z = self._sample_multimodal(self.total_collected_latents).to(self.gpu_id)
            self.accumulated_prior = self._sample_multimodal(self.total_collected_latents).to(self.gpu_id)
        else: 
            # Ensure on proper device because loading from pickle
            self.accumulated_z = self.accumulated_z.to(self.gpu_id)
            self.accumulated_prior = self.accumulated_prior.to(self.gpu_id)

        # Running tab of update index
        self.next_update_index = 0

        # Watch with WandB
        wandb.watch(self.wae)
        
    def _set_to_train(self):
        self.wae.train()

    def _set_to_eval(self):
        self.wae.eval()

    def _zero_all_grads(self):
        self.opt_wae.zero_grad()
        self.opt_cls.zero_grad()

    def _save_checkpoint(self, epoch, delete_old_checkpoints, **kwargs):
            
        print("CHECKPOINT SAVE")

        # Create new directory for this epoch
        base_checkpoint_dir = self.model_dir + f"/checkpoints"
        check_epoch_dir = base_checkpoint_dir + f"/Epoch_{str(epoch)}"

        print("Saving wae model weights")

        ### WAE CHECKPOINT 
        check_core_dir = check_epoch_dir + "/core_checkpoints"
        if not os.path.exists(check_core_dir): os.makedirs(check_core_dir)

        # Save model
        ckp = self.wae.module.state_dict()
        check_path = check_core_dir + "/checkpoint_epoch" + str(epoch) + "_wae.pt"
        torch.save(ckp, check_path)

        # Save optimizers
        opt_ckp = self.opt_wae.state_dict()
        opt_path = check_core_dir + "/checkpoint_epoch" + str(epoch) + "_wae_opt.pt"
        torch.save(opt_ckp, opt_path)

        opt_ckp_cls = self.opt_cls.state_dict()
        opt_path_cls = check_core_dir + "/checkpoint_epoch" + str(epoch) + "_cls_opt.pt"
        torch.save(opt_ckp_cls, opt_path_cls)

        # Save Running Latents 
        latents_path = check_core_dir + "/checkpoint_epoch" +str(epoch) + "_running_latents.pkl"
        output_obj = open(latents_path, 'wb')
        pickle.dump(self.accumulated_z, output_obj)
        output_obj.close()
        print("Saved running latents")

        prior_path = check_core_dir + "/checkpoint_epoch" +str(epoch) + "_running_prior.pkl"
        output_obj = open(prior_path, 'wb')
        pickle.dump(self.accumulated_prior, output_obj)
        output_obj.close()
        print("Saved running prior")


        print(f"Epoch {epoch} | Training checkpoint saved at {check_epoch_dir}")

        if delete_old_checkpoints:
            utils_functions.delete_old_checkpoints(dir = base_checkpoint_dir, curr_epoch = epoch, **kwargs)
            print("Deleted old checkpoints, except epochs at end of reguaization annealing period")

    def _sample_multimodal(self, num_samps):
        """
        Used to initialize the learnable prior 

        Generates high-dimensional samples from a multimodal distribution using Gamma distributions.
        The same shape and scale parameters are used for every latent dimension within a mode.

        Args:
            num_samps (int): Number of samples to generate.
            self.multimodal_shapes (list or torch.Tensor): List or tensor of shape parameters (k) for the modes.
            self.multimodal_scales (list or torch.Tensor): List or tensor of scale parameters (θ) for the modes.
            self.multimodal_weights (list or torch.Tensor): List or tensor of weights for the modes, summing to 1.
            self.latent_dim (int): Dimensionality of the latent space (e.g., 2048).

        Returns:
            torch.Tensor: High-dimensional samples from the multimodal Gamma distribution, of shape (num_samps, self.latent_dim).

        Example usage:
            num_samps = 1000
            self.multimodal_shapes = [2, 5, 1]  # Shape parameters (k) for 3 modes
            self.multimodal_scales = [1, 2, 0.5]  # Scale parameters (θ) for 3 modes
            self.multimodal_weights = [0.2, 0.5, 0.3]  # Weights for the 3 modes, adds to 1
            self.latent_dim = 2048  # High-dimensional latent space

            samples = self._sample_multimodal(num_samps)
        """
        # Convert inputs to tensors if they are not already
        if not isinstance(self.multimodal_shapes, torch.Tensor):
            self.multimodal_shapes = torch.tensor(self.multimodal_shapes, dtype=torch.float32)
        if not isinstance(self.multimodal_scales, torch.Tensor):
            self.multimodal_scales = torch.tensor(self.multimodal_scales, dtype=torch.float32)
        if not isinstance(self.multimodal_weights, torch.Tensor):
            self.multimodal_weights = torch.tensor(self.multimodal_weights, dtype=torch.float32)

        # Determine which mode to sample from for each sample in the batch
        mode_indices = torch.multinomial(self.multimodal_weights, num_samples=num_samps, replacement=True)

        # Generate samples from the chosen modes
        samples = torch.zeros((num_samps, self.latent_dim))  # Shape: (num_samps, latent_dim)
        for i in range(len(self.multimodal_weights)):
            mask = mode_indices == i
            num_samples_from_mode = mask.sum()
            if num_samples_from_mode > 0:
                # Sample from a Gamma distribution for the current mode
                # Use the same shape and scale for all latent dimensions
                gamma_samples = torch.distributions.Gamma(
                    concentration=self.multimodal_shapes[i],  # Shape parameter (k)
                    rate=1 / self.multimodal_scales[i]       # Rate parameter (1/θ)
                ).sample((num_samples_from_mode, self.latent_dim))
                samples[mask] = gamma_samples
        
        return samples
        
    def _sample_running_window(self, latent, prior, random_observed_sampling, **kwargs):
        self.accumulated_z = self.accumulated_z.detach() # Ensure detached before sampling
        self.accumulated_prior = self.accumulated_prior.detach()
        num_new = latent.shape[0]
        num_past = self.running_reg_passes - num_new
        
        if num_new == self.running_reg_passes:
            return latent, prior

        elif num_new > self.running_reg_passes:
            raise Exception(f"latent.shape[0] > self.running_reg_passes ({num_new} > {self.running_reg_passes})")

        if random_observed_sampling: # RANDOM
            random_past_idxs = torch.randperm(self.accumulated_z.shape[0])[:num_past] 
            past_observed_samples = self.accumulated_z[random_past_idxs, :]

            random_past_idxs = torch.randperm(self.accumulated_prior.shape[0])[:num_past] 
            past_prior_samples = self.accumulated_prior[random_past_idxs, :]

        else: # NOT RANDOM
            # Must check for rollover
            if (self.next_update_index - num_past) >= 0:
                past_observed_samples = self.accumulated_z[self.next_update_index - num_past : self.next_update_index, :] # Use the next update index to know where the latest past samples are
                past_prior_samples = self.accumulated_prior[self.next_update_index - num_past : self.next_update_index, :]

            else: # Straddles end/start of indexes
                num_past_from_start = self.next_update_index
                num_past_from_end = num_past - num_past_from_start

                past_observed_samples_from_start = self.accumulated_z[:num_past_from_start, :]
                past_observed_samples_from_end = self.accumulated_z[-num_past_from_end:, :]

                past_prior_samples_from_start = self.accumulated_prior[:num_past_from_start, :]
                past_prior_samples_from_end = self.accumulated_prior[-num_past_from_end:, :]

                past_observed_samples = torch.cat([past_observed_samples_from_end, past_observed_samples_from_start])
                past_prior_samples = torch.cat([past_prior_samples_from_end, past_prior_samples_from_start])

        return torch.cat([past_observed_samples, latent]), torch.cat([past_prior_samples, prior])

    def _update_reg_window(
        self,
        latent,
        prior
        ):

        num_new_updates = latent.shape[0]
        if (self.next_update_index + num_new_updates) < self.total_collected_latents:
            self.accumulated_z[self.next_update_index: self.next_update_index + num_new_updates, :] = latent
            self.accumulated_prior[self.next_update_index: self.next_update_index + num_new_updates, :] = prior
            self.next_update_index = self.next_update_index + num_new_updates

        else: # Rollover
            residual_num = (self.next_update_index + num_new_updates) % self.total_collected_latents
            end_num = num_new_updates - residual_num

            self.accumulated_z[self.next_update_index: self.next_update_index + end_num, :] = latent[:end_num, :]
            self.accumulated_z[0: residual_num, :] = latent[end_num:, :]

            self.accumulated_prior[self.next_update_index: self.next_update_index + end_num, :] = prior[:end_num, :]
            self.accumulated_prior[0: residual_num, :] = prior[end_num:, :]

            self.next_update_index = residual_num

        # Detach to allow next backpass
        self.accumulated_z = self.accumulated_z.detach() 
        self.accumulated_prior = self.accumulated_prior.detach() 

    def _run_export_embeddings(
        self, 
        dataset_curr, 
        dataset_string,
        attention_dropout,
        num_dataloader_workers,
        max_batch_size,
        inference_batch_mult,
        padded_channels,
        **kwargs):

        '''
        This will run inference (ONLY encoder) and save the latent space per file to a .pkl
        Unlike _run_train_epoch (which is random data pulls), this function will sequentially iterate through entire file (one patient at a time)
        NOTE: This function takes in a *Dataset*, NOT a *Dataloader* like in _run_train_epoch

        OUTPUT
        The output files will be smoothed according to the following variables:
        inference_window_sec_list: e.g. [60, 20] 
        inference_stride_sec_list: e.g. [30, 5] 
    
        '''

        print(f"[GPU{self.gpu_id}] encode_token_samples: {self.encode_token_samples}")

        ### ALL/FULL FILES - LATENT ONLY - FULL TRANSFORMER CONTEXT ### 
        print("WARNING: Setting alpha = 0")
        self.classifier_alpha = 0

        # Go through every subject in this dataset
        for pat_idx in range(0,len(dataset_curr.pat_ids)):
            dataset_curr.set_pat_curr(pat_idx)
            dataloader_curr =  utils_functions.prepare_dataloader(dataset_curr, batch_size=max_batch_size, num_workers=num_dataloader_workers)

            # Go through every file in dataset
            file_count = 0
            for data_tensor, file_name, file_class_label in dataloader_curr: # Hash done outside data.py for single pat inference

                file_count = file_count + len(file_name)

                num_channels_curr = data_tensor.shape[1]

                # Create the sequential latent sequence array for the file 
                num_samples_in_forward = self.transformer_seq_length * self.encode_token_samples
                num_windows_in_file = data_tensor.shape[2] / num_samples_in_forward
                assert (num_windows_in_file % 1) == 0
                num_windows_in_file = int(num_windows_in_file)
                num_samples_in_forward = int(num_samples_in_forward)

                # Prep the output tensor and put on GPU
                files_latents = torch.zeros([data_tensor.shape[0], num_windows_in_file, self.latent_dim]).to(self.gpu_id)

                # Put whole file on GPU
                data_tensor = data_tensor.to(self.gpu_id)

                for w in range(num_windows_in_file):
                    
                    # Print Status
                    print_interval = 100
                    if (self.gpu_id == 0) & (w % print_interval == 0):
                        sys.stdout.write(f"\r{dataset_string}: Pat {pat_idx}/{len(dataset_curr.pat_ids)-1}, File {file_count - len(file_name)}:{file_count}/{len(dataset_curr)/self.world_size - 1}  * GPUs (DDP), Intrafile Iter {w}/{num_windows_in_file}          ") 
                        sys.stdout.flush() 

                    # Generate hashes for feedforward conditioning (single pat, so outside data.py)
                    rand_modifer = 0 # 0 for inference
                    hash_pat_embedding, hash_channel_order = utils_functions.hash_to_vector(
                        input_string=dataset_curr.pat_ids[pat_idx], 
                        num_channels=num_channels_curr, 
                        latent_dim=self.latent_dim, 
                        modifier=rand_modifer,
                        hash_output_range=self.hash_output_range)
                    
                    # Collect sequential embeddings for transformer by running sequential raw data windows through BSE N times 

                    # TODO: get rid of for loop like random data generator 
                    x = torch.zeros(data_tensor.shape[0], self.transformer_seq_length, padded_channels, self.encode_token_samples).to(self.gpu_id)
                    start_idx = w * num_samples_in_forward
                    for embedding_idx in range(0, self.transformer_seq_length):
                        # Pull out data for this window - NOTE: no hashing
                        end_idx = start_idx + self.encode_token_samples * embedding_idx + self.encode_token_samples 
                        x[:, embedding_idx, :num_channels_curr, :] = data_tensor[:, hash_channel_order, end_idx-self.encode_token_samples : end_idx]

                    ### WAE ENCODER
                    # Forward pass in stacked batch through WAE encoder
                    # latent, _, _ = self.wae(x[:, :-1, :, :], reverse=False, alpha=self.classifier_alpha)   # 1 shifted just to be aligned with training style
                    latent, _, _ = self.wae(x, reverse=False, alpha=self.classifier_alpha) # No shift if not causal masking
                    files_latents[:, w, :] = torch.mean(latent, dim=1)

                # After file complete, pacmap_window/stride the file and save each file from batch seperately
                # Seperate directory for each win/stride combination
                # First pull off GPU and convert to numpy
                files_latents = files_latents.cpu().numpy()
                for i in range(len(self.inference_window_sec_list)):

                    win_sec_curr = self.inference_window_sec_list[i]
                    stride_sec_curr = self.inference_stride_sec_list[i]
                    sec_in_forward = num_samples_in_forward/self.FS

                    if (win_sec_curr < sec_in_forward) or (stride_sec_curr < sec_in_forward):
                        raise Exception("Window or stride is too small compared to input sequence to encoder")
                    
                    num_latents_in_win = win_sec_curr / sec_in_forward
                    assert (num_latents_in_win % 1) == 0
                    num_latents_in_win = int(num_latents_in_win)

                    num_latents_in_stride = stride_sec_curr / sec_in_forward
                    assert (num_latents_in_stride % 1) == 0 
                    num_latents_in_stride = int(num_latents_in_stride)

                    # May not go in evenly, that is ok
                    num_strides_in_file = int((files_latents.shape[1] - num_latents_in_win) / num_latents_in_stride) 
                    windowed_file_latent = np.zeros([data_tensor.shape[0], num_strides_in_file, self.latent_dim])
                    for s in range(num_strides_in_file):
                        windowed_file_latent[:, s, :] = np.mean(files_latents[:, s*num_latents_in_stride: s*num_latents_in_stride + num_latents_in_win], axis=1)

                    # Save each windowed latent in a pickle for each file
                    for b in range(data_tensor.shape[0]):
                        filename_curr = file_name[b]
                        save_dir = f"{self.model_dir}/latent_files/Epoch{self.epoch}/{win_sec_curr}SecondWindow_{stride_sec_curr}SecondStride/{dataset_string}"
                        if not os.path.exists(save_dir): os.makedirs(save_dir)
                        output_obj = open(f"{save_dir}/{filename_curr}_latent_{win_sec_curr}secWindow_{stride_sec_curr}secStride.pkl", 'wb')
                        pickle.dump(windowed_file_latent[b, :, :], output_obj)
                        output_obj.close()

    def _run_train_epoch( ### RANDOM SUBSET OF FILES ###    
        self, 
        dataloader_curr, 
        dataset_string,
        attention_dropout,
        num_dataloader_workers,
        max_batch_size,
        padded_channels,
        val_finetune,
        val_unseen,
        realtime_latent_printing,
        realtime_printing_interval,
        **kwargs):

        '''
        This function is for training the model. 
        It will pull: random pats/random files/random portions of file

        IMPORTANT: Takes in a *DataLoader*, not a *Dataset*
        '''

        print(f"[GPU{self.gpu_id}] encode_token_samples: {self.encode_token_samples}")
        
        iter_curr = 0
        total_iters = len(dataloader_curr)
        for x, file_name, file_class_label, hash_channel_order, hash_pat_embedding in dataloader_curr: 

            # Put the data and labels on GPU
            x = x.to(self.gpu_id)
            file_class_label = file_class_label.to(self.gpu_id)
            hash_pat_embedding = hash_pat_embedding.to(self.gpu_id)
        
            # For Training: Update the Regulizer multiplier (BETA), and Learning Rate according for Heads Models and Core Model
            self.reg_weight, self.curr_LR_core, self.curr_LR_cls, self.sparse_weight, self.classifier_weight, self.classifier_alpha = utils_functions.LR_and_weight_schedules(
                epoch=self.epoch, iter_curr=iter_curr, iters_per_epoch=total_iters, **self.kwargs)
            
            if (not val_finetune) & (not val_unseen): 
                self.opt_wae.param_groups[0]['lr'] = self.curr_LR_core
                self.opt_cls.param_groups[0]['lr'] = self.curr_LR_cls
            
            else: # For validation, do not consider classifier
                self.classifier_alpha = 0
                self.opt_cls.param_groups[0]['lr'] = 0

            # Check for NaNs
            if torch.isnan(x).any(): raise Exception(f"ERROR: found nans in one of these files: {file_name}")

            ### WAE ENCODER: 1-shifted
            # latent, class_probs_mean_of_latent, attW = self.wae(x[:, :-1, :, :], reverse=False, alpha=self.classifier_alpha)
            latent, prior, class_probs_mean_of_latent, attW = self.wae(x, reverse=False, alpha=self.classifier_alpha) # No shift if not causal masking
            
            ### WAE DECODER: 1-shifted & Transformer Encoder Concat Shifted (Need to prime first embedding with past context)
            x_hat = self.wae(latent, reverse=True, hash_pat_embedding=hash_pat_embedding)  

            # Pseudobatch the latent for regularization loss
            latent_batched = latent.reshape(latent.shape[0]* latent.shape[1], -1)
            prior_batched = prior.reshape(latent.shape[0]* latent.shape[1], -1)

            # # Sample from the recent observed & prior (otherwise batch would be ripple in ocean - without some past info batch cannot capure meaningful shape of latent)
            observed_samples, prior_samples = self._sample_running_window(latent_batched, prior_batched, **kwargs) # Will require grad now that blended in latent
               
            # LOSSES
            recon_loss = loss_functions.recon_loss_function(
                x=x, # No shift if not causal masking
                x_hat=x_hat,
                recon_weight=self.recon_weight)

            reg_loss = loss_functions.sinkhorn_loss( # Just on this local batch compared to Prior
                observed = observed_samples, # From accumulated_z
                prior = prior_samples, # From Prior
                weight = self.reg_weight,
                sinkhorn_blur = self.sinkhorn_blur,
                wasserstein_order = self.wasserstein_order)

            adversarial_loss = loss_functions.adversarial_loss_function(
                probs=class_probs_mean_of_latent, 
                labels=file_class_label,
                classifier_weight=self.classifier_weight)

            loss = recon_loss + reg_loss + adversarial_loss 

            # Not currently used, but is nice to see
            sparse_loss = loss_functions.sparse_l1_reg(
                z=latent, 
                sparse_weight=self.sparse_weight, 
                **kwargs)

            # Do not backprop for pure validation (i.e. val unseen), but do for training & finetuning
            if not val_unseen: 
                self._zero_all_grads()
                loss.backward()    
                self.opt_wae.step()
                self.opt_cls.step()
                self._update_reg_window(latent_batched, prior_batched) # Update the buffers & detach() 

            # Realtime terminal info and WandB 
            if (iter_curr%self.recent_display_iters==0):
                if val_finetune: state_str = "VAL FINETUNE"
                elif val_unseen: state_str = "VAL UNSEEN"
                else: state_str = "TRAIN"
                now_str = datetime.datetime.now().strftime("%I:%M%p-%B/%d/%Y")
                if (self.gpu_id == 1):
                    sys.stdout.write(
                        f"\r{now_str} [GPU{str(self.gpu_id)}]: {state_str}, EPOCH {self.epoch}, Iter [BatchSize: {x.shape[0]}] {iter_curr}/{total_iters}, " + 
                        f"MeanLoss: {round(loss.detach().item(), 2)}                 ")
                    sys.stdout.flush() 

                # Log to WandB
                wandb.define_metric('Steps')
                wandb.define_metric("*", step_metric="Steps")
                train_step = self.epoch * int(total_iters) + iter_curr
                if (not val_finetune) & (not val_unseen):
                    metrics = dict(
                        train_attention_dropout=attention_dropout,
                        train_loss=loss,
                        train_recon_loss=recon_loss, 
                        train_reg_loss=reg_loss, 
                        train_adversarial_loss=adversarial_loss,
                        train_sparse_loss=sparse_loss,
                        train_LR_wae=self.opt_wae.param_groups[0]['lr'], 
                        train_LR_classifier=self.opt_cls.param_groups[0]['lr'], 
                        train_reg_Beta=self.reg_weight, 
                        train_sinkhorn_blur=self.sinkhorn_blur,
                        train_running_reg_passes=self.running_reg_passes,
                        train_ReconWeight=self.recon_weight,
                        train_AdversarialAlpha=self.classifier_alpha,
                        train_Sparse_weight=self.sparse_weight,
                        train_epoch=self.epoch)

                elif val_finetune:
                    metrics = dict(
                        val_finetune_attention_dropout=attention_dropout,
                        val_finetune_loss=loss, 
                        val_finetune_recon_loss=recon_loss, 
                        val_finetune_reg_loss=reg_loss, 
                        val_finetune_sparse_loss=sparse_loss,
                        val_finetune_LR_wae=self.opt_wae.param_groups[0]['lr'], 
                        val_finetune_LR_classifier=self.opt_cls.param_groups[0]['lr'], 
                        val_finetune_reg_Beta=self.reg_weight, 
                        val_finetune_sinkhorn_blur=self.sinkhorn_blur,
                        val_finetune_ReconWeight=self.recon_weight,
                        val_finetune_AdversarialAlpha=self.classifier_alpha,
                        val_finetune_Sparse_weight=self.sparse_weight,
                        val_finetune_epoch=self.epoch)

                elif val_unseen:
                    metrics = dict(
                        val_unseen_attention_dropout=attention_dropout,
                        val_unseen_loss=loss, 
                        val_unseen_recon_loss=recon_loss, 
                        val_unseen_reg_loss=reg_loss, 
                        val_unseen_sparse_loss=sparse_loss,
                        val_unseen_LR_wae=self.opt_wae.param_groups[0]['lr'], 
                        val_unseen_LR_classifier=self.opt_cls.param_groups[0]['lr'], 
                        val_unseen_reg_Beta=self.reg_weight, 
                        val_unseen_sinkhorn_blur=self.sinkhorn_blur,
                        val_unseen_ReconWeight=self.recon_weight,
                        val_unseen_AdversarialAlpha=self.classifier_alpha,
                        val_unseen_Sparse_weight=self.sparse_weight,
                        val_unseen_epoch=self.epoch)

            wandb.log({**metrics, 'Steps': train_step})

            # Realtime latent visualizations
            if realtime_latent_printing & ((iter_curr + 1) % realtime_printing_interval == 0):
                    if self.gpu_id == 0:
                        if torch.isnan(loss).any():
                            print("WARNING: Loss is nan, no plots can be made")
                        else:
                            utils_functions.print_latent_realtime(
                                latent = observed_samples.cpu().detach().numpy(), 
                                prior = prior_samples.cpu().detach().numpy(),
                                savedir = self.model_dir + f"/realtime_plots/{dataset_string}/realtime_latents",
                                epoch = self.epoch,
                                iter_curr = iter_curr,
                                file_name = file_name,
                                **kwargs)
                            utils_functions.print_recon_realtime(
                                # x=x[:, 1 + self.num_encode_concat_transformer_tokens:, :, :], 
                                x=x,
                                x_hat=x_hat, 
                                savedir = self.model_dir + f"/realtime_plots/{dataset_string}/realtime_recon",
                                epoch = self.epoch,
                                iter_curr = iter_curr,
                                file_name = file_name,
                                **kwargs)
                            utils_functions.print_attention_realtime(
                                epoch = self.epoch, 
                                iter_curr = iter_curr,
                                pat_idxs = file_class_label, 
                                scores_byLayer_meanHeads = attW, 
                                savedir = self.model_dir + f"/realtime_plots/{dataset_string}/realtime_attention", 
                                **kwargs)
                            utils_functions.print_prior_params_realtime(
                                epoch = self.epoch, 
                                iter_curr = iter_curr,
                                shapes = (self.wae.module.prior.k1, self.wae.module.prior.k2),
                                scales = (self.wae.module.prior.theta1, self.wae.module.prior.theta2),
                                alphas = self.wae.module.prior.alpha,
                                savedir = self.model_dir + f"/realtime_plots/{dataset_string}/realtime_priorparams", 
                                **kwargs)

                            # NOTE: for finetuning, will still be guess the training patients, and TODO: idxs in dataset are wrong for class labels anyway
                            if (not val_unseen) & (not val_finetune): # Will not have accumulated for val_unseen
                                # utils_functions.print_confusion_realtime(
                                #     class_probs = self.accumulated_class_probs,
                                #     class_labels = self.accumulated_labels,
                                #     savedir = self.model_dir + f"/realtime_plots/{dataset_string}/realtime_confusion",
                                #     epoch = self.epoch,
                                #     iter_curr = iter_curr,
                                #     **kwargs)

                                utils_functions.print_classprobs_realtime(
                                    class_probs = class_probs_mean_of_latent,
                                    class_labels = file_class_label,
                                    savedir = self.model_dir + f"/realtime_plots/{dataset_string}/realtime_classprobs",
                                    epoch = self.epoch,
                                    iter_curr = iter_curr,
                                    file_name = file_name,
                                    **kwargs)

            # Advance the iteration counter (one iter per complete patient loop - i.e. one backward pass)
            iter_curr = iter_curr + 1

        # Plot the full running latents
        utils_functions.plot_observed_latents(
            gpu_id = self.gpu_id,
            prior = self.accumulated_prior.cpu().numpy(),
            observed = self.accumulated_z.cpu().numpy(),
            savedir = self.model_dir + f"/realtime_plots/{dataset_string}/observed_latents",
            epoch = self.epoch,
            **kwargs)
        
        print(f"[GPU{str(self.gpu_id)}] at end of epoch")
        # barrier()
        # gc.collect()
        # torch.cuda.empty_cache()
            
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