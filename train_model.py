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
# from torch.utils.data._utils import shared_memory_cleanup

# Local Imports
from utilities import latent_plotting
from utilities import utils_functions
from utilities import loss_functions
from data import SEEG_Tornado_Dataset
from models.WAE import WAE

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
            **kwargs)

    else: # If no val, just make a train dataset
        train_pats_dirs = all_pats_dirs
        train_pats_list = all_pats_list

        # Dummy
        valfinetune_dataset = []
        valunseen_dataset = []

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
        **kwargs)
         
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

    return train_dataset, valfinetune_dataset, valunseen_dataset, wae, opt_wae, opt_cls

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
    finetune_inference: bool, 
    PaCMAP_model_to_infer = [],
    wae_state_dict_prev_path = [],
    wae_opt_state_dict_prev_path = [],
    cls_opt_state_dict_prev_path = [],
    barycenter_path = [],
    running_latent_path = [],
    epochs_to_train: int = -1,

    **kwargs):

    '''
    Highest level loop to run train epochs, validate, pacmap... etc. 

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
    train_dataset, valfinetune_dataset, valunseen_dataset, wae, opt_wae, opt_cls = load_train_objs(gpu_id=gpu_id, **kwargs) 
    
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

        # Load barycenter and running latents
        with open(barycenter_path, "rb") as f: barycenter = pickle.load(f)
        with open(running_latent_path, "rb") as f: accumulated_z = pickle.load(f)
        print(f"[GPU{gpu_id}] Barycenter and Running Latents loaded from checkpoints")
    
    else:
        barycenter = []
        accumulated_z = []

    # Create the training object
    trainer = Trainer(
        world_size=world_size,
        gpu_id=gpu_id, 
        wae=wae, 
        opt_wae=opt_wae,
        opt_cls=opt_cls,
        start_epoch=start_epoch,
        train_dataset=train_dataset, 
        valfinetune_dataset=valfinetune_dataset,
        valunseen_dataset=valunseen_dataset,
        PaCMAP_model_to_infer=PaCMAP_model_to_infer,
        wandb_run=wandb_run,
        finetune_inference=finetune_inference,
        barycenter=barycenter,
        accumulated_z=accumulated_z,
        **kwargs)
    
    # Run through all epochs
    for epoch in range(start_epoch, epochs_to_train):
        trainer.epoch = epoch

        # Full INFERENCE on all data
        if (epoch > 0) & ((trainer.epoch + 1) % trainer.inference_every == 0):

            # Save pre-finetune model/opt weights
            if finetune_inference:
                wae_dict = trainer.wae.module.state_dict()
                wae_opt_dict = trainer.opt_wae.state_dict()
                cls_opt_dict = trainer.opt_cls.state_dict()

                # FINETUNE on beginning of validation patients (currently only one epoch)
                # Set to train and change LR to validate settings
                trainer._set_to_train()
                trainer.opt_wae.param_groups[0]['lr'] = LR_val_wae
                trainer.opt_cls.param_groups[0]['lr'] = LR_val_cls
                trainer._run_epoch(
                    dataset_curr = trainer.valfinetune_dataset, 
                    dataset_string = "valfinetune",
                    all_files_latent_only = False, # this will run every file for every patient instead of subsampling (changes how dataloaders are made)
                    val_finetune = True,
                    val_unseen = False,
                    num_rand_hashes = val_num_rand_hashes,
                    **kwargs)

                dataset_list = [trainer.train_dataset, trainer.valfinetune_dataset, trainer.valunseen_dataset]
                dataset_strs = ["train", "valfinetune", "valunseen"]

            # No finetune, only run data for Train dataset
            else:   
                dataset_list = [trainer.train_dataset]
                dataset_strs = ["train"]
            
            # INFERENCE on all selected datasets
            trainer._set_to_eval()
            with torch.no_grad():
                for d in range(0, len(dataset_list)):
                    trainer._run_epoch(
                        dataset_curr = dataset_list[d], 
                        dataset_string = dataset_strs[d],
                        all_files_latent_only = True, # this will run every file for every patient instead of subsampling (changes how dataloaders are made)
                        val_finetune = False,
                        val_unseen = False,
                        num_rand_hashes = -1,
                        **kwargs)

            # # After inferenece, run the PACMAP
            # # Only initiate for one GPU process - but calculations are done on CPU
            # if (gpu_id == 0): 
            #     # New pacmap run for every win/stride combination
            #     for i in range(len(trainer.inference_window_sec_list)):
            #         utils_functions.run_pacmap(
            #             dataset_strs=dataset_strs, 
            #             epoch=epoch, 
            #             win_sec=trainer.inference_window_sec_list[i], 
            #             stride_sec=trainer.inference_stride_sec_list[i], 
            #             latent_subdir=f"/latent_files/Epoch{epoch}", 
            #             **kwargs)

            # Restore model/opt weights to pre-finetune
            if finetune_inference:
                trainer.wae.module.load_state_dict(wae_dict)
                trainer.opt_wae.load_state_dict(wae_opt_dict)
                trainer.opt_cls.load_state_dict(cls_opt_dict)

            print(f"GPU{str(trainer.gpu_id)} at post inference barrier")
            barrier()
        
        # TRAIN
        trainer._set_to_train()
        trainer._run_epoch(
            dataset_curr = trainer.train_dataset, 
            dataset_string = "train",
            all_files_latent_only = False, # this will run every file for every patient instead of subsampling (changes how dataloaders are made)
            val_finetune = False,
            val_unseen = False,
            **kwargs)
        
        # CHECKPOINT
        # After every train epoch, optionally delete old checkpoints
        print(f"GPU{str(trainer.gpu_id)} at pre checkpoint save barrier")
        barrier()
        if trainer.gpu_id == 0: trainer._save_checkpoint(trainer.epoch, **kwargs)

        # VALIDATE 
        # (skip if it's a pacmap epoch because it will already have been done)
        if ((trainer.epoch + 1) % trainer.val_every == 0) & ((trainer.epoch + 1) % trainer.inference_every != 0):
            # Save pre-finetune model/opt weights
            wae_dict = trainer.wae.module.state_dict()
            wae_opt_dict = trainer.opt_wae.state_dict()
            cls_opt_dict = trainer.opt_cls.state_dict()

            # FINETUNE on beginning of validation patients (currently only one epoch)
            # Set to train and change LR to validate settings
            trainer._set_to_train()
            trainer.opt_wae.param_groups[0]['lr'] = LR_val_wae
            trainer.opt_cls.param_groups[0]['lr'] = LR_val_cls
            trainer._run_epoch(
                dataset_curr = trainer.valfinetune_dataset, 
                dataset_string = "valfinetune",
                all_files_latent_only = False, # this will run every file for every patient instead of subsampling (changes how dataloaders are made)
                val_finetune = True,
                val_unseen = False,
                **kwargs)

            # Inference on UNSEEN portion of validation patients 
            trainer._set_to_eval()
            with torch.no_grad():
                trainer._run_epoch(
                    dataset_curr = trainer.valunseen_dataset, 
                    dataset_string = "valunseen",
                    all_files_latent_only = False, # this will run every file for every patient instead of subsampling (changes how dataloaders are made)
                    val_finetune = False,
                    val_unseen = True,
                    **kwargs)

            # Restore model/opt weights to pre-finetune
            trainer.wae.module.load_state_dict(wae_dict)
            trainer.opt_wae.load_state_dict(wae_opt_dict)
            trainer.opt_cls.load_state_dict(cls_opt_dict)

    # Kill the process after training loop completes
    print(f"[GPU{gpu_id}]: End of train loop, killing subprocess")
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
        opt_wae: torch.optim.Optimizer,
        opt_cls: torch.optim.Optimizer,
        wandb_run,
        model_dir: str,
        val_every: int,
        inference_every: int,
        finetune_inference: bool,
        latent_dim: int,
        autoencode_samples: int,
        num_samples: int,
        transformer_seq_length: int,
        num_encode_concat_transformer_tokens: int,
        transformer_start_pos: int,
        FS: int,
        intrapatient_dataset_style: list,
        # transformer_weight: float,
        recon_weight: float,
        atd_file: str,
        PaCMAP_model_to_infer,
        inference_window_sec_list: list,
        inference_stride_sec_list: list,
        recent_display_iters: int,
        running_reg_passes: int,
        classifier_num_pats: int,
        sinkhorn_blur: int,
        optimizer_forward_passes: int,
        barycenter,
        accumulated_z,
        **kwargs
    ) -> None:
        self.world_size = world_size
        self.gpu_id = gpu_id
        self.wae = wae
        self.start_epoch = start_epoch
        self.train_dataset = train_dataset
        self.valfinetune_dataset = valfinetune_dataset
        self.valunseen_dataset = valunseen_dataset
        self.opt_wae = opt_wae
        self.opt_cls = opt_cls
        self.model_dir = model_dir
        self.val_every = val_every
        self.inference_every = inference_every
        self.finetune_inference = finetune_inference
        self.latent_dim = latent_dim
        self.autoencode_samples = autoencode_samples
        self.num_samples = num_samples
        self.transformer_seq_length = transformer_seq_length
        self.num_encode_concat_transformer_tokens = num_encode_concat_transformer_tokens
        self.transformer_start_pos = transformer_start_pos
        self.FS = FS
        self.intrapatient_dataset_style = intrapatient_dataset_style
        self.curr_LR_core = -1
        self.recon_weight = recon_weight
        self.atd_file = atd_file
        self.PaCMAP_model_to_infer = PaCMAP_model_to_infer
        self.inference_window_sec_list = inference_window_sec_list
        self.inference_stride_sec_list = inference_stride_sec_list
        self.recent_display_iters = recent_display_iters
        self.running_reg_passes = running_reg_passes
        self.classifier_num_pats = classifier_num_pats
        self.sinkhorn_blur = sinkhorn_blur
        self.optimizer_forward_passes = optimizer_forward_passes
        self.barycenter = barycenter
        self.accumulated_z = accumulated_z
        self.wandb_run = wandb_run
        self.kwargs = kwargs

        assert len(self.inference_window_sec_list) == len(self.inference_stride_sec_list)

        self.reg_weight = -1 # dummy variable, only needed when debugging and training is skipped

        # Number of iterations per file
        self.num_windows = int((self.num_samples - self.transformer_seq_length * self.autoencode_samples - self.autoencode_samples)/self.autoencode_samples) - 2

        # Set up wae & transformer with DDP
        self.wae = DDP(wae, device_ids=[gpu_id])   # find_unused_parameters=True
                    
        # Running Regulizer window for latent data
        if self.barycenter == []:
            self.accumulated_z = Gamma(kwargs['gamma_shape'], 1/kwargs['gamma_scale']).sample((self.running_reg_passes, self.latent_dim)).to(self.gpu_id)
            self.accumulated_prior = Gamma(kwargs['gamma_shape'], 1/kwargs['gamma_scale']).sample((self.running_reg_passes, self.latent_dim)).to(self.gpu_id)
        
        else:
            self.accumulated_prior = self.barycenter
            if self.accumulated_z == []: raise Exception("Error, accumulated_z is [], should be filled if barycenter was passed in")

        # Running class labels for classifier
        self.accumulated_labels = torch.zeros(self.running_reg_passes, dtype=torch.int64).to(self.gpu_id)
        self.accumulated_class_probs = torch.zeros(self.running_reg_passes, self.classifier_num_pats).to(self.gpu_id)
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

        # Save Barycenter 
        barycenter_path = check_core_dir + "/checkpoint_epoch" + str(epoch) + "_barycenter.pkl"
        output_obj = open(barycenter_path, 'wb')
        pickle.dump(self.barycenter, output_obj)
        output_obj.close()
        print("Saved barycenter")

        # Save Running Latents 
        latents_path = check_core_dir + "/checkpoint_epoch" +str(epoch) + "_running_latents.pkl"
        output_obj = open(latents_path, 'wb')
        pickle.dump(self.accumulated_z, output_obj)
        output_obj.close()
        print("Saved running latents")


        print(f"Epoch {epoch} | Training checkpoint saved at {check_epoch_dir}")

        if delete_old_checkpoints:
            utils_functions.delete_old_checkpoints(dir = base_checkpoint_dir, curr_epoch = epoch, **kwargs)
            print("Deleted old checkpoints, except epochs with PaCMAP/HDBSCAN models")

    def _sample_barycenter(self, z, **kwargs):
        # Randomly select indices from the barycenter samples
        indices = torch.randint(0, self.barycenter.shape[0], (z.shape[0],))
    
        # Sample from the barycenter using the selected indices
        return self.barycenter[indices]

    def _update_barycenter(self, gamma_shape, gamma_scale, num_barycenter_iters, sinkhorn_blur, **kwargs):

        # Sample from gamma
        gamma_samples = torch.distributions.Gamma(gamma_shape, 1/gamma_scale).sample((self.accumulated_z.shape[0], self.accumulated_z.shape[1])).to(self.accumulated_z)

        # Compute pairwise cost matrix (Euclidean distance)
        cost_matrix = torch.cdist(self.accumulated_z, gamma_samples, p=2)
        cost_matrix = cost_matrix / cost_matrix.max() 
        
        # Compute optimal transport plan using Sinkhorn
        transport_plan = loss_functions.sinkhorn(cost_matrix, blur=sinkhorn_blur, n_iter=num_barycenter_iters)
        
        # Update barycenter as the weighted average of gamma samples
        self.barycenter = torch.matmul(transport_plan.T, gamma_samples)  # Shape: (16384, 1024)

        print(f"[GPU{self.gpu_id}] Barycenter updated: {self.barycenter.shape}")
        
    def _update_reg_window(
        self,
        mean_latent, 
        new_prior_samples,
        file_class_label,
        class_probs_mean_of_latent
        ):

        num_new_updates = mean_latent.shape[0]
        if (self.next_update_index + num_new_updates) < self.running_reg_passes:
            self.accumulated_z[self.next_update_index: self.next_update_index + num_new_updates, :] = mean_latent
            self.accumulated_prior[self.next_update_index: self.next_update_index + num_new_updates, :] = new_prior_samples
            self.accumulated_labels[self.next_update_index: self.next_update_index + num_new_updates] = file_class_label
            self.accumulated_class_probs[self.next_update_index: self.next_update_index + num_new_updates, :] = class_probs_mean_of_latent
            
            self.next_update_index = self.next_update_index + num_new_updates

        # Rollover
        else:
            residual_num = (self.next_update_index + num_new_updates) % self.running_reg_passes
            end_num = num_new_updates - residual_num
            self.accumulated_z[self.next_update_index: self.next_update_index + end_num, :] = mean_latent[:end_num, :]
            self.accumulated_z[0: residual_num, :] = mean_latent[end_num:, :]

            self.accumulated_prior[self.next_update_index: self.next_update_index + end_num, :] = new_prior_samples[:end_num, :]
            self.accumulated_prior[0: residual_num, :] = new_prior_samples[end_num:, :]

            self.accumulated_labels[self.next_update_index: self.next_update_index + end_num] = file_class_label[:end_num]
            self.accumulated_labels[0: residual_num] = file_class_label[end_num:]

            self.accumulated_class_probs[self.next_update_index: self.next_update_index + end_num, :] = class_probs_mean_of_latent[:end_num, :]
            self.accumulated_class_probs[0: residual_num, :] = class_probs_mean_of_latent[end_num:, :]

            self.next_update_index = residual_num

    def _run_epoch(
        self, 
        dataset_curr, 
        dataset_string,
        attention_dropout,
        all_files_latent_only,
        num_dataloader_workers,
        max_batch_size,
        inference_batch_mult,
        padded_channels,
        val_finetune,
        val_unseen,
        realtime_latent_printing,
        realtime_printing_interval,
        **kwargs):

        print(f"[GPU{self.gpu_id}] Autoencode_samples: {self.autoencode_samples}")

        try:
            ### ALL/FULL FILES - LATENT ONLY - FULL TRANSFORMER CONTEXT ### 
            # This setting is used for inference
            # If wanting all files from every patient, need to run patients serially
            if all_files_latent_only: 

                print("WARNING: Setting alpha = 1")
                self.classifier_alpha = 1

                # Check for erroneous configs
                if val_finetune or val_unseen:
                    raise Exception("ERROR: innapropriate config: if running all files then val_finetune/val_unseen must all be False")

                # Go through every subject in this dataset
                for pat_idx in range(0,len(dataset_curr.pat_ids)):
                    dataset_curr.set_pat_curr(pat_idx)
                    dataloader_curr =  utils_functions.prepare_dataloader(dataset_curr, batch_size=max_batch_size * inference_batch_mult, num_workers=num_dataloader_workers)

                    # Go through every file in dataset
                    file_count = 0
                    for data_tensor, file_name, file_class_label in dataloader_curr: # Hash done outside data.py for single pat inference

                        file_count = file_count + len(file_name)

                        num_channels_curr = data_tensor.shape[1]

                        # Create the sequential latent sequence array for the file 
                        num_samples_in_forward = self.transformer_seq_length * self.autoencode_samples
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
                                modifier=rand_modifer)
                            
                            # Collect sequential embeddings for transformer by running sequential raw data windows through BSE N times 
                            x = torch.zeros(data_tensor.shape[0], self.transformer_seq_length, padded_channels, self.autoencode_samples).to(self.gpu_id)
                            start_idx = w * num_samples_in_forward
                            for embedding_idx in range(0, self.transformer_seq_length):
                                # Pull out data for this window - NOTE: no hashing
                                end_idx = start_idx + self.autoencode_samples * embedding_idx + self.autoencode_samples 
                                x[:, embedding_idx, :num_channels_curr, :] = data_tensor[:, hash_channel_order, end_idx-self.autoencode_samples : end_idx]

                            ### WAE ENCODER
                            # Forward pass in stacked batch through WAE encoder
                            latent, _ = self.wae(x[:, :-1, :, :], reverse=False, alpha=self.classifier_alpha)   # 1 shifted just to be aligned with training style
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

            ### RANDOM SUBSET OF FILES ###
            # This setting is used for regular training 
            else: 
                dataset_curr.set_pat_curr(-1) # -1 sets to random generation
                num_pats_curr = dataset_curr.get_pat_count()
                dataloader_curr = utils_functions.prepare_dataloader(dataset_curr, batch_size=None, num_workers=num_dataloader_workers)
                dataloader_curr.sampler.set_epoch(self.epoch) # Ensure new file order is pulled


                # Update the Barycenter (average between empiric latent distriubution and given prior (e.g. Gamma) for this epoch
                self._update_barycenter(**kwargs)

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

                    # Check for NaNs
                    if torch.isnan(x).any(): raise Exception(f"ERROR: found nans in one of these files: {file_name}")

                    ### WAE ENCODER: 1-shifted
                    latent, class_probs_mean_of_latent = self.wae(x[:, :-1, :, :], reverse=False, alpha=self.classifier_alpha)
                    
                    ### WAE DECODER: 1-shifted & Transformer Encoder Concat Shifted (Need to prime first embedding with past context)
                    x_hat = self.wae(latent, reverse=True, hash_pat_embedding=hash_pat_embedding)  

                    # Average the latents to get single embedding per encoder input
                    mean_latent = torch.mean(latent, dim=1)

                    # # Check if new run, if so will need to fill buffers
                    # if not self.buffer_filled: # NOTE: This logic breaks if buffer cannot be filled within one epoch
                    #     if (self.next_update_index + mean_latent.shape[0]) >= self.running_reg_passes: 
                    #         self.buffer_filled = True
                    #         print("Buffers filled, proceeding with backprop from here on")
                    #     else:
                    #         sys.stdout.write(f"\r[Backprop Skipped] Filling running window Regulizer and Classifier buffers {self.next_update_index}/{self.running_reg_passes}")
                    #         sys.stdout.flush() 
                    #         self.accumulated_z = self.accumulated_z.detach() # No backprop on these data
                    #         self.accumulated_class_probs = self.accumulated_class_probs.detach() # No backprop on these data

                    # Sample from the Barycenter
                    new_prior_samples = self._sample_barycenter(z=mean_latent, **kwargs)  

                    # Update the buffers
                    self._update_reg_window(mean_latent, new_prior_samples, file_class_label, class_probs_mean_of_latent)

                    # LOSSES
                    recon_loss = loss_functions.recon_loss_function(
                        x=x[:, 1 + self.num_encode_concat_transformer_tokens :, :, :], # opposite 1-shifted & Transformer Encoder Concat Shifted 
                        x_hat=x_hat,
                        recon_weight=self.recon_weight)

                    reg_loss = loss_functions.sinkhorn_loss( # Just on this local batch compared to Barycenter
                        z_real = mean_latent, 
                        z_fake = new_prior_samples, # From Barycenter
                        weight = self.reg_weight,
                        sinkhorn_blur = self.sinkhorn_blur)

                    adversarial_loss = loss_functions.adversarial_loss_function(
                        probs=class_probs_mean_of_latent, 
                        labels=file_class_label,
                        classifier_weight=self.classifier_weight)

                    # Not currently used
                    sparse_loss = loss_functions.sparse_l1_reg(
                        z=latent, 
                        sparse_weight=self.sparse_weight, 
                        **kwargs)

                    # AFTER EACH FORWARD PASS
                    self._zero_all_grads()
                    loss = recon_loss + reg_loss + adversarial_loss 
                    loss.backward()         
                    self.accumulated_z = self.accumulated_z.detach() # Detach to allow next backpass
                    self.accumulated_class_probs = self.accumulated_class_probs.detach() # Detach to allow next backpass

                    # Step optimizer at desired number of froward passes
                    if (iter_curr%self.optimizer_forward_passes==0):
                        self.opt_wae.step()
                        self.opt_cls.step()

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
                                train_AdversarialWeight=self.classifier_weight,
                                train_AdversarialAlpha=self.classifier_alpha,
                                train_Sparse_weight=self.sparse_weight,
                                train_epoch=self.epoch)

                        elif val_finetune:
                            metrics = dict(
                                val_finetune_attention_dropout=attention_dropout,
                                val_finetune_loss=loss, 
                                val_finetune_recon_loss=recon_loss, 
                                val_finetune_reg_loss=reg_loss, 
                                val_finetune_adversarial_loss=adversarial_loss,
                                val_finetune_sparse_loss=sparse_loss,
                                val_finetune_LR_wae=self.opt_wae.param_groups[0]['lr'], 
                                val_finetune_LR_classifier=self.opt_cls.param_groups[0]['lr'], 
                                val_finetune_reg_Beta=self.reg_weight, 
                                val_finetune_sinkhorn_blur=self.sinkhorn_blur,
                                val_finetune_ReconWeight=self.recon_weight,
                                val_finetune_AdversarialWeight=self.classifier_weight,
                                val_finetune_AdversarialAlpha=self.classifier_alpha,
                                val_finetune_Sparse_weight=self.sparse_weight,
                                val_finetune_epoch=self.epoch)

                        elif val_unseen:
                            metrics = dict(
                                val_unseen_attention_dropout=attention_dropout,
                                val_unseen_loss=loss, 
                                val_unseen_recon_loss=recon_loss, 
                                val_unseen_reg_loss=reg_loss, 
                                val_unseen_adversarial_loss=adversarial_loss,
                                val_unseen_sparse_loss=sparse_loss,
                                val_unseen_LR_wae=self.opt_wae.param_groups[0]['lr'], 
                                val_unseen_LR_classifier=self.opt_cls.param_groups[0]['lr'], 
                                val_unseen_reg_Beta=self.reg_weight, 
                                val_unseen_sinkhorn_blur=self.sinkhorn_blur,
                                val_unseen_ReconWeight=self.recon_weight,
                                val_unseen_AdversarialWeight=self.classifier_weight,
                                val_unseen_AdversarialAlpha=self.classifier_alpha,
                                val_unseen_Sparse_weight=self.sparse_weight,
                                val_unseen_epoch=self.epoch)

                    wandb.log({**metrics, 'Steps': train_step})

                    # Advance the iteration counter (one iter per complete patient loop - i.e. one backward pass)
                    iter_curr = iter_curr + 1

                    # Realtime latent visualizations
                    if realtime_latent_printing & ((iter_curr + 1) % realtime_printing_interval == 0):
                        if self.gpu_id == 0:
                            if torch.isnan(loss).any():
                                print("WARNING: Loss is nan, no plots can be made")
                            else:
                                utils_functions.print_latent_realtime(
                                    latent = self.accumulated_z.cpu().detach().numpy(), 
                                    prior = self.accumulated_prior.cpu().detach().numpy(),
                                    pat_labels = self.accumulated_labels.cpu().detach().numpy(),
                                    savedir = self.model_dir + f"/realtime_plots/{dataset_string}/realtime_latents",
                                    epoch = self.epoch,
                                    iter_curr = iter_curr,
                                    file_name = file_name,
                                    **kwargs)
                                utils_functions.print_recon_realtime(
                                    x=x[:, 1 + self.num_encode_concat_transformer_tokens:, :, :], 
                                    x_hat=x_hat, 
                                    savedir = self.model_dir + f"/realtime_plots/{dataset_string}/realtime_recon",
                                    epoch = self.epoch,
                                    iter_curr = iter_curr,
                                    file_name = file_name,
                                    **kwargs)
                                utils_functions.print_confusion_realtime(
                                    class_probs = self.accumulated_class_probs,
                                    class_labels = self.accumulated_labels,
                                    savedir = self.model_dir + f"/realtime_plots/{dataset_string}/realtime_confusion",
                                    epoch = self.epoch,
                                    iter_curr = iter_curr,
                                    **kwargs)
                                utils_functions.print_classprobs_realtime(
                                    class_probs = class_probs_mean_of_latent,
                                    class_labels = file_class_label,
                                    savedir = self.model_dir + f"/realtime_plots/{dataset_string}/realtime_classprobs",
                                    epoch = self.epoch,
                                    iter_curr = iter_curr,
                                    file_name = file_name,
                                    **kwargs)
    
        finally:
            print(f"[GPU{str(self.gpu_id)}] at 'finally' section after epoch")
            barrier()
            # for worker in dataloader_curr._workers: worker.terminate()
            del dataloader_curr
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