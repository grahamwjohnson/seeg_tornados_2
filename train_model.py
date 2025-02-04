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
    
    return train_dataset, valfinetune_dataset, valunseen_dataset, vae, opt_vae 

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
    train_runs_per_file: int,
    valfinetune_runs_per_file: int,
    valunseen_runs_per_file: int,
    train_num_rand_hashes: int,
    val_num_rand_hashes: int,
    LR_val_vae: float,
    finetune_pacmap: bool, 
    PaCMAP_model_to_infer = [],
    vae_state_dict_prev_path = [],
    vae_opt_state_dict_prev_path = [],
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
    train_dataset, valfinetune_dataset, valunseen_dataset, vae, opt_vae = load_train_objs(gpu_id=gpu_id, **kwargs) 
    
    # Load the model/opt states if not first epoch & if in training mode
    if (start_epoch > 0):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}

        # Load in VAE Core weights and opts
        vae_state_dict_prev = torch.load(vae_state_dict_prev_path, map_location=map_location)
        vae.load_state_dict(vae_state_dict_prev)
        vae_opt_state_dict_prev = torch.load(vae_opt_state_dict_prev_path, map_location=map_location)
        opt_vae.load_state_dict(vae_opt_state_dict_prev)

        print("Model and Opt weights loaded from checkpoints")

    # Create the training object
    trainer = Trainer(
        world_size=world_size,
        gpu_id=gpu_id, 
        vae=vae, 
        opt_vae=opt_vae,
        start_epoch=start_epoch,
        train_dataset=train_dataset, 
        valfinetune_dataset=valfinetune_dataset,
        valunseen_dataset=valunseen_dataset,
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

                # FINETUNE on beginning of validation patients (currently only one epoch)
                # Set to train and change LR to validate settings
                trainer._set_to_train()
                trainer.opt_vae.param_groups[0]['lr'] = LR_val_vae
                trainer._run_epoch(
                    dataset_curr = trainer.valfinetune_dataset, 
                    dataset_string = "valfinetune",
                    batchsize=trainer.wdecode_batch_size,
                    runs_per_file = valfinetune_runs_per_file, 
                    all_files_latent_only = False, # this will run every file for every patient instead of subsampling (changes how dataloaders are made)
                    val_finetune = True,
                    val_unseen = False,
                    backprop = True,
                    num_rand_hashes = val_num_rand_hashes,
                    **kwargs)

            # # INFERENCE on all datasets
            dataset_list = [trainer.train_dataset, trainer.valfinetune_dataset, trainer.valunseen_dataset]
            dataset_strs = ["train", "valfinetune", "valunseen"]
            trainer._set_to_eval()
            with torch.no_grad():
                for d in range(0, len(dataset_list)):
                    trainer._run_epoch(
                        dataset_curr = dataset_list[d], 
                        dataset_string = dataset_strs[d],
                        batchsize=trainer.onlylatent_batch_size,
                        runs_per_file = -1, 
                        all_files_latent_only = True, # this will run every file for every patient instead of subsampling (changes how dataloaders are made)
                        val_finetune = False,
                        val_unseen = False,
                        backprop = False,
                        num_rand_hashes = -1,
                        **kwargs)

            # After inferenece, run the PACMAP
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

            print(f"GPU{str(trainer.gpu_id)} at post PaCMAP barrier")
            barrier()
        
        # TRAIN
        trainer._set_to_train()
        trainer._run_epoch(
            dataset_curr = trainer.train_dataset, 
            dataset_string = "train",
            batchsize=trainer.wdecode_batch_size,
            runs_per_file = train_runs_per_file, 
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

            # FINETUNE on beginning of validation patients (currently only one epoch)
            # Set to train and change LR to validate settings
            trainer._set_to_train()
            trainer.opt_vae.param_groups[0]['lr'] = LR_val_vae
            trainer._run_epoch(
                dataset_curr = trainer.valfinetune_dataset, 
                dataset_string = "valfinetune",
                batchsize=trainer.wdecode_batch_size,
                runs_per_file = valfinetune_runs_per_file, 
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
                    runs_per_file = valunseen_runs_per_file, 
                    all_files_latent_only = False, # this will run every file for every patient instead of subsampling (changes how dataloaders are made)
                    val_finetune = False,
                    val_unseen = True,
                    backprop = False,
                    num_rand_hashes = val_num_rand_hashes,
                    **kwargs)

            # Restore model/opt weights to pre-finetune
            trainer.vae.module.load_state_dict(vae_dict)
            trainer.opt_vae.load_state_dict(vae_opt_dict)

    # Kill the process after training loop completes
    print(f"[GPU{gpu_id}]: End of train loop, killing subprocess")
    wandb.finish()
    destroy_process_group() 

class Trainer:
    def __init__(
        self,
        world_size: int,
        gpu_id: int,
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
        val_every: int,
        pacmap_every: int,
        finetune_pacmap: bool,
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
        pre_PaCMAP_window_sec_list: list,
        pre_PaCMAP_stride_sec_list: list,
        recent_display_iters: int,
        **kwargs
    ) -> None:
        self.world_size = world_size
        self.gpu_id = gpu_id
        self.vae = vae
        self.start_epoch = start_epoch
        self.train_dataset = train_dataset
        self.valfinetune_dataset = valfinetune_dataset
        self.valunseen_dataset = valunseen_dataset
        self.opt_vae = opt_vae
        self.wdecode_batch_size = wdecode_batch_size
        self.onlylatent_batch_size = onlylatent_batch_size
        self.model_dir = model_dir
        self.val_every = val_every
        self.pacmap_every = pacmap_every
        self.finetune_pacmap = finetune_pacmap
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
                    
        # Watch with WandB
        wandb.watch(self.vae)
        
    def _set_to_train(self):
        self.vae.train()

    def _set_to_eval(self):
        self.vae.eval()

    def _zero_all_grads(self):
        self.opt_vae.zero_grad()

    def _save_checkpoint(self, epoch, delete_old_checkpoints, **kwargs):
            
        print("CHECKPOINT SAVE")

        # Create new directory for this epoch
        base_checkpoint_dir = self.model_dir + f"/checkpoints"
        check_epoch_dir = base_checkpoint_dir + f"/Epoch_{str(epoch)}"

        print("Saving vae model weights")

        ### VAE CHECKPOINT 
        check_core_dir = check_epoch_dir + "/core_checkpoints"
        if not os.path.exists(check_core_dir): os.makedirs(check_core_dir)

        # Save model
        ckp = self.vae.module.state_dict()
        check_path = check_core_dir + "/checkpoint_epoch" +str(epoch) + "_vae.pt"
        torch.save(ckp, check_path)

        # Save optimizer
        opt_ckp = self.opt_vae.state_dict()
        opt_path = check_core_dir + "/checkpoint_epoch" +str(epoch) + "_vae_opt.pt"
        torch.save(opt_ckp, opt_path)

        print(f"Epoch {epoch} | Training checkpoint saved at {check_epoch_dir}")

        if delete_old_checkpoints:
            utils_functions.delete_old_checkpoints(dir = base_checkpoint_dir, curr_epoch = epoch)
            print("Deleted old checkpoints, except epochs with PaCMAP/HDBSCAN models")

    def _train_start_idxs(self, runs_per_file):

        last_possible_start_idx = self.num_samples - (self.transformer_seq_length + 1) * self.autoencode_samples 

        start_idxs = np.zeros(runs_per_file, dtype=int)
        for i in range(runs_per_file):
            np.random.seed(seed=None) 
            start_idxs[i] = np.random.randint(0, last_possible_start_idx+1)
        
        return start_idxs

    def _run_epoch(
        self, 
        dataset_curr, 
        dataset_string,
        batchsize,
        attention_dropout,
        runs_per_file, 
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

        ### ALL/FULL FILES - LATENT ONLY - FULL TRANSFORMER CONTEXT ### 
        # This setting is used for inference
        # If wanting all files from every patient, need to run patients serially
        if all_files_latent_only: 

            # Check for erroneous configs
            if backprop or val_finetune or val_unseen:
                raise Exception("ERROR: innapropriate config: if running all files then backprop/val_finetune/val_unseen must all be False")

            # Go through every subject in this dataset
            for pat_idx in range(0,len(dataset_curr.pat_ids)):
                dataset_curr.set_pat_curr(pat_idx)
                dataloader_curr =  utils_functions.prepare_dataloader(dataset_curr, batch_size=batchsize, num_workers=num_dataloader_workers_SEQUENTIAL)
                
                # Go through every file in dataset
                file_count = 0
                for data_tensor, file_name in dataloader_curr:

                    file_count = file_count + len(file_name)

                    # Create the sequential latent sequence array for the file 
                    num_samples_in_forward = self.transformer_seq_length * self.autoencode_samples
                    num_windows_in_file = data_tensor.shape[2] / num_samples_in_forward
                    assert (num_windows_in_file % 1) == 0
                    num_windows_in_file = int(num_windows_in_file)
                    num_samples_in_forward = int(num_samples_in_forward)

                    # Prep the output tensor and put on GPU
                    file_means = torch.zeros([data_tensor.shape[0], num_windows_in_file, self.latent_dim]).to(self.gpu_id)

                    # Put whole file on GPU
                    data_tensor = data_tensor.to(self.gpu_id)

                    for w in range(num_windows_in_file):
                        
                        # Print Status
                        print_interval = 100
                        if (self.gpu_id == 0) & (w % print_interval == 0):
                            sys.stdout.write(f"\r{dataset_string}: Pat {pat_idx}/{len(dataset_curr.pat_ids)-1}, File {file_count - len(file_name)}:{file_count}/{len(dataset_curr)/self.world_size - 1}  * GPUs (DDP), Intrafile Iter {w}/{num_windows_in_file}          ") 
                            sys.stdout.flush() 

                        # HASHING: Get the patient channel order and patid_embedding for this channel order
                        np.random.seed(seed=None) 
                        # rand_modifer = int(random.uniform(0, num_rand_hashes -1))
                        rand_modifer = 0 # For Inference
                        hash_pat_embedding, hash_channel_order = utils_functions.hash_to_vector(
                            input_string=dataset_curr.pat_ids[pat_idx], 
                            num_channels=data_tensor.shape[1], 
                            latent_dim=self.latent_dim, 
                            modifier=rand_modifer)
                        
                        # Collect sequential embeddings for transformer by running sequential raw data windows through BSE N times 
                        x = torch.zeros(data_tensor.shape[0], self.transformer_seq_length, data_tensor.shape[1], self.autoencode_samples).to(self.gpu_id)
                        start_idx = w * num_samples_in_forward
                        for embedding_idx in range(0, self.transformer_seq_length):
                            # Pull out data for this window - NOTE: no hashing
                            end_idx = start_idx + self.autoencode_samples * embedding_idx + self.autoencode_samples 
                            x[:, embedding_idx, :, :] = data_tensor[:, hash_channel_order, end_idx-self.autoencode_samples : end_idx]

                         ### VAE ENCODER
                        # Forward pass in stacked batch through VAE encoder
                        mean_batched, _, _ = self.vae(x, reverse=False)

                        # Split the batched dimension and stack into sequence dimension [batch, seq, latent_dims]
                        # NOTE: you lose the priming tokens needed by transformer
                        mean = torch.stack(torch.split(mean_batched, self.transformer_seq_length - self.num_encode_concat_transformer_tokens, dim=0), dim=0)
                        file_means[:, w, :] = torch.mean(mean, dim=1)
                        
                        # Split the batched dimension and stack into sequence dimension [batch, seq, latent_dims]
                        # mean_seq = torch.split(mean_batched, pseudobatch_onlylatent, dim=0)
                        # file_mean[:, w * pseudobatch_onlylatent:  w * pseudobatch_onlylatent + pseudobatch_onlylatent, :] = torch.stack(mean_seq, dim=0).cpu().numpy()

                    # After file complete, pacmap_window/stride the file and save each file from batch seperately
                    # Seperate directory for each win/stride combination
                    # First pull off GPU and convert to numpy
                    file_means = file_means.cpu().numpy()
                    for i in range(len(self.pre_PaCMAP_window_sec_list)):

                        win_sec_curr = self.pre_PaCMAP_window_sec_list[i]
                        stride_sec_curr = self.pre_PaCMAP_stride_sec_list[i]
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
                        num_strides_in_file = int((file_means.shape[1] - num_latents_in_win) / num_latents_in_stride) 
                        windowed_file_latent = np.zeros([data_tensor.shape[0], num_strides_in_file, self.latent_dim])
                        for s in range(num_strides_in_file):
                            windowed_file_latent[:, s, :] = np.mean(file_means[:, s*num_latents_in_stride: s*num_latents_in_stride + num_latents_in_win], axis=1)

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
        # Can run the patients in parallel with this setting
        else: 
            dataset_curr.set_pat_curr(-1) # -1 enables all pat mode

            # Build dataloader from dataset
            dataloader_curr =  utils_functions.prepare_dataloader(dataset_curr, batch_size=batchsize, num_workers=num_dataloader_workers_SEQUENTIAL)
            dataloader_curr.sampler.set_epoch(self.epoch) 
            num_pats_curr = dataloader_curr.dataset.get_pat_count()
            
            # Get miniepoch start indexes. Same random indexes will be used for all patients' files.
            start_idxs = self._train_start_idxs(runs_per_file=runs_per_file)
            total_train_iters = int(len(dataloader_curr) * len(start_idxs) * num_pats_curr) # 
            iter_curr = 0 # Iteration across all sequential trian files

            for data_tensor_by_pat, file_name_by_pat in dataloader_curr: # Paralell random file pull accross patients
                for start_idx in start_idxs: # Same start_idx for all patients (has no biological meaning)

                    # For Training: Update the KL multiplier (BETA), and Learning Rate according for Heads Models and Core Model
                    self.KL_multiplier, self.curr_LR_core, self.sparse_weight = utils_functions.LR_and_weight_schedules(
                        epoch=self.epoch, iter_curr=iter_curr, iters_per_epoch=int(total_train_iters), **self.kwargs)
                    
                    # Update LR to schedule
                    if (not val_finetune) & (not val_unseen):
                        self.opt_vae.param_groups[0]['lr'] = self.curr_LR_core

                    # Reset the cumulative losses & zero gradients
                    kld_loss = 0
                    recon_loss = 0
                    self._zero_all_grads()

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

                        ### VAE ENCODER: 1-shifted
                        mean_batched, logvar_batched, latent_batched = self.vae(x[:, :-1, :, :], reverse=False)
                        
                        ### VAE DECODER: 1-shifted & Transformer Encoder Concat Shifted (Need to prime first embedding with past context)
                        x_hat_batched = self.vae(latent_batched, reverse=True, hash_pat_embedding=hash_pat_embedding, out_channels=x.shape[2])  
                        x_hat = torch.split(x_hat_batched, self.transformer_seq_length - self.num_encode_concat_transformer_tokens - 1, dim=0)
                        x_hat = torch.stack(x_hat, dim=0)
 
                        # LOSSES: Intra-Patient 
                        recon_loss = loss_functions.recon_loss_function(
                            x=x[:, 1 + self.num_encode_concat_transformer_tokens:, :, :], # opposite 1-shifted & Transformer Encoder Concat Shifted 
                            x_hat=x_hat,
                            recon_weight=self.recon_weight)

                        kld_loss = loss_functions.kld_loss_function(
                            mean=mean_batched, 
                            logvar=logvar_batched,
                            KL_multiplier=self.KL_multiplier)

                        sparse_loss = loss_functions.sparse_l1_reg(
                            z=latent_batched, 
                            sparse_weight=self.sparse_weight, 
                            **kwargs)

                        # Intrapatient backprop
                        loss = recon_loss + kld_loss # + sparse_loss + kld_loss                      ################ KLD LOSS INCLUDED ?????????? ##############
                        if backprop: loss.backward()

                        # Realtime terminal info and WandB 
                        if (iter_curr%self.recent_display_iters==0):
                            if val_finetune: state_str = "VAL FINETUNE"
                            elif val_unseen: state_str = "VAL UNSEEN"
                            else: state_str = "TRAIN"
                            now_str = datetime.datetime.now().strftime("%I:%M%p-%B/%d/%Y")
                            if (self.gpu_id == 1):
                                sys.stdout.write(
                                    f"\r{now_str} [GPU{str(self.gpu_id)}]: {state_str}, EPOCH {self.epoch}, Iter [BatchSize:{batchsize}]: " + 
                                    str(iter_curr) + "/" + str(total_train_iters) + 
                                    ", MeanLoss: " + str(round(loss.detach().item(), 2))) 
                                sys.stdout.flush() 

                            # Log to WandB
                            wandb.define_metric('Steps')
                            wandb.define_metric("*", step_metric="Steps")
                            train_step = self.epoch * int(total_train_iters) + iter_curr
                            if (not val_finetune) & (not val_unseen):
                                metrics = dict(
                                    train_attention_dropout=attention_dropout,
                                    train_loss=loss,
                                    train_recon_loss=recon_loss, 
                                    train_kld_loss=kld_loss, 
                                    train_sparse_loss=sparse_loss,
                                    train_LR_encoder=self.opt_vae.param_groups[0]['lr'], 
                                    train_KL_Beta=self.KL_multiplier, 
                                    train_ReconWeight=self.recon_weight,
                                    train_Sparse_weight=self.sparse_weight,
                                    train_epoch=self.epoch)

                            elif val_finetune:
                                metrics = dict(
                                    val_finetune_attention_dropout=attention_dropout,
                                    val_finetune_loss=loss, 
                                    val_finetune_recon_loss=recon_loss, 
                                    val_finetune_kld_loss=kld_loss, 
                                    val_finetune_sparse_loss=sparse_loss,
                                    val_finetune_LR_encoder=self.opt_vae.param_groups[0]['lr'], 
                                    val_finetune_KL_Beta=self.KL_multiplier, 
                                    val_finetune_ReconWeight=self.recon_weight,
                                    val_finetune_Sparse_weight=self.sparse_weight,
                                    val_finetune_epoch=self.epoch)

                            elif val_unseen:
                                metrics = dict(
                                    val_unseen_attention_dropout=attention_dropout,
                                    val_unseen_loss=loss, 
                                    val_unseen_recon_loss=recon_loss, 
                                    val_unseen_kld_loss=kld_loss, 
                                    val_unseen_sparse_loss=sparse_loss,
                                    val_unseen_LR_encoder=self.opt_vae.param_groups[0]['lr'], 
                                    val_unseen_KL_Beta=self.KL_multiplier, 
                                    val_unseen_ReconWeight=self.recon_weight,
                                    val_unseen_Sparse_weight=self.sparse_weight,
                                    val_unseen_epoch=self.epoch)

                            wandb.log({**metrics, 'Steps': train_step})

                        # Advance the iteration counter (one iter per complete patient loop - i.e. one backward pass)
                        iter_curr = iter_curr + 1

                        # Realtime latent visualizations
                        if realtime_latent_printing & ((iter_curr + 1) % realtime_printing_interval == 0):
                            if self.gpu_id == 0:
                                logvar = torch.split(logvar_batched, self.transformer_seq_length - self.num_encode_concat_transformer_tokens - 1, dim=0) # 1-shifted & Transformer Encoder Concat Shifted 
                                logvar = torch.stack(logvar, dim=0)
                                mean = torch.split(mean_batched, self.transformer_seq_length - self.num_encode_concat_transformer_tokens - 1, dim=0) # 1-shifted & Transformer Encoder Concat Shifted 
                                mean = torch.stack(mean, dim=0)
                                utils_functions.print_latent_realtime(
                                    mu = mean.cpu().detach().numpy(), 
                                    logvar = logvar.cpu().detach().numpy(),
                                    savedir = self.model_dir + f"/realtime_plots/{dataset_string}/realtime_latents",
                                    epoch = self.epoch,
                                    iter_curr = iter_curr,
                                    pat_id = dataset_curr.pat_ids[pat_idx],
                                    **kwargs)
                                utils_functions.print_recon_realtime(
                                    x=x[:, 1 + self.num_encode_concat_transformer_tokens:, :, :], 
                                    x_hat=x_hat, 
                                    savedir = self.model_dir + f"/realtime_plots/{dataset_string}/realtime_recon",
                                    epoch = self.epoch,
                                    iter_curr = iter_curr,
                                    pat_id = dataset_curr.pat_ids[pat_idx],
                                    **kwargs
                                )
            
                        # ### WITHIN PATIENT LOOP ###
                        # # Step optimizers after single patient
                        # self.opt_vae.step()
                    
                    ### AFTER PATIENT LOOP ###
                    # Step optimizers after all patients have been backpropgated
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