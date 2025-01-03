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

# Local Imports
from utilities import latent_plotting
from utilities import utils_functions
from utilities import loss_functions
from data import SEEG_Tornado_Dataset
from models.Transformer import ModelArgs, Transformer
from models.VAE_core import VAE_Enc, VAE_Dec
from models.VAE_heads import BSE_Enc_Head, BSE_Dec_Head, Head_Optimizers


######
torch.autograd.set_detect_anomaly(True)


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

    # BSE
    core_weight_decay, 
    head_weight_decay, 
    
    # transformer
    max_seq_len, 
    max_batch_size,
    n_layers,
    n_heads, 
    multiple_of,
    adamW_wd,
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
    kwargs['dataset_pic_dir'] = kwargs['model_dir'] + '/dataset_bargraphs_BSE/Train_Grouped'
    train_set = SEEG_Tornado_Dataset(
        gpu_id=gpu_id, 
        pat_list=train_pats_list,
        pat_dirs=train_pats_dirs,
        intrapatient_dataset_style=intrapatient_dataset_style, 
        hour_dataset_range=train_hour_dataset_range,
        **kwargs)
    
    print(f"[GPU{str(gpu_id)}] Generating VALIDATION FINETUNE dataset")
    kwargs['dataset_pic_dir'] = kwargs['model_dir'] + '/dataset_bargraphs_BSE/Val_Grouped_Finetune'
    val_finetune_set = SEEG_Tornado_Dataset(
        gpu_id=gpu_id, 
        pat_list=val_pats_list,
        pat_dirs=val_pats_dirs,
        intrapatient_dataset_style=intrapatient_dataset_style, 
        hour_dataset_range=val_finetune_hour_dataset_range,
        **kwargs)

    print(f"[GPU{str(gpu_id)}] Generating VALIDATION UNSEEN dataset")
    kwargs['dataset_pic_dir'] = kwargs['model_dir'] + '/dataset_bargraphs_BSE/Val_Grouped_Unseen'
    val_unseen_set = SEEG_Tornado_Dataset(
        gpu_id=gpu_id, 
        pat_list=val_pats_list,
        pat_dirs=val_pats_dirs,
        intrapatient_dataset_style=intrapatient_dataset_style, 
        hour_dataset_range=val_unseen_hour_dataset_range,
        **kwargs)
     
    # ### VAE HEADS ###

    # Train
    train_enc_heads = [-1]*len(train_pats_list)
    train_dec_heads = [-1]*len(train_pats_list)
    for i in range(0, len(train_pats_list)):
        train_enc_heads[i] = BSE_Enc_Head(pat_id=train_pats_list[i], num_channels=train_set.pat_num_channels[i], **kwargs).to(gpu_id)
        train_dec_heads[i] = BSE_Dec_Head(pat_id=train_pats_list[i], num_channels=train_set.pat_num_channels[i], **kwargs).to(gpu_id)
    
    train_heads = (train_enc_heads, train_dec_heads)
    opts_train = Head_Optimizers(heads=train_heads, wd=head_weight_decay, lr=kwargs['LR_min_heads'])

    # Val
    val_enc_heads = [-1]*len(val_pats_list)
    val_dec_heads = [-1]*len(val_pats_list)
    for i in range(0, len(val_pats_list)):
        val_enc_heads[i] = BSE_Enc_Head(pat_id=val_pats_list[i], num_channels=val_finetune_set.pat_num_channels[i], **kwargs).to(gpu_id)
        val_dec_heads[i] = BSE_Dec_Head(pat_id=val_pats_list[i], num_channels=val_finetune_set.pat_num_channels[i], **kwargs).to(gpu_id)
    
    val_heads = (val_enc_heads, val_dec_heads)
    opts_val = Head_Optimizers(heads=val_heads, wd=head_weight_decay, lr=kwargs['LR_min_heads'])

    ### VAE Enc ###
    vae_enc = VAE_Enc(gpu_id=gpu_id, **kwargs) 
    vae_enc = vae_enc.to(gpu_id) 
    opt_enc = torch.optim.AdamW(vae_enc.parameters(), lr=kwargs['LR_min_core'], weight_decay=core_weight_decay)
    
    ### VAE Dec ###
    vae_dec = VAE_Dec(gpu_id=gpu_id, **kwargs) 
    vae_dec = vae_dec.to(gpu_id) 
    opt_dec = torch.optim.AdamW(vae_dec.parameters(), lr=kwargs['LR_min_core'], weight_decay=core_weight_decay)

    ### Transformer ###
    transformer = Transformer(ModelArgs(device=gpu_id, **kwargs))
    transformer = transformer.to(gpu_id)
    transformer_opt = torch.optim.AdamW(transformer.parameters(), lr=kwargs['LR_min_transformer'], weight_decay=adamW_wd)
    print(f"[GPU{gpu_id}] transformer loaded")

    return train_set, val_finetune_set, val_unseen_set, transformer, transformer_opt, vae_enc, vae_dec, train_heads, val_heads, opt_enc, opt_dec, opts_train, opts_val  #infer_set

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
    PaCMAP_model_to_infer = [],
    enc_state_dict_prev_path = [],
    enc_opt_state_dict_prev_path = [],
    dec_state_dict_prev_path = [],
    dec_opt_state_dict_prev_path = [],
    transformer_state_dict_prev_path = [],
    transformer_opt_state_dict_prev_path = [],
    heads_prev_dir = [],
    epochs_to_train: int = -1,
    **kwargs):

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
    train_dataset, valfinetune_dataset, valunseen_dataset, transformer, transformer_opt, vae_enc, vae_dec, train_heads, val_heads, opt_enc, opt_dec, opts_train, opts_val = load_train_objs(gpu_id=gpu_id, **kwargs) 
    
    # Load the model/opt/sch states if not first epoch & if in training mode
    if (start_epoch > 0):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}

        # Load in VAE Core weights and opts
        enc_state_dict_prev = torch.load(enc_state_dict_prev_path, map_location=map_location)
        vae_enc.load_state_dict(enc_state_dict_prev)
        enc_opt_state_dict_prev = torch.load(enc_opt_state_dict_prev_path, map_location=map_location)
        opt_enc.load_state_dict(enc_opt_state_dict_prev)

        dec_state_dict_prev = torch.load(dec_state_dict_prev_path, map_location=map_location)
        vae_dec.load_state_dict(dec_state_dict_prev)
        dec_opt_state_dict_prev = torch.load(dec_opt_state_dict_prev_path, map_location=map_location)
        opt_dec.load_state_dict(dec_opt_state_dict_prev)

        # Load in Transformer model weights and opt
        transformer_state_dict_prev = torch.load(transformer_state_dict_prev_path, map_location=map_location)
        transformer.load_state_dict(transformer_state_dict_prev)
        transformer_opt_state_dict_prev = torch.load(transformer_opt_state_dict_prev_path, map_location=map_location)
        transformer_opt.load_state_dict(transformer_opt_state_dict_prev)

        # Load in train heads and opts, val heads are never pretrained by design
        enc_head_weight_files = glob.glob(heads_prev_dir + "/*enc_head.pt")
        enc_head_opt_files = glob.glob(heads_prev_dir + "/*enc_head_opt.pt")
        dec_head_weight_files = glob.glob(heads_prev_dir + "/*dec_head.pt")
        dec_head_opt_files = glob.glob(heads_prev_dir + "/*dec_head_opt.pt")

        # Sort the file names to line up with pat idxs
        enc_head_weight_files.sort()
        enc_head_opt_files.sort()
        dec_head_weight_files.sort()
        dec_head_opt_files.sort()

        for pat_idx in range(len(train_heads[0])):
            # Load model weights for heads (enc, dec)
            filename_enc = enc_head_weight_files[pat_idx]
            pat_idx_in_filename = int(filename_enc.split("/")[-1].split("_")[2].replace("patidx",""))
            if pat_idx_in_filename != pat_idx: raise Exception("Pat idx mismatch in head state loading")
            train_heads[0][pat_idx].load_state_dict(torch.load(filename_enc, map_location=map_location))

            filename_dec = dec_head_weight_files[pat_idx]
            pat_idx_in_filename = int(filename_dec.split("/")[-1].split("_")[2].replace("patidx",""))
            if pat_idx_in_filename != pat_idx: raise Exception("Pat idx mismatch in head state loading")
            train_heads[1][pat_idx].load_state_dict(torch.load(filename_dec, map_location=map_location))

            # Load the optimizers
            filename_enc_OPT = enc_head_opt_files[pat_idx]
            pat_idx_in_filename = int(filename_enc_OPT.split("/")[-1].split("_")[2].replace("patidx",""))
            if pat_idx_in_filename != pat_idx: raise Exception("Pat idx mismatch in head state loading")
            opts_train.enc_head_opts[pat_idx].load_state_dict(torch.load(filename_enc_OPT, map_location=map_location))

            filename_dec_OPT = dec_head_opt_files[pat_idx]
            pat_idx_in_filename = int(filename_dec_OPT.split("/")[-1].split("_")[2].replace("patidx",""))
            if pat_idx_in_filename != pat_idx: raise Exception("Pat idx mismatch in head state loading")
            opts_train.dec_head_opts[pat_idx].load_state_dict(torch.load(filename_dec_OPT, map_location=map_location))

        print("Core and Head Weights and Opts loaded from checkpoints")

    trainer = Trainer(
        world_size=world_size,
        gpu_id=gpu_id, 
        transformer=transformer,
        transformer_opt=transformer_opt,
        vae_enc=vae_enc, 
        vae_dec=vae_dec, 
        train_heads=train_heads,
        val_heads=val_heads,
        start_epoch=start_epoch,
        train_dataset=train_dataset, 
        valfinetune_dataset=valfinetune_dataset,
        valunseen_dataset=valunseen_dataset,
        opt_enc=opt_enc,
        opt_dec=opt_dec, 
        opts_train=opts_train,
        opts_val=opts_val,
        wdecode_batch_size=wdecode_batch_size,
        onlylatent_batch_size=onlylatent_batch_size,
        PaCMAP_model_to_infer=PaCMAP_model_to_infer,
        wandb_run=wandb_run,
        **kwargs)
    
    # Run through all epochs
    for epoch in range(start_epoch, epochs_to_train):
        trainer.epoch = epoch
        
        # PACMAP
        if (trainer.epoch + 1) % trainer.pacmap_every == 0:
            trainer._pacmap(**kwargs)
            # Checkpoint after PACMAP, do not save finetuned model weights
            print(f"GPU{str(trainer.gpu_id)} at pre checkpoint save barrier")
            barrier()
            if trainer.gpu_id == 0: trainer._save_checkpoint(trainer.epoch, saveModels=False, savePaCMAP=True, **kwargs)
        

        # QUICK RECON
        if (trainer.epoch + 1) % trainer.quick_recon_val_every == 0:
            trainer._quick_recon(**kwargs)

        # TRAIN
        trainer._set_to_train()
        trainer._run_epoch(
                dataset_curr = trainer.train_dataset, 
                batchsize=trainer.wdecode_batch_size,
                heads_curr = trainer.train_heads,
                head_opts_curr = trainer.opts_train,
                random_bool = True, # will subsample and randomize
                subsample_file_factor_curr = train_subsample_file_factor, # only valid if 'random_bool' is True
                only_latent = False, # Do not run decoder or transformer
                save_latents = False, # will save the latents to tmp_directory
                all_files_bool = False, # this will run every file for every patient instead of subsampling (changes how dataloaders are made)
                val_finetune = False,
                **kwargs)
        
        # Checkpoint after every train epoch, optionally delete old checkpoints
        print(f"GPU{str(trainer.gpu_id)} at pre checkpoint save barrier")
        barrier()
        if trainer.gpu_id == 0: trainer._save_checkpoint(trainer.epoch, saveModels=True, savePaCMAP=False, **kwargs)

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
        transformer_opt: torch.optim.Optimizer,
        vae_enc: torch.nn.Module,
        vae_dec: torch.nn.Module,
        train_heads: tuple,
        val_heads: tuple,
        start_epoch: int,
        train_dataset: SEEG_Tornado_Dataset,
        valfinetune_dataset: SEEG_Tornado_Dataset,
        valunseen_dataset: SEEG_Tornado_Dataset,
        opt_enc: torch.optim.Optimizer,
        opt_dec: torch.optim.Optimizer,
        opts_train,
        opts_val,
        wdecode_batch_size: int,
        onlylatent_batch_size: int,
        wandb_run,
        model_dir: str,
        quick_recon_val_every: int,
        finetune_quickrecon: bool,
        pacmap_every: int,
        finetune_pacmap: bool,
        pic_save_dir: str,
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
        pre_PaCMAP_window_sec: float,
        pre_PaCMAP_stride_sec: float,
        recent_display_iters: int,
        **kwargs
    ) -> None:
        self.world_size = world_size
        self.gpu_id = gpu_id
        self.transformer = transformer
        self.transformer_opt = transformer_opt
        self.vae_enc = vae_enc
        self.vae_dec = vae_dec
        self.train_heads = train_heads
        self.val_heads = val_heads
        self.start_epoch = start_epoch
        self.train_dataset = train_dataset
        self.valfinetune_dataset = valfinetune_dataset
        self.valunseen_dataset = valunseen_dataset
        self.opt_enc = opt_enc
        self.opt_dec = opt_dec
        self.opts_train = opts_train
        self.opts_val = opts_val
        self.wdecode_batch_size = wdecode_batch_size
        self.onlylatent_batch_size = onlylatent_batch_size
        self.model_dir = model_dir
        self.quick_recon_val_every = quick_recon_val_every
        self.finetune_quickrecon = finetune_quickrecon
        self.pacmap_every = pacmap_every
        self.finetune_pacmap = finetune_pacmap
        self.pic_save_dir = pic_save_dir
        self.latent_dim = latent_dim
        self.autoencode_samples = autoencode_samples
        self.num_samples = num_samples
        self.transformer_seq_length = transformer_seq_length
        self.FS = FS
        self.intrapatient_dataset_style = intrapatient_dataset_style
        self.curr_LR_core = -1
        self.curr_LR_heads = -1
        self.transformer_weight = transformer_weight
        self.recon_weight = recon_weight
        self.atd_file = atd_file
        self.PaCMAP_model_to_infer = PaCMAP_model_to_infer
        self.pre_PaCMAP_window_sec = pre_PaCMAP_window_sec
        self.pre_PaCMAP_stride_sec = pre_PaCMAP_stride_sec
        self.recent_display_iters = recent_display_iters
        self.wandb_run = wandb_run
        self.kwargs = kwargs

        self.KL_multiplier = -1 # dummy variable, only needed when debugging and training is skipped

        # Set up Core/Heads & transformer with DDP
        self.vae_enc = DDP(vae_enc, device_ids=[gpu_id])   # find_unused_parameters=True
        self.vae_dec = DDP(vae_dec, device_ids=[gpu_id])   # find_unused_parameters=True
        self.transformer = DDP(transformer, device_ids=[gpu_id])   # find_unused_parameters=True
        
        self.train_heads = [[-1]*len(train_heads[i]) for i in range(len(train_heads))]
        for i in range(0, len(train_heads)):
            for j in range(len(train_heads[i])):
                self.train_heads[i][j] = DDP(train_heads[i][j], device_ids=[gpu_id])

        self.val_heads = [[-1]*len(val_heads[i]) for i in range(len(val_heads))]
        for i in range(0, len(val_heads)):
            for j in range(len(val_heads[i])):
                self.val_heads[i][j] = DDP(val_heads[i][j], device_ids=[gpu_id])

        self.num_train_pats = len(train_heads[0]) 
        self.train_pat_ids = [self.train_heads[0][i].module.pat_id for i in range(len(self.train_heads[0]))]

        self.num_val_pats = len(val_heads[0])
        self.val_pat_ids = [self.val_heads[0][i].module.pat_id for i in range(len(self.val_heads[0]))]
                
        # Watch with WandB
        # TODO: watch heads as well?
        wandb.watch(self.vae_enc)
        wandb.watch(self.vae_dec)
        wandb.watch(self.transformer)
        
    def _set_to_train(self):
        self.vae_enc.train()
        self.vae_dec.train()
        self.transformer.train()
        self._set_heads_to_train(self.train_heads)
        self._set_heads_to_train(self.val_heads)

    def _set_to_eval(self):
        self.vae_enc.eval()
        self.vae_dec.eval()
        self.transformer.eval()
        self._set_heads_to_eval(self.train_heads)
        self._set_heads_to_eval(self.val_heads)

    def _set_heads_to_train(self, head_tuple):
        for i in range(0, len(head_tuple)):
            for j in range(0, len(head_tuple[i])):
                head_tuple[i][j].train()
                           
    def _set_heads_to_eval(self, head_tuple):
        for i in range(0, len(head_tuple)):
            for j in range(0, len(head_tuple[i])):
                head_tuple[i][j].eval()

    def _zero_all_grads(self):
        self.opt_enc.zero_grad()
        self.opt_dec.zero_grad()
        self.transformer_opt.zero_grad()
        self.opts_train.zero_grad()
        self.opts_val.zero_grad()

    def _save_checkpoint(self, epoch, saveModels, savePaCMAP, delete_old_checkpoints, head_names, **kwargs):
            
            print("CHECKPOINT SAVE")

            # Create new directory for this epoch
            base_checkpoint_dir = self.model_dir + f"/checkpoints"
            check_epoch_dir = base_checkpoint_dir + f"/Epoch_{str(epoch)}"

            # MODEL SAVES
            if saveModels:

                print("Saving core/head model weights")

                ### CORE CHECKPOINT 
                check_core_dir = check_epoch_dir + "/core_checkpoints"
                if not os.path.exists(check_core_dir): os.makedirs(check_core_dir)

                # Save core model
                ckp = self.vae_enc.module.state_dict()
                check_path = check_core_dir + "/checkpoint_epoch" +str(epoch) + "_vae_enc.pt"
                torch.save(ckp, check_path)
                
                ckp = self.vae_dec.module.state_dict()
                check_path = check_core_dir + "/checkpoint_epoch" +str(epoch) + "_vae_dec.pt"
                torch.save(ckp, check_path)

                # Save opt core
                opt_ckp = self.opt_enc.state_dict()
                opt_path = check_core_dir + "/checkpoint_epoch" +str(epoch) + "_vae_enc_opt.pt"
                torch.save(opt_ckp, opt_path)

                opt_ckp = self.opt_dec.state_dict()
                opt_path = check_core_dir + "/checkpoint_epoch" +str(epoch) + "_vae_dec_opt.pt"
                torch.save(opt_ckp, opt_path)

                ### HEADS CHECKPOINT
                check_heads_dir = check_epoch_dir + "/heads_checkpoints"
                if not os.path.exists(check_heads_dir): os.makedirs(check_heads_dir)

                # Save train heads model
                if len(head_names) != len(self.train_heads): raise Exception(f"Got {len(self.train_heads)} from self.train_heads, but expected {len(head_names)} based on provided head names in config.yml file")
                for head_style in range(0, len(self.train_heads)):
                    for pat_idx in range(0, len(self.train_heads[head_style])):
                        ckp = self.train_heads[head_style][pat_idx].module.state_dict()
                        check_path = check_heads_dir + "/checkpoint_epoch" +str(epoch) + f"_patidx{pat_idx}_{head_names[head_style]}_head.pt"
                        torch.save(ckp, check_path)

                # Opts are indexed by head name, thus must iterate #TODO not hardcoded
                for pat_idx in range(0, len(self.train_heads[0])):
                    opt_ckp = self.opts_train.enc_head_opts[pat_idx].state_dict()
                    opt_path = check_heads_dir + "/checkpoint_epoch" +str(epoch) + f"_patidx{pat_idx}_enc_head_opt.pt"
                    torch.save(opt_ckp, opt_path)

                    opt_ckp = self.opts_train.dec_head_opts[pat_idx].state_dict()
                    opt_path = check_heads_dir + "/checkpoint_epoch" +str(epoch) + f"_patidx{pat_idx}_dec_head_opt.pt"
                    torch.save(opt_ckp, opt_path)

                ### Transformer ###

                print("Saving Transformer model weights")

                # Save transformer model
                check_transformer_dir = check_epoch_dir + "/transformer_checkpoints"
                if not os.path.exists(check_transformer_dir): os.makedirs(check_transformer_dir)
                ckp = self.transformer.module.state_dict()
                check_path = check_transformer_dir + "/checkpoint_epoch" +str(epoch) + "_transformer.pt"
                torch.save(ckp, check_path)
                
                # Save transformer optimizer
                opt_ckp = self.transformer_opt.state_dict()
                opt_path = check_transformer_dir + "/checkpoint_epoch" +str(epoch) + "_transformer_opt.pt"
                torch.save(opt_ckp, opt_path)

                print(f"Epoch {epoch} | Training checkpoint saved at {check_epoch_dir}")


            ### PACMAP & HDBSCAN
            if savePaCMAP:

                print("Saving PaCMAP models")

                # Path
                pacmap_dir = check_epoch_dir + "/pacmap"
                if not os.path.exists(pacmap_dir): os.makedirs(pacmap_dir) 

                # Save the PaCMAP model for use in inference
                PaCMAP_common_prefix = pacmap_dir + "/checkpoint_epoch" +str(epoch) + "_PaCMAP"
                pacmap.save(self.PaCMAP, PaCMAP_common_prefix)

                PaCMAP_common_prefix_MedDim = pacmap_dir + "/checkpoint_epoch" +str(epoch) + "_PaCMAP_MedDim"
                pacmap.save(self.PaCMAP_MedDim, PaCMAP_common_prefix_MedDim)

                pre_PaCMAP_window_sec_path = pacmap_dir + "/checkpoint_epoch" +str(epoch) + "_pre_PaCMAP_window_sec.pkl"
                output_obj2 = open(pre_PaCMAP_window_sec_path, 'wb')
                pickle.dump(self.pre_PaCMAP_window_sec, output_obj2)
                output_obj2.close()

                pre_PaCMAP_stride_sec_path = pacmap_dir + "/checkpoint_epoch" +str(epoch) + "_pre_PaCMAP_stride_sec.pkl"
                output_obj3 = open(pre_PaCMAP_stride_sec_path, 'wb')
                pickle.dump(self.pre_PaCMAP_stride_sec, output_obj3)
                output_obj3.close()
                print("Saved PaCMAP 2-dim and MedDim models")

                hdbscan_path = pacmap_dir + "/checkpoint_epoch" +str(epoch) + "_hdbscan.pkl"
                output_obj4 = open(hdbscan_path, 'wb')
                pickle.dump(self.HDBSCAN, output_obj4)
                output_obj4.close()
                print("Saved HDBSCAN model")

                pca_path = pacmap_dir + "/checkpoint_epoch" +str(epoch) + "_PCA.pkl"
                output_obj5 = open(pca_path, 'wb')
                pickle.dump(self.pca, output_obj5)
                output_obj5.close()
                print("Saved PCA model")

                reorder_path = pacmap_dir + "/checkpoint_epoch" +str(epoch) + "_cluster_reorder_indexes.pkl"
                output_obj6 = open(reorder_path, 'wb')
                pickle.dump(self.cluster_reorder_indexes, output_obj6)
                output_obj6.close()
                print("Saved cluster reorder indexes")

                xylim_path = pacmap_dir + "/checkpoint_epoch" +str(epoch) + "_xy_lims.pkl"
                output_obj7 = open(xylim_path, 'wb')
                pickle.dump(self.xy_lims, output_obj7)
                output_obj7.close()
                print("Saved xy_lims for PaCMAP")

                xylims_RAWDIMS_path = pacmap_dir + "/checkpoint_epoch" +str(epoch) + "_xy_lims_RAW_DIMS.pkl"
                output_obj8 = open(xylims_RAWDIMS_path, 'wb')
                pickle.dump(self.xy_lims_RAW_DIMS, output_obj8)
                output_obj8.close()
                print("Saved xy_lims RAW DIMS")

                xylims_PCA_path = pacmap_dir + "/checkpoint_epoch" +str(epoch) + "_xy_lims_PCA.pkl"
                output_obj9 = open(xylims_PCA_path, 'wb')
                pickle.dump(self.xy_lims_PCA, output_obj9)
                output_obj9.close()
                print("Saved xy_lims PCA")

                infoscore_path = pacmap_dir + "/checkpoint_epoch" +str(epoch) + "_info_score_idxs.pkl"
                output_obj10 = open(infoscore_path, 'wb')
                pickle.dump(self.info_score_idxs, output_obj10)
                output_obj10.close()
                print("Saved info score indexes for raw dims")

            if delete_old_checkpoints:
                utils_functions.delete_old_checkpoints(dir = base_checkpoint_dir, curr_epoch = epoch)
                print("Deleted old checkpoints, except epochs with PaCMAP/HDBSCAN models")

    def _quick_recon(self, **kwargs):

        # Train Pats pre-valfinetune

        if self.finetune_quickrecon:
             raise Exception("TODO")

            # Val Finetune

            # Val Unseen

            # Train Pats post-valfinetune

        raise Exception("TODO")

    def _pacmap(self, **kwargs):

        if self.finetune_pacmap:
            raise Exception("TODO")

        
        raise Exception("TODO")

    def _train_start_idxs(self, subsample_file_factor, random_bool):

        np.random.seed(seed=None) # should replace with Generator for newer code
        
        if random_bool: frame_shift = int(random.uniform(0, self.autoencode_samples -1))
        else: frame_shift = 0
        
        start_idxs = np.arange(0,self.num_windows - self.transformer_seq_length - 1) * self.autoencode_samples + frame_shift
        if random_bool: np.random.shuffle(start_idxs)

        if random_bool: start_idxs = start_idxs[0::subsample_file_factor]
        
        return start_idxs

    def _run_epoch(
        self, 
        dataset_curr, 
        batchsize,
        heads_curr,
        head_opts_curr,
        random_bool, # will subsample and randomize
        subsample_file_factor_curr, # only valid if 'random' is True
        only_latent,
        save_latents, 
        all_files_bool,
        num_dataloader_workers_SEQUENTIAL,
        val_finetune,
        realtime_latent_printing,
        realtime_printing_interval,
        **kwargs):

        print(f"autoencode_samples: {self.autoencode_samples}")

        ### ALL FILES ### 
        # If wanting all files from every patiet, need to run patients serially
        if all_files_bool: 
            # Go through every subject 
            for pat_idx in range(0,len(dataset_curr.pat_ids)):
                dataset_curr.set_pat_curr(pat_idx)
                dataloader_curr =  utils_functions.prepare_dataloader(dataset_curr, batch_size=batchsize, num_workers=num_dataloader_workers_SEQUENTIAL)
                raise Exception("not coded up")

        
        ### SUBSET OF FILES ###
        # Can run the patients in parallel
        else: 
            dataset_curr.set_pat_curr(-1) # -1 enables all pat mode

            # Build dataloader from dataset
            dataloader_curr =  utils_functions.prepare_dataloader(dataset_curr, batch_size=batchsize, num_workers=num_dataloader_workers_SEQUENTIAL)
            num_pats_curr = dataloader_curr.dataset.get_pat_count()
            
            # Number of iterations per file
            self.num_windows = int((self.num_samples - self.transformer_seq_length * self.autoencode_samples - self.autoencode_samples)/self.autoencode_samples) - 2

            # Get miniepoch start indexes. Same random indexes will be used for all patients' files.
            start_idxs = self._train_start_idxs(random_bool=random_bool, subsample_file_factor=subsample_file_factor_curr)
            total_train_iters = int(len(dataloader_curr) * len(start_idxs) * num_pats_curr) # 
            iter_curr = 0 # Iteration across all sequential trian files

            for data_tensor_by_pat, file_name_by_pat in dataloader_curr: # Paralell random file pull accross patients
                for start_idx in start_idxs: # Same start_idx for all patients (has no biological meaning)

                    # Update the KL multiplier (BETA), and Learning Rate according for Heads Models and Core Model
                    self.KL_multiplier, self.curr_LR_core, self.curr_LR_heads, self.transformer_LR = utils_functions.LR_and_weight_schedules(
                        epoch=self.epoch, iter_curr=iter_curr, iters_per_epoch=int(total_train_iters), **self.kwargs)
                    
                    # Update Core and Heads LR
                    self.opt_enc.param_groups[0]['lr'] = self.curr_LR_core
                    self.opt_dec.param_groups[0]['lr'] = self.curr_LR_core
                    head_opts_curr.set_all_lr(self.curr_LR_heads)
                    self.transformer_opt.param_groups[0]['lr'] = self.transformer_LR

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

                        # Pull out the patient's heads
                        enc_head=heads_curr[0][pat_idx] # Heads are [enc, dec]
                        dec_head=heads_curr[1][pat_idx]

                        # Pull patient's data
                        data_tensor = data_tensor_by_pat[pat_idx]

                        # Reset the data vars for Transformer Sequence and put on GPU
                        x = torch.zeros(data_tensor.shape[0], self.transformer_seq_length, data_tensor.shape[1], self.autoencode_samples).to(self.gpu_id)

                        # Collect sequential embeddings for transformer by running sequential raw data windows through BSE N times 
                        for embedding_idx in range(0, self.transformer_seq_length):

                            # Pull out data for this window
                            end_idx = start_idx + self.autoencode_samples * embedding_idx + self.autoencode_samples 
                            x[:, embedding_idx, :, :] = data_tensor[:, :, end_idx-self.autoencode_samples : end_idx]

                        # Stack the vars into batch dimension 
                        x_batched = x.reshape([x.shape[0]*x.shape[1], x.shape[2], x.shape[3]])

                        # To remember how to split back to original:
                        # a = torch.split(x_decode_shifted_batched, self.transformer_seq_length-1, dim=0)
                        # b = torch.stack(a, dim=0)

                        ### VAE ENCODER
                        # Forward pass in stacked batch through head then VAE encoder
                        x_posthead = enc_head(x_batched)
                        mean_batched, logvar_batched, latent_batched, = self.vae_enc(x_posthead)

                        # Split the batched dimension and stack into sequence dimension [batch, seq, latent_dims]
                        latent_seq = torch.split(latent_batched, self.transformer_seq_length, dim=0)
                        latent_seq = torch.stack(latent_seq, dim=0)
  
                        ### TRANSFORMER
                        # Run sequence through transformer and get transformer loss
                        predicted_embeddings = self.transformer(latent_seq[:, :-1, :])
                        
                        ### VAE DECODER
                        # Run the predicted embeddings through decoder
                        predicted_embeddings_batched = predicted_embeddings.reshape(predicted_embeddings.shape[0]*predicted_embeddings.shape[1], predicted_embeddings.shape[2])                        
                        core_out = self.vae_dec(predicted_embeddings_batched)  
                        x_hat_batched = dec_head(core_out)
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

                        # Intrapatient backprop
                        loss = recon_loss + kld_loss + transformer_loss             ################ direct TRANSFORMER LOSS INCLUDED ?????????? ##############
                        loss.backward()

                        # Realtime info as epoch is running
                        if (iter_curr%self.recent_display_iters==0):
                            if val_finetune: state_str = "VAL FINETUNE"
                            else: state_str = "TRAIN"
                            now_str = datetime.datetime.now().strftime("%I:%M%p-%B/%d/%Y")
                            if (self.gpu_id == 1):
                                sys.stdout.write(f"\r{now_str} [GPU{str(self.gpu_id)}]: {state_str}, Iter [BatchSize:{batchsize}]: " + 
                                                str(iter_curr) + "/" + str(total_train_iters) + 
                                                ", MeanLoss: " + str(round(loss.detach().item(), 2)) + ", [" + 
                                                    "Rec: " + str(round(recon_loss.detach().item(), 2)) + " + " + 
                                                    "KLD: {:0.3e}".format(kld_loss.detach().item(), 2) + " + " +
                                                    "Trnsfr {:0.3e}".format(transformer_loss.detach().item(), 2) + "], " + 
                                                    "Core LR: {:0.3e}".format(self.opt_enc.param_groups[0]['lr']) + 
                                                    ", Head LR: {:0.3e}".format(head_opts_curr.get_lr()) + 
                                                    ", Transformer LR: {:0.3e}".format(self.transformer_opt.param_groups[0]['lr']) +
                                                ", Beta: {:0.3e}".format(self.KL_multiplier) + "         ")
                                sys.stdout.flush() 

                            # Log to WandB
                            wandb.define_metric('Steps')
                            wandb.define_metric("*", step_metric="Steps")
                            train_step = self.epoch * int(total_train_iters) + iter_curr
                            if not val_finetune:
                                metrics = dict(
                                    train_loss=loss,
                                    train_transformer_loss=transformer_loss,
                                    train_recon_loss=recon_loss, 
                                    train_kld_loss=kld_loss, 
                                    train_LR_encoder=self.opt_enc.param_groups[0]['lr'], 
                                    train_LR_decoder=self.opt_dec.param_groups[0]['lr'], 
                                    train_LR_transformer=self.transformer_opt.param_groups[0]['lr'],
                                    train_LR_heads=head_opts_curr.get_lr(),
                                    train_KL_Beta=self.KL_multiplier, 
                                    train_ReconWeight=self.recon_weight,
                                    train_Transformer_weight=self.transformer_weight,
                                    train_epoch=self.epoch)
                            else:
                                metrics = dict(
                                    val_finetune_loss=loss, 
                                    val_finetune_transformer_loss=transformer_loss,
                                    val_finetune_recon_loss=recon_loss, 
                                    val_finetune_kld_loss=kld_loss, 
                                    val_finetune_LR_heads=head_opts_curr.get_lr(),
                                    val_finetune_LR_encoder=self.opt_enc.param_groups[0]['lr'], 
                                    val_finetune_LR_decoder=self.opt_dec.param_groups[0]['lr'], 
                                    val_finetune_LR_transformer=self.transformer_opt.param_groups[0]['lr'],
                                    val_finetune_KL_Beta=self.KL_multiplier, 
                                    val_finetune_ReconWeight=self.recon_weight,
                                    val_finetune_Transformer_weight=self.transformer_weight,
                                    val_finetune_epoch=self.epoch)

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
                                    savedir = self.model_dir + "/realtime_latents",
                                    epoch = self.epoch,
                                    iter_curr = iter_curr,
                                    pat_id = dataset_curr.pat_ids[pat_idx],
                                    **kwargs)
                                utils_functions.print_recon_realtime(
                                    x_decode_shifted=x[:, 1:, :, :], 
                                    x_hat=x_hat, 
                                    savedir = self.model_dir + "/realtime_recon",
                                    epoch = self.epoch,
                                    iter_curr = iter_curr,
                                    pat_id = dataset_curr.pat_ids[pat_idx],
                                    **kwargs
                                )
            
                    ### AFTER PATIENT LOOP ###
                    # Step optimizers after all patients have been backpropgated
                    self.transformer_opt.step()
                    self.opt_enc.step()
                    self.opt_dec.step()
                    for pat_idx in np.arange(0,num_pats_curr): 
                        head_opts_curr.step(pat_idx)
                        
        return self
        

if __name__ == "__main__":

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