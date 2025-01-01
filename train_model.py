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
from utilities import loss
from data import SEEG_Tornado_Dataset
from models.Transformer import ModelArgs, Transformer
from models.VAE_core import BSE_Middle_VAE
from models.VAE_heads import BSE_Enc_Head, BSE_Dec_Head, BSE_Dec_Hint_Prep, Head_Optimizers


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
    eon, 
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
    transformer_dim,
    adamW_wd,
    transformer_LR,
    
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
        eon=eon, 
        intrapatient_dataset_style=intrapatient_dataset_style, 
        hour_dataset_range=train_hour_dataset_range,
        single_pat_seq = False, 
        **kwargs)
    
    print(f"[GPU{str(gpu_id)}] Generating VALIDATION FINETUNE dataset")
    kwargs['dataset_pic_dir'] = kwargs['model_dir'] + '/dataset_bargraphs_BSE/Val_Grouped_Finetune'
    val_finetune_set = SEEG_Tornado_Dataset(
        gpu_id=gpu_id, 
        eon=eon, 
        pat_list=val_pats_list,
        pat_dirs=val_pats_dirs,
        intrapatient_dataset_style=intrapatient_dataset_style, 
        hour_dataset_range=val_finetune_hour_dataset_range,
        single_pat_seq = False,  
        **kwargs)

    print(f"[GPU{str(gpu_id)}] Generating VALIDATION UNSEEN dataset")
    kwargs['dataset_pic_dir'] = kwargs['model_dir'] + '/dataset_bargraphs_BSE/Val_Grouped_Unseen'
    val_unseen_set = SEEG_Tornado_Dataset(
        gpu_id=gpu_id, 
        eon=eon, 
        pat_list=val_pats_list,
        pat_dirs=val_pats_dirs,
        intrapatient_dataset_style=intrapatient_dataset_style, 
        hour_dataset_range=val_unseen_hour_dataset_range,
        single_pat_seq = False,  
        **kwargs)
    
    print(f"[GPU{str(gpu_id)}] Generating INDIVIDUAL TRAIN dataset")
    kwargs['dataset_pic_dir'] = kwargs['model_dir'] + '/dataset_bargraphs_BSE/Train_Individual'
    ind_train_datasets = [-1]*len(train_pats_list)
    for pat_idx in range(0, len(train_pats_list)):
        ind_train_datasets[pat_idx] = SEEG_Tornado_Dataset(
            gpu_id=gpu_id, 
            eon=eon, 
            pat_list=[train_pats_list[pat_idx]],
            pat_dirs=[train_pats_dirs[pat_idx]],
            intrapatient_dataset_style=intrapatient_dataset_style, 
            hour_dataset_range=train_hour_dataset_range,
            single_pat_seq = True,  
            **kwargs)

    print(f"[GPU{str(gpu_id)}] Generating INDIVIDUAL VALIDATION FINETUNE dataset")
    kwargs['dataset_pic_dir'] = kwargs['model_dir'] + '/dataset_bargraphs_BSE/Val_Individual_Finetune'
    ind_val_finetune_datasets = [-1]*len(val_pats_list)
    for pat_idx in range(0, len(val_pats_list)):
        ind_val_finetune_datasets[pat_idx] = SEEG_Tornado_Dataset(
            gpu_id=gpu_id, 
            eon=eon, 
            pat_list=[val_pats_list[pat_idx]],
            pat_dirs=[val_pats_dirs[pat_idx]],
            intrapatient_dataset_style=intrapatient_dataset_style, 
            hour_dataset_range=val_finetune_hour_dataset_range,
            single_pat_seq = True,  
            **kwargs)

    print(f"[GPU{str(gpu_id)}] Generating INDIVIDUAL VALIDATION UNSEEN dataset")
    kwargs['dataset_pic_dir'] = kwargs['model_dir'] + '/dataset_bargraphs_BSE/Val_Individual_Unseen'
    ind_val_unseen_datasets = [-1]*len(val_pats_list)
    for pat_idx in range(0, len(val_pats_list)):
        ind_val_unseen_datasets[pat_idx] = SEEG_Tornado_Dataset(
            gpu_id=gpu_id, 
            eon=eon, 
            pat_list=[val_pats_list[pat_idx]],
            pat_dirs=[val_pats_dirs[pat_idx]],
            intrapatient_dataset_style=intrapatient_dataset_style, 
            hour_dataset_range=val_unseen_hour_dataset_range,
            single_pat_seq = True,  
            **kwargs)
    
    # Build the model with swappable heads
    train_enc_heads = [-1]*len(train_pats_list)
    train_dec_heads = [-1]*len(train_pats_list)
    train_hint_preppers = [-1]*len(train_pats_list)
    for i in range(0, len(train_pats_list)):
        train_enc_heads[i] = BSE_Enc_Head(pat_id=train_pats_list[i], num_channels=train_set.pat_num_channels[i], **kwargs).to(gpu_id)
        train_dec_heads[i] = BSE_Dec_Head(pat_id=train_pats_list[i], num_channels=train_set.pat_num_channels[i], **kwargs).to(gpu_id)
        train_hint_preppers[i] = BSE_Dec_Hint_Prep(pat_id=train_pats_list[i], num_channels=train_set.pat_num_channels[i], **kwargs).to(gpu_id)
    train_heads = (train_enc_heads, train_dec_heads, train_hint_preppers)

    val_enc_heads = [-1]*len(val_pats_list)
    val_dec_heads = [-1]*len(val_pats_list)
    val_hint_preppers = [-1]*len(val_pats_list)
    for i in range(0, len(val_pats_list)):
        val_enc_heads[i] = BSE_Enc_Head(pat_id=val_pats_list[i], num_channels=val_finetune_set.pat_num_channels[i], **kwargs).to(gpu_id)
        val_dec_heads[i] = BSE_Dec_Head(pat_id=val_pats_list[i], num_channels=val_finetune_set.pat_num_channels[i], **kwargs).to(gpu_id)
        val_hint_preppers[i] = BSE_Dec_Hint_Prep(pat_id=val_pats_list[i], num_channels=val_finetune_set.pat_num_channels[i], **kwargs).to(gpu_id)
    val_heads = (val_enc_heads, val_dec_heads, val_hint_preppers)

    # Build the core model
    vae_core = BSE_Middle_VAE(gpu_id=gpu_id, **kwargs) 
    vae_core = vae_core.to(gpu_id) # move to GPU here to avoid opt_core problems when loading states

    # Build the optimizers, one for core and individual opts for swappable heads
    opt_core = torch.optim.AdamW(vae_core.parameters(), lr=kwargs['LR_min_core'], weight_decay=core_weight_decay)
    opts_train = Head_Optimizers(heads=train_heads, wd=head_weight_decay, lr=kwargs['LR_min_heads'])
    opts_val = Head_Optimizers(heads=val_heads, wd=head_weight_decay, lr=kwargs['LR_min_heads'])
    

    ### Transformer ###
    
    # Make the transformer based on Llama3
    model_args: ModelArgs = ModelArgs(
        dim=transformer_dim,
        vae_dim=kwargs['latent_dim'],
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        n_layers=n_layers,
        n_heads=n_heads,
        multiple_of=multiple_of,
        device=gpu_id
    )
    transformer = Transformer(model_args)
    transformer = transformer.to(gpu_id)
    # opt = torch.optim.Adam(transformer.parameters(), lr=LR_min)
    transformer_opt = torch.optim.AdamW(transformer.parameters(), lr=transformer_LR, weight_decay=adamW_wd)
    print(f"[GPU{gpu_id}] transformer loaded")

    return train_set, val_finetune_set, val_unseen_set, ind_train_datasets, ind_val_finetune_datasets, ind_val_unseen_datasets, transformer, transformer_opt, vae_core, train_heads, val_heads, opt_core, opts_train, opts_val  #infer_set




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
    PaCMAP_model_to_infer = [],
    core_state_dict_prev_path = [],
    core_opt_state_dict_prev_path = [],
    heads_prev_dir = [],
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

    print(f"[GPU{str(gpu_id)}] Loading training objects (datasets, model, opt_core)")
    train_dataset, val_finetune_dataset, val_unseen_dataset, ind_train_datasets, ind_val_finetune_datasets, ind_val_unseen_datasets, transformer, transformer_opt, vae_core, train_heads, val_heads, opt_core, opts_train, opts_val  = load_train_objs(gpu_id=gpu_id, eon=eon, **kwargs) 
    

    # Build dataloaders from datasets
    seq_workers = kwargs['num_dataloader_workers_SEQUENTIAL']
    train_wdecode_dataloader =  utils_functions.prepare_dataloader(train_dataset, batch_size=wdecode_batch_size, num_workers=seq_workers)
    train_onlylatent_dataloader = utils_functions.prepare_dataloader(train_dataset, batch_size=onlylatent_batch_size, num_workers=seq_workers, droplast=False)
    val_finetune_wdecode_dataloader =  utils_functions.prepare_dataloader(val_finetune_dataset, batch_size=wdecode_batch_size, num_workers=seq_workers) 
    val_finetune_onlylatent_dataloader =  utils_functions.prepare_dataloader(val_finetune_dataset, batch_size=onlylatent_batch_size, num_workers=seq_workers, droplast=False) 
    val_unseen_wdecode_dataloader =  utils_functions.prepare_dataloader(val_unseen_dataset, batch_size=wdecode_batch_size, num_workers=seq_workers) 
    val_unseen_onlylatent_dataloader =  utils_functions.prepare_dataloader(val_unseen_dataset, batch_size=onlylatent_batch_size, num_workers=seq_workers, droplast=False) 
    ind_train_wdecode_dataloaders = [utils_functions.prepare_dataloader(ind_train_datasets[i], batch_size=wdecode_batch_size, num_workers=seq_workers) for i in range(len(ind_train_datasets))] 
    ind_train_onlylatent_dataloaders = [utils_functions.prepare_dataloader(ind_train_datasets[i], batch_size=onlylatent_batch_size, num_workers=seq_workers, droplast=False) for i in range(len(ind_train_datasets))] 
    ind_val_finetune_wdecode_dataloaders = [utils_functions.prepare_dataloader(ind_val_finetune_datasets[i], batch_size=wdecode_batch_size, num_workers=seq_workers) for i in range(len(ind_val_finetune_datasets))] 
    ind_val_finetune_onlylatent_dataloaders = [utils_functions.prepare_dataloader(ind_val_finetune_datasets[i], batch_size=onlylatent_batch_size, num_workers=seq_workers, droplast=False) for i in range(len(ind_val_finetune_datasets))] 
    ind_val_unseen_wdecode_dataloaders = [utils_functions.prepare_dataloader(ind_val_unseen_datasets[i], batch_size=wdecode_batch_size, num_workers=seq_workers) for i in range(len(ind_val_unseen_datasets))] 
    ind_val_unseen_onlylatent_dataloaders = [utils_functions.prepare_dataloader(ind_val_unseen_datasets[i], batch_size=onlylatent_batch_size, num_workers=seq_workers, droplast=False) for i in range(len(ind_val_unseen_datasets))] 

    # Load the model/opt/sch states if not first epoch & if in training mode
    if (start_epoch > 0):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}
        core_state_dict_prev = torch.load(core_state_dict_prev_path, map_location=map_location)
        vae_core.load_state_dict(core_state_dict_prev)
        core_opt_state_dict_prev = torch.load(core_opt_state_dict_prev_path, map_location=map_location)
        opt_core.load_state_dict(core_opt_state_dict_prev)

        # Load in train heads and opts, val heads are never pretrained by design
        enc_head_weight_files = glob.glob(heads_prev_dir + "/*enc_head.pt")
        enc_head_opt_files = glob.glob(heads_prev_dir + "/*enc_head_opt.pt")
        dec_head_weight_files = glob.glob(heads_prev_dir + "/*dec_head.pt")
        dec_head_opt_files = glob.glob(heads_prev_dir + "/*dec_head_opt.pt")
        hinter_head_weight_files = glob.glob(heads_prev_dir + "/*hinter_head.pt")
        hinter_head_opt_files = glob.glob(heads_prev_dir + "/*hinter_head_opt.pt")

        # Sort the file names to line up with pat idxs
        enc_head_weight_files.sort()
        enc_head_opt_files.sort()
        dec_head_weight_files.sort()
        dec_head_opt_files.sort()
        hinter_head_weight_files.sort()
        hinter_head_opt_files.sort()

        for pat_idx in range(len(train_heads[0])):
            # Load model weights for heads (enc, dec, hinter)
            filename_enc = enc_head_weight_files[pat_idx]
            pat_idx_in_filename = int(filename_enc.split("/")[-1].split("_")[2].replace("patidx",""))
            if pat_idx_in_filename != pat_idx: raise Exception("Pat idx mismatch in head state loading")
            train_heads[0][pat_idx].load_state_dict(torch.load(filename_enc, map_location=map_location))

            filename_dec = dec_head_weight_files[pat_idx]
            pat_idx_in_filename = int(filename_dec.split("/")[-1].split("_")[2].replace("patidx",""))
            if pat_idx_in_filename != pat_idx: raise Exception("Pat idx mismatch in head state loading")
            train_heads[1][pat_idx].load_state_dict(torch.load(filename_dec, map_location=map_location))

            filename_hinter = hinter_head_weight_files[pat_idx]
            pat_idx_in_filename = int(filename_hinter.split("/")[-1].split("_")[2].replace("patidx",""))
            if pat_idx_in_filename != pat_idx: raise Exception("Pat idx mismatch in head state loading")
            train_heads[2][pat_idx].load_state_dict(torch.load(filename_hinter, map_location=map_location))

            # Load the optimizers
            filename_enc_OPT = enc_head_opt_files[pat_idx]
            pat_idx_in_filename = int(filename_enc_OPT.split("/")[-1].split("_")[2].replace("patidx",""))
            if pat_idx_in_filename != pat_idx: raise Exception("Pat idx mismatch in head state loading")
            opts_train.enc_head_opts[pat_idx].load_state_dict(torch.load(filename_enc_OPT, map_location=map_location))

            filename_dec_OPT = dec_head_opt_files[pat_idx]
            pat_idx_in_filename = int(filename_dec_OPT.split("/")[-1].split("_")[2].replace("patidx",""))
            if pat_idx_in_filename != pat_idx: raise Exception("Pat idx mismatch in head state loading")
            opts_train.dec_head_opts[pat_idx].load_state_dict(torch.load(filename_dec_OPT, map_location=map_location))

            filename_hinter_OPT = hinter_head_opt_files[pat_idx]
            pat_idx_in_filename = int(filename_hinter_OPT.split("/")[-1].split("_")[2].replace("patidx",""))
            if pat_idx_in_filename != pat_idx: raise Exception("Pat idx mismatch in head state loading")
            opts_train.hint_preppers_opts[pat_idx].load_state_dict(torch.load(filename_hinter_OPT, map_location=map_location))

        print("Core and Head Weights and Opts loaded from checkpoints")

    trainer = Trainer(
        world_size=world_size,
        gpu_id=gpu_id, 
        transformer=transformer,
        transformer_opt=transformer_opt,
        vae_core=vae_core, 
        train_heads=train_heads,
        val_heads=val_heads,
        eon=eon,
        start_epoch=start_epoch,
        train_wdecode_dataloader=train_wdecode_dataloader, 
        train_onlylatent_dataloader=train_onlylatent_dataloader,
        val_finetune_wdecode_dataloader=val_finetune_wdecode_dataloader,
        val_finetune_onlylatent_dataloader=val_finetune_onlylatent_dataloader,
        val_unseen_wdecode_dataloader=val_unseen_wdecode_dataloader,
        val_unseen_onlylatent_dataloader=val_unseen_onlylatent_dataloader,
        ind_train_wdecode_dataloaders = ind_train_wdecode_dataloaders,
        ind_train_onlylatent_dataloaders = ind_train_onlylatent_dataloaders,
        ind_val_finetune_wdecode_dataloaders = ind_val_finetune_wdecode_dataloaders,
        ind_val_finetune_onlylatent_dataloaders = ind_val_finetune_onlylatent_dataloaders,
        ind_val_unseen_wdecode_dataloaders = ind_val_unseen_wdecode_dataloaders,
        ind_val_unseen_onlylatent_dataloaders = ind_val_unseen_onlylatent_dataloaders,
        opt_core=opt_core, 
        opts_train=opts_train,
        opts_val=opts_val,
        wdecode_batch_size=wdecode_batch_size,
        onlylatent_batch_size=onlylatent_batch_size,
        PaCMAP_model_to_infer=PaCMAP_model_to_infer,
        wandb_run=wandb_run,
        **kwargs)
    
    # Run training & val
    if not kwargs['run_inference_now']: trainer._train_and_val(**kwargs)

    print(f"[GPU{gpu_id}]: End of main loop, killing process")
    wandb.finish()
    destroy_process_group() 



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