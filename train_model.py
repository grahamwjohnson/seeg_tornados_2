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
    train_dataset, val_finetune_dataset, val_unseen_dataset, ind_train_datasets, ind_val_finetune_datasets, ind_val_unseen_datasets, lbm, lbm_opt, vae_core, train_heads, val_heads, opt_core, opts_train, opts_val  = load_train_objs(gpu_id=gpu_id, eon=eon, **kwargs) 
    
    # Print VAE CORE model info (only do once on gpu_id==0)
    if gpu_id == 0: 
        utils_functions.print_model_summary(model=vae_core)

        # Print the estimated LBM model size
        utils_functions.print_model_summary(model=lbm)

        # Save some code
        save_code_path = kwargs['model_dir'] + '/code_archived'
        if not os.path.exists(save_code_path): os.makedirs(save_code_path)
        shutil.copyfile(vae_core.get_script_filename(), save_code_path + '/model_code.py')
        shutil.copyfile(train_dataset.get_script_filename(), save_code_path + '/data_code.py')
        shutil.copyfile(__file__, save_code_path + '/train_code.py')

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
        lbm=lbm,
        lbm_opt=lbm_opt,
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