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
from models/Transformer import ModelArgs, Transformer
from models/VAE_core import BSE_Middle_VAE
from models/VAE_heads import BSE_Enc_Head, BSE_Dec_Head, BSE_Dec_Hint_Prep, Head_Optimizers

















if __name__ == "__main__":

    # Print to console
    print("\n\n***************** MAIN START ******************\n\n")
    print(datetime.datetime.now().strftime("%I:%M%p-%B/%d/%Y"))
    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'    

    # Needed?
    mp.set_start_method('spawn', force=True)
    mp_lock = mp.Lock()

    # Read in configuration file
    config_f = 'train_config.yml'
    with open(config_f, "r") as f: kwargs = yaml.load(f,Loader=yaml.FullLoader)
    kwargs = utils_functions.exec_kwargs(kwargs) # Execute the arithmatic build into kwargs and reassign kwargs

    # All Time Data file to get event timestamps
    kwargs['root_save_dir'] = utils_functions.assemble_model_save_path(**kwargs)
    # kwargs['data_dir'] = kwargs['root_save_dir'] + kwargs['data_dir_subfolder']
    kwargs['tmp_model_dir'] = os.getcwd() + kwargs['tmp_file_dir']                             
    kwargs['run_params_dir_name'] = utils_functions.get_training_dir_name(**kwargs)

    # Set world size to number of GPUs in system available to CUDA
    world_size = torch.cuda.device_count()
        
    # Determine if past or future is being autoencoded
    if kwargs['mini_batch_window_size'] > kwargs['decode_samples']: porf = "future"
    else: porf = "present" 
    
    # How compressed is the latent space compared to output of decoder
    # comp = round(kwargs['latent_dim'] /(kwargs['decode_samples'] * kwargs['num_channels'])  , 3)
    run_notes = utils_functions.make_runnotes(porf=porf, **kwargs) 

    # Call the initialization script to start new run or continue existing run
    start_eon, start_epoch, kwargs = utils_functions.initialize_run(run_notes=run_notes, **kwargs)

    # *** RUN ***
    # This will run training for only inference, new run, or continue run 
    for eon in range(start_eon, kwargs['num_eons']):

        print("Attempting to kill any remaining GPU processes")
        os.system('nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -n1 kill -9')

        # Get the timestamp ID for this run (will be used to resume wandb logging if this is a restarted training)
        s = kwargs['model_dir'].split("/")[-1]
        timestamp_id = ''.join(map(str, s))
        run_name = '_'.join(map(str,s.split('_')[0:2]))

        # Spawn subprocesses with start/join (mp.spawn causes memory sigdev errors??)
        ctx = mp.get_context('spawn') # necessary to use context if have set_start_method anove?
        children = []
        for i in range(world_size):
            subproc = ctx.Process(target=main, args=(i, world_size, eon, start_epoch, kwargs, run_name, timestamp_id), kwargs=kwargs)
            children.append(subproc)
            subproc.start()

        for i in range(world_size):
            children[i].join()

        # If within the training loop, pull previous models from tmp directory (must reset here because previous paths could be otherwise set for a continuation run)
        kwargs['model_state_dict_prev_path'] = kwargs['tmp_model_dir'] + '/model_state_dict.pt'
        kwargs['opt_state_dict_prev_path'] = kwargs['tmp_model_dir'] + '/opt_state_dict.pt'


    print("End of script")