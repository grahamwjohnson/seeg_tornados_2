'''
@author: grahamwjohnson
Developed between 2023-2025

Main script to train the Brain State Embedder (BSE)
'''

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.utils.data import DataLoader
import sys  
import os
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
from utilities import loss_functions
from data import SEEG_Tornado_Dataset
from models.GMVAE import GMVAE, GaussianProcessPrior, Discriminator

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
    num_dataloader_workers,

    # GM-VAE Optimizer
    posterior_weight_decay, 
    adamW_beta1,
    adamW_beta2,

    # Prior Optimizer
    prior_weight_decay, 
    adamW_beta1_prior, 
    adamW_beta2_prior,

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

    """
    Initializes training objects, including datasets, dataloaders, the GM-VAE model, and its optimizer.
    Splits patient data into training and validation sets, configures datasets, and sets up the model and optimizer.

    Args:
        gpu_id (int): GPU ID for training.
        pat_dir (str): Directory containing patient data.
        train_val_pat_perc (list): Percentage split for training and validation patients.
        intrapatient_dataset_style (list): Dataset style and split (e.g., [1, 0] for all data, train split).
        train_hour_dataset_range (list): Time range for training data.
        val_finetune_hour_dataset_range (list): Time range for validation finetune data.
        val_unseen_hour_dataset_range (list): Time range for validation unseen data.
        num_dataloader_workers (int): Number of workers for dataloaders.
        posterior_weight_decay (float): Weight decay for posterior parameters.
        adamW_beta1, adamW_beta2 (float): AdamW optimizer parameters for posterior.
        prior_weight_decay (float): Weight decay for prior parameters.
        adamW_beta1_prior, adamW_beta2_prior (float): AdamW optimizer parameters for prior.
        classifier_weight_decay (float): Weight decay for classifier parameters.
        classifier_adamW_beta1, classifier_adamW_beta2 (float): AdamW optimizer parameters for classifier.
        train_num_rand_hashes (int): Random hashes for training dataset.
        val_num_rand_hashes (int): Random hashes for validation dataset.
        train_forward_passes (int): Forward passes for training dataset.
        valfinetune_forward_passes (int): Forward passes for validation finetune dataset.
        valunseen_forward_passes (int): Forward passes for validation unseen dataset.
        **kwargs: Additional arguments for dataset and model configuration.

    Returns:
        train_dataset, train_dataloader: Training dataset and dataloader.
        valfinetune_dataset, valfinetune_dataloader: Validation finetune dataset and dataloader.
        valunseen_dataset, valunseen_dataloader: Validation unseen dataset and dataloader.
        gmvae: Initialized GM-VAE model.
        opt_gmvae: Optimizer for the GM-VAE model.
    """

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
         
    ### GMVAE ###
    gmvae = GMVAE(gpu_id=gpu_id, **kwargs) 
    gmvae = gmvae.to(gpu_id) 

    ### DISCRIMINATOR ###
    disc = Discriminator(gpu_id=gpu_id, **kwargs)
    disc = disc.to(gpu_id)

    # Separate the parameters into two groups
    gmvae_params = []
    prior_params = []
    classifier_params = []

    # Iterate through the model parameters
    for name, param in gmvae.named_parameters():
        # Check if the parameter is part of the encoder submodule
        if 'adversarial_classifier' in name:
            classifier_params.append(param)
        elif 'prior' in name:
            prior_params.append(param)
        else:
            gmvae_params.append(param)

    # Separate the parameters
    param_groups = [
        {"params": gmvae_params, "lr":  kwargs['LR_min_posterior'], "weight_decay": posterior_weight_decay, "betas": (adamW_beta1, adamW_beta2)},  
        {"params": prior_params, "lr":  kwargs['LR_min_prior'], "weight_decay": prior_weight_decay, "betas": (adamW_beta1_prior, adamW_beta2_prior)},  
        {"params": classifier_params, "lr": kwargs['LR_min_classifier'], "weight_decay": classifier_weight_decay, "betas":(classifier_adamW_beta1, classifier_adamW_beta2)}  
    ]
    opt_gmvae = torch.optim.AdamW(param_groups)
    opt_disc = torch.optim.AdamW(disc.parameters())

    return train_dataset, train_dataloader, valfinetune_dataset, valfinetune_dataloader, valunseen_dataset, valunseen_dataloader, gmvae, disc, opt_gmvae, opt_disc

def main(  
    # Ordered variables
    gpu_id: int, 
    world_size: int, 
    config, # aka kwargs
    
    # Passed by kwargs
    run_name: str,
    timestamp_id: int,
    start_epoch: int,
    LR_val_gmvae: float,
    LR_val_prior: float,
    LR_val_cls: float,
    LR_val_disc: float,
    val_finetine_bool: bool,
    finetune_inference: bool, 
    finetune_inference_epochs: int,
    singlebatch_printing_interval_val,
    singlebatch_printing_interval_train,
    gmvae_state_dict_prev_path = [],
    gmvae_opt_state_dict_prev_path = [],
    disc_state_dict_prev_path = [],
    disc_opt_state_dict_prev_path = [],
    running_mean_path = [],
    running_logvar_path = [],
    running_zmeaned_path = [],
    running_mogpreds_path = [],
    running_patidxs_path = [],
    epochs_to_train: int = -1,
    **kwargs):
    
    """
    Main training loop for the GM-VAE model with support for distributed data parallelism (DDP) 
    and model inference.

    This script handles the entire process of training, validation, and inference for the GM-VAE model. 
    It initializes key components such as datasets, models, and optimizers, and sets up the Distributed Data 
    Parallel (DDP) framework to facilitate multi-GPU training. The training process includes optional fine-tuning 
    of the model on validation data, as well as inference on the entire dataset to export embeddings.

    Key Features:
    - Initializes WandB for experiment tracking.
    - Handles multi-GPU distributed training with DDP setup.
    - Loads and saves model checkpoints during training.
    - Supports fine-tuning on validation datasets before inference.
    - Exports embeddings for multiple datasets after training.
    - Manages the training and validation loops, including dynamic learning rate adjustments.
    - Handles data generators and ensures that the model continues training from previous states when necessary.

    Arguments:
    - gpu_id: GPU identifier for multi-GPU training.
    - world_size: Number of GPUs used in training.
    - config: Configuration dictionary with hyperparameters and settings.
    - run_name, timestamp_id, and other parameters for experiment tracking and model fine-tuning.
    - Various paths for loading pre-trained model weights, running statistics, and checkpoints.

    Functions:
    - Main loop through epochs, including training, validation, and inference stages.
    - Dynamic loading and saving of model weights.
    - Regular saving of checkpoints and management of data generators.
    """

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
    train_dataset, train_dataloader, valfinetune_dataset, valfinetune_dataloader, valunseen_dataset, valunseen_dataloader, gmvae, disc, opt_gmvae, opt_disc = load_train_objs(gpu_id=gpu_id, **kwargs) 
    
    # Load the model/opt states if not first epoch & if in training mode
    if (start_epoch > 0):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}

        # Load in GMVAE weights and opts
        gmvae_state_dict_prev = torch.load(gmvae_state_dict_prev_path, map_location=map_location)
        gmvae.load_state_dict(gmvae_state_dict_prev)
        gmvae_opt_state_dict_prev = torch.load(gmvae_opt_state_dict_prev_path, map_location=map_location)
        opt_gmvae.load_state_dict(gmvae_opt_state_dict_prev)
        print(f"[GPU{gpu_id}] GM-VAE Model and Opt weights loaded from checkpoints")

        # Load in Discriminator weights and opts
        disc_state_dict_prev = torch.load(disc_state_dict_prev_path, map_location=map_location)
        disc.load_state_dict(disc_state_dict_prev)
        disc_opt_state_dict_prev = torch.load(disc_opt_state_dict_prev_path, map_location=map_location)
        opt_disc.load_state_dict(disc_opt_state_dict_prev)
        print(f"[GPU{gpu_id}] Discriminator and Opt weights loaded from checkpoints")

        # Load running mean 
        with open(running_mean_path, "rb") as f: accumulated_mean = pickle.load(f)
        print(f"[GPU{gpu_id}] Running Means loaded from checkpoints")

        # Load running logvar
        with open(running_logvar_path, "rb") as f: accumulated_logvar = pickle.load(f)
        print(f"[GPU{gpu_id}] Running Logvars loaded from checkpoints")

        # Load running zmeaned
        with open(running_zmeaned_path, "rb") as f: accumulated_zmeaned = pickle.load(f)
        print(f"[GPU{gpu_id}] Running Z Token-Averaged loaded from checkpoints")

        # Load running mog predictions
        with open(running_mogpreds_path, "rb") as f: accumulated_mogpreds = pickle.load(f)
        print(f"[GPU{gpu_id}] Running MoG Predictions loaded from checkpoints")

        # Load running patidxs
        with open(running_patidxs_path, "rb") as f: accumulated_patidxs = pickle.load(f)
        print(f"[GPU{gpu_id}] Running MoG Predictions loaded from checkpoints")
    
    else:
        accumulated_mean = []
        accumulated_logvar = []
        accumulated_zmeaned = []
        accumulated_mogpreds = []
        accumulated_patidxs = []

    # Create the training object
    trainer = Trainer(
        world_size=world_size,
        gpu_id=gpu_id, 
        gmvae=gmvae, 
        disc=disc,
        opt_gmvae=opt_gmvae,
        opt_disc=opt_disc,
        start_epoch=start_epoch,
        train_dataset=train_dataset, 
        train_dataloader=train_dataloader,
        valfinetune_dataset=valfinetune_dataset,
        valfinetune_dataloader=valfinetune_dataloader,
        valunseen_dataset=valunseen_dataset,
        valunseen_dataloader=valunseen_dataloader,
        wandb_run=wandb_run,
        finetune_inference=finetune_inference,
        accumulated_mean=accumulated_mean,
        accumulated_logvar=accumulated_logvar,
        accumulated_zmeaned=accumulated_zmeaned,
        accumulated_mogpreds=accumulated_mogpreds,
        accumulated_patidxs=accumulated_patidxs,
        **kwargs)

    # Kill the val data generators if val_every is set artificially high
    if (kwargs['val_every'] > 999) & (kwargs['inference_every'] > 999) & (trainer.valfinetune_dataset != None) & (trainer.valunseen_dataset != None):
        print(f"[GPU{str(trainer.gpu_id)}] WARNING: val_every is {kwargs['val_every']}, which is over arbitrary limit of 999, thus going to kill val generators")
        trainer.valfinetune_dataset.kill_generator()
        trainer.valunseen_dataset.kill_generator()
    
    # Main loop through all epochs
    for epoch in range(start_epoch, epochs_to_train):
        trainer.epoch = epoch

        # INFERENCE on all data
        if (epoch > 0) & ((trainer.epoch + 1) % trainer.inference_every == 0):

            if trainer.valunseen_dataset != None: trainer.valunseen_dataset.kill_generator()
            trainer.train_dataset.kill_generator()
            print(f"[GPU{trainer.gpu_id}] Killed Train/ValUnseen random generators for inference mode")

            # Save pre-finetune model/opt weights
            if finetune_inference:
                gmvae_dict = trainer.gmvae.module.state_dict()
                disc_dict = trainer.disc.module.state_dict()
                gmvae_opt_dict = trainer.opt_gmvae.state_dict()
                disc_opt_dict = trainer.opt_disc.state_dict()
                accumulated_mean = trainer.accumulated_mean
                accumulated_logvar = trainer.accumulated_logvar
                accumulated_zmeaned = trainer.accumulated_zmeaned
                accumulated_mogpreds = trainer.accumulated_mogpreds
                accumulated_patidxs = trainer.accumulated_patidxs

                # FINETUNE on beginning of validation patients (currently only one epoch)
                # Set to train and change LR to validate settings
                trainer._set_to_train()
                trainer.opt_gmvae.param_groups[0]['lr'] = LR_val_gmvae
                trainer.opt_gmvae.param_groups[1]['lr'] = LR_val_prior
                trainer.opt_gmvae.param_groups[2]['lr'] = LR_val_cls
                trainer.opt_disc['lr'] = LR_val_disc
                for finetune_epoch in range(finetune_inference_epochs):
                    trainer.epoch = epoch + finetune_epoch
                    trainer._run_train_epoch(
                        dataloader_curr = trainer.valfinetune_dataloader, 
                        dataset_string = "valfinetune",
                        val_finetune = True,
                        val_unseen = False,
                        singlebatch_printing_interval = singlebatch_printing_interval_val,
                        **kwargs)

                # Finetuned, so setup inference for all datasets
                dataset_list = [trainer.train_dataset, trainer.valfinetune_dataset, trainer.valunseen_dataset]
                dataset_strs = ["train", "valfinetune", "valunseen"]

            else: # No finetune, only run data for Train and ValUnseen datasets (running on val_finetune would just create confusion later when using embeddings)
                dataset_list = [trainer.train_dataset, trainer.valunseen_dataset]
                dataset_strs = ["train", "valunseen"]
            
            # INFERENCE on all selected datasets, kill finetune now that we also do not need it
            if trainer.valfinetune_dataset != None: trainer.valfinetune_dataset.kill_generator()
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
                trainer.gmvae.module.load_state_dict(gmvae_dict)
                trainer.disc.module.load_state_dict(disc_dict)
                trainer.opt_gmvae.load_state_dict(gmvae_opt_dict)
                trainer.opt_disc.load_state_dict(disc_opt_dict)
                trainer.accumulated_mean = accumulated_mean
                trainer.accumulated_logvar = accumulated_logvar
                trainer.accumulated_zmeaned = accumulated_zmeaned
                trainer.accumulated_mogpreds = accumulated_mogpreds
                trainer.accumulated_patidxs = accumulated_patidxs

            
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
            singlebatch_printing_interval = singlebatch_printing_interval_train,
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
                gmvae_dict = trainer.gmvae.module.state_dict()
                disc_dict = trainer.disc.module.state_dict()
                gmvae_opt_dict = trainer.opt_gmvae.state_dict()
                disc_opt_dict = trainer.opt_disc.state_dict()
                accumulated_mean = trainer.accumulated_mean
                accumulated_logvar = trainer.accumulated_logvar
                accumulated_zmeaned = trainer.accumulated_zmeaned
                accumulated_mogpreds = trainer.accumulated_mogpreds
                accumulated_patidxs = trainer.accumulated_patidxs

                # FINETUNE on beginning of validation patients (currently only one epoch)
                # Set to train and change LR to validate settings
                trainer._set_to_train()
                trainer.opt_gmvae.param_groups[0]['lr'] = LR_val_gmvae
                trainer.opt_gmvae.param_groups[1]['lr'] = LR_val_prior
                trainer.opt_gmvae.param_groups[2]['lr'] = LR_val_cls
                trainer.opt_disc['lr'] = LR_val_disc
                trainer._run_train_epoch(
                    dataloader_curr = trainer.valfinetune_dataloader, 
                    dataset_string = "valfinetune",
                    val_finetune = True,
                    val_unseen = False,
                    singlebatch_printing_interval = singlebatch_printing_interval_val,
                    **kwargs)

            # UNSEEN portion of validation patients 
            trainer._set_to_eval()
            with torch.inference_mode():
                trainer._run_train_epoch(
                    dataloader_curr = trainer.valunseen_dataloader, 
                    dataset_string = "valunseen",
                    val_finetune = False,
                    val_unseen = True,
                    singlebatch_printing_interval = singlebatch_printing_interval_val,
                    **kwargs)

            # Restore model/opt weights to pre-finetune
            if val_finetine_bool:
                trainer.gmvae.module.load_state_dict(gmvae_dict)
                trainer.disc.module.load_state_dict(disc_dict)
                trainer.opt_gmvae.load_state_dict(gmvae_opt_dict)
                trainer.opt_disc.load_state_dict(disc_opt_dict)
                trainer.accumulated_mean = accumulated_mean
                trainer.accumulated_logvar = accumulated_logvar
                trainer.accumulated_zmeaned = accumulated_zmeaned
                trainer.accumulated_mogpreds = accumulated_mogpreds
                trainer.accumulated_patidxs = accumulated_patidxs

    # Kill the process after training loop completes
    print(f"[GPU{gpu_id}]: End of train loop, killing 'main' subprocess")
    wandb.finish()
    destroy_process_group() 

class Trainer:
    def __init__(
        self,
        world_size: int,
        gpu_id: int,
        gmvae: torch.nn.Module,
        disc: torch.nn.Module,
        start_epoch: int,
        train_dataset: SEEG_Tornado_Dataset,
        valfinetune_dataset: SEEG_Tornado_Dataset,
        valunseen_dataset: SEEG_Tornado_Dataset,
        train_dataloader: DataLoader,
        valfinetune_dataloader: DataLoader,
        valunseen_dataloader: DataLoader,
        opt_gmvae: torch.optim.Optimizer,
        opt_disc: torch.optim.Optimizer,
        wandb_run,
        model_dir: str,
        val_every: int,
        inference_every: int,
        finetune_inference: bool,
        latent_dim: int,
        prior_mog_components: int,
        encode_token_samples: int,
        num_samples: int,
        transformer_seq_length: int,
        transformer_start_pos: int,
        FS: int,
        hash_output_range: tuple,
        intrapatient_dataset_style: list,
        atd_file: str,
        inference_window_sec_list: list,
        inference_stride_sec_list: list,
        recent_display_iters: int,
        total_collected_latents: int,
        classifier_num_pats: int,
        accumulated_mean,
        accumulated_logvar,
        accumulated_zmeaned,
        accumulated_mogpreds,
        accumulated_patidxs,
        **kwargs
    ) -> None:
        self.world_size = world_size
        self.gpu_id = gpu_id
        self.gmvae = gmvae
        self.disc = disc
        self.start_epoch = start_epoch
        self.train_dataset = train_dataset
        self.valfinetune_dataset = valfinetune_dataset
        self.valunseen_dataset = valunseen_dataset
        self.train_dataloader = train_dataloader
        self.valfinetune_dataloader = valfinetune_dataloader
        self.valunseen_dataloader = valunseen_dataloader
        self.opt_gmvae = opt_gmvae
        self.opt_disc = opt_disc
        self.model_dir = model_dir
        self.val_every = val_every
        self.inference_every = inference_every
        self.finetune_inference = finetune_inference
        self.latent_dim = latent_dim
        self.prior_mog_components = prior_mog_components
        self.encode_token_samples = encode_token_samples
        self.num_samples = num_samples
        self.transformer_seq_length = transformer_seq_length
        self.transformer_start_pos = transformer_start_pos
        self.FS = FS
        self.hash_output_range = hash_output_range
        self.intrapatient_dataset_style = intrapatient_dataset_style
        self.curr_LR_posterior = -1
        self.atd_file = atd_file
        self.inference_window_sec_list = inference_window_sec_list
        self.inference_stride_sec_list = inference_stride_sec_list
        self.recent_display_iters = recent_display_iters
        self.total_collected_latents = total_collected_latents
        self.classifier_num_pats = classifier_num_pats
        self.accumulated_mean = accumulated_mean
        self.accumulated_logvar = accumulated_logvar
        self.accumulated_zmeaned = accumulated_zmeaned
        self.accumulated_mogpreds = accumulated_mogpreds
        self.accumulated_patidxs = accumulated_patidxs
        self.wandb_run = wandb_run
        self.kwargs = kwargs

        """
        Initialization method for the GM-VAE training object.

        This method sets up all necessary configurations for training a GM-VAE model 
        in a distributed environment. It initializes model parameters, datasets, 
        dataloaders, optimization settings, and various training-related variables. 
        Additionally, it sets up the model for Distributed Data Parallel (DDP) and 
        prepares accumulated variables to track and manage latent data throughout 
        the training process.

        Key Initializations:
        - Distributed training setup with DDP.
        - Model and optimization state (GM-VAE, optimizer).
        - Guassian Process prior for temporal regularization across sequential tokens
        - Training and validation datasets, dataloaders, and sampling strategies.
        - Accumulated statistics for latent data and prior distribution.
        - Integration with WandB for model monitoring.

        Assertions and checks are performed to ensure correct configurations and 
        prevent potential runtime errors.
        """

        # Check variables that will cause delayed crashes 
        assert len(self.inference_window_sec_list) == len(self.inference_stride_sec_list)

        self.kl_weight = -1 # dummy variable, only needed when debugging and training is skipped

        # Number of iterations per file
        self.num_windows = int((self.num_samples - self.transformer_seq_length * self.encode_token_samples - self.encode_token_samples)/self.encode_token_samples) - 2

        # Set up gmvae & transformer with DDP
        self.gmvae = DDP(gmvae, device_ids=[gpu_id])   # find_unused_parameters=True
        self.disc = DDP(disc, device_ids=[gpu_id])
        self.gp_prior = GaussianProcessPrior(self.gpu_id, self.transformer_seq_length, self.latent_dim, **kwargs)
                    
        # Running Regulizer window for latent data
        if self.accumulated_mean == []:
            # If first initialziing, then start off with random
            self.accumulated_mean = torch.randn(self.total_collected_latents, self.prior_mog_components, self.latent_dim).to(self.gpu_id)
            self.accumulated_logvar = torch.randn(self.total_collected_latents, self.prior_mog_components, self.latent_dim).to(self.gpu_id)
            self.accumulated_zmeaned = torch.randn(self.total_collected_latents, self.latent_dim).to(self.gpu_id)
            self.accumulated_mogpreds = torch.softmax(torch.randn(self.total_collected_latents, self.prior_mog_components), dim=1).to(self.gpu_id)
            self.accumulated_patidxs = (torch.ones(self.total_collected_latents) * -1).to(self.gpu_id)
        else: 
            # Ensure on proper device because loading from pickle
            self.accumulated_mean = self.accumulated_mean.to(self.gpu_id)
            self.accumulated_logvar = self.accumulated_logvar.to(self.gpu_id)
            self.accumulated_zmeaned = self.accumulated_zmeaned.to(self.gpu_id)
            self.accumulated_mogpreds = self.accumulated_mogpreds.to(self.gpu_id)
            self.accumulated_patidxs = self.accumulated_patidxs.to(self.gpu_id)

        # Running tab of update index
        self.next_update_index = 0

        # Watch with WandB
        wandb.watch(self.gmvae)
        wandb.watch(self.disc)
        
    def _set_to_train(self):
        self.gmvae.train()

    def _set_to_eval(self):
        self.gmvae.eval()

    def _zero_all_grads(self):
        self.opt_gmvae.zero_grad()

    def _save_checkpoint(self, epoch, delete_old_checkpoints, **kwargs):
        """
        Saves a checkpoint of the GM-VAE model, optimizer, and accumulated statistics (means, logvars, etc.)
        at the specified epoch. Optionally deletes old checkpoints to save disk space.

        Args:
            epoch (int): Current epoch number.
            delete_old_checkpoints (bool): If True, deletes old checkpoints except those at the end of
                regularization annealing periods.
            **kwargs: Additional arguments for checkpoint deletion.

        Saves:
            - GM-VAE model weights and optimizer state.
            - Accumulated statistics: running means, logvars, z-meaned, MoG predictions, and patient indices.
        """

        print("CHECKPOINT SAVE")

        # Create new directory for this epoch
        base_checkpoint_dir = self.model_dir + f"/checkpoints"
        check_epoch_dir = base_checkpoint_dir + f"/Epoch_{str(epoch)}"

        print("Saving gmvae model weights")

        ### GMVAE CHECKPOINT 
        check_posterior_dir = check_epoch_dir + "/posterior_checkpoints"
        if not os.path.exists(check_posterior_dir): os.makedirs(check_posterior_dir)

        # Save GMVAE model
        ckp = self.gmvae.module.state_dict()
        check_path = check_posterior_dir + "/checkpoint_epoch" + str(epoch) + "_gmvae.pt"
        torch.save(ckp, check_path)

        # Save DISC model
        ckp = self.disc.module.state_dict()
        check_path = check_posterior_dir + "/checkpoint_epoch" + str(epoch) + "_disc.pt"
        torch.save(ckp, check_path)

        # Save GMVAE optimizer
        opt_ckp = self.opt_gmvae.state_dict()
        opt_path = check_posterior_dir + "/checkpoint_epoch" + str(epoch) + "_gmvae_opt.pt"
        torch.save(opt_ckp, opt_path)

        # Save DISC optimizer
        opt_ckp = self.opt_disc.state_dict()
        opt_path = check_posterior_dir + "/checkpoint_epoch" + str(epoch) + "_disc_opt.pt"
        torch.save(opt_ckp, opt_path)

        # Save DISC optimizer
        opt_ckp = self.opt_disc.state_dict()
        opt_path = check_posterior_dir + "/checkpoint_epoch" + str(epoch) + "_disc_opt.pt"
        torch.save(opt_ckp, opt_path)

        # Save Running Mean 
        latents_path = check_posterior_dir + "/checkpoint_epoch" +str(epoch) + "_running_means.pkl"
        output_obj = open(latents_path, 'wb')
        pickle.dump(self.accumulated_mean, output_obj)
        output_obj.close()
        print("Saved running means")

        # Save Running Logvar 
        latents_path = check_posterior_dir + "/checkpoint_epoch" +str(epoch) + "_running_logvars.pkl"
        output_obj = open(latents_path, 'wb')
        pickle.dump(self.accumulated_logvar, output_obj)
        output_obj.close()
        print("Saved running logvars")

        # Save Running Z
        latents_path = check_posterior_dir + "/checkpoint_epoch" +str(epoch) + "_running_zmeaned.pkl"
        output_obj = open(latents_path, 'wb')
        pickle.dump(self.accumulated_zmeaned, output_obj)
        output_obj.close()
        print("Saved running Z Token-Meaned")

        # Save Running MoG Predictions 
        latents_path = check_posterior_dir + "/checkpoint_epoch" +str(epoch) + "_running_mogpreds.pkl"
        output_obj = open(latents_path, 'wb')
        pickle.dump(self.accumulated_mogpreds, output_obj)
        output_obj.close()
        print("Saved running MoG Predictions")

        # Save Running Pat Idxs
        latents_path = check_posterior_dir + "/checkpoint_epoch" +str(epoch) + "_running_patidxs.pkl"
        output_obj = open(latents_path, 'wb')
        pickle.dump(self.accumulated_patidxs, output_obj)
        output_obj.close()
        print("Saved running Pat Idxs")

        print(f"Epoch {epoch} | Training checkpoint saved at {check_epoch_dir}")

        if delete_old_checkpoints:
            utils_functions.delete_old_checkpoints(dir = base_checkpoint_dir, curr_epoch = epoch, **kwargs)
            print("Deleted old checkpoints, except epochs at end of reguaization annealing period")
    
    def _update_reg_window(self, mean, logvar, zmeaned, mogpreds, patidxs):

        """
        This function collects and updates the most recent encoder outputs for plotting purposes. 
        Historically, it was used for maintaining a running regularization window, but the cost 
        became prohibitive after switching to outputting K means and K logvars. As a result, 
        this function is now only used for visualization of the encoder outputs.

        The function accumulates the following data:
        - `mean`: The means from the encoder output.
        - `logvar`: The log variances from the encoder output.
        - `zmeaned`: The reparameterized means for latent variable z.
        - `mogpreds`: The predictions from the mixture of Gaussians.
        - `patidxs`: The patient indices associated with the data.

        The accumulated data is stored in predefined arrays, and once the arrays are full, 
        the function rolls over to overwrite the oldest data. This ensures that the window 
        always contains the most recent `total_collected_latents` updates.

        ### Key Features:
        - **Window-based Accumulation:** The function maintains a running window of the 
        most recent encoder outputs.
        - **Rollover Mechanism:** When the window is full, older data is overwritten, 
        keeping the window size constant.
        - **Detachment for Backpropagation:** The accumulated data is detached from the 
        computation graph, allowing it to be used for plotting while preventing 
        gradients from flowing through it in subsequent backward passes.

        ### Parameters:
        - `mean` (torch.Tensor): The means output by the encoder.
        - `logvar` (torch.Tensor): The log variances output by the encoder.
        - `zmeaned` (torch.Tensor): The reparameterized means of the latent variable z.
        - `mogpreds` (torch.Tensor): The mixture of Gaussian predictions from the encoder.
        - `patidxs` (torch.Tensor): The patient indices associated with the current batch.

        ### Outputs:
        - The function updates the accumulated encoder outputs, which are stored in 
        `self.accumulated_mean`, `self.accumulated_logvar`, `self.accumulated_zmeaned`, 
        `self.accumulated_mogpreds`, and `self.accumulated_patidxs`.

        ### Notes:
        - **Averaging:** This function requires that the encoder outputs be averaged across tokens before being passed in.
        - **Plotting Purpose:** The primary purpose of this function is for visualization and monitoring of the encoder's behavior.
        - **Memory Efficiency:** By detaching the accumulated values, the function ensures that 
        memory is managed efficiently while still allowing the data to be used for plotting.

        """

        num_new_updates = mean.shape[0]
        if (self.next_update_index + num_new_updates) < self.total_collected_latents:
            self.accumulated_mean[self.next_update_index: self.next_update_index + num_new_updates, :, :] = mean
            self.accumulated_logvar[self.next_update_index: self.next_update_index + num_new_updates, :, :] = logvar
            self.accumulated_zmeaned[self.next_update_index: self.next_update_index + num_new_updates, :] = zmeaned
            self.accumulated_mogpreds[self.next_update_index: self.next_update_index + num_new_updates, :] = mogpreds
            self.accumulated_patidxs[self.next_update_index: self.next_update_index + num_new_updates] = patidxs
            self.next_update_index = self.next_update_index + num_new_updates

        else: # Rollover
            residual_num = (self.next_update_index + num_new_updates) % self.total_collected_latents
            end_num = num_new_updates - residual_num

            self.accumulated_mean[self.next_update_index: self.next_update_index + end_num, :] = mean[:end_num, :, :]
            self.accumulated_mean[0: residual_num, :] = mean[end_num:, :, :]

            self.accumulated_logvar[self.next_update_index: self.next_update_index + end_num, :] = logvar[:end_num, :, :]
            self.accumulated_logvar[0: residual_num, :] = logvar[end_num:, :, :]

            self.accumulated_zmeaned[self.next_update_index: self.next_update_index + end_num, :] = zmeaned[:end_num, :]
            self.accumulated_zmeaned[0: residual_num, :] = zmeaned[end_num:, :]

            self.accumulated_mogpreds[self.next_update_index: self.next_update_index + end_num, :] = mogpreds[:end_num, :]
            self.accumulated_mogpreds[0: residual_num, :] = mogpreds[end_num:, :]

            self.accumulated_patidxs[self.next_update_index: self.next_update_index + end_num] = patidxs[:end_num]
            self.accumulated_patidxs[0: residual_num] = patidxs[end_num:]

            self.next_update_index = residual_num

        # Detach to allow next backpass
        self.accumulated_mean = self.accumulated_mean.detach() 
        self.accumulated_logvar = self.accumulated_logvar.detach() 
        self.accumulated_zmeaned = self.accumulated_zmeaned.detach() 
        self.accumulated_mogpreds = self.accumulated_mogpreds.detach() 
        self.accumulated_patidxs = self.accumulated_patidxs.detach() 

    def remove_padded_channels(self, x: torch.Tensor, hash_channel_order):
        """
        Removes padded channels (-1) from x for each sample in the batch.
        
        Args:
            x: Input tensor of shape [batch, tokens, max_channels, seq_len].
            hash_channel_order: List of lists where -1 indicates padded channels.
        
        Returns:
            List of tensors with padded channels removed (one per batch sample).
        """
        # Convert hash_channel_order to a tensor
        channel_order_tensor = torch.tensor(hash_channel_order, dtype=torch.long).to(self.gpu_id)  # [batch, max_channels]
        valid_mask = (channel_order_tensor != -1)  # [batch, max_channels]

        # Filter out padded channels per sample
        x_nopad = []
        for i in range(x.shape[0]):  # Loop over batch
            x_nopad.append(x[i, :, valid_mask[i], :])  # [tokens, valid_channels, seq_len]

        return x_nopad

    def _run_export_embeddings(
        self, 
        dataset_curr, 
        dataset_string,
        num_dataloader_workers,
        max_batch_size,
        padded_channels,
        **kwargs):

        """
        This function runs inference with the GMVAE encoder (only the encoder) and saves 
        the resulting latent space embeddings for each file, in the form of pickle files. 
        Unlike the `_run_train_epoch` function, which pulls random data, this function 
        iterates sequentially through the entire dataset (one patient at a time). 

        The function processes the dataset and saves latent embeddings using the GMVAE encoder 
        for each patient, file, and window-stride configuration. The embeddings are smoothed 
        according to specified window and stride sizes.

        ### Key Features:
        - **Sequential Inference:** This function sequentially processes data for each patient 
        and file, as opposed to random data pulls.
        - **Latent Embedding Calculation:** It calculates latent embeddings for each file's 
        data and saves them as pickle files.
        - **Window-Stride Smoothing:** The latent embeddings are smoothed according to the 
        window and stride sizes defined by `inference_window_sec_list` and 
        `inference_stride_sec_list`.
        - **Conditioning:** Hashes for each patient and channel are generated to condition 
        the feedforward process.
        - **GPU Support:** The function processes the data on GPU for efficient computations.

        ### Parameters:
        - `dataset_curr` (Dataset): The dataset object that holds the current patient data 
        to be processed.
        - `dataset_string` (str): A string identifier for the dataset (for tracking or naming).
        - `attention_dropout` (float): Dropout rate for attention layers (though not directly 
        used in this function).
        - `num_dataloader_workers` (int): Number of worker threads for loading data batches.
        - `max_batch_size` (int): The maximum batch size used for inference.
        - `inference_batch_mult` (float): A multiplier for adjusting batch size during inference.
        - `padded_channels` (int): The number of padded channels in the data tensor (to 
        accommodate variable input sizes).
        - `**kwargs` (dict): Additional arguments that may be passed (not used directly here).

        ### Outputs:
        - For each patient, the function processes their data and saves the computed latent 
        embeddings into pickle files. 
        - The files are organized according to the window and stride configurations, with 
        directories and filenames reflecting these settings (e.g., `latent_60secWindow_30secStride.pkl`).
        - Each file’s latent space embeddings are saved after the GMVAE encoder processes the 
        raw data windows with the specified stride and window sizes.

        ### Notes:
        - **Data Processing:** The function processes files one by one per patient. Data for 
        each file is passed through the GMVAE encoder to extract the latent space.
        - **Windowing and Striding:** After processing each file, the latent space is divided 
        into overlapping windows based on the stride and window settings, smoothing the 
        embeddings as necessary.
        - **File Structure:** Latent embeddings are saved in directories named after the 
        window and stride sizes, and each file's latent embeddings are saved separately.

        """

        print(f"[GPU{self.gpu_id}] encode_token_samples: {self.encode_token_samples}")

        ### ALL/FULL FILES - LATENT ONLY - FULL TRANSFORMER CONTEXT ### 
        print("WARNING: Setting alpha = 0")
        self.classifier_alpha = 0

        # Go through every subject in this dataset
        for pat_idx in range(0,len(dataset_curr.pat_ids)):
            dataset_curr.set_pat_curr(pat_idx)
            _, pat_id_curr, _, _ = dataset_curr.get_pat_curr()
            dataloader_curr =  utils_functions.prepare_dataloader(dataset_curr, batch_size=max_batch_size, num_workers=num_dataloader_workers)

            # Go through every file in dataset
            file_count = 0
            for data_tensor, file_name, _ in dataloader_curr: # Hash done outside data.py for single pat inference

                file_count = file_count + len(file_name)

                num_channels_curr = data_tensor.shape[1]

                # Create the sequential latent sequence array for the file 
                num_samples_in_forward = self.transformer_seq_length * self.encode_token_samples
                num_windows_in_file = data_tensor.shape[2] / num_samples_in_forward
                assert (num_windows_in_file % 1) == 0
                num_windows_in_file = int(num_windows_in_file)
                num_samples_in_forward = int(num_samples_in_forward)

                # Prep the output tensor and put on GPU
                files_means = torch.zeros([data_tensor.shape[0], num_windows_in_file, self.latent_dim]).to(self.gpu_id)
                files_logvars = torch.zeros([data_tensor.shape[0], num_windows_in_file, self.latent_dim]).to(self.gpu_id)
                files_mogpreds = torch.zeros([data_tensor.shape[0], num_windows_in_file, self.prior_mog_components]).to(self.gpu_id)

                # Put whole file on GPU
                data_tensor = data_tensor.to(self.gpu_id)

                for w in range(num_windows_in_file):
                    
                    # Print Status
                    print_interval = 100
                    if (self.gpu_id == 0) & (w % print_interval == 0):
                        sys.stdout.write(f"\r{dataset_string}: Pat {pat_idx}/{len(dataset_curr.pat_ids)-1}, File {file_count - len(file_name)}:{file_count}/{len(dataset_curr)/self.world_size - 1}  * GPUs (DDP), Intrafile Iter {w}/{num_windows_in_file}          ") 
                        sys.stdout.flush() 
                    
                    x = torch.zeros(data_tensor.shape[0], self.transformer_seq_length, padded_channels, self.encode_token_samples).to(self.gpu_id)

                    # PAD THE CHANNELS ACCORDING TO HASH
                    # NOTE: Just running for modifier '0' for inference
                    inference_modifier = 0
                    _, hash_channel_order = utils_functions.hash_to_vector(
                        input_string = pat_id_curr,
                        num_channels = num_channels_curr,
                        padded_channels = padded_channels,
                        latent_dim = self.latent_dim,
                        modifier = inference_modifier, 
                        hash_output_range = self.hash_output_range)

                    raise Exception("Need to pad and order channels according to hash")

                    start_idx = w * num_samples_in_forward
                    for embedding_idx in range(0, self.transformer_seq_length):
                        # Pull out data for this window - NOTE: no hashing random order, channels just put in order with zero pad at end
                        end_idx = start_idx + self.encode_token_samples * embedding_idx + self.encode_token_samples 
                        x[:, embedding_idx, :num_channels_curr, :] = data_tensor[:, :, end_idx-self.encode_token_samples : end_idx]

                    ### GMVAE ENCODER
                    # Forward pass in stacked batch through GMVAE encoder
                    # latent, _, _ = self.gmvae(x[:, :-1, :, :], reverse=False, alpha=self.classifier_alpha)   # 1 shifted just to be aligned with training style
                    _, mean_pseudobatch, logvar_pseudobatch, mogpreds_pseudobatch, _ = self.gmvae(x, reverse=False) # No shift if not causal masking
                    
                    # Theoretical Levels of detail to save (Token or Token-Meaned level):
                    # 1) Save mogpreds [CURRENTLY SAVED at Token-Mean level]
                    # 2) Save the weighted means from each component [CURRENTLY SAVED at Token-Mean level]
                    # 3) Save weighted means and uncertainty with weighted logvars [CURRENTLY SAVED at Token-Mean level]
                    # 4) Or just save one component of interest 

                    # Reshape back to token level 
                    mogpreds = mogpreds_pseudobatch.split(self.transformer_seq_length, dim=0)
                    mogpreds = torch.stack(mogpreds, dim=0)
                    mean = mean_pseudobatch.split(self.transformer_seq_length, dim=0)
                    mean = torch.stack(mean, dim=0)
                    logvar = logvar_pseudobatch.split(self.transformer_seq_length, dim=0)
                    logvar = torch.stack(logvar, dim=0)

                    # Weight the means using mogpreds
                    # mogpreds shape: [batch_size, seq_length, num_components]
                    # mean shape: [batch_size, seq_length, num_components, latent_dim]
                    # Expand mogpreds to match the latent_dim for broadcasting
                    mogpreds_expanded = mogpreds.unsqueeze(-1)  # Shape: [batch_size, seq_length, num_components, 1]

                    # Weight the means & logvars
                    weighted_means = mean * mogpreds_expanded  # Shape: [batch_size, seq_length, num_components, latent_dim]
                    weighted_logvars = logvar * mogpreds_expanded 

                    # Sum over the components to get the final weighted mean for each token
                    weighted_means_summed = torch.sum(weighted_means, dim=2)  # Shape: [batch_size, seq_length, latent_dim]
                    weighted_logvars_summed = torch.sum(weighted_logvars, dim=2)  # Shape: [batch_size, seq_length, latent_dim]

                    # Take the mean across the token dimension (optional, if you want to reduce further)
                    mean_tokenmeaned = torch.mean(weighted_means_summed, dim=1)  # Shape: [batch_size, latent_dim]
                    logvar_tokenmeaned = torch.mean(weighted_logvars_summed, dim=1)  # Shape: [batch_size, latent_dim]

                    # Save the results
                    files_means[:, w, :] = mean_tokenmeaned
                    files_logvars[:, w, :] = logvar_tokenmeaned
                    files_mogpreds[:, w, :] = torch.mean(mogpreds, dim=1)  # Save the mean mogpreds across tokens

                # After file complete, pacmap_window/stride the file and save each file from batch seperately
                # Seperate directory for each win/stride combination
                # First pull off GPU and convert to numpy
                files_means = files_means.cpu().numpy()
                files_logvars = files_logvars.cpu().numpy()
                files_mogpreds = files_mogpreds.cpu().numpy()
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
                    num_strides_in_file = int((files_means.shape[1] - num_latents_in_win) / num_latents_in_stride) 
                    windowed_file_means = np.zeros([data_tensor.shape[0], num_strides_in_file, self.latent_dim])
                    windowed_file_logvars = np.zeros([data_tensor.shape[0], num_strides_in_file, self.latent_dim])
                    windowed_file_mogpreds = np.zeros([data_tensor.shape[0], num_strides_in_file, self.prior_mog_components])
                    for s in range(num_strides_in_file):
                        windowed_file_means[:, s, :] = np.mean(files_means[:, s*num_latents_in_stride: s*num_latents_in_stride + num_latents_in_win, :], axis=1)
                        windowed_file_logvars[:, s, :] = np.mean(files_logvars[:, s*num_latents_in_stride: s*num_latents_in_stride + num_latents_in_win, :], axis=1)
                        windowed_file_mogpreds[:, s, :] = np.mean(files_mogpreds[:, s*num_latents_in_stride: s*num_latents_in_stride + num_latents_in_win, :], axis=1)

                    # Save each windowed latent in a pickle for each file
                    for b in range(data_tensor.shape[0]):
                        filename_curr = file_name[b]
                        save_dir = f"{self.model_dir}/latent_files/Epoch{self.epoch}/{win_sec_curr}SecondWindow_{stride_sec_curr}SecondStride/{dataset_string}"
                        if not os.path.exists(save_dir): os.makedirs(save_dir)
                        output_obj = open(f"{save_dir}/{filename_curr}_latent_{win_sec_curr}secWindow_{stride_sec_curr}secStride.pkl", 'wb')
                        save_dict = {
                            'windowed_weighted_means': windowed_file_means[b, :, :],
                            'windowed_weighted_logvars': windowed_file_logvars[b, :, :],
                            'windowed_mogpreds': windowed_file_mogpreds[b, :, :]}
                        pickle.dump(save_dict, output_obj)
                        output_obj.close()

    def _run_train_epoch( ### RANDOM SUBSET OF FILES ###    
        self, 
        dataloader_curr, 
        dataset_string,
        attention_dropout,
        val_finetune,
        val_unseen,
        singlebatch_latent_printing,
        singlebatch_printing_interval,
        **kwargs):

        """
        This function is responsible for training the model for one complete epoch. It iterates through the batches 
        in the given dataloader, performs forward and backward passes, computes loss, and updates model parameters 
        through gradient descent.

        The function performs the following steps:
        1. Fetches random subsets of data (random files, portions of files, etc.) from the DataLoader.
        2. Transfers data to the GPU for efficient computation.
        3. Updates learning rates and weights for various components of the model based on the current epoch and iteration.
        4. Passes the data through the GM-VAE encoder and decoder to compute the reconstructed output.
        5. Computes various loss components such as reconstruction loss, KL divergence, mean matching, logvar matching,
           posterior entropy loss, and adversarial loss.
        6. Backpropagates the gradients and updates the model parameters using the optimizer.
        7. Logs the training metrics and loss values to WandB for visualization and tracking.
        8. Optionally prints visualizations for latent representations, reconstructions, and attention weights for 
           debugging or inspection purposes.
        9. Optionally updates buffers that accumulate various statistics about the model’s predictions.
        10. Handles validation modes (fine-tuning or unseen data) where the classifier component is frozen, and no 
            backpropagation is performed for evaluation.

        Key Parameters:
        IMPORTANT: Takes in a *DataLoader*, not a *Dataset*

        - dataloader_curr: The DataLoader that provides the data for the current epoch. It is expected to return batches
          of data containing input tensors, file names, class labels, channel orders, and patient embeddings.
        - dataset_string: A string identifying the dataset being used (for logging and file saving purposes).
        - attention_dropout: A dropout rate applied during attention mechanisms to regularize the model.
        - val_finetune: A boolean indicating whether the model is being fine-tuned on validation data.
        - val_unseen: A boolean indicating whether the model is being evaluated on unseen data.
        - singlebatch_latent_printing: A boolean flag that determines whether latent representations for a single batch 
          should be printed.
        - singlebatch_printing_interval: The frequency (in terms of iterations) at which latent printing occurs.
        - kwargs: Additional keyword arguments that might include configuration parameters such as learning rate schedules, 
          regularization weights, or Gumbel-Softmax temperature.

        Returns:
        - None. The function performs in-place updates to the model and logs training metrics during the epoch.

        """

        self.curr_LR_discriminator = kwargs['LR_disc']

        print(f"[GPU{self.gpu_id}] encode_token_samples: {self.encode_token_samples}")

        iter_curr = 0
        total_iters = len(dataloader_curr)
        for x, file_name, file_class_label, hash_channel_order, _ in dataloader_curr: 

            # Put the data and labels on GPU
            x = x.to(self.gpu_id)
            file_class_label = file_class_label.to(self.gpu_id)
        
            # LR & WEIGHT SCHEDULES
            self.gumbel_softmax_temp, self.mean_match_weight, self.logvar_match_weight, self.mse_weight, self.kl_weight, self.gp_weight, self.curr_LR_posterior, self.curr_LR_prior, self.curr_LR_cls, self.posterior_mogpreds_entropy_weight, self.posterior_mogpreds_intersequence_diversity_weight, self.classifier_weight, self.classifier_alpha = utils_functions.LR_and_weight_schedules(
                epoch=self.epoch, iter_curr=iter_curr, iters_per_epoch=total_iters, **self.kwargs)
            
            if (not val_finetune) & (not val_unseen): 
                self.opt_gmvae.param_groups[0]['lr'] = self.curr_LR_posterior
                self.opt_gmvae.param_groups[1]['lr'] = self.curr_LR_prior
                self.opt_gmvae.param_groups[2]['lr'] = self.curr_LR_cls

                self.opt_disc.param_groups[0]['lr'] = self.curr_LR_discriminator
            else: 
                self.classifier_alpha = 0 # For validation, do not consider classifier
                self.opt_gmvae.param_groups[2]['lr'] = 0

            # Set the temperature in the model itself
            self.gmvae.module.set_temp(self.gumbel_softmax_temp)

            # Check for NaNs
            if torch.isnan(x).any(): raise Exception(f"ERROR: found nans in one of these files: {file_name}")

            # DISCRIMINATOR TRAINING
            total_disc_loss = 0
            for _ in range(kwargs['discriminator_training_iters']):
                self.opt_disc.zero_grad()
                with torch.no_grad():
                    z_posterior, _, _, _, _ = self.gmvae(x, reverse=False)
                    z_prior = self.gmvae.module.prior.sample_prior(z_posterior.shape[0])
                disc_loss = loss_functions.discriminator_loss(z_posterior, z_prior, self.disc)
                disc_loss.backward()
                self.opt_disc.step()
                total_disc_loss += disc_loss.item() 
            disc_loss = total_disc_loss / kwargs['discriminator_training_iters']

            # GM-VAE ENCODER: 
            z_pseudobatch, mean_pseudobatch, logvar_pseudobatch, mogpreds_pseudobatch, attW = self.gmvae(x, reverse=False) # No 1-shift if not causal masking

            # Reshape variables back to token level 
            z_token = z_pseudobatch.split(self.transformer_seq_length, dim=0)
            z_token = torch.stack(z_token, dim=0)
            mogpreds = mogpreds_pseudobatch.split(self.transformer_seq_length, dim=0)
            mogpreds = torch.stack(mogpreds, dim=0)
            mean = mean_pseudobatch.split(self.transformer_seq_length, dim=0)
            mean = torch.stack(mean, dim=0)
            logvar = logvar_pseudobatch.split(self.transformer_seq_length, dim=0)
            logvar = torch.stack(logvar, dim=0)

            # GMVAE DECODER - at Token Level
            x_hat = self.gmvae(z_token, reverse=True)  

            # CLASSIFIER - on the mean of Z
            z_tokenmeaned = torch.mean(z_token, dim=1)
            class_probs_mean_of_latent = self.gmvae.module.adversarial_classifier(z_tokenmeaned, alpha=self.classifier_alpha)
   
            # REMOVE PADDING - otherwise would reward recon loss for patients with fewer channels
            x_nopad_list = self.remove_padded_channels(x, hash_channel_order) 
            x_hat_nopad_list = self.remove_padded_channels(x_hat, hash_channel_order) 

            # LOSSES
            mse_loss = loss_functions.recon_loss(
                x=x_nopad_list, 
                x_hat=x_hat_nopad_list,
                mse_weight=self.mse_weight)
            
            gmvae_adversarial_loss = loss_functions.gmvae_adversarial_loss(
                z_posterior=z_pseudobatch,
                discriminator=self.disc,
                beta=self.kl_weight)
            
            neg_gp_log_prob = (-1) * self.gp_prior(z_token, self.gp_weight) # On sequential token-level of z posterior

            mean_match_loss = loss_functions.mean_matching_loss(
                mean_posterior = mean_pseudobatch,
                mean_prior = self.gmvae.module.prior.means,
                weight=self.mean_match_weight)

            logvar_match_loss = loss_functions.logvar_matching_loss(
                logvar_posterior = logvar_pseudobatch,
                logvar_prior = self.gmvae.module.prior.logvars,
                weight= self.logvar_match_weight) 

            posterior_mogpreds_entropy_loss = loss_functions.posterior_mogpreds_entropy_loss(
                mogpreds=mogpreds,
                posterior_mogpreds_entropy_weight=self.posterior_mogpreds_entropy_weight,
                **kwargs)

            posterior_mogpreds_intersequence_diversity_loss = loss_functions.posterior_mogpreds_intersequence_diversity_loss(
                mogpreds=mogpreds,
                weight=self.posterior_mogpreds_intersequence_diversity_weight)  

            prior_entropy = loss_functions.prior_entropy_regularization(
                weights = torch.softmax(self.gmvae.module.prior.weightlogits, dim=0), 
                logvars = self.gmvae.module.prior.logvars, 
                **kwargs)
            
            prior_repulsion = loss_functions.prior_repulsion_regularization(
                weights = torch.softmax(self.gmvae.module.prior.weightlogits, dim=0), 
                means = self.gmvae.module.prior.means, 
                **kwargs)

            patient_adversarial_loss = loss_functions.patient_adversarial_loss_function(
                probs=class_probs_mean_of_latent, 
                labels=file_class_label,
                classifier_weight=self.classifier_weight)

            # Accumulate all losses
            loss = mse_loss + gmvae_adversarial_loss, neg_gp_log_prob + mean_match_loss + logvar_match_loss + posterior_mogpreds_entropy_loss + posterior_mogpreds_intersequence_diversity_loss + prior_entropy + prior_repulsion + patient_adversarial_loss 

            # For plotting visualization purposes
            mean_tokenmeaned = torch.mean(mean, dim=1)
            logvar_tokenmeaned = torch.mean(logvar, dim=1)
            mogpreds_logsofmaxed_tokenmeaned = torch.mean(mogpreds, dim=1)

            # Do not backprop for pure validation (i.e. val unseen), but do for training & finetuning
            if not val_unseen: 
                self._zero_all_grads()
                loss.backward()    
                torch.nn.utils.clip_grad_norm_(self.gmvae.module.parameters(), max_norm=1.0)
                self.opt_gmvae.step()
                self._update_reg_window(mean_tokenmeaned, logvar_tokenmeaned, z_tokenmeaned, mogpreds_logsofmaxed_tokenmeaned, file_class_label) # Update the buffers & detach() 

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
                        train_recon_MSE_loss=mse_loss, 
                        train_discriminator_loss=disc_loss,
                        train_gmvae_adversarial_loss=gmvae_adversarial_loss,
                        train_neg_gp_log_prob = neg_gp_log_prob,
                        train_neg_gp_weight = self.gp_weight,
                        train_mean_match_loss = mean_match_loss,
                        train_logvar_match_loss = logvar_match_loss,
                        train_mean_match_weight = self.mean_match_weight,
                        train_logvar_match_weight = self.logvar_match_weight,
                        train_posterior_mogpreds_entropy_loss=posterior_mogpreds_entropy_loss,
                        train_posterior_mogpreds_intersequence_diversity_weight = self.posterior_mogpreds_intersequence_diversity_weight,
                        train_posterior_mogpreds_entropy_weight = self.posterior_mogpreds_entropy_weight,
                        train_posterior_mogpreds_intersequence_diversity_loss = posterior_mogpreds_intersequence_diversity_loss,
                        train_gumbel_softmax_temperature=self.gumbel_softmax_temp, # TODO: probably should schedule an annealing pattern
                        train_prior_entropy_loss=prior_entropy,
                        train_prior_repulsion_loss=prior_repulsion,
                        train_patient_adversarial_loss=patient_adversarial_loss,
                        train_LR_gmvae=self.opt_gmvae.param_groups[0]['lr'], 
                        train_LR_disc=self.opt_disc.param_groups[0]['lr'], 
                        train_LR_prior=self.opt_gmvae.param_groups[1]['lr'], 
                        train_LR_classifier=self.opt_gmvae.param_groups[2]['lr'],
                        train_KL_weight=self.kl_weight, 
                        train_ReconMSE_Weight=self.mse_weight,
                        train_AdversarialAlpha=self.classifier_alpha,
                        train_epoch=self.epoch)

                elif val_finetune:
                    metrics = dict(
                        val_finetune_attention_dropout=attention_dropout,
                        val_finetune_loss=loss, 
                        val_finetune_recon_MSE_loss=mse_loss, 
                        # val_finetune_KL_divergence=KL_divergence, 
                        val_finetune_neg_gp_weight = self.gp_weight,
                        val_finetune_neg_gp_log_prob = neg_gp_log_prob,
                        val_finetune_mean_match_loss = mean_match_loss,
                        val_finetune_logvar_match_loss = logvar_match_loss,
                        val_finetune_mean_match_weight = self.mean_match_weight,
                        val_finetune_logvar_match_weight = self.logvar_match_weight,
                        val_finetune_LR_gmvae=self.opt_gmvae.param_groups[0]['lr'], 
                        val_finetune_LR_disc=self.opt_disc.param_groups[0]['lr'], 
                        val_finetune_LR_prior=self.opt_gmvae.param_groups[1]['lr'], 
                        val_finetune_LR_classifier=self.opt_gmvae.param_groups[2]['lr'],
                        val_finetune_KL_weight=self.kl_weight, 
                        val_finetune_ReconMSE_Weight=self.mse_weight,
                        val_finetune_AdversarialAlpha=self.classifier_alpha,
                        val_finetune_epoch=self.epoch)

                elif val_unseen:
                    metrics = dict(
                        val_unseen_attention_dropout=attention_dropout,
                        val_unseen_loss=loss, 
                        val_unseen_recon_MSE_loss=mse_loss, 
                        # val_unseen_KL_divergence=KL_divergence, 
                        val_unseen_neg_gp_weight = self.gp_weight,
                        val_unseen_neg_gp_log_prob = neg_gp_log_prob,
                        val_unseen_mean_match_loss = mean_match_loss,
                        val_unseen_logvar_match_loss = logvar_match_loss,
                        val_unseen_mean_match_weight = self.mean_match_weight,
                        val_unseen_logvar_match_weight = self.logvar_match_weight,
                        val_unseen_LR_gmvae=self.opt_gmvae.param_groups[0]['lr'], 
                        val_unseen_LR_disc=self.opt_disc.param_groups[0]['lr'], 
                        val_unseen_LR_prior=self.opt_gmvae.param_groups[1]['lr'],
                        val_unseen_LR_classifier=self.opt_gmvae.param_groups[2]['lr'],
                        val_unseen_KL_weight=self.kl_weight, 
                        val_unseen_ReconMSE_Weight=self.mse_weight,
                        val_unseen_AdversarialAlpha=self.classifier_alpha,
                        val_unseen_epoch=self.epoch)

            try:
                wandb.log({**metrics, 'Steps': train_step})
            except Exception as e:
                print(f"An error occurred during WandB logging': {e}")

            # Realtime latent visualizations
            if singlebatch_latent_printing & ((iter_curr + 1) % singlebatch_printing_interval == 0):
                try:
                    if self.gpu_id == 0:
                        if torch.isnan(loss).any():
                            print("WARNING: Loss is nan, no plots can be made")
                        else:
                            utils_functions.print_latent_singlebatch(
                                epoch = self.epoch, 
                                iter_curr = iter_curr,
                                mean = mean.cpu().detach().numpy(), 
                                logvar = logvar.cpu().detach().numpy(),
                                mogpreds = mogpreds.cpu().detach().numpy(),
                                prior_means = self.gmvae.module.prior.means.detach().cpu().numpy(), 
                                prior_logvars = self.gmvae.module.prior.logvars.detach().cpu().numpy(), 
                                prior_weights = torch.softmax(self.gmvae.module.prior.weightlogits, dim=0).detach().cpu().numpy(), 
                                savedir = self.model_dir + f"/plots/{dataset_string}/latents", 
                                **kwargs)
                            utils_functions.print_recon_singlebatch(
                                x=x, 
                                x_hat=x_hat, 
                                savedir = self.model_dir + f"/plots/{dataset_string}/recon",
                                epoch = self.epoch,
                                iter_curr = iter_curr,
                                file_name = file_name,
                                **kwargs)
                            utils_functions.print_attention_singlebatch(
                                epoch = self.epoch, 
                                iter_curr = iter_curr,
                                pat_idxs = file_class_label, 
                                scores_byLayer_meanHeads = attW, 
                                savedir = self.model_dir + f"/plots/{dataset_string}/attention", 
                                **kwargs)

                            # NOTE: for finetuning, will still be guess the training patients, and TODO: idxs in dataset are wrong for class labels anyway
                            if (not val_unseen) & (not val_finetune): # Will not have accumulated for val_unseen
                                utils_functions.print_classprobs_singlebatch(
                                    class_probs = class_probs_mean_of_latent,
                                    class_labels = file_class_label,
                                    savedir = self.model_dir + f"/plots/{dataset_string}/classprobs",
                                    epoch = self.epoch,
                                    iter_curr = iter_curr,
                                    file_name = file_name,
                                    **kwargs)

                                utils_functions.print_patposteriorweights_cumulative( 
                                    mogpreds = self.accumulated_mogpreds.cpu().detach().numpy(),
                                    patidxs = self.accumulated_patidxs.cpu().detach().numpy(),
                                    patname_list = dataloader_curr.dataset.pat_ids,
                                    savedir = self.model_dir + f"/plots/{dataset_string}/patposteriorweights",
                                    epoch = self.epoch,
                                    iter_curr = iter_curr,
                                    **kwargs)
                except Exception as e:
                    print(f"An error occurred during realtime plotting': {e}")
                                
            # Advance the iteration counter (one iter per complete patient loop - i.e. one backward pass)
            iter_curr = iter_curr + 1

        # Plot the accumulated running posterior outputs at end of epoch 
        # Already meaned at the token level
        # Not necessarily everything from epoch, the number of accumulated samples is defined by 'total_collected_latents'
        try:
            utils_functions.plot_posterior( # Plot all GPUs
                gpu_id = self.gpu_id, 
                prior_means = self.gmvae.module.prior.means.detach().cpu().numpy(), 
                prior_logvars = self.gmvae.module.prior.logvars.detach().cpu().numpy(), 
                prior_weights = torch.softmax(self.gmvae.module.prior.weightlogits, dim=0).detach().cpu().numpy(), 
                encoder_means = self.accumulated_mean.detach().cpu().numpy(),
                encoder_logvars = self.accumulated_logvar.detach().cpu().numpy(),
                encoder_zmeaned = self.accumulated_zmeaned.detach().cpu().numpy(),
                encoder_mogpreds = self.accumulated_mogpreds.detach().cpu().numpy(),
                savedir = self.model_dir + f"/plots/{dataset_string}/posterior",
                epoch = self.epoch,
                **kwargs)
            
            if self.gpu_id == 0: utils_functions.plot_prior(
                prior_means = self.gmvae.module.prior.means.detach().cpu().numpy(), 
                prior_logvars = self.gmvae.module.prior.logvars.detach().cpu().numpy(), 
                prior_weights = torch.softmax(self.gmvae.module.prior.weightlogits, dim=0).detach().cpu().numpy(), 
                savedir = self.model_dir + f"/plots/{dataset_string}/prior",
                epoch = self.epoch,
                **kwargs)
            
            print(f"[GPU{str(self.gpu_id)}] at end of epoch")
            # barrier()
            # gc.collect()
            # torch.cuda.empty_cache()
        except Exception as e:
            print(f"An error occurred during end-of-epoch plotting': {e}")
            
if __name__ == "__main__":

    """
    Main script to initialize and run a distributed training session.

    This script performs the following steps:
    1. Sets the Python hash seed for reproducibility.
    2. Loads configuration settings from a YAML file (`train_config.yml`).
    3. Executes arithmetic build in the configuration and sets up the environment.
    4. Spawns multiple subprocesses (using the `spawn` method) to execute the main training function in parallel across multiple processes.
    5. Each subprocess runs the training with distributed settings, ensuring synchronization and coordination across workers.
    6. Waits for all subprocesses to complete before printing "End of script."

    Key functions:
    - Loading and executing configuration settings (`exec_kwargs`, `run_setup`).
    - Parallelizing the training process across multiple workers using multiprocessing.

    Note: The script avoids using `mp.spawn` due to potential memory errors and directly manages subprocesses.
    """

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