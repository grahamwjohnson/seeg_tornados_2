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
import os
import wandb
import numpy as np
import datetime
import pickle
import glob
import yaml
import wandb

# Local Imports
from utilities import latent_plotting
from utilities import utils_functions
from utilities import loss_functions
from data import SEEG_Tornado_Dataset
from models.VAE import VAE


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
    num_dataloader_workers,

    # VAE Optimizers
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
         
    ### VAE ###
    vae = VAE(gpu_id=gpu_id, **kwargs) 
    vae = vae.to(gpu_id) 

    # Separate the parameters into two groups
    vae_params = []
    classifier_params = []

    # Iterate through the model parameters
    for name, param in vae.named_parameters():
        # Check if the parameter is part of the encoder submodule
        if 'adversarial_classifier' in name:
            classifier_params.append(param)
        else:
            vae_params.append(param)

    # Separate the parameters
    param_groups = [
        {"params": vae_params, "lr":  kwargs['LR_min_core'], "weight_decay": core_weight_decay, "betas": (adamW_beta1, adamW_beta2)},  
        {"params": classifier_params, "lr": kwargs['LR_min_classifier'], "weight_decay": classifier_weight_decay, "betas":(classifier_adamW_beta1, classifier_adamW_beta2)}  
    ]
    opt_vae = torch.optim.AdamW(param_groups)

    return train_dataset, train_dataloader, valfinetune_dataset, valfinetune_dataloader, valunseen_dataset, valunseen_dataloader, vae, opt_vae

def main(  
    # Ordered variables
    gpu_id: int, 
    world_size: int, 
    config, # aka kwargs
    
    # Passed by kwargs
    run_name: str,
    timestamp_id: int,
    start_epoch: int,
    LR_val_vae: float,
    LR_val_cls: float,
    val_finetine_bool: bool,
    finetune_inference: bool, 
    finetune_inference_epochs: int,
    realtime_printing_interval_val,
    realtime_printing_interval_train,
    vae_state_dict_prev_path = [],
    vae_opt_state_dict_prev_path = [],
    running_mean_path = [],
    running_logvar_path = [],
    running_zmeaned_path = [],
    running_mogpreds_path = [],
    running_patidxs_path = [],
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
    train_dataset, train_dataloader, valfinetune_dataset, valfinetune_dataloader, valunseen_dataset, valunseen_dataloader, vae, opt_vae = load_train_objs(gpu_id=gpu_id, **kwargs) 
    
    # Load the model/opt states if not first epoch & if in training mode
    if (start_epoch > 0):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}

        # Load in VAE weights and opts
        vae_state_dict_prev = torch.load(vae_state_dict_prev_path, map_location=map_location)
        vae.load_state_dict(vae_state_dict_prev)
        vae_opt_state_dict_prev = torch.load(vae_opt_state_dict_prev_path, map_location=map_location)
        opt_vae.load_state_dict(vae_opt_state_dict_prev)
        print(f"[GPU{gpu_id}] Model and Opt weights loaded from checkpoints")

        # Load running mean 
        with open(running_mean_path, "rb") as f: accumulated_mean = pickle.load(f)
        print(f"[GPU{gpu_id}] Running Means loaded from checkpoints")

        # Load running logvar
        with open(running_logvar_path, "rb") as f: accumulated_logvar = pickle.load(f)
        print(f"[GPU{gpu_id}] Running Logvars loaded from checkpoints")

        # Load running zmeaned
        with open(running_zmeaned_path, "rb") as f: accumulated_zmeaned = pickle.load(f)
        print(f"[GPU{gpu_id}] Running Z Token-Averaged loaded from checkpoints")

        # Load running logvar
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
        vae=vae, 
        opt_vae=opt_vae,
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

        # Full INFERENCE on all data
        if (epoch > 0) & ((trainer.epoch + 1) % trainer.inference_every == 0):

            trainer.valunseen_dataset.kill_generator()
            trainer.train_dataset.kill_generator()
            print(f"[GPU{trainer.gpu_id}] Killed Train/ValUnseen random generators for inference mode")

            # Save pre-finetune model/opt weights
            if finetune_inference:
                vae_dict = trainer.vae.module.state_dict()
                vae_opt_dict = trainer.opt_vae.state_dict()
                accumulated_mean = trainer.accumulated_mean
                accumulated_logvar = trainer.accumulated_logvar
                accumulated_zmeaned = trainer.accumulated_zmeaned
                accumulated_mogpreds = trainer.accumulated_mogpreds
                accumulated_patidxs = trainer.accumulated_patidxs

                # FINETUNE on beginning of validation patients (currently only one epoch)
                # Set to train and change LR to validate settings
                trainer._set_to_train()
                trainer.opt_vae.param_groups[0]['lr'] = LR_val_vae
                trainer.opt_vae.param_groups[1]['lr'] = LR_val_cls
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
                trainer.vae.module.load_state_dict(vae_dict)
                trainer.opt_vae.load_state_dict(vae_opt_dict)
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
                vae_dict = trainer.vae.module.state_dict()
                vae_opt_dict = trainer.opt_vae.state_dict()
                accumulated_mean = trainer.accumulated_mean
                accumulated_logvar = trainer.accumulated_logvar
                accumulated_zmeaned = trainer.accumulated_zmeaned
                accumulated_mogpreds = trainer.accumulated_mogpreds
                accumulated_patidxs = trainer.accumulated_patidxs

                # FINETUNE on beginning of validation patients (currently only one epoch)
                # Set to train and change LR to validate settings
                trainer._set_to_train()
                trainer.opt_vae.param_groups[0]['lr'] = LR_val_vae
                trainer.opt_vae.param_groups[1]['lr'] = LR_val_cls
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
                trainer.vae.module.load_state_dict(vae_dict)
                trainer.opt_vae.load_state_dict(vae_opt_dict)
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
        vae: torch.nn.Module,
        start_epoch: int,
        train_dataset: SEEG_Tornado_Dataset,
        valfinetune_dataset: SEEG_Tornado_Dataset,
        valunseen_dataset: SEEG_Tornado_Dataset,
        train_dataloader: DataLoader,
        valfinetune_dataloader: DataLoader,
        valunseen_dataloader: DataLoader,
        opt_vae: torch.optim.Optimizer,
        wandb_run,
        model_dir: str,
        val_every: int,
        inference_every: int,
        finetune_inference: bool,
        latent_dim: int,
        mog_components: int,
        encode_token_samples: int,
        num_samples: int,
        transformer_seq_length: int,
        transformer_start_pos: int,
        FS: int,
        hash_output_range: tuple,
        intrapatient_dataset_style: list,
        recon_weight: float,
        atd_file: str,
        inference_window_sec_list: list,
        inference_stride_sec_list: list,
        recent_display_iters: int,
        total_collected_latents: int,
        classifier_num_pats: int,
        accumulated_mean,
        accumulated_logvar,
        accumulated_zmeaned,
        **kwargs
    ) -> None:
        self.world_size = world_size
        self.gpu_id = gpu_id
        self.vae = vae
        self.start_epoch = start_epoch
        self.train_dataset = train_dataset
        self.valfinetune_dataset = valfinetune_dataset
        self.valunseen_dataset = valunseen_dataset
        self.train_dataloader = train_dataloader
        self.valfinetune_dataloader = valfinetune_dataloader
        self.valunseen_dataloader = valunseen_dataloader
        self.opt_vae = opt_vae
        self.model_dir = model_dir
        self.val_every = val_every
        self.inference_every = inference_every
        self.finetune_inference = finetune_inference
        self.latent_dim = latent_dim
        self.mog_components = mog_components
        self.encode_token_samples = encode_token_samples
        self.num_samples = num_samples
        self.transformer_seq_length = transformer_seq_length
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
        self.classifier_num_pats = classifier_num_pats
        self.accumulated_mean = accumulated_mean
        self.accumulated_logvar = accumulated_logvar
        self.accumulated_zmeaned = accumulated_zmeaned
        self.wandb_run = wandb_run
        self.kwargs = kwargs

        # Check variables that will cause delayed crashes 
        assert len(self.inference_window_sec_list) == len(self.inference_stride_sec_list)

        self.reg_weight = -1 # dummy variable, only needed when debugging and training is skipped

        # Number of iterations per file
        self.num_windows = int((self.num_samples - self.transformer_seq_length * self.encode_token_samples - self.encode_token_samples)/self.encode_token_samples) - 2

        # Set up vae & transformer with DDP
        self.vae = DDP(vae, device_ids=[gpu_id])   # find_unused_parameters=True
                    
        # Running Regulizer window for latent data
        if self.accumulated_mean == []:
            # If first initialziing, then start off with random
            self.accumulated_mean = torch.randn(self.total_collected_latents, self.mog_components, self.latent_dim).to(self.gpu_id)
            self.accumulated_logvar = torch.randn(self.total_collected_latents, self.mog_components, self.latent_dim).to(self.gpu_id)
            self.accumulated_zmeaned = torch.randn(self.total_collected_latents, self.latent_dim).to(self.gpu_id)
            self.accumulated_mogpreds = torch.softmax(torch.randn(self.total_collected_latents, self.mog_components), dim=1).to(self.gpu_id)
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
        check_path = check_core_dir + "/checkpoint_epoch" + str(epoch) + "_vae.pt"
        torch.save(ckp, check_path)

        # Save optimizer
        opt_ckp = self.opt_vae.state_dict()
        opt_path = check_core_dir + "/checkpoint_epoch" + str(epoch) + "_vae_opt.pt"
        torch.save(opt_ckp, opt_path)

        # Save Running Mean 
        latents_path = check_core_dir + "/checkpoint_epoch" +str(epoch) + "_running_means.pkl"
        output_obj = open(latents_path, 'wb')
        pickle.dump(self.accumulated_mean, output_obj)
        output_obj.close()
        print("Saved running means")

        # Save Running Logvar 
        latents_path = check_core_dir + "/checkpoint_epoch" +str(epoch) + "_running_logvars.pkl"
        output_obj = open(latents_path, 'wb')
        pickle.dump(self.accumulated_logvar, output_obj)
        output_obj.close()
        print("Saved running logvars")

        # Save Running Z
        latents_path = check_core_dir + "/checkpoint_epoch" +str(epoch) + "_running_zmeaned.pkl"
        output_obj = open(latents_path, 'wb')
        pickle.dump(self.accumulated_zmeaned, output_obj)
        output_obj.close()
        print("Saved running Z Token-Meaned")

        # Save Running MoG Predictions 
        latents_path = check_core_dir + "/checkpoint_epoch" +str(epoch) + "_running_mogpreds.pkl"
        output_obj = open(latents_path, 'wb')
        pickle.dump(self.accumulated_mogpreds, output_obj)
        output_obj.close()
        print("Saved running MoG Predictions")

        # Save Running Pat Idxs
        latents_path = check_core_dir + "/checkpoint_epoch" +str(epoch) + "_running_patidxs.pkl"
        output_obj = open(latents_path, 'wb')
        pickle.dump(self.accumulated_patidxs, output_obj)
        output_obj.close()
        print("Saved running Pat Idxs")

        print(f"Epoch {epoch} | Training checkpoint saved at {check_epoch_dir}")

        if delete_old_checkpoints:
            utils_functions.delete_old_checkpoints(dir = base_checkpoint_dir, curr_epoch = epoch, **kwargs)
            print("Deleted old checkpoints, except epochs at end of reguaization annealing period")
    
    def _update_reg_window(
        self,
        mean,
        logvar,
        zmeaned,
        mogpreds,
        patidxs):

        '''
        Collect most recent encoder outputs
        Purely for plotting purpioses right now
        (Historically was used as running regularization window, 
        but cost got too expensive when switched to outputing K means and K logvars)

        NOTE: Must be averaged across tokens before this function call
        '''

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
                        padded_channels=padded_channels,
                        latent_dim=self.latent_dim, 
                        modifier=rand_modifer,
                        hash_output_range=self.hash_output_range)
                    
                    # Collect sequential embeddings for transformer by running sequential raw data windows through BSE N times 
                    raise Exception("Hash channel order now has -1 in it, need to code up")

                    # TODO: get rid of for loop like random data generator 
                    x = torch.zeros(data_tensor.shape[0], self.transformer_seq_length, padded_channels, self.encode_token_samples).to(self.gpu_id)
                    start_idx = w * num_samples_in_forward
                    for embedding_idx in range(0, self.transformer_seq_length):
                        # Pull out data for this window - NOTE: no hashing
                        end_idx = start_idx + self.encode_token_samples * embedding_idx + self.encode_token_samples 
                        x[:, embedding_idx, :num_channels_curr, :] = data_tensor[:, hash_channel_order, end_idx-self.encode_token_samples : end_idx]

                    ### VAE ENCODER
                    # Forward pass in stacked batch through VAE encoder
                    # latent, _, _ = self.vae(x[:, :-1, :, :], reverse=False, alpha=self.classifier_alpha)   # 1 shifted just to be aligned with training style
                    mean, _, _, _, _ = self.vae(x, reverse=False, alpha=self.classifier_alpha) # No shift if not causal masking
                    files_latents[:, w, :] = torch.mean(mean, dim=1)

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
        
            # LR & WEIGHT SCHEDULES
            self.reg_weight, self.curr_LR_core, self.curr_LR_cls, self.sparse_weight, self.classifier_weight, self.classifier_alpha = utils_functions.LR_and_weight_schedules(
                epoch=self.epoch, iter_curr=iter_curr, iters_per_epoch=total_iters, **self.kwargs)
            if (not val_finetune) & (not val_unseen): 
                self.opt_vae.param_groups[0]['lr'] = self.curr_LR_core
                self.opt_vae.param_groups[1]['lr'] = self.curr_LR_cls
            else: 
                self.classifier_alpha = 0 # For validation, do not consider classifier
                self.opt_vae.param_groups[1]['lr'] = 0

            # Check for NaNs
            if torch.isnan(x).any(): raise Exception(f"ERROR: found nans in one of these files: {file_name}")

            # VAE ENCODER: 1-shifted
            # latent, class_probs_mean_of_latent, attW = self.vae(x[:, :-1, :, :], reverse=False, alpha=self.classifier_alpha)
            mean, logvar, mogpreds, attW = self.vae(x, reverse=False) # No 1-shift if not causal masking
            
            # REPARAMETERIZATION to Z (token-level) with MoG Selection/Loss
            reg_loss, z_token = loss_functions.mog_loss(
                encoder_means=mean, 
                encoder_logvars=logvar, 
                encoder_mogpreds=mogpreds,
                mog_prior=self.vae.module.prior, 
                weight=self.reg_weight,
                **kwargs)

            # VAE DECODER - at Token Level
            x_hat = self.vae(z_token, reverse=True, hash_pat_embedding=hash_pat_embedding)  

            # CLASSIFIER - on the mean of mus
            z_meaned = torch.mean(z_token, dim=1)
            class_probs_mean_of_latent = self.vae.module.adversarial_classifier(z_meaned, alpha=self.classifier_alpha)

            # LOSSES
            recon_loss = loss_functions.recon_loss_function(
                x=x, # No shift if not causal masking
                x_hat=x_hat,
                recon_weight=self.recon_weight)

            reg_entropy = loss_functions.mog_entropy_regularization(
                weights = self.vae.module.prior.weights, 
                logvars = self.vae.module.prior.logvars, 
                **kwargs)
            
            reg_repulsion = loss_functions.mog_repulsion_regularization(
                weights = self.vae.module.prior.weights, 
                means = self.vae.module.prior.means, 
                **kwargs)

            adversarial_loss = loss_functions.adversarial_loss_function(
                probs=class_probs_mean_of_latent, 
                labels=file_class_label,
                classifier_weight=self.classifier_weight)

            loss = recon_loss + reg_loss + reg_entropy + reg_repulsion + adversarial_loss 

            # Not currently used, but is nice to see
            sparse_loss = loss_functions.sparse_l1_reg(
                z=z_meaned, 
                sparse_weight=self.sparse_weight, 
                **kwargs)

            # For plotting visualization purposes
            mean_tokenmeaned = torch.mean(mean, dim=1)
            logvar_tokenmeaned = torch.mean(logvar, dim=1)
            mogpreds_tokenmeaned = torch.mean(mogpreds, dim=1)

            # Do not backprop for pure validation (i.e. val unseen), but do for training & finetuning
            if not val_unseen: 
                self._zero_all_grads()
                loss.backward()    
                torch.nn.utils.clip_grad_norm_(self.vae.module.prior.parameters(), max_norm=1.0) # Gradient clipping for MoG prior, prevent excessive updates even with strong regularization 
                self.opt_vae.step()
                self._update_reg_window(mean_tokenmeaned, logvar_tokenmeaned, z_meaned, mogpreds_tokenmeaned, file_class_label) # Update the buffers & detach() 

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
                        train_reg_entropy=reg_entropy,
                        train_reg_repulsion=reg_repulsion,
                        train_adversarial_loss=adversarial_loss,
                        train_sparse_loss=sparse_loss,
                        train_LR_vae=self.opt_vae.param_groups[0]['lr'], 
                        train_LR_classifier=self.opt_vae.param_groups[1]['lr'],
                        train_reg_Beta=self.reg_weight, 
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
                        val_finetune_LR_vae=self.opt_vae.param_groups[0]['lr'], 
                        val_finetune_LR_classifier=self.opt_vae.param_groups[1]['lr'],
                        val_finetune_reg_Beta=self.reg_weight, 
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
                        val_unseen_LR_vae=self.opt_vae.param_groups[0]['lr'], 
                        val_unseen_LR_classifier=self.opt_vae.param_groups[1]['lr'],
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
                                epoch = self.epoch, 
                                iter_curr = iter_curr,
                                mean = mean.cpu().detach().numpy(), 
                                logvar = logvar.cpu().detach().numpy(),
                                mogpreds = mogpreds.cpu().detach().numpy(),
                                savedir = self.model_dir + f"/realtime_plots/{dataset_string}/realtime_latents", 
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

                                utils_functions.print_patmogweights_realtime(
                                    mogpreds = self.accumulated_mogpreds.cpu().detach().numpy(),
                                    patidxs = self.accumulated_patidxs.cpu().detach().numpy(),
                                    savedir = self.model_dir + f"/realtime_plots/{dataset_string}/realtime_patmogweights",
                                    epoch = self.epoch,
                                    iter_curr = iter_curr,
                                    **kwargs)
                                


            # Advance the iteration counter (one iter per complete patient loop - i.e. one backward pass)
            iter_curr = iter_curr + 1

        # Plot the full running latents at end of epoch 
        # ALready meaned at the token level
        # NOTE: not necessarily everything from epoch, the number of accumulated samples is defined by 'total_collected_latents'
        utils_functions.plot_mog_and_encoder_means(
            gpu_id = self.gpu_id, 
            encoder_means = self.accumulated_mean.detach().cpu().numpy(),
            encoder_logvars = self.accumulated_logvar.detach().cpu().numpy(),
            encoder_zmeaned = self.accumulated_zmeaned.detach().cpu().numpy(),
            encoder_mogpreds = self.accumulated_mogpreds.detach().cpu().numpy(),
            savedir = self.model_dir + f"/realtime_plots/{dataset_string}/observed_latents",
            epoch = self.epoch,
            **kwargs)
        
        if self.gpu_id == 0: utils_functions.plot_prior(
            mog_means = self.vae.module.prior.means.detach().cpu().numpy(), 
            mog_logvars = self.vae.module.prior.logvars.detach().cpu().numpy(), 
            mog_weights = self.vae.module.prior.weights.detach().cpu().numpy(), 
            savedir = self.model_dir + f"/realtime_plots/{dataset_string}/prior",
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