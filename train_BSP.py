'''
@author: grahamwjohnson
April 2025

This script is intended to be used with a fully trained BSE pulled from GitHub using torch.hub in a tagged release.
It will run all of your preprocessed data through the BSE to get embeddings.
This is intended to be used for retrospective data anylyses on your dataset. 

IMPORTANT: Data must be preprocessed using the provided preprocessing pipeline. 
Specifically, histogram equalization is used and the BSE will not function properly without proper input distribution of data. 

'''
# Package imports 
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier
import yaml, sys, os, pickle, numpy as np, datetime, glob, random
import wandb
from torch.utils.data import DataLoader

# Local imports
from utilities import utils_functions
from utilities import loss_functions
from models.BSP import BSP, BSV
from data import SEEG_BSP_Dataset

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
    bsp_batchsize,
    
    bsp_LR,
    bsp_weight_decay,
    bsp_adamW_beta1,
    bsp_adamW_beta2, 

    bsv_LR,
    bsv_weight_decay,
    bsv_adamW_beta1,
    bsv_adamW_beta2, 
    
    num_dataloader_workers=0,
    **kwargs):

    # Load the train dataset (N sequential epochs of size for BSE)
    train_dataset = SEEG_BSP_Dataset(gpu_id, **kwargs)
    train_dataloader, _ = utils_functions.prepare_ddp_dataloader(train_dataset, batch_size=bsp_batchsize, num_workers=num_dataloader_workers)

    # Load the pretrained Brain-State Embedder (BSE) from GitHub and put on GPU, and initialize DDP
    bse, disc = get_bse(gpu_id=gpu_id, bsp_batchsize=bsp_batchsize, **kwargs)
    # bse = bse.to(gpu_id) 
    # bse.gpu_id = gpu_id
    # bse.transformer_encoder.freqs_cis = bse.transformer_encoder.freqs_cis.to(gpu_id)
    DDP(bse, device_ids=[gpu_id])
    DDP(disc, device_ids=[gpu_id])

    # Load the BSP, BSV & optimizers for each
    bsp = BSP(gpu_id=gpu_id, **kwargs) 
    bsp = bsp.to(gpu_id)
    bsv = BSV(gpu_id=gpu_id, **kwargs)
    bsv = bsv.to(gpu_id)

    opt_bsp = torch.optim.AdamW(bsp.parameters(), lr=bsp_LR, betas=(bsp_adamW_beta1, bsp_adamW_beta2), weight_decay=bsp_weight_decay)
    opt_bsv = torch.optim.AdamW(bsv.parameters(), lr=bsv_LR, betas=(bsv_adamW_beta1, bsv_adamW_beta2), weight_decay=bsv_weight_decay)

    return train_dataloader, bse, disc, bsp, bsv, opt_bsp, opt_bsv 

def get_bse(models_codename, gpu_id, bsp_transformer_seq_length, bsp_batchsize, **kwargs):
    torch.hub.set_dir('./.torch_hub_cache') # Set a local cache directory for testing

    # Load the BSE model with pretrained weights from GitHub
    bse, disc, _, _, _ = torch.hub.load(
        'grahamwjohnson/seeg_tornados_2',
        'load_lbm',
        codename=models_codename,
        gpu_id=gpu_id,
        pretrained=True,
        load_bse=True, 
        load_discriminator=True,
        load_bsp=False,
        load_bsv=False,
        load_pacmap=False,
        trust_repo='check',
        max_batch_size=bsp_transformer_seq_length*bsp_batchsize, # update for pseudobatching
        # force_reload=True
    )

    return bse, disc
        
def main(
    gpu_id,
    world_size,
    config, # aka kwargs

    timestamp_id,
    run_name,
    start_epoch,

    hard_reset_bsv_opt,

    bsp_epochs_to_train = 99999,
    bsp_state_dict_prev_path = [],
    bsp_opt_state_dict_prev_path = [],
    bsv_state_dict_prev_path = [],
    bsv_opt_state_dict_prev_path = [],
    running_bsp_mu_path = [],
    running_bsp_logvar_path = [],
    running_bsv_z_path = [],
    running_bsp_filenames_path = [],
    running_bsp_startidxs_path = [],
    running_bsv_mu_path = [],
    running_bsv_logvar_path = [],
    running_bsv_filenames_path = [],
    running_bsv_startidxs_path = [],

    **kwargs):

    # Initialize new WandB here and group GPUs together with DDP
    wandb.require("service")
    wandb_run = wandb.init(
        resume="allow",
        id=f"{timestamp_id}_GPU{gpu_id}",
        name=f"{run_name}_GPU{gpu_id}",
        project="Tornados_LBM",
        dir=kwargs['model_dir'], 
        group='DDP', 
        config=config)

    # Set the number of threads for this MP subprocess
    torch.set_num_threads(kwargs['subprocess_num_threads'])

    # Initialize DDP 
    ddp_setup(gpu_id, world_size)

    # Load the BSP and BSV, as well as optimizers for each
    train_dataloader, bse, disc, bsp, bsv, opt_bsp, opt_bsv  = load_train_objs(gpu_id, **kwargs)

    # Load the Brain-State Predictor (BSP) model/opt states if not first epoch & if in training mode
    if (start_epoch > 0):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}

        # Load in BSP weights and opts
        bsp_state_dict_prev = torch.load(bsp_state_dict_prev_path, map_location=map_location)
        bsp.load_state_dict(bsp_state_dict_prev)
        bsp_opt_state_dict_prev = torch.load(bsp_opt_state_dict_prev_path, map_location=map_location)
        opt_bsp.load_state_dict(bsp_opt_state_dict_prev)
        print(f"[GPU{gpu_id}] BSP Model and Opt weights loaded from checkpoints")

        # Load in BSV weights and opts
        if not hard_reset_bsv_opt:
            bsv_state_dict_prev = torch.load(bsv_state_dict_prev_path, map_location=map_location)
            bsv.load_state_dict(bsv_state_dict_prev)
            bsv_opt_state_dict_prev = torch.load(bsv_opt_state_dict_prev_path, map_location=map_location)
            opt_bsv.load_state_dict(bsv_opt_state_dict_prev)
            print(f"[GPU{gpu_id}] BSV Model and Opt weights loaded from checkpoints")
        else:
            print("WARNING: HARD RESET OF BSV & OPTIMIZER")


        ### BSP running embeddings

        # Load running embeddings for BSP
        with open(running_bsp_mu_path, "rb") as f: bsp_epoch_mu = pickle.load(f)
        with open(running_bsp_logvar_path, "rb") as f: bsp_epoch_logvar = pickle.load(f)
        print(f"[GPU{gpu_id}] Running Embeddings loaded from checkpoints")

        # Load running filenames for BSP
        with open(running_bsp_filenames_path, "rb") as f: bsp_epoch_filenames = pickle.load(f)
        print(f"[GPU{gpu_id}] Running Filenames loaded from checkpoints")

        # Load running start idx offsets for BSP
        with open(running_bsp_startidxs_path, "rb") as f: bsp_epoch_start_idx_offset = pickle.load(f)
        print(f"[GPU{gpu_id}] Running Start Idx Offsets loaded from checkpoints")


        ### BSV running embeddings

        # Load running embeddings for BSV
        with open(running_bsv_mu_path, "rb") as f: bsv_epoch_mu = pickle.load(f)
        with open(running_bsv_logvar_path, "rb") as f: bsv_epoch_logvar = pickle.load(f)
        with open(running_bsv_z_path, "rb") as f: bsv_epoch_z = pickle.load(f)
        print(f"[GPU{gpu_id}] Running Embeddings loaded from checkpoints")

        # Load running filenames for BSV
        with open(running_bsv_filenames_path, "rb") as f: bsv_epoch_filenames = pickle.load(f)
        print(f"[GPU{gpu_id}] Running Filenames loaded from checkpoints")

        # Load running start idx offsets for BSV
        with open(running_bsv_startidxs_path, "rb") as f: bsv_epoch_start_idx_offset = pickle.load(f)
        print(f"[GPU{gpu_id}] Running Start Idx Offsets loaded from checkpoints")

    else:
        bsp_epoch_mu = []
        bsp_epoch_logvar = []
        bsp_epoch_filenames = []
        bsp_epoch_start_idx_offset = []

        bsv_epoch_mu = []
        bsv_epoch_logvar = []
        bsv_epoch_z = []
        bsv_epoch_filenames = []
        bsv_epoch_start_idx_offset = []

    # Create the trainer object
    trainer = Trainer(
        world_size=world_size,
        gpu_id=gpu_id, 
        bse=bse, 
        disc=disc,
        bsp=bsp,
        bsv=bsv,
        opt_bsp=opt_bsp,
        opt_bsv=opt_bsv,
        start_epoch=start_epoch,
        train_dataloader=train_dataloader,

        bsp_epoch_mu=bsp_epoch_mu,
        bsp_epoch_logvar=bsp_epoch_logvar,
        bsp_epoch_filenames=bsp_epoch_filenames,
        bsp_epoch_start_idx_offset=bsp_epoch_start_idx_offset,

        bsv_epoch_mu=bsv_epoch_mu,
        bsv_epoch_logvar=bsv_epoch_logvar,
        bsv_epoch_z=bsv_epoch_z,
        bsv_epoch_filenames=bsv_epoch_filenames,
        bsv_epoch_start_idx_offset=bsv_epoch_start_idx_offset,

        wandb_run=wandb_run,
        **kwargs)

    # Main loop through all epochs
    for epoch in range(start_epoch, bsp_epochs_to_train):
        trainer.epoch = epoch
        
        # TRAIN
        trainer._set_to_train()
        trainer._run_train_epoch(
            dataloader_curr = trainer.train_dataloader, 
            dataset_string = "train",
            **kwargs)
        
        # CHECKPOINT
        # After every train epoch, optionally delete old checkpoints
        if trainer.gpu_id == 0: trainer._save_checkpoint(trainer.epoch, **kwargs)
        print(f"GPU{str(trainer.gpu_id)} at post checkpoint save barrier")
        barrier()

    # Kill the process after training loop completes
    print(f"[GPU{gpu_id}]: End of train loop, killing 'main' subprocess")
    wandb.finish()
    destroy_process_group() 

class Trainer:
    def __init__(
        self,
        world_size: int,
        gpu_id: int,
        bse: torch.nn.Module,
        disc: torch.nn.Module,
        bsp: torch.nn.Module,
        bsv: torch.nn.Module,
        opt_bsp: torch.optim.Optimizer,
        opt_bsv: torch.optim.Optimizer,
        start_epoch: int,
        train_dataloader: DataLoader,

        prior_mog_components,

        bsp_latent_dim,
        bsp_running_kld_length,
        bsp_epoch_mu,
        bsp_epoch_logvar,
        bsp_epoch_filenames,
        bsp_epoch_start_idx_offset,

        bsv_running_kld_length,
        bsv_epoch_mu,
        bsv_epoch_logvar,
        bsv_epoch_z,
        bsv_epoch_filenames,
        bsv_epoch_start_idx_offset,

        wandb_run,
        model_dir: str,
        transformer_seq_length: int, # BSE
        encode_token_samples: int,
        bsp_transformer_seq_length: int,
        transformer_start_pos: int,
        atd_file: str,
        bsp_recent_display_iters: int,

        bsp_disc_weight,
        bsp_loss_weight,
        seeg_loss_weight,

        bsp_anneal_epochs_to_max, 
        bsp_anneal_epochs_at_max, 
        bsp_anneal_max_weight, 
        bsp_anneal_min_weight,

        bsv_anneal_epochs_to_max, 
        bsv_anneal_epochs_at_max, 
        bsv_anneal_max_weight, 
        bsv_anneal_min_weight,
        
        **kwargs
    ) -> None:
        self.world_size = world_size
        self.gpu_id = gpu_id
        self.bse = bse
        self.disc = disc
        self.bse_prior_mog_components = prior_mog_components
        self.epoch = start_epoch
        self.train_dataloader = train_dataloader
        self.bsp_latent_dim = bsp_latent_dim
        self.bsp_running_kld_length = bsp_running_kld_length
        self.bsv_running_kld_length = bsv_running_kld_length
        self.wandb_run = wandb_run
        self.model_dir = model_dir
        self.bse_transformer_seq_length = transformer_seq_length
        self.bse_encode_token_samples = encode_token_samples
        self.bsp_transformer_seq_length = bsp_transformer_seq_length
        self.transformer_start_pos = transformer_start_pos
        self.atd_file = atd_file
        self.recent_display_iters = bsp_recent_display_iters
        self.bsp_disc_weight = bsp_disc_weight
        self.seeg_loss_weight = seeg_loss_weight
        self.bsp_loss_weight = bsp_loss_weight
        self.bsv_output_dim = kwargs['bsv_dims'][-1]
        self.kwargs = kwargs

        # Set up bsp & bsv with DDP
        self.bsp = DDP(bsp, device_ids=[gpu_id])   # find_unused_parameters=True
        self.opt_bsp = opt_bsp
        self.bsv = DDP(bsv, device_ids=[gpu_id])
        self.opt_bsv = opt_bsv

        # Just assume that we will start overwriting the BSP embeddings at beginning
        self.running_bsp_index = 0
        self.running_bsv_index = 0

        # Initialize the BSP/BSV emebeddings for the epoch
        if bsv_epoch_mu == []:
            self.bsp_epoch_mu = torch.randn(self.bsp_running_kld_length * self.train_dataloader.batch_size, self.bsp_transformer_seq_length, self.bsp_latent_dim).to(self.gpu_id) 
            self.bsp_epoch_logvar = torch.randn(self.bsp_running_kld_length * self.train_dataloader.batch_size, self.bsp_transformer_seq_length, self.bsp_latent_dim).to(self.gpu_id) 
            self.bsp_epoch_filenames = [""] * (self.bsp_running_kld_length * self.train_dataloader.batch_size)
            self.bsp_epoch_start_idx_offset = torch.ones((self.bsp_running_kld_length * self.train_dataloader.batch_size), dtype=torch.int64).to(self.gpu_id)            
            
            self.bsv_epoch_mu = torch.randn(len(self.train_dataloader) * self.train_dataloader.batch_size, self.bsp_transformer_seq_length - 1, self.bsv_output_dim).to(self.gpu_id) 
            self.bsv_epoch_logvar = torch.randn(len(self.train_dataloader) * self.train_dataloader.batch_size, self.bsp_transformer_seq_length - 1, self.bsv_output_dim).to(self.gpu_id) 
            self.bsv_epoch_z = torch.randn(len(self.train_dataloader) * self.train_dataloader.batch_size, self.bsp_transformer_seq_length - 1, self.bsv_output_dim).to(self.gpu_id)
            self.bsv_epoch_filenames = [""] * (len(self.train_dataloader) * self.train_dataloader.batch_size)
            self.bsv_epoch_start_idx_offset = torch.ones((len(self.train_dataloader) * self.train_dataloader.batch_size), dtype=torch.int64).to(self.gpu_id)
       
        else: 
            self.bsp_epoch_mu = bsp_epoch_mu.to(gpu_id)
            self.bsp_epoch_logvar = bsp_epoch_logvar.to(gpu_id)
            self.bsp_epoch_filenames = bsp_epoch_filenames
            self.bsp_epoch_start_idx_offset = bsp_epoch_start_idx_offset.to(self.gpu_id)

            self.bsv_epoch_mu = bsv_epoch_mu.to(gpu_id)
            self.bsv_epoch_logvar = bsv_epoch_logvar.to(gpu_id)
            self.bsv_epoch_z = bsv_epoch_z.to(gpu_id)
            self.bsv_epoch_filenames = bsv_epoch_filenames
            self.bsv_epoch_start_idx_offset = bsv_epoch_start_idx_offset.to(self.gpu_id)

        self.bsp_annealer = utils_functions.CyclicalAnnealingWeight(bsp_anneal_epochs_to_max, bsp_anneal_epochs_at_max, bsp_anneal_max_weight, bsp_anneal_min_weight)
        self.bsv_annealer = utils_functions.CyclicalAnnealingWeight(bsv_anneal_epochs_to_max, bsv_anneal_epochs_at_max, bsv_anneal_max_weight, bsv_anneal_min_weight)

        # Watch with WandB
        wandb.watch(self.bsp)
        wandb.watch(self.bsv)

    def _set_to_train(self):
        self.bsp.train()
        self.bsv.train()
    
    def _set_to_eval(self):
        self.bsp.eval()
        self.bsv.eval()

    def store_BSP_embeddings(self, bsp_mu, bsp_logvar, filename, start_idx_offset):
        self.bsp_epoch_mu = self.bsp_epoch_mu.detach()
        self.bsp_epoch_logvar = self.bsp_epoch_logvar.detach()

        batch_size = bsp_mu.shape[0]
        total_capacity = len(self.bsp_epoch_filenames)  # Could optionally replace with a fixed total_capacity

        # Compute start and end indices with wraparound
        start_idx = self.running_bsp_index % total_capacity
        end_idx = start_idx + batch_size

        if end_idx <= total_capacity:
            # Normal write
            self.bsp_epoch_mu[start_idx:end_idx, :, :] = bsp_mu
            self.bsp_epoch_logvar[start_idx:end_idx, :, :] = bsp_logvar
            self.bsp_epoch_filenames[start_idx:end_idx] = filename
            self.bsp_epoch_start_idx_offset[start_idx:end_idx] = start_idx_offset
        else:
            # Wraparound case
            overflow = end_idx - total_capacity
            part1 = batch_size - overflow

            # First segment
            self.bsp_epoch_mu[start_idx:total_capacity, :, :] = bsp_mu[:part1]
            self.bsp_epoch_logvar[start_idx:total_capacity, :, :] = bsp_logvar[:part1]
            self.bsp_epoch_filenames[start_idx:total_capacity] = filename[:part1]
            self.bsp_epoch_start_idx_offset[start_idx:total_capacity] = start_idx_offset[:part1]

            # Wrapped segment
            self.bsp_epoch_mu[0:overflow, :, :] = bsp_mu[part1:]
            self.bsp_epoch_logvar[0:overflow, :, :] = bsp_logvar[part1:]
            self.bsp_epoch_filenames[0:overflow] = filename[part1:]
            self.bsp_epoch_start_idx_offset[0:overflow] = start_idx_offset[part1:]

        # Update index counter
        self.running_bsp_index += batch_size

    def store_BSV_embeddings(self, bsv_mu, bsv_logvar, bsv_z, filename, start_idx_offset):
        self.bsv_epoch_mu = self.bsv_epoch_mu.detach()
        self.bsv_epoch_logvar = self.bsv_epoch_logvar.detach()
        self.bsv_epoch_z = self.bsv_epoch_z.detach()

        batch_size = bsv_mu.shape[0]
        total_capacity = len(self.bsv_epoch_filenames)  # Could optionally replace with a fixed total_capacity

        # Compute start and end indices with wraparound
        start_idx = self.running_bsv_index % total_capacity
        end_idx = start_idx + batch_size

        if end_idx <= total_capacity:
            # Normal write
            self.bsv_epoch_mu[start_idx:end_idx, :, :] = bsv_mu
            self.bsv_epoch_logvar[start_idx:end_idx, :, :] = bsv_logvar
            self.bsv_epoch_z[start_idx:end_idx, :, :] = bsv_z
            self.bsv_epoch_filenames[start_idx:end_idx] = filename
            self.bsv_epoch_start_idx_offset[start_idx:end_idx] = start_idx_offset
        else:
            # Wraparound case
            overflow = end_idx - total_capacity
            part1 = batch_size - overflow

            # First segment
            self.bsv_epoch_mu[start_idx:total_capacity, :, :] = bsv_mu[:part1]
            self.bsv_epoch_logvar[start_idx:total_capacity, :, :] = bsv_logvar[:part1]
            self.bsv_epoch_z[start_idx:total_capacity, :, :] = bsv_z[:part1]
            self.bsv_epoch_filenames[start_idx:total_capacity] = filename[:part1]
            self.bsv_epoch_start_idx_offset[start_idx:total_capacity] = start_idx_offset[:part1]

            # Wrapped segment
            self.bsv_epoch_mu[0:overflow, :, :] = bsv_mu[part1:]
            self.bsv_epoch_logvar[0:overflow, :, :] = bsv_logvar[part1:]
            self.bsv_epoch_z[0:overflow, :, :] = bsv_z[part1:]
            self.bsv_epoch_filenames[0:overflow] = filename[part1:]
            self.bsv_epoch_start_idx_offset[0:overflow] = start_idx_offset[part1:]

        # Update index counter
        self.running_bsv_index += batch_size

    def _remove_padded_channels(self, x: torch.Tensor, hash_channel_order):
        """
        Removes padded channels (-1) from x for each sample in the batch.
        
        Args:
            x: Input tensor of shape [batch, tokens, max_channels, seq_len].
            hash_channel_order: 2D tensor where -1 indicates padded channels.
        
        Returns:
            List of tensors with padded channels removed (one per batch sample).
        """

        # Convert hash_channel_order to list of lists
        hash_channel_order = hash_channel_order.to(torch.int).tolist()

        # Convert hash_channel_order to a tensor
        channel_order_tensor = torch.tensor(hash_channel_order, dtype=torch.long).to(self.gpu_id)  # [batch, max_channels]
        valid_mask = (channel_order_tensor != -1)  # [batch, max_channels]

        # Filter out padded channels per sample
        x_nopad = []
        for i in range(x.shape[0]):  # Loop over batch
            x_nopad.append(x[i, :, valid_mask[i], :])  # [tokens, valid_channels, seq_len]

        return x_nopad

    def _run_train_epoch(
        self,
        dataloader_curr,
        dataset_string,
        bsp_singlebatch_printing_interval_train,
        bsp_LR,
        bsv_LR,
        **kwargs):

        iter_curr = 0
        total_iters = len(dataloader_curr)

        for x, filename, pat_idxs, start_idx_offset, rand_ch_orders in dataloader_curr:  

            # Update VAE cyclical annealing weights
            self.bsp_annealer.update_weight(self.epoch, iter_curr, total_iters)
            self.bsv_annealer.update_weight(self.epoch, iter_curr, total_iters)

            # Update LR
            self.opt_bsp.param_groups[0]['lr'] = bsp_LR
            self.opt_bsv.param_groups[0]['lr'] = bsv_LR

            ### BSE ### 
            # Pretrained
            with torch.no_grad():
                self.bse.eval()
                x = x.to(self.gpu_id)
                post_bse_z = torch.zeros(x.shape[0], self.bsp_transformer_seq_length, x.shape[2], self.bse.latent_dim, dtype=torch.float32).to(self.gpu_id)
                for b in range(x.shape[0]): # One batch index at a time to not have to double pseudobatch
                    x_in = x[b, :, :, :, :]
                    z_pseudobatch, _, _, mogpreds_pseudobatch_softmax, _ = self.bse(x_in, reverse=False) 
                    z_split = z_pseudobatch.split(self.bse_transformer_seq_length, dim=0) # Reshape variables back to token level 
                    post_bse_z[b, :, :, :] = torch.stack(z_split, dim=0)

            ### BSP ###
            post_bse2p_mu, post_bse2p_logvar, post_bse2p_z, post_bsp, bsp_attW, post_bsp2e = self.bsp(post_bse_z) # Not 1-shifted, but will 1-shift within BSP (i.e. after BSE2P)
            self.store_BSP_embeddings(post_bse2p_mu, post_bse2p_logvar, filename, start_idx_offset + self.bse_transformer_seq_length * self.bse_encode_token_samples) 
            
            # Discriminator to encourage decoded BSP2E to be on same manifold as post BSE 
            self.disc.eval()
            post_bsp2e_pseudopseudobatch = post_bsp2e.reshape(post_bsp2e.shape[0]*post_bsp2e.shape[1]*post_bsp2e.shape[2], post_bsp2e.shape[3])
            disc_fake_loss = loss_functions.bse_adversarial_loss(
                z_posterior=post_bsp2e_pseudopseudobatch,
                discriminator=self.disc, # Try to fool it
                beta=self.bsp_disc_weight)

            # Reconstruct back to raw SEEG by using pretrained BSE decoder
            post_bsp2e_pseudobatch = post_bsp2e.reshape(post_bsp2e.shape[0] * post_bsp2e.shape[1], post_bsp2e.shape[2], post_bsp2e.shape[3])
            x_hat_pseudobatch = self.bse(post_bsp2e_pseudobatch, reverse=True)

            # Reconstruct short-circuit BSE output to raw SEEG for comparison
            with torch.no_grad():
                post_bse_z_pseudobatch = post_bse_z[:,1:,:,:].reshape(post_bse_z.shape[0] * (post_bse_z.shape[1]-1), post_bse_z.shape[2], post_bse_z.shape[3])
                x_hat_shortcircuit_pseudobatch = self.bse(post_bse_z_pseudobatch, reverse=True)

            # Reshape original x for loss and plotting
            x_shift_pseudobatch = x[:, 1:, :, :, :].reshape(x.shape[0] * x.shape[1]-1, -1, x.shape[3], x.shape[4]) 

            # REMOVE PADDING - otherwise would reward recon loss for patients with fewer channels
            ch_order_pseudobatch = rand_ch_orders[:, 1:, :].reshape(rand_ch_orders.shape[0] * (rand_ch_orders.shape[1] -1), rand_ch_orders.shape[2])
            x_nopad_list = self._remove_padded_channels(x_shift_pseudobatch, ch_order_pseudobatch) 
            x_hat_nopad_list = self._remove_padded_channels(x_hat_pseudobatch, ch_order_pseudobatch) 

            # BSP Loss
            bsp_recon_loss = loss_functions.mse_loss(post_bse_z[:,1:,:,:], post_bsp2e) * self.bsp_loss_weight
            seeg_recon_loss = loss_functions.recon_loss(x=x_nopad_list, x_hat=x_hat_nopad_list, mse_weight=self.seeg_loss_weight)
            bsp_mu_recent = utils_functions.circular_slice_tensor(self.bsp_epoch_mu, self.running_bsp_index - self.bsp_running_kld_length, self.running_bsp_index)
            bsp_logvar_recent = utils_functions.circular_slice_tensor(self.bsp_epoch_logvar, self.running_bsp_index - self.bsp_running_kld_length, self.running_bsp_index)
            bsp_mu_reshaped = bsp_mu_recent.reshape(bsp_mu_recent.shape[0] * bsp_mu_recent.shape[1], -1)
            bsp_logvar_reshaped = bsp_logvar_recent.reshape(bsp_logvar_recent.shape[0] * bsp_logvar_recent.shape[1], -1)
            bsp_kld_loss = loss_functions.bsp_kld_loss(bsp_mu_reshaped, bsp_logvar_reshaped, **kwargs) * self.bsp_annealer.get_weight()
            bsp_loss = bsp_recon_loss + bsp_kld_loss + seeg_recon_loss + disc_fake_loss

            ### BSV ###
            bsv_dec, bsv_mu, bsv_logvar, bsv_z = self.bsv(post_bse2p_z[:, :-1, :].detach())
            self.store_BSV_embeddings(bsv_mu, bsv_logvar, bsv_z, filename, start_idx_offset + self.bse_transformer_seq_length * self.bse_encode_token_samples) # 1-sifted

            # BSV Loss 
            # For KLD: only pull out N most recent based on 'bsv_running_kld_length'
            bsv_recon_loss = loss_functions.mse_loss(post_bse2p_z[:, :-1, :].detach(), bsv_dec)
            bsv_mu_recent = utils_functions.circular_slice_tensor(self.bsv_epoch_mu, self.running_bsv_index - self.bsv_running_kld_length, self.running_bsv_index)
            bsv_logvar_recent = utils_functions.circular_slice_tensor(self.bsv_epoch_logvar, self.running_bsv_index - self.bsv_running_kld_length, self.running_bsv_index)
            bsv_mu_reshaped = bsv_mu_recent.reshape(bsv_mu_recent.shape[0] * bsv_mu_recent.shape[1], -1)
            bsv_logvar_reshaped = bsv_logvar_recent.reshape(bsv_logvar_recent.shape[0] * bsv_logvar_recent.shape[1], -1)
            bsv_kld_loss = loss_functions.bsv_kld_loss(bsv_mu_reshaped, bsv_logvar_reshaped, **kwargs) * self.bsv_annealer.get_weight()
            bsv_loss = bsv_recon_loss + bsv_kld_loss

            # Step optimizers
            self.opt_bsp.zero_grad()
            self.opt_bsv.zero_grad()
            bsp_loss.backward()  
            bsv_loss.backward()  
            torch.nn.utils.clip_grad_norm_(self.bsp.module.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.bsv.module.parameters(), max_norm=1.0)
            self.opt_bsp.step() 
            self.opt_bsv.step()

            # Realtime terminal info and WandB 
            if (iter_curr%self.recent_display_iters==0):
                now_str = datetime.datetime.now().strftime("%I:%M%p-%B/%d/%Y")
                if (self.gpu_id == 0):
                    sys.stdout.write(
                        f"\r{now_str} [GPU{str(self.gpu_id)}]: {dataset_string}, EPOCH {self.epoch}, Iter [BatchSize: {x.shape[0]}] {iter_curr}/{total_iters}, " + 
                        f"MeanBSPLoss: {round(bsp_loss.detach().item(), 2)}, MeanBSVLoss: {round(bsv_loss.detach().item(), 2)}                 ")
                    sys.stdout.flush() 

                # Log to WandB
                wandb.define_metric('Steps')
                wandb.define_metric("*", step_metric="Steps")
                train_step = self.epoch * int(total_iters) + iter_curr
                metrics = dict(
                    # train_bsp_pred_loss=bsp_pred_loss,
                    train_bsp_recon_loss=bsp_recon_loss,
                    train_bsp_kld_loss=bsp_kld_loss,
                    # train_bsp_gp_loss = bsp_gp_loss,
                    train_disc_fake_loss = disc_fake_loss,
                    train_bsp_disc_weight = self.bsp_disc_weight,
                    train_bsp_loss=bsp_loss,
                    train_bsp_loss_weight=self.bsp_loss_weight,
                    train_bsp_seeg_recon_loss=seeg_recon_loss,
                    train_bsp_seeg_loss_weight=self.seeg_loss_weight,
                    train_bsp_kld_weight=self.bsp_annealer.get_weight(),
                    train_bsv_recon_loss=bsv_recon_loss,
                    train_bsv_kld_loss=bsv_kld_loss,
                    train_bsv_loss=bsv_loss,
                    train_bsv_kld_weight=self.bsv_annealer.get_weight(),
                    train_LR_bsp=self.opt_bsp.param_groups[0]['lr'], 
                    train_LR_bsv=self.opt_bsv.param_groups[0]['lr'], 
                    train_epoch=self.epoch)
                try:
                    wandb.log({**metrics, 'Steps': train_step})
                except Exception as e:
                    print(f"An error occurred during WandB logging': {e}")

            # Plotting
            if (self.gpu_id == 0) & ((iter_curr + 1) % bsp_singlebatch_printing_interval_train == 0):
                utils_functions.print_BSP_attention_singlebatch(
                    gpu_id=self.gpu_id,
                    epoch = self.epoch, 
                    iter_curr = iter_curr,
                    pat_idxs = pat_idxs, 
                    scores_byLayer_meanHeads = bsp_attW, 
                    savedir = self.model_dir + f"/plots/{dataset_string}/attention", 
                    **kwargs)
                utils_functions.print_BSP_recon_singlebatch(
                    gpu_id=self.gpu_id,
                    epoch = self.epoch, 
                    iter_curr = iter_curr,
                    pat_idxs = pat_idxs, 
                    mu = post_bse_z[:,1:,:,:], 
                    post_bsp2e = post_bsp2e,
                    savedir = self.model_dir + f"/plots/{dataset_string}/bsp_recon", 
                    **kwargs)
                utils_functions.print_BSV_recon_singlebatch(
                    gpu_id=self.gpu_id,
                    epoch = self.epoch, 
                    iter_curr = iter_curr,
                    pat_idxs = pat_idxs, 
                    post_bse2p = post_bse2p_z[:, :-1, :],
                    post_bsp = post_bsp,
                    bsv_dec = bsv_dec,
                    savedir = self.model_dir + f"/plots/{dataset_string}/bsv_recon", 
                    **kwargs)
                utils_functions.print_recon_singlebatch(
                    x=x_shift_pseudobatch, 
                    x_hat=x_hat_pseudobatch, 
                    savedir = self.model_dir + f"/plots/{dataset_string}/seeg_recon",
                    epoch = self.epoch,
                    iter_curr = iter_curr,
                    file_name = None, # TODO: modify plotting function for BSP
                    **kwargs)
                utils_functions.print_recon_singlebatch(
                    x=x_shift_pseudobatch, 
                    x_hat=x_hat_shortcircuit_pseudobatch, 
                    savedir = self.model_dir + f"/plots/{dataset_string}/seeg_recon_shortcircuit",
                    epoch = self.epoch,
                    iter_curr = iter_curr,
                    file_name = None, # TODO: modify plotting function for BSP
                    **kwargs)

                x_hat_shortcircuit_pseudobatch

            iter_curr = iter_curr + 1

        # # After epoch completes, plot 2D BSV output
        # print("Plotting BSV 2D outputs for epoch")
        # utils_functions.print_BSV_ND_embeddings(
        #     gpu_id=self.gpu_id,
        #     embeddings=self.bsv_epoch_mu,
        #     filenames=self.bsv_epoch_filenames,
        #     start_idx_offset=self.bsv_epoch_start_idx_offset,
        #     epoch=self.epoch,
        #     iter_curr=iter_curr,
        #     savedir=self.model_dir + f"/plots/{dataset_string}/bsv_embeddings",
        #     **kwargs)

        # After epoch completes, plot 1D BSV output
        print("Plotting BSV 1D outputs for epoch")
        utils_functions.print_BSV_1D_embeddings(
            gpu_id=self.gpu_id,
            embeddings=self.bsv_epoch_z,
            filenames=self.bsv_epoch_filenames,
            start_idx_offset=self.bsv_epoch_start_idx_offset,
            epoch=self.epoch,
            iter_curr=iter_curr,
            savedir=self.model_dir + f"/plots/{dataset_string}/bsv_embeddings_z",
            **kwargs)
        utils_functions.print_BSV_1D_embeddings(
            gpu_id=self.gpu_id,
            embeddings=self.bsv_epoch_mu,
            filenames=self.bsv_epoch_filenames,
            start_idx_offset=self.bsv_epoch_start_idx_offset,
            epoch=self.epoch,
            iter_curr=iter_curr,
            savedir=self.model_dir + f"/plots/{dataset_string}/bsv_embeddings_mu",
            **kwargs)

    def _save_checkpoint(self, epoch, delete_old_checkpoints, **kwargs):

        print("CHECKPOINT SAVE")

        ### BSP CHECKPOINT 
        # Create new directory for this epoch
        base_checkpoint_dir_BSP = self.model_dir + f"/bsp_checkpoints"
        check_epoch_dir = base_checkpoint_dir_BSP + f"/Epoch_{str(epoch)}"
        if not os.path.exists(check_epoch_dir): os.makedirs(check_epoch_dir)

        # Save BSP model
        ckp = self.bsp.module.state_dict()
        check_path = check_epoch_dir + "/checkpoint_epoch" + str(epoch) + "_bsp.pt"
        torch.save(ckp, check_path)

        # Save BSP optimizer
        opt_ckp = self.opt_bsp.state_dict()
        opt_path = check_epoch_dir + "/checkpoint_epoch" + str(epoch) + "_bsp_opt.pt"
        torch.save(opt_ckp, opt_path)

        # Save BSV model
        ckp = self.bsv.module.state_dict()
        check_path = check_epoch_dir + "/checkpoint_epoch" + str(epoch) + "_bsv.pt"
        torch.save(ckp, check_path)

        # Save BSV optimizer
        opt_ckp = self.opt_bsv.state_dict()
        opt_path = check_epoch_dir + "/checkpoint_epoch" + str(epoch) + "_bsv_opt.pt"
        torch.save(opt_ckp, opt_path)


        # Save the running BSP embeddings and filenames
        embeddings_path = check_epoch_dir + "/checkpoint_epoch" +str(epoch) + "_running_bsp_mu.pkl"
        output_obj = open(embeddings_path, 'wb')
        pickle.dump(self.bsp_epoch_mu, output_obj)
        output_obj.close()

        embeddings_path = check_epoch_dir + "/checkpoint_epoch" +str(epoch) + "_running_bsp_logvar.pkl"
        output_obj = open(embeddings_path, 'wb')
        pickle.dump(self.bsp_epoch_logvar, output_obj)
        output_obj.close()
        print("Saved BSP running embeddings")

        filename_path = check_epoch_dir + "/checkpoint_epoch" +str(epoch) + "_running_bsp_filenames.pkl"
        output_obj = open(filename_path, 'wb')
        pickle.dump(self.bsp_epoch_filenames, output_obj)
        output_obj.close()
        print("Saved BSP running filenames")

        filename_path = check_epoch_dir + "/checkpoint_epoch" +str(epoch) + "_running_bsp_start_idxs.pkl"
        output_obj = open(filename_path, 'wb')
        pickle.dump(self.bsp_epoch_start_idx_offset, output_obj)
        output_obj.close()
        print("Saved BSP running start offset idxs")


        # Save the running BSV embeddings and filenames
        embeddings_path = check_epoch_dir + "/checkpoint_epoch" +str(epoch) + "_running_bsv_mu.pkl"
        output_obj = open(embeddings_path, 'wb')
        pickle.dump(self.bsv_epoch_mu, output_obj)
        output_obj.close()

        embeddings_path = check_epoch_dir + "/checkpoint_epoch" +str(epoch) + "_running_bsv_logvar.pkl"
        output_obj = open(embeddings_path, 'wb')
        pickle.dump(self.bsv_epoch_logvar, output_obj)
        output_obj.close()

        embeddings_path = check_epoch_dir + "/checkpoint_epoch" +str(epoch) + "_running_bsv_z.pkl"
        output_obj = open(embeddings_path, 'wb')
        pickle.dump(self.bsv_epoch_z, output_obj)
        output_obj.close()
        print("Saved BSV running embeddings")

        filename_path = check_epoch_dir + "/checkpoint_epoch" +str(epoch) + "_running_bsv_filenames.pkl"
        output_obj = open(filename_path, 'wb')
        pickle.dump(self.bsv_epoch_filenames, output_obj)
        output_obj.close()
        print("Saved BSV running filenames")

        filename_path = check_epoch_dir + "/checkpoint_epoch" +str(epoch) + "_running_bsv_start_idxs.pkl"
        output_obj = open(filename_path, 'wb')
        pickle.dump(self.bsv_epoch_start_idx_offset, output_obj)
        output_obj.close()
        print("Saved BSV running start offset idxs")


        print(f"Epoch {epoch} | Training checkpoint saved at {check_epoch_dir}")

        if delete_old_checkpoints:
            utils_functions.delete_old_checkpoints(dir = base_checkpoint_dir_BSP, curr_epoch = epoch, **kwargs)
            print("Deleted old checkpoints, except epochs at end of reguaization annealing period")
    
if __name__ == "__main__":

    # Set the hash seed 
    os.environ['PYTHONHASHSEED'] = '1234'  

    # Read in configuration file & setup the run
    config_f = 'config.yml'
    with open(config_f, "r") as f: kwargs = yaml.load(f,Loader=yaml.FullLoader)
    kwargs = utils_functions.exec_kwargs(kwargs) # Execute the arithmatic build into kwargs and reassign kwargs
    world_size, kwargs = utils_functions.bsp_run_setup(**kwargs)

    # Spawn subprocesses with start/join (mp.spawn causes memory sigdev errors??)
    ctx = mp.get_context('spawn') # necessary to use context if have set_start_method anove?
    children = []
    for i in range(world_size):
        subproc = ctx.Process(target=main, args=(i, world_size, kwargs), kwargs=kwargs)
        children.append(subproc)
        subproc.start()

    for i in range(world_size):
        children[i].join()

