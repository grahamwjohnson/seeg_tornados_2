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
from utilities import manifold_utilities
from utilities import loss_functions
from data import SEEG_BSP_Dataset
from models.BSP import BSP



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
    bsp_source_dir, 
    bsp_num_dataloader_workers,
    bsp_batchsize,
    bsp_weight_decay, 
    bsp_adamW_beta1,
    bsp_adamW_beta2,
    som_precomputed_path,
    **kwargs):

    # Sequential dataset used to run inference on train data 
    print(f"[GPU{str(gpu_id)}] Generating TRAIN dataset")
    kwargs['dataset_pic_dir'] = None
    train_dataset = SEEG_BSP_Dataset(
        gpu_id=gpu_id, 
        bsp_source_dir=bsp_source_dir,
        data_logger_enabled=True,
        data_logger_file=f"{kwargs['log_dir']}/data_forward_pass_log_Train_GPU{gpu_id}.jsonl.gz",
        **kwargs)

    ### Random DataLoaders ###
    train_dataloader, _ = utils_functions.prepare_ddp_dataloader(train_dataset, batch_size=bsp_batchsize, num_workers=bsp_num_dataloader_workers)
         
    ### BSP ###
    bsp = BSP(gpu_id=gpu_id, **kwargs) 
    bsp = bsp.to(gpu_id) 

    ### Optimizer ###
    param_groups = [{"params": bsp.parameters(), "lr":  kwargs['bsp_LR'], "weight_decay": bsp_weight_decay, "betas": (bsp_adamW_beta1, bsp_adamW_beta2)}]
    opt_bsp = torch.optim.AdamW(param_groups)

    ### Kohonen self-organizing map ###
    som, _ = manifold_utilities.load_kohonen(som_precomputed_path, gpu_id)
    som.reset_device(gpu_id)

    return train_dataloader, bsp, opt_bsp, som

def main(  
    # Ordered variables
    gpu_id: int, 
    world_size: int, 
    config, # aka kwargs
    
    # Passed by kwargs
    run_name: str,
    timestamp_id: int,
    start_epoch: int,
    bsp_singlebatch_printing_interval_train: int,
    bsp_autoregressive_plot_steps: int,
    bsp_state_dict_prev_path = [],
    bsp_opt_state_dict_prev_path = [],
    bsp_epochs_to_train: int = -1,
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

    print(f"[GPU{str(gpu_id)}] Loading training objects (datasets, models, optimizers)")
    train_dataloader, bsp, opt_bsp, som = load_train_objs(gpu_id=gpu_id, **kwargs) 
    
    # Load the model/opt states if not first epoch & if in training mode
    if (start_epoch > 0):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}

        # Load in BSP weights and opts
        bsp_state_dict_prev = torch.load(bsp_state_dict_prev_path, map_location=map_location)
        bsp.load_state_dict(bsp_state_dict_prev)
        bsp_opt_state_dict_prev = torch.load(bsp_opt_state_dict_prev_path, map_location=map_location)
        opt_bsp.load_state_dict(bsp_opt_state_dict_prev)
        opt_bsp.param_groups[0]['lr'] =  kwargs['bsp_LR']
        print(f"[GPU{gpu_id}] GM-VAE Model and Opt weights loaded from checkpoints")
    
    # Create the training object
    trainer = Trainer(
        world_size=world_size,
        gpu_id=gpu_id, 
        bsp=bsp, 
        opt_bsp=opt_bsp,
        som=som,
        start_epoch=start_epoch,
        train_dataloader=train_dataloader,
        wandb_run=wandb_run,
        **kwargs)
    
    # Main loop through all epochs
    for epoch in range(start_epoch, bsp_epochs_to_train):
        trainer.epoch = epoch
        
        # TRAIN
        trainer._set_to_train()
        trainer.train_dataloader.dataset.set_future_buffer(0)
        trainer._run_train_epoch(
            dataloader_curr = trainer.train_dataloader, 
            dataset_string = "train",
            singlebatch_printing_interval = bsp_singlebatch_printing_interval_train,
            **kwargs)
        
        # CHECKPOINT
        # After every train epoch, optionally delete old checkpoints
        if trainer.gpu_id == 0: trainer._save_checkpoint(trainer.epoch, **kwargs)
        print(f"GPU{str(trainer.gpu_id)} at post checkpoint save")

        # # AUTOREGRESSIVE PLOT
        # if gpu_id == 0:
        #     trainer._set_to_eval()
        #     trainer.train_dataloader.dataset.set_future_buffer(bsp_autoregressive_plot_steps)
        #     print("Running and plotting autoregression for model evaluation")
        #     with torch.inference_mode():
        #         trainer._autoregress_plot(
        #             epoch = epoch,
        #             dataloader_curr = trainer.train_dataloader, 
        #             dataset_string = "train",
        #             bsp_autoregressive_plot_steps=bsp_autoregressive_plot_steps,
        #             **kwargs)
        #     print(f"[GPU{gpu_id}] at post-autoregression plotting barrier")

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
        bsp: torch.nn.Module,
        start_epoch: int,
        train_dataloader: DataLoader,
        opt_bsp: torch.optim.Optimizer,
        som,
        wandb_run,
        model_dir: str,
        latent_dim: int,
        bsp_transformer_seq_length: int,
        bsp_transformer_start_pos: int,
        FS: int,
        recent_display_iters: int,
        **kwargs
    ) -> None:
        self.world_size = world_size
        self.gpu_id = gpu_id
        self.bsp = bsp
        self.start_epoch = start_epoch
        self.train_dataloader = train_dataloader
        self.opt_bsp = opt_bsp
        self.som = som
        self.model_dir = model_dir
        self.latent_dim = latent_dim
        self.bsp_transformer_seq_length = bsp_transformer_seq_length
        self.bsp_transformer_start_pos = bsp_transformer_start_pos
        self.FS = FS
        self.recent_display_iters = recent_display_iters
        self.wandb_run = wandb_run
        self.kwargs = kwargs

        # Setup DDP
        self.bsp = DDP(bsp, device_ids=[gpu_id]) 

        # Watch with WandB
        wandb.watch(self.bsp)

    def _set_to_train(self):
        self.bsp.train()

    def _set_to_eval(self):
        self.bsp.eval()

    def _autoregress_plot(
        self,
        epoch,
        dataloader_curr, 
        dataset_string,
        bsp_autoregressive_plot_steps,
        som_plot_data_path,
        bsp_autogressive_subbatch_to_plot,
        num_autoregress_batches = 1,
        **kwargs):
        
        autoregress_batch_idx = 0
        for x, _, pat_idxs in dataloader_curr: 
            x = x.to(self.gpu_id)
            full_output = torch.zeros_like(x)
            full_output[:, 0:self.bsp_transformer_seq_length, :] = x[:, 0:self.bsp_transformer_seq_length, :].clone().detach() # fill the running vector with initial context

            auto_step = 0
            for i in range(bsp_autoregressive_plot_steps):
                context_curr = full_output[:,auto_step:auto_step+self.bsp_transformer_seq_length,:].clone().detach()
                pred, _ = self.bsp(context_curr)
                full_output[:,auto_step+self.bsp_transformer_seq_length,:] = pred[:, -1, :].clone().detach()
                auto_step += 1

            # How many from batch to plot?
            num_to_plot = min(bsp_autogressive_subbatch_to_plot, x.shape[0])

            # Plot the autoregressed predictions
            # Include 1 point overlap in predictions and ground truth for plotting purposes 
            save_dir = self.model_dir + f"/plots/{dataset_string}/autoregression"
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            for b in range(num_to_plot):
                try:
                    pred_plot_axis = manifold_utilities.plot_kohonen_prediction(
                        gpu_id=self.gpu_id,
                        save_dir = save_dir, 
                        som = self.som, 
                        plot_data_path = som_plot_data_path, 
                        epoch = epoch,
                        batch_idx = b,
                        pat_id = pat_idxs[b],
                        context = x[b, 0:self.bsp_transformer_seq_length, :], 
                        ground_truth_future=x[b, self.bsp_transformer_seq_length-1:, :], 
                        predictions=full_output[b, self.bsp_transformer_seq_length-1:, :], 
                        undo_log=True, 
                        smoothing_factor=10)  
                except:
                    print(f"Plotting batch {b} failed")

            # Kill after desired number of batches
            autoregress_batch_idx += 1
            if autoregress_batch_idx >= num_autoregress_batches: break

    def _save_checkpoint(self, epoch, delete_old_checkpoints, **kwargs):

        print("CHECKPOINT SAVE")

        # Create new directory for this epoch
        base_checkpoint_dir = self.model_dir + f"/checkpoints"
        check_epoch_dir = base_checkpoint_dir + f"/Epoch_{str(epoch)}"

        print("Saving bsp model weights")

        ### BSP CHECKPOINT 
        check_bsp_dir = check_epoch_dir + "/bsp_checkpoints"
        if not os.path.exists(check_bsp_dir): os.makedirs(check_bsp_dir)

        # Save BSP model
        ckp = self.bsp.module.state_dict()
        check_path = check_bsp_dir + "/checkpoint_epoch" + str(epoch) + "_bsp.pt"
        torch.save(ckp, check_path)

        # Save BSP optimizer
        opt_ckp = self.opt_bsp.state_dict()
        opt_path = check_bsp_dir + "/checkpoint_epoch" + str(epoch) + "_bsp_opt.pt"
        torch.save(opt_ckp, opt_path)

        print(f"Epoch {epoch} | Training checkpoint saved at {check_epoch_dir}")

        if delete_old_checkpoints:
            utils_functions.delete_old_checkpoints(dir = base_checkpoint_dir, curr_epoch = epoch, **kwargs)
            print("Deleted old checkpoints")
    
    def _run_train_epoch(
        self, 
        dataloader_curr, 
        dataset_string,
        singlebatch_printing_interval,
        **kwargs):

        iter_curr = 0
        total_iters = len(dataloader_curr)
        for x, _, pat_idxs in dataloader_curr: 
            x = x.to(self.gpu_id)
            x_pred, attW = self.bsp(x[:, :-1, :]) # No end token
            loss = loss_functions.bsp_loss(x[:, 1:, :], x_pred, **kwargs) # Shift forward by 1
            self.opt_bsp.zero_grad()
            loss.backward()    
            torch.nn.utils.clip_grad_norm_(self.bsp.module.parameters(), max_norm=1.0)
            self.opt_bsp.step()

            # Realtime terminal info and WandB 
            if (iter_curr%self.recent_display_iters==0):
                state_str = dataset_string
                now_str = datetime.datetime.now().strftime("%I:%M%p-%B/%d/%Y")
                if (self.gpu_id == 0):
                    sys.stdout.write(
                        f"\r{now_str} [GPU{str(self.gpu_id)}]: {state_str}, EPOCH {self.epoch}, Iter [BatchSize: {x.shape[0]}] {iter_curr}/{total_iters}, " + 
                        f"Loss: {round(loss.detach().item(), 2)}                 ")
                    sys.stdout.flush() 

                # Log to WandB
                wandb.define_metric('Steps')
                wandb.define_metric("*", step_metric="Steps")
                train_step = self.epoch * int(total_iters) + iter_curr
                metrics = dict(
                    train_loss=loss,
                    train_LR_bsp=self.opt_bsp.param_groups[0]['lr'], 
                    train_epoch=self.epoch)
                
                try: wandb.log({**metrics, 'Steps': train_step})
                except Exception as e: print(f"An error occurred during WandB logging': {e}")
            
            # Advance the iteration counter (one iter per complete patient loop - i.e. one backward pass)
            iter_curr = iter_curr + 1

            if (self.gpu_id == 0) and (iter_curr%singlebatch_printing_interval==0):
                try:
                    utils_functions.print_BSP_attention_singlebatch(
                        gpu_id=self.gpu_id,
                        epoch = self.epoch, 
                        iter_curr = iter_curr,
                        pat_idxs = pat_idxs, 
                        scores_byLayer_meanHeads = attW, 
                        savedir = self.model_dir + f"/plots/{dataset_string}/attention", 
                        **kwargs)
                except:
                    print(f"Attention plotting failed")

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






