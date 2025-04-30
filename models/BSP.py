import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchinfo import summary
import torch.distributions as dist
import yaml

# Local imports
from .Transformer import ModelArgs, Transformer, RMSNorm

class BSP(nn.Module):
 
    def __init__(
        self, 
        gpu_id,
        bsp_latent_dim,
        bsp_n_heads,
        bsp_n_layers,
        bsp_ffn_dim_multiplier,
        bsp_max_batch_size,
        bsp_max_seq_len,
        bsp_transformer_activation,
        bsp_transformer_start_pos,
        **kwargs):
        super(BSP, self).__init__()

        self.gpu_id = gpu_id
        self.bsp_latent_dim = bsp_latent_dim
        self.bsp_transformer_activation = bsp_transformer_activation
        self.bsp_n_heads = bsp_n_heads
        self.bsp_n_layers = bsp_n_layers
        self.bsp_ffn_dim_multiplier = bsp_ffn_dim_multiplier
        self.bsp_max_batch_size=bsp_max_batch_size
        self.bsp_max_seq_len = bsp_max_seq_len

        self.bsp_transformer_start_pos = bsp_transformer_start_pos

        # Transformer - dimension is same as output of cross attention
        self.transformer = Transformer(ModelArgs(
            device=self.gpu_id, 
            dim=self.bsp_latent_dim, 
            n_heads=self.bsp_n_heads,
            n_layers=self.bsp_n_layers,
            ffn_dim_multiplier=self.bsp_ffn_dim_multiplier,
            max_batch_size=self.bsp_max_batch_size,
            max_seq_len=self.bsp_max_seq_len,
            activation=self.bsp_transformer_activation))

    def forward(self, x):

        y, attW = self.transformer(x, start_pos=self.bsp_transformer_start_pos, return_attW=True, causal_mask_bool=True, self_mask=False)

        return y, attW 

def bsp_print_models_flow(x, **kwargs):
    '''
    Builds models on CPU and prints sizes of forward passes with random data as inputs

    No losses are computed, just data flow through model

    Helpful for debugging or understanding data sizes as it flows through the model
    
    '''

    # Build the BSP
    bsp = BSP(gpu_id='cpu', **kwargs) 

    print(f"INPUT\n"f"x:{x.shape}")
    y, attW = bsp(x)  
    summary(bsp, input_size=(x.shape), depth=999, device="cpu")
    print(
    f"x:{x.shape}\n",
    f"attW:{attW.shape}\n")
    
    del bsp


