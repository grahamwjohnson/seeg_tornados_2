import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchinfo import summary
import torch.distributions as dist
import yaml

# Local imports
from .Transformer import ModelArgs, Transformer, RMSNorm

class LinearDimStepper(nn.Module):
    def __init__(self, in_dim, out_dim, norm_factory=RMSNorm, act=nn.SiLU):
        super().__init__()
        assert in_dim % 2 == 0 and out_dim % 2 == 0, "Dimensions must be divisible by 2"
        assert in_dim >= out_dim, "Input dim must be greater than or equal to output dim"

        dims = [in_dim]
        while dims[-1] > out_dim:
            next_dim = max(out_dim, dims[-1] // 2)
            dims.append(next_dim)

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # skip norm + act after final layer
                layers.append(norm_factory(dims[i+1]))
                layers.append(act())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class BSV(nn.Module):
    def __init__(self, gpu_id, bsv_dims, **kwargs): 
        super().__init__()
        self.gpu_id = gpu_id
        self.dims = bsv_dims

        self.encoder_layers = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()

        for i in range(len(bsv_dims) - 2):
            self.encoder_layers.append(nn.Linear(bsv_dims[i], bsv_dims[i + 1]))
            self.encoder_norms.append(RMSNorm(bsv_dims[i + 1]))

        final_enc_dim = bsv_dims[-2]
        latent_dim = bsv_dims[-1]
        self.fc_mu = nn.Linear(final_enc_dim, latent_dim)
        self.fc_logvar = nn.Linear(final_enc_dim, latent_dim)

        self.decoder_layers = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()
        reversed_dims = list(reversed(bsv_dims))

        for i in range(len(reversed_dims) - 1):
            self.decoder_layers.append(nn.Linear(reversed_dims[i], reversed_dims[i + 1]))
            if i < len(reversed_dims) - 2:
                self.decoder_norms.append(RMSNorm(reversed_dims[i + 1]))

    def encode(self, x):
        for i, layer in enumerate(self.encoder_layers):
            x = F.silu(layer(x))
            x = self.encoder_norms[i](x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        for i, layer in enumerate(self.decoder_layers):
            z = layer(z)
            if i < len(self.decoder_layers) - 1:
                z = F.silu(z)
                z = self.decoder_norms[i](z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

class BSE2P(nn.Module): # A Growing Transformer
    def __init__(
        self, 
        gpu_id,
        transformer_dim,
        bsp_transformer_seq_length,
        bse2p_transformer_dim,
        bsp_latent_dim, 
        bse2p_layers,
        bse2p_num_heads,
        bsp2e_ffn_dim_multiplier,
        bsp2e_max_batch_size,
        bsp2e_transformer_seq_length,
        bsp2e_transformer_activation,
        dropout=0.1,
        **kwargs):
        super(BSE2P, self).__init__()

        self.gpu_id = gpu_id
        self.bse_transformer_dim = transformer_dim

        self.bsp_transformer_seq_length = bsp_transformer_seq_length
        
        self.bse2p_transformer_dim = bse2p_transformer_dim
        self.bse2p_layers = bse2p_layers
        self.bse2p_num_heads = bse2p_num_heads

        self.bsp_latent_dim = bsp_latent_dim
        self.bsp2e_ffn_dim_multiplier = bsp2e_ffn_dim_multiplier
        self.bsp2e_max_batch_size = bsp2e_max_batch_size
        self.bsp2e_transformer_seq_length = bsp2e_transformer_seq_length
        self.bsp2e_transformer_activation = bsp2e_transformer_activation

        self.bse2p_mlp = LinearDimStepper(self.bse_transformer_dim, self.bse2p_transformer_dim)
        
        self.bse2p_transformer = Transformer(ModelArgs(
            device=self.gpu_id, 
            dim=self.bse2p_transformer_dim, 
            n_heads=self.bse2p_num_heads,
            n_layers=self.bse2p_layers,
            ffn_dim_multiplier=self.bsp2e_ffn_dim_multiplier,
            max_batch_size=self.bsp2e_max_batch_size,
            max_seq_len=self.bsp2e_transformer_seq_length,
            activation=self.bsp2e_transformer_activation))

        self.pre_vae_hidden_dim = max(4 * self.bse_transformer_dim, 4 * self.bsp_latent_dim)
        self.num_sampled_tokens = self.pre_vae_hidden_dim / self.bse2p_transformer_dim
        assert self.num_sampled_tokens % 1 == 0
        self.num_sampled_tokens = int(self.num_sampled_tokens)

        # VAE layers
        self.mu_mlp = nn.Linear(self.pre_vae_hidden_dim, self.bsp_latent_dim)
        self.logvar_mlp = nn.Linear(self.pre_vae_hidden_dim, self.bsp_latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):

        # Reshape input to pseudobatch the transformer sequence dimension
        # Converts [batch, TransSeq, FS, BSE latnet dim] --> [batch * TransSeq, FS, BSE latnet dim]
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])

        # Shrink latent dim
        x = self.bse2p_mlp(x)

        # Run through Transformer to attend across lowest timescale tokens
        x = self.bse2p_transformer(x, start_pos=0, causal_mask_bool=False, self_mask=False, return_attW=False)  # No masking

        # Re-shape back to [Batch, TransSeq, FS, BSP latent dim]
        # Take last token
        x = x.reshape(-1, self.bsp_transformer_seq_length, x.shape[1], x.shape[2])

        # Get evenly spaced indices
        B, C, T, D = x.shape
        indices = torch.linspace(0, T - 1, steps=self.num_sampled_tokens).long()

        # Sample those indices and concatenate along the last dim
        x = x[:, :, indices, :]  # shape: (B, C, num_samples, D)
        x = x.reshape(B, C, self.num_sampled_tokens * D)  # concatenate sampled tokens into last dim

        # Send through VAE layers
        mu = self.mu_mlp(x)
        logvar = self.logvar_mlp(x)
        z = self.reparameterize(mu, logvar)

        return mu, logvar, z

class BSP2E(nn.Module): # A Growing Transformer
    def __init__(
        self, 
        gpu_id,
        bsp_transformer_seq_length,
        latent_dim,
        bsp_latent_dim, 
        bse2p_layers,
        bse2p_num_heads,
        bsp2e_ffn_dim_multiplier,
        bsp2e_max_batch_size,
        bsp2e_transformer_seq_length,
        bsp2e_transformer_activation,
        bsp2e_hidden_dims,
        dropout=0.1,
        **kwargs):
        super(BSP2E, self).__init__()

        self.gpu_id = gpu_id
        self.bse_latent_dim = latent_dim
        self.bsp_transformer_seq_length = bsp_transformer_seq_length
        self.bsp_latent_dim = bsp_latent_dim
        self.bse2p_layers = bse2p_layers
        self.bse2p_num_heads = bse2p_num_heads
        self.bsp2e_ffn_dim_multiplier = bsp2e_ffn_dim_multiplier
        self.bsp2e_max_batch_size = bsp2e_max_batch_size
        self.bsp2e_transformer_seq_length = bsp2e_transformer_seq_length
        self.bsp2e_transformer_activation = bsp2e_transformer_activation
        self.bsp2e_hidden_dims = bsp2e_hidden_dims

        layers = []
        prev_dim = 1
        for dim in self.bsp2e_hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(RMSNorm(dim))
            layers.append(nn.SiLU())
            prev_dim = dim
        self.mlp_token_expander = nn.Sequential(*layers)
        
        self.mlp_latentdim = nn.Sequential(
            nn.Linear(self.bsp_latent_dim, self.bsp_latent_dim * 4),
            RMSNorm(self.bsp_latent_dim * 4),
            nn.SiLU(),
            nn.Linear(self.bsp_latent_dim * 4, self.bse_latent_dim * 4),
            RMSNorm(self.bse_latent_dim * 4),
            nn.SiLU(),
            nn.Linear(self.bse_latent_dim * 4, self.bse_latent_dim))

        self.time_dim = self.bsp2e_hidden_dims[-1]
        self.mlp_timedim = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim * 4),
            RMSNorm(self.time_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_dim * 4, self.time_dim * 4),
            RMSNorm(self.time_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_dim * 4, self.time_dim))
        
    def forward(self, x):

        # Reshape input to pseudobatch the transformer sequence dimension
        # Converts [batch, TransSeq, FS, BSE latent dim] --> [batch * TransSeq, FS, BSE latent dim]
        x = x.unsqueeze(1)
        x = self.mlp_token_expander(x.transpose(1,2)).transpose(1,2)
        x = self.mlp_latentdim(x)
        x = x.transpose(1,2)
        x = self.mlp_timedim(x)
        x = x.transpose(1,2)

        return x

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

        # Build the BSE2P
        self.bse2p = BSE2P(gpu_id=self.gpu_id, bsp_latent_dim=bsp_latent_dim, **kwargs)

        # Build the BSP
        self.bsp_latent_dim = bsp_latent_dim
        self.bsp_transformer_activation = bsp_transformer_activation
        self.bsp_n_heads = bsp_n_heads
        self.bsp_n_layers = bsp_n_layers
        self.bsp_ffn_dim_multiplier = bsp_ffn_dim_multiplier
        self.bsp_max_batch_size=bsp_max_batch_size
        self.bsp_max_seq_len = bsp_max_seq_len

        self.bsp_transformer_start_pos = bsp_transformer_start_pos

        # Transformer for BSP
        self.transformer = Transformer(ModelArgs(
            device=self.gpu_id, 
            dim=self.bsp_latent_dim, 
            n_heads=self.bsp_n_heads,
            n_layers=self.bsp_n_layers,
            ffn_dim_multiplier=self.bsp_ffn_dim_multiplier,
            max_batch_size=self.bsp_max_batch_size,
            max_seq_len=self.bsp_max_seq_len,
            activation=self.bsp_transformer_activation))
        
        # Build the BSP2E to undo the transformer embedding
        self.bsp2e = BSP2E(gpu_id=gpu_id, bsp_latent_dim=bsp_latent_dim, **kwargs)

    def forward(self, x): # Input is from weighted mu output of BSE

        # Run through BSE2P
        post_bse2p_mu, post_bse2p_logvar, post_bse2p_z = self.bse2p(x)

        # Run through BSP (1-shifted)
        post_bsp, bsp_attW = self.transformer(post_bse2p_z[:, :-1, :], start_pos=self.bsp_transformer_start_pos, return_attW=True, causal_mask_bool=True, self_mask=False)

        # Run through BSP2E to decode back to post BSE size
        post_bsp_pseudobatch = post_bsp.reshape(post_bsp.shape[0]*post_bsp.shape[1], post_bsp.shape[2])
        post_bsp2e = self.bsp2e(post_bsp_pseudobatch)
        post_bsp2e = post_bsp2e.reshape(x.shape[0], -1, x.shape[2], x.shape[3])

        return post_bse2p_mu, post_bse2p_logvar, post_bse2p_z, post_bsp, bsp_attW, post_bsp2e

def bsp_print_models_flow(x, **kwargs):
    '''
    Builds models on CPU and prints sizes of forward passes with random data as inputs

    No losses are computed, just data flow through model

    Helpful for debugging or understanding data sizes as it flows through the model
    
    '''

    # Build the BSP
    bsp = BSP(gpu_id='cpu', **kwargs) 

    print(f"INPUT\n"f"x:{x.shape}")
    post_bse2p_mu, post_bse2p_logvar, post_bse2p_z, post_bsp, bsp_attW, post_bsp2e = bsp(x)  
    summary(bsp, input_size=(x.shape), depth=999, device="cpu")
    print(
        f"post_bse2p:{post_bse2p_mu.shape}\n",
        f"post_bsp:{post_bsp.shape}\n",
        f"attW:{bsp_attW.shape}\n",
        f"post_bsp2e:{post_bsp2e.shape}\n")
    
    # Build the BSV
    bsv = BSV(gpu_id='cpu', **kwargs)

    bsv_dec, mu, logvar, _ = bsv(post_bse2p_mu.detach())
    summary(bsv, input_size=(post_bse2p_mu.shape), depth=999, device="cpu")
    print(
        f"mu: {mu.shape}\n",
        f"logvar: {logvar.shape}\n"
        f"bsv_dec: {bsv_dec.shape}\n")
    
    del bsp, bsv







