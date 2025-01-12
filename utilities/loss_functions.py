import torch
from torch import nn
import heapq

def recon_loss_function(x, x_hat, recon_weight):

    # recon_loss = LogCosh_weight * LogCosh_loss_fn(x, x_hat) 
    loss_fn = nn.MSELoss(reduction='mean')
    recon_loss = recon_weight * loss_fn(x, x_hat) 

    return recon_loss

def kld_loss_function(mean, logvar, KL_multiplier):

    # VAE KL divergence
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp(), dim=1))
    kld_loss = KL_multiplier * kld_loss

    return kld_loss  

def transformer_loss_function(target_embeddings, out_embeddings, transformer_weight):
    
    # criterion = nn.CosineSimilarity(dim = 2)
    # transformer_loss = 1 - criterion(target_embeddings, out_embeddings).mean()
    loss_fn = nn.MSELoss(reduction='mean')
    transformer_loss = transformer_weight * loss_fn(target_embeddings, out_embeddings) 

    return transformer_weight * transformer_loss #/ in_embeddings.shape[0] / in_embeddings.shape[1] # normalize by batch size and seq length

def simple_mean_latent_loss(latent, mean_loss_weight, **kwargs):
    return torch.abs(mean_loss_weight * torch.mean(latent))


