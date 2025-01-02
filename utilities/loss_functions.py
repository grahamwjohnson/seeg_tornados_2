import torch
from torch import nn
import heapq

def recon_loss_function(x, x_hat, recon_weight):

    batch_size = x.shape[0]
    num_channels = x.shape[1]

    # recon_loss = LogCosh_weight * LogCosh_loss_fn(x, x_hat) 
    loss_fn = nn.MSELoss(reduction='sum')
    recon_loss_sum = recon_weight * loss_fn(x, x_hat) / batch_size # Normalize to batch size

    return recon_loss_sum/num_channels  # Normalize to num_channels to compare across patients

def kld_loss_function(mean, logvar, KL_multiplier):

    # VAE KL divergence
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp(), dim=1))
    kld_loss = KL_multiplier * kld_loss

    return kld_loss  

def transformer_loss_function(target_embeddings, out_embeddings, transformer_weight):
    
    criterion = nn.CosineSimilarity(dim = 2)
    transformer_loss = 1 - criterion(target_embeddings, out_embeddings).mean()

    return transformer_weight * transformer_loss #/ in_embeddings.shape[0] / in_embeddings.shape[1] # normalize by batch size and seq length
