import torch
from torch import nn
from geomloss import SamplesLoss

'''
@author: grahamwjohnson
'''

def recon_loss_function(x, x_hat, recon_weight):
    # recon_loss = LogCosh_weight * LogCosh_loss_fn(x, x_hat) 
    loss_fn = nn.MSELoss(reduction='mean')
    # loss_fn = nn.L1Loss(reduction='mean')
    recon_loss = loss_fn(x, x_hat) 
    return recon_weight * recon_loss

# def transformer_loss_function(target_embeddings, out_embeddings, transformer_weight):
#     criterion = nn.CosineSimilarity(dim = 2)
#     transformer_loss = 1 - criterion(target_embeddings, out_embeddings).mean()
#     # loss_fn = nn.MSELoss(reduction='mean')
#     # transformer_loss = transformer_weight * loss_fn(target_embeddings, out_embeddings) 
#     return transformer_weight * transformer_loss #/ in_embeddings.shape[0] / in_embeddings.shape[1] # normalize by batch size and seq length 

def sparse_l1_reg(z, sparse_weight, **kwargs):
    l1_penalty = torch.sum(torch.abs(z))  # L1 norm
    return sparse_weight * l1_penalty
    
def adversarial_loss_function(probs, labels, classifier_weight):
    adversarial_loss = nn.functional.cross_entropy(probs, labels) / torch.log(torch.tensor(probs.shape[1]))
    return classifier_weight * adversarial_loss

def sinkhorn_loss(observed, prior, weight, sinkhorn_blur, wasserstein_order):

    if torch.isnan(observed).any(): raise ValueError("NaN detected in OBSERVED tensors for Sinkhorn loss!")
    if torch.isinf(observed).any(): raise ValueError("Inf detected in OBSERVED tensors for Sinkhorn loss!")
    if torch.isnan(prior).any(): raise ValueError("NaN detected in PRIOR tensors for Sinkhorn loss!")
    if torch.isinf(prior).any(): raise ValueError("Inf detected in PRIOR tensors for Sinkhorn loss!")
    
    # # **Compute Standard Sinkhorn Loss**
    loss_fn = SamplesLoss(loss="sinkhorn", p=wasserstein_order, blur=sinkhorn_blur)
    sinkhorn_loss = loss_fn(observed, prior)  # Standard Sinkhorn loss

    return sinkhorn_loss * weight  # Apply manual re-weighting
   



