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

def sinkhorn_loss(observed, prior, weight, sinkhorn_blur, wasserstein_alpha):

    if torch.isnan(observed).any(): raise ValueError("NaN detected in OBSERVED tensors for Sinkhorn loss!")
    if torch.isinf(observed).any(): raise ValueError("Inf detected in OBSERVED tensors for Sinkhorn loss!")
    if torch.isnan(prior).any(): raise ValueError("NaN detected in PRIOR tensors for Sinkhorn loss!")
    if torch.isinf(prior).any(): raise ValueError("Inf detected in PRIOR tensors for Sinkhorn loss!")
    
    # W1
    loss_fn = SamplesLoss(loss="sinkhorn", p=1, blur=sinkhorn_blur)
    W1_loss = loss_fn(observed, prior)  # Standard Sinkhorn loss

    if wasserstein_alpha > 0: # Some mixing of W2 present
        # W2
        loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=sinkhorn_blur)
        W2_loss = loss_fn(observed, prior)  # Standard Sinkhorn loss
        
        loss = wasserstein_alpha * W2_loss + (1- wasserstein_alpha) * W1_loss 

        return weight * loss   # Apply manual re-weighting
    
    else:
        return weight * W1_loss

def mog_weight_reg(prior_weights, mog_weight_reg_beta):
    num_components = prior_weights.shape[0]
    even_weight = 1/num_components
    mean_weight_diff = torch.mean(torch.abs(prior_weights - even_weight))
    return mog_weight_reg_beta * mean_weight_diff


   



