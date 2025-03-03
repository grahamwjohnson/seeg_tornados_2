import torch
from torch import nn
import heapq
import torch.nn.functional as F
from geomloss import SamplesLoss

def recon_loss_function(x, x_hat, recon_weight):
    # recon_loss = LogCosh_weight * LogCosh_loss_fn(x, x_hat) 
    loss_fn = nn.MSELoss(reduction='mean')
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

def sinkhorn_loss(observed, prior, weight, sinkhorn_blur, wasserstein_order, tail_penalty_lambda):
    
    """
    Compute an asymmetric Sinkhorn divergence where movements from the tail 
    (higher values) toward lower values are penalized.
    
    :param observed: Tensor of shape [batch_size, d] (model output, latent samples)
    :param prior: Tensor of shape [batch_size, d] (prior distribution, e.g., Gamma)
    :param tail_penalty_lambda: Strength of asymmetry (higher = stronger penalty)
    :param wasserstein_order: Wasserstein order (1, 2, default W2)
    :param blur: Sinkhorn regularization (higher = more smoothing)
    :return: Asymmetric Sinkhorn divergence
    """

    # **Apply Asymmetric Penalty**: Penalize movement from high observed → low prior
    penalty_factor = 1 + tail_penalty_lambda * (observed.mean(dim=1, keepdim=True) > prior.mean(dim=1, keepdim=True)).float()
    
    # **Modify Observed Samples Instead of Cost Matrix**
    observed_scaled = observed * penalty_factor  # Penalizes tail movement

    # Compute standard Sinkhorn Loss using GeomLoss
    loss_fn = SamplesLoss(loss="sinkhorn", p=wasserstein_order, blur=sinkhorn_blur)
    loss = loss_fn(observed_scaled, prior)  # Use modified observed distribution

    norm_loss = loss * (sinkhorn_blur ** wasserstein_order) # To compare across blur/order settings

    return norm_loss * weight  # Manual re-weighting
    
    # OLD
    # criterion = SamplesLoss(loss="sinkhorn", p=wasserstein_order, blur=sinkhorn_blur)
    # loss = criterion(observed, prior)

    # return loss * weight

