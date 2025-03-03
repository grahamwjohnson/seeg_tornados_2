import torch
from torch import nn
import heapq
import torch.nn.functional as F
from utilities.sinkhorn import sinkhorn

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

def sinkhorn_loss(observed, prior, weight, sinkhorn_eps, wasserstein_order, max_sinkhorn_iters, **kwargs):

    loss, corrs_x_to_y, corr_y_to_x = sinkhorn(
        x = observed, 
        y = prior, 
        p = wasserstein_order,
        w_x = None,
        w_y = None,
        eps = sinkhorn_eps,
        max_iters = max_sinkhorn_iters, 
        stop_thresh = 1e-5,
        verbose=False)

    return loss * weight

