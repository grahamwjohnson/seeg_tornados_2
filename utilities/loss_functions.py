import torch
from torch import nn
import heapq
import torch.nn.functional as F
# from geomloss import SamplesLoss  # GeomLoss provides Sinkhorn divergence
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
    # sinkhorn = SamplesLoss(loss="sinkhorn", p=wasserstein_order, blur=sinkhorn_blur)  # Blur controls entropy strength, if potentials is True, returns Transport plan
    # loss = sinkhorn(z_real, z_fake) 

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

# def sinkhorn_asymmetric_loss(x, y, weight, sinkhorn_blur, batch_sinkhorn_asymmetry_lambda_factor, wasserstein_order=2, max_iter=50, **kwargs):
#     """Compute Sinkhorn divergence with an asymmetric cost function in PyTorch."""
    
#     # Compute asymmetric cost matrix
#     C = torch.cdist(x, y, p=wasserstein_order)  # Standard W1 cost (Euclidean by default)
#     # C = C / C.max()
#     # C = C / C.mean(dim=1, keepdim=True)
    
#     # Now compute the mask comparing samples, not element-wise features
#     # We'll compare the sums across all features
#     x_mean = x.mean(dim=1, keepdim=True)  # Shape (batch_size, 1)
#     y_mean = y.mean(dim=1, keepdim=True)  # Shape (batch_size, 1)

#     # Create mask where x_sum > y_sum for each pair of samples
#     mask = (x_mean > y_mean.T).float()  # Shape (batch_size, batch_size)
#     C = C * (1 + (batch_sinkhorn_asymmetry_lambda_factor - 1) * mask)  # Increase cost for contraction

#     # Initialize potentials for Sinkhorn
#     a = torch.ones(x.size(0), device=x.device) / x.size(0)  # Uniform weights for x
#     b = torch.ones(y.size(0), device=y.device) / y.size(0)  # Uniform weights for y
    
#     # Potentials u and v should match the size of a and b
#     u = torch.ones_like(a) / a.size(0)  # u is the potential for x
#     v = torch.ones_like(b) / b.size(0)  # v is the potential for y

#     # Sinkhorn iteration
#     for _ in range(max_iter):
#         # The matmul should be done on a (batch_size x batch_size) transport matrix
#         u = a / (torch.matmul(torch.exp(-C / sinkhorn_blur), v) + 1e-9)
#         v = b / (torch.matmul(torch.exp(-C.T / sinkhorn_blur), u) + 1e-9)

#     # Compute transport plan
#     transport_plan = torch.exp(-C / sinkhorn_blur) * (u[:, None] * v[None, :])

#     # Return Sinkhorn loss weighted by provided weight
#     loss = (transport_plan * C).sum() * weight

#     return loss