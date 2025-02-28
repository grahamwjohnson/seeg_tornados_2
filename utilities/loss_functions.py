import torch
from torch import nn
import heapq
import torch.nn.functional as F
from geomloss import SamplesLoss  # GeomLoss provides Sinkhorn divergence


def recon_loss_function(x, x_hat, recon_weight):

    # recon_loss = LogCosh_weight * LogCosh_loss_fn(x, x_hat) 
    loss_fn = nn.MSELoss(reduction='mean')
    recon_loss = loss_fn(x, x_hat) 

    return recon_weight * recon_loss

def kld_loss_function(mean, logvar, KL_multiplier):

    # Batch the sequence dimension
    mean_batched = mean.reshape(mean.shape[0] * mean.shape[1], mean.shape[2])
    logvar_batched =  logvar.reshape(logvar.shape[0] * logvar.shape[1], logvar.shape[2])

    # VAE KL divergence
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar_batched - mean_batched**2 - logvar_batched.exp(), dim=1))

    return KL_multiplier * kld_loss  

def transformer_loss_function(target_embeddings, out_embeddings, transformer_weight):
    
    criterion = nn.CosineSimilarity(dim = 2)
    transformer_loss = 1 - criterion(target_embeddings, out_embeddings).mean()
    # loss_fn = nn.MSELoss(reduction='mean')
    # transformer_loss = transformer_weight * loss_fn(target_embeddings, out_embeddings) 

    return transformer_weight * transformer_loss #/ in_embeddings.shape[0] / in_embeddings.shape[1] # normalize by batch size and seq length

def sparse_l1_reg(z, sparse_weight, **kwargs):
    
    l1_penalty = torch.sum(torch.abs(z))  # L1 norm

    return sparse_weight * l1_penalty
    
def adversarial_loss_function(probs, labels, classifier_weight):

    # Class probs comes in as [batch, seq, num_classes] softmax
    # Change to [batch * seq, num_classes] softmax
    # class_probs_batched = class_probs.reshape(class_probs.shape[0] * class_probs.shape[1], -1)

    # Must repeat the file labels for entire sequence
    # labels_repeated = file_class_label.unsqueeze(1).repeat(1, class_probs.shape[1])
    # labels_batched = torch.squeeze(labels_repeated.reshape(labels_repeated.shape[0] * labels_repeated.shape[1], -1))
    # adversarial_loss = nn.functional.cross_entropy(class_probs_batched, labels_batched)

    adversarial_loss = nn.functional.cross_entropy(probs, labels) / torch.log(torch.tensor(probs.shape[1]))

    return classifier_weight * adversarial_loss

def sinkhorn_loss(z_real, z_fake, weight, sinkhorn_blur):
    sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=sinkhorn_blur)  # Blur controls entropy strength
    loss = sinkhorn(z_real, z_fake)  # Computes Sinkhorn loss instead of MMD

    return loss * weight

def sinkhorn(cost_matrix, blur, n_iter):
    """
    Compute the optimal transport plan using the Sinkhorn algorithm.
    
    Args:
        cost_matrix (torch.Tensor): Pairwise cost matrix (batch_size x batch_size).
        blur (float): Entropy regularization parameter.
        n_iter (int): Number of Sinkhorn iterations.
    
    Returns:
        torch.Tensor: Optimal transport plan.
    """
    K = torch.exp(-cost_matrix / blur)  # Gibbs kernel
    u = torch.ones_like(K[:, 0])  # Initialize dual variables
    
    for _ in range(n_iter):
        v = 1.0 / (K @ u + 1e-8)  # Update v
        u = 1.0 / (K @ v + 1e-8)  # Update u
    
    return u[:, None] * K * v[None, :]  # Transport plan
