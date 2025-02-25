import torch
from torch import nn
import heapq

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

def median_heuristic(x, y):
    """
    Compute the median pairwise distance between samples in x and y.
    """
    pairwise_dist = torch.cdist(x, y)  # Compute pairwise distances
    return torch.median(pairwise_dist)

# Gaussian kernel for MMD
def gaussian_kernel(x, y, sigma):
    """
    Compute the Gaussian kernel matrix between two sets of samples.
    
    Args:
        x: Tensor of shape (n_samples, n_features).
        y: Tensor of shape (m_samples, n_features).
        sigma: Bandwidth parameter for the Gaussian kernel.
    
    Returns:
        Kernel matrix of shape (n_samples, m_samples).
    """
    # Compute pairwise squared distances
    x_sq = torch.sum(x ** 2, dim=1, keepdim=True)
    y_sq = torch.sum(y ** 2, dim=1, keepdim=True)
    xy = torch.matmul(x, y.t())
    dist_sq = x_sq + y_sq.t() - 2 * xy
    
    # Compute Gaussian kernel
    return torch.exp(-dist_sq / (2 * sigma ** 2))

def mmd_loss_function(x, y, weight):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of samples.
    
    Args:
        x: Tensor of shape (n_samples, n_features).
        y: Tensor of shape (m_samples, n_features).
        sigma: Bandwidth parameter for the Gaussian kernel.
    
    Returns:
        MMD value (scalar).
    """

    sigma = median_heuristic(x, y)

    # Compute kernel matrices
    xx = gaussian_kernel(x, x, sigma)
    yy = gaussian_kernel(y, y, sigma)
    xy = gaussian_kernel(x, y, sigma)
    
    # Compute MMD
    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd * weight, sigma