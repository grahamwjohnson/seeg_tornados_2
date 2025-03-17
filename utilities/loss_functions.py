import torch
from torch import nn
from geomloss import SamplesLoss

'''
@author: grahamwjohnson
'''

def mog_entropy_regularization(weights, logvars, entropy_weight, **kwargs):
    """
    Computes the entropy of the MoG prior, considering weights and log-variances.

    Args:
        weights (torch.Tensor): Unnormalized mixture weights (K,).
        logvars (torch.Tensor): Log-variances of Gaussians (K, D).

    Returns:
        torch.Tensor: Negative entropy regularization loss.
    """
    K, D = logvars.shape  # Number of components, Latent dimension

    # Convert unnormalized weights to probabilities
    probs = torch.softmax(weights, dim=0)  # Shape: (K,)

    # Gaussian Entropy: H = 0.5 * (D * (1 + log(2π)) + sum(logvars))
    gaussian_entropies = 0.5 * (D * (1 + torch.log(2 * torch.tensor(torch.pi))) + logvars.sum(dim=-1))  # (K,)

    # Weighted sum of Gaussian entropies
    mog_entropy = torch.sum(probs * gaussian_entropies)  # Scalar

    return entropy_weight * (-mog_entropy) # Negative entropy (minimization) 

def mog_repulsion_regularization(weights, means, repulsion_weight, **kwargs):
    """
    Computes the entropy of the MoG prior, considering weights, means, and log-variances.

    Args:
        weights (torch.Tensor): Unnormalized mixture weights (K,).
        means (torch.Tensor): Means of the Gaussian components (K, D).

    Returns:
        torch.Tensor: Repulsion regularization loss.
    """

    # Convert unnormalized weights to probabilities
    probs = torch.softmax(weights, dim=0)  # Shape: (K,)

    # Pairwise Mean Separation (Repulsive Regularization)
    mean_diffs = means.unsqueeze(0) - means.unsqueeze(1)  # Shape: (K, K, D)
    mean_sq_dists = torch.sum(mean_diffs ** 2, dim=-1)  # Shape: (K, K)
    
    repulsion_term = torch.sum(probs.unsqueeze(0) * probs.unsqueeze(1) * torch.exp(-mean_sq_dists))  # Scalar

    return repulsion_weight * repulsion_term  

def mog_loss(mean, logvar, mog_prior, weight, monte_carlo_samples, **kwargs):
    """
    Compute the KL divergence between q(z|x) and the MoG prior p(z).
    mean: Encoder mean, shape (batch_size, D)
    logvar: Encoder log-variance, shape (batch_size, D)
    mog_prior: Instance of MoGPrior
    z_samples: Number of samples for Monte Carlo approximation
    """
    batch_size, D = mean.shape

    # Sample from q(z|x) using the reparameterization trick
    std = torch.exp(0.5 * logvar)  # Standard deviation
    eps = torch.randn(monte_carlo_samples, batch_size, D, device=mean.device)  # Random noise
    z = mean + eps * std  # Shape: (z_samples, batch_size, D)

    # Compute log q(z|x)
    log_qzx = -0.5 * (
        logvar +
        torch.pow(z - mean, 2) / torch.exp(logvar) +
        D * torch.log(2 * torch.tensor(torch.pi))
    ).sum(dim=-1)  # Shape: (z_samples, batch_size)

    # Compute log p(z) under the MoG prior
    log_pz = mog_prior(z.view(-1, D))  # Shape: (z_samples * batch_size,)
    log_pz = log_pz.view(monte_carlo_samples, batch_size)  # Shape: (z_samples, batch_size)

    # KL divergence: log q(z|x) - log p(z)
    kl_div = log_qzx - log_pz  # Shape: (z_samples, batch_size)

    # Average over samples and batch
    kl_div = kl_div.mean()  # Scalar

    return weight * kl_div / (batch_size * D)

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

# def KL_divergence(mu, logvar, weight):
#     if torch.isnan(mu).any(): raise ValueError("NaN detected in OBSERVED tensors for Sinkhorn loss!")
#     if torch.isinf(mu).any(): raise ValueError("Inf detected in OBSERVED tensors for Sinkhorn loss!")
#     if torch.isnan(logvar).any(): raise ValueError("NaN detected in PRIOR tensors for Sinkhorn loss!")
#     if torch.isinf(logvar).any(): raise ValueError("Inf detected in PRIOR tensors for Sinkhorn loss!")

#     batch_size, latent_dim = mu.shape

#     # KL Divergence loss
#     kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
#     return weight * kl_divergence / (batch_size *  latent_dim)



