import torch
from torch import nn
import torch.nn.functional as F

'''
@author: grahamwjohnson
'''

# RECONSTRUCTION

def recon_loss(x: list[torch.Tensor], x_hat: list[torch.Tensor], mse_weight: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes MSE, FD1, and FD2 losses on filtered (non-padded) tensors.
    
    Args:
        x: List of tensors (per batch) with shape [tokens, valid_channels, seq_len].
        x_hat: Reconstructed tensors (same shapes as x).
        mse_weight: Weight for MSE loss.

    Returns:
        mse_loss
    """
    mse_losses = []

    for x_sample, x_hat_sample in zip(x, x_hat):
        # MSE Loss (mean over all dimensions)
        mse = (x_hat_sample - x_sample).pow(2).mean()
        mse_losses.append(mse)

    # Average losses across the batch
    mse_loss = torch.stack(mse_losses).mean()

    return mse_loss * mse_weight

# POSTERIOR vs. PRIOR

def gmvae_kl_loss(z, encoder_means, encoder_logvars, encoder_mogpreds, prior_means, prior_logvars, prior_weights, weight, **kwargs):
    """
    GM-VAE KL-Divergence Loss Function with Prior Parameters

    Parameters:
        z (Tensor): Latent variable sampled from the encoder (batch, latent_dimension).
        encoder_means (Tensor): Means of the Gaussian mixture components (batch, mog_component, latent_dimension).
        encoder_logvars (Tensor): Log variances of the Gaussian mixture components (batch, mog_component, latent_dimension).
        encoder_mogpreds (Tensor): Mixture probabilities (already softmaxed) (batch, mog_component).
        prior_means (Tensor): Means of the prior Gaussian components (mog_component, latent_dimension).
        prior_logvars (Tensor): Log variances of the prior Gaussian components (mog_component).
        prior_weights (Tensor): Mixture weights of the prior distribution (mog_component). ALREADY softmaxed before being passed in. 

    Returns:
        kl_loss (Tensor): KL divergence loss term.
    """
    batch_size, mog_component, latent_dim = encoder_means.shape

    # Expand z to align with encoder_means and encoder_logvars
    z = z.unsqueeze(1)  # Reshape z to (batch, 1, latent_dimension)

    # Compute log probability of z under encoder components
    log_prob_encoder = -0.5 * (encoder_logvars + (z - encoder_means) ** 2 / encoder_logvars.exp())
    log_prob_encoder = log_prob_encoder.mean(dim=2)  # Sum over the latent dimensions (batch, mog_component)

    # Compute log probability of z under prior components
    prior_means = prior_means.unsqueeze(0)  # (1, mog_component, latent_dimension)
    prior_logvars = prior_logvars.unsqueeze(0)  # (1, mog_component, latent_dimension)
    
    log_prob_prior = -0.5 * (prior_logvars + (z - prior_means) ** 2 / prior_logvars.exp())
    log_prob_prior = log_prob_prior.mean(dim=2)  # (batch, mog_component)

    # Incorporate prior mixture weights
    log_prior_weights = torch.log(prior_weights + 1e-6)  # Add small value for numerical stability
    log_prob_prior += log_prior_weights  # (batch, mog_component)

    # Compute KL divergence
    kl_divergence = torch.sum(encoder_mogpreds * (log_prob_encoder - log_prob_prior), dim=1)  # (batch)

    return kl_divergence.mean() * weight  # Return the mean KL loss across the batch

def logvar_matching_loss(logvar_posterior, logvar_prior, weight):
    """
    Compute the variance-matching loss between posterior and prior log-variances.
    
    Args:
        logvar_posterior: Posterior log-variances, shape [batch_size, K, latent_dim].
        logvar_prior: Prior log-variances, shape [K, latent_dim].
        weight: Weight for the variance-matching loss.
    
    Returns:
        Variance-matching loss (scalar).
    """
    # Expand prior logvars to match the batch size of posterior logvars
    batch_size = logvar_posterior.size(0)
    logvar_prior_expanded = logvar_prior.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, K, latent_dim]

    # Compute mean squared error (MSE) between posterior and prior logvars
    loss = F.mse_loss(logvar_posterior, logvar_prior_expanded)

    # Apply weight
    return weight * loss

def mean_matching_loss(mean_posterior, mean_prior, weight):
    """
    Compute the mean-matching loss between posterior and prior means.
    
    Args:
        mean_posterior: Posterior means, shape [batch_size, K, latent_dim].
        mean_prior: Prior means, shape [K, latent_dim].
        weight: Weight for the mean-matching loss.
    
    Returns:
        Mean-matching loss (scalar).
    """
    
    # Expand prior means to match the batch size of posterior means
    batch_size = mean_posterior.size(0)
    mean_prior_expanded = mean_prior.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, K, latent_dim]

    # Compute mean squared error (MSE) between posterior and prior means
    loss = F.mse_loss(mean_posterior, mean_prior_expanded, reduction='mean')

    # Apply weight
    return weight * loss


# POSTERIOR only

def posterior_mogpreds_entropy_loss(mogpreds, posterior_mogpreds_entropy_weight, **kwargs):
    """
    Compute the entropy loss for MoG predictions, promoting entropy across the entire dataset.
    mogpreds: MoG component probabilities, shape (batch_size, T, K)
    posterior_mogpreds_entropy_weight: Weight for the entropy loss
    """
    # Ensure mogpreds is a valid probability distribution
    assert torch.all(mogpreds >= 0), "mogpreds contains negative values"
    assert torch.allclose(mogpreds.sum(dim=-1, keepdim=True), torch.ones_like(mogpreds.sum(dim=-1, keepdim=True))), "mogpreds does not sum to 1"

    # Clamp mogpreds to avoid log(0) issues
    mogpreds = torch.clamp(mogpreds, min=1e-10, max=1.0)

    # Compute the average probability of each component across all samples
    # Shape: (batch_size * T, K) -> (K,)
    aggregated_probs = mogpreds.mean(dim=(0, 1))  # Average over batch & time

    # Compute entropy across all MoG components
    entropy = -torch.mean(aggregated_probs * torch.log(aggregated_probs))

    # Return the negative entropy (maximize entropy to promote diverse component usage)
    return -posterior_mogpreds_entropy_weight * entropy

def posterior_mogpreds_intersequence_diversity_loss(mogpreds, weight, threshold=0.5, smoothness=10.0):
    """
    Compute the diversity loss for MoG predictions, promoting sequences to be far apart in the latent space.
    The loss is rescaled so that it is 0 below a certain threshold and approaches the original loss gradually above the threshold.

    Args:
        mogpreds: MoG component probabilities, shape (batch_size, T, K)
        weight: Weight for the diversity loss
        threshold: Threshold below which the loss is 0 (default: 0.5)
        smoothness: Controls how smoothly the loss transitions from 0 to the original value (default: 10.0)
    """
    # Ensure mogpreds is a valid probability distribution
    assert torch.all(mogpreds >= 0), "mogpreds contains negative values"
    assert torch.allclose(mogpreds.sum(dim=-1), torch.ones_like(mogpreds.sum(dim=-1))), "mogpreds does not sum to 1"

    # Clamp mogpreds to avoid log(0)
    mogpreds = torch.clamp(mogpreds, min=1e-10, max=1.0)

    # Compute the mean prediction for each sequence (across time steps)
    # Shape: (batch_size, T, K) -> (batch_size, K)
    mean_mogpreds = mogpreds.mean(dim=1)

    # Compute pairwise cosine similarity between sequences
    # Shape: (batch_size, K) -> (batch_size, batch_size)
    cosine_sim = torch.nn.functional.cosine_similarity(
        mean_mogpreds.unsqueeze(1),  # Shape: (batch_size, 1, K)
        mean_mogpreds.unsqueeze(0),  # Shape: (1, batch_size, K)
        dim=-1
    )

    # Exclude self-similarity (diagonal elements)
    batch_size = mean_mogpreds.shape[0]
    mask = 1 - torch.eye(batch_size, device=mean_mogpreds.device)  # Mask for off-diagonal elements
    cosine_sim = cosine_sim * mask

    # Compute the average pairwise similarity (excluding self-similarity)
    avg_pairwise_sim = cosine_sim.sum() / (batch_size * (batch_size - 1))

    # Rescale the loss to approach 0 below the threshold
    if avg_pairwise_sim < threshold:
        # Smoothly transition the loss from 0 to the original value
        rescale_factor = torch.sigmoid(smoothness * (avg_pairwise_sim - threshold))
        diversity_loss = avg_pairwise_sim * rescale_factor
    else:
        # Use the original loss value above the threshold
        diversity_loss = avg_pairwise_sim

    # Return the diversity loss (weighted)
    return weight * diversity_loss


# PRIOR only

def prior_entropy_regularization(weights, logvars, prior_entropy_weight, **kwargs):
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

    return prior_entropy_weight * (-mog_entropy) # Negative entropy (minimization) 

def prior_repulsion_regularization(weights, means, prior_repulsion_weight, **kwargs):
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

    return prior_repulsion_weight * repulsion_term  


# ADVERSARIAL

def adversarial_loss_function(probs, labels, classifier_weight):
    adversarial_loss = nn.functional.cross_entropy(probs, labels) / torch.log(torch.tensor(probs.shape[1]))
    return classifier_weight * adversarial_loss


