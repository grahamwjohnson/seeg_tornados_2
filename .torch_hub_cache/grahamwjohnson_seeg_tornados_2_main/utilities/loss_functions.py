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
    Computes MSE losses on filtered (non-padded) tensors. 
    Each patient will have different number of channels, that is why this is a list.
    Padded channels were stripped prior to this function. 
    
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

def discriminator_loss(z_posterior, z_prior, discriminator):
    # Discriminator should classify posterior samples as fake (0) and prior samples as real (1)
    output_posterior = discriminator(z_posterior)
    fake_loss = F.binary_cross_entropy(output_posterior, torch.zeros_like(output_posterior))
    output_prior = discriminator(z_prior)
    real_loss = F.binary_cross_entropy(output_prior, torch.ones_like(output_prior))
    total_discriminator_loss = (real_loss + fake_loss) / 2
    return total_discriminator_loss, real_loss, fake_loss

def bse_adversarial_loss(z_posterior, discriminator, beta):
    # We want the discriminator to be unable to distinguish posterior samples from prior samples
    discriminator_output_posterior = discriminator(z_posterior)
    adversarial_loss = -torch.mean(torch.log(discriminator_output_posterior + 1e-8)) # Try to make output close to 1

    return beta * adversarial_loss

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

def entropy_based_intersequence_diversity_loss(mogpreds, weight, epsilon=1e-8):

    batchsize = mogpreds.shape[0]

    # Ensure mogpreds is a valid probability distribution
    assert torch.all(mogpreds >= 0), "mogpreds contains negative values"
    assert torch.allclose(mogpreds.sum(dim=-1), torch.ones_like(mogpreds.sum(dim=-1))), "mogpreds does not sum to 1"

    # Compute the mean component prediction for each sequence (across time steps)
    # Shape: (batch_size, T, K) -> (batch_size, K)
    mean_mogpreds = mogpreds.mean(dim=1)

    # Compute the difference in mean predictions across all pairs of batch indexes (upper triangle)
    mean_abs_diffs = 0
    for i in range(batchsize):
        for j in range(i + 1, batchsize):
            mean_abs_diffs += torch.mean(torch.abs(mean_mogpreds[j, :] - mean_mogpreds[i, :]))

    diversity_loss = 1 - mean_abs_diffs / ((batchsize ** 2 - batchsize)/2) # Divide by number of comparisons to standardize across batchsizes

    return weight * diversity_loss

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


# ADVERSARIAL CLASSIFIER

def patient_adversarial_loss_function(probs, labels, classifier_weight):
    """
    Try to learn which patient is being embedded in latent space, then feed the reverse of that
    gradient up to encoder adversarially (with reverse graident layer in GMVAE)
    """
    adversarial_loss = nn.functional.cross_entropy(probs, labels) / torch.log(torch.tensor(probs.shape[1]))
    return classifier_weight * adversarial_loss


# Prediction loss for BSP
def cosine_loss(a: torch.Tensor, b: torch.Tensor, reduction='mean') -> torch.Tensor:
    """
    Compute cosine loss between two embeddings of shape 
    [batch, big_seq, fine_seq, latent_dim].

    Args:
        a (torch.Tensor): First embedding tensor.
        b (torch.Tensor): Second embedding tensor.
        reduction (str): 'mean', 'sum', or 'none'.

    Returns:
        torch.Tensor: Scalar loss if reduced, else tensor of per-element losses.
    """
    # Flatten all but the last dimension using reshape for non-contiguous tensors
    a_flat = a.reshape(-1, a.size(-1))
    b_flat = b.reshape(-1, b.size(-1))

    # Compute cosine similarity and convert to loss
    cosine_sim = F.cosine_similarity(a_flat, b_flat, dim=-1)
    loss = 1 - cosine_sim  # cosine loss = 1 - cosine similarity

    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss.reshape(*a.shape[:-1])
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")

# Recon Losses for BSP & BSV
def mse_loss(a, b, reduction='mean'):
    """
    Computes mean squared error (MSE) between tensors a and b.

    Args:
        a (Tensor): Predicted tensor.
        b (Tensor): Target tensor.
        reduction (str): 'mean', 'sum', or 'none'.

    Returns:
        Tensor: The computed loss.
    """
    loss = F.mse_loss(a, b, reduction=reduction)
    return loss

def bsv_kld_loss(mu, logvar, **kwargs):
    """
    Compute the KL divergence between N(mu, sigma^2) and N(0, I).
    
    Args:
        mu (Tensor): Mean tensor from the encoder (B, D)
        logvar (Tensor): Log-variance tensor from the encoder (B, D)
        
    Returns:
        Tensor: Scalar KLD loss
    """
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kld.mean()

def bsp_kld_loss(mu, logvar, **kwargs):
    """
    Compute the KL divergence between N(mu, sigma^2) and N(0, I)
    for inputs of shape (B, S, D), where:
        B = batch size,
        S = sequence length,
        D = latent dimension.

    Args:
        mu (Tensor): Mean tensor from the encoder (B, S, D)
        logvar (Tensor): Log-variance tensor from the encoder (B, S, D)
        bsv_kld_weight (float): Weighting factor for the KL divergence.

    Returns:
        Tensor: Scalar KLD loss
    """
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # (B, S)
    return kld.mean()  # Average over batch and sequence


def rbf_kernel_matrix(t, lengthscale, variance):
    """
    Compute the RBF kernel matrix for time points `t` of shape [T].
    Returns [T, T] kernel matrix.
    """
    T = t.shape[0]
    t = t.view(-1, 1)  # [T, 1]
    dists = (t - t.T) ** 2  # [T, T]
    K = variance * torch.exp(-0.5 * dists / (lengthscale**2))
    return K

def gp_prior_loss(z, bsp_lengthscale, bsp_variance, bsp_gp_weight, noise=1e-4, **kwargs):
    """
    z: [B, T, D]
    Vectorized GP prior loss over all latent dimensions and batch elements.
    """
    B, T, D = z.shape
    device = z.device
    dtype = z.dtype

    # Compute kernel matrix [T, T], same for all batch since time points are identical
    t = torch.arange(T, dtype=dtype, device=device)
    K = rbf_kernel_matrix(t, bsp_lengthscale, bsp_variance)
    K += noise * torch.eye(T, device=device)

    # Cholesky decomposition [T, T]
    L = torch.linalg.cholesky(K)

    # We need to solve for each batch and each latent dim:
    # For that, reshape z to [T, B*D] so cholesky_solve can batch solve
    z_reshaped = z.permute(1, 0, 2).reshape(T, B * D)  # [T, B*D]

    # Solve linear system for each column independently
    solve1 = torch.cholesky_solve(z_reshaped, L)  # [T, B*D]

    # Compute Mahalanobis term for each batch and dim: sum over T for each column
    mahalanobis = (z_reshaped * solve1).sum(dim=0)  # [B*D]

    # Reshape back to [B, D]
    mahalanobis = mahalanobis.view(B, D)  # [B, D]

    # Log determinant is scalar, same for all batch/dim
    log_det_K = 2.0 * torch.sum(torch.log(torch.diagonal(L)))

    # Compute per-batch GP loss: sum over dims
    loss_per_batch = 0.5 * (mahalanobis.sum(dim=1) + D * log_det_K + D * T * torch.log(torch.tensor(2 * torch.pi, device=device, dtype=dtype)))

    # Average over batch, normalize over D and T (optional)
    loss = loss_per_batch.mean() / (D * T)

    return loss * bsp_gp_weight

