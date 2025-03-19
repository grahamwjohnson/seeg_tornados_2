import torch
from torch import nn
from geomloss import SamplesLoss

'''
@author: grahamwjohnson
'''

def mogpreds_intersequence_diversity_loss(mogpreds_softmax, mogpreds_diversity_weight, **kwargs):
    """
    Compute the diversity loss for MoG predictions, promoting sequences to be far apart in the latent space.
    mogpreds_softmax: MoG component probabilities, shape (batch_size, T, K)
    mogpreds_diversity_weight: Weight for the diversity loss
    """
    # Ensure mogpreds_softmax is a valid probability distribution
    assert torch.all(mogpreds_softmax >= 0), "mogpreds_softmax contains negative values"
    assert torch.allclose(mogpreds_softmax.sum(dim=-1), torch.ones_like(mogpreds_softmax.sum(dim=-1))), "mogpreds_softmax does not sum to 1"

    # Clamp mogpreds_softmax to avoid log(0)
    mogpreds_softmax = torch.clamp(mogpreds_softmax, min=1e-10, max=1.0)

    # Compute the mean prediction for each sequence (across time steps)
    # Shape: (batch_size, T, K) -> (batch_size, K)
    mean_mogpreds = mogpreds_softmax.mean(dim=1)

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

    # Diversity loss: minimize average pairwise similarity
    diversity_loss = avg_pairwise_sim

    # Return the diversity loss (weighted)
    return mogpreds_diversity_weight * diversity_loss

def mogpreds_intrasequence_consistency_loss(mogpreds_softmax, mogpreds_consistency_weight, **kwargs):
    """
    Compute the consistency loss for MoG predictions, rewarding similar mogpreds WITHIN sequences.
    mogpreds_softmax: MoG component probabilities, shape (batch_size, T, K)
    mogpreds_consistency_weight: Weight for the consistency loss
    """
    # Ensure mogpreds_softmax is a valid probability distribution
    assert torch.all(mogpreds_softmax >= 0), "mogpreds_softmax contains negative values"
    assert torch.allclose(mogpreds_softmax.sum(dim=-1), torch.ones_like(mogpreds_softmax.sum(dim=-1))), "mogpreds_softmax does not sum to 1"

    # Clamp mogpreds_softmax to avoid log(0)
    mogpreds_softmax = torch.clamp(mogpreds_softmax, min=1e-10, max=1.0)

    # Compute the variance of mogpreds within each sequence
    # Shape: (batch_size, T, K) -> (batch_size, K)
    variance = torch.var(mogpreds_softmax, dim=1)

    # Average the variance across components and sequences
    consistency_loss = variance.mean()

    # Return the consistency loss (weighted)
    return mogpreds_consistency_weight * consistency_loss

def mogpreds_entropy_loss(mogpreds_softmax, mogpreds_entropy_weight, **kwargs):
    """
    Compute the entropy loss for MoG predictions, promoting entropy ACROSS sequences.
    mogpreds_softmax: MoG component probabilities, shape (batch_size, T, K)
    mogpreds_entropy_weight: Weight for the entropy loss
    """
    # Ensure mogpreds_softmax is a valid probability distribution
    assert torch.all(mogpreds_softmax >= 0), "mogpreds_softmax contains negative values"
    assert torch.allclose(mogpreds_softmax.sum(dim=-1), torch.ones_like(mogpreds_softmax.sum(dim=-1))), "mogpreds_softmax does not sum to 1"

    # Clamp mogpreds_softmax to avoid log(0)
    mogpreds_softmax = torch.clamp(mogpreds_softmax, min=1e-10, max=1.0)

    # Aggregate predictions across sequences (batch and time dimensions)
    # Shape: (batch_size * T, K) -> (K,)
    aggregated_probs = mogpreds_softmax.mean(dim=(0, 1))  # Average over batch and time

    # Compute entropy of the aggregated distribution
    entropy = -torch.sum(aggregated_probs * torch.log(aggregated_probs))

    # Return the negative entropy (to maximize entropy across sequences)
    return (-1) * mogpreds_entropy_weight * entropy

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

def mog_loss(encoder_means, encoder_logvars, encoder_mogpreds, mog_prior, weight, gumbel_softmax_temperature, **kwargs):
    """
    Compute the KL divergence between q(z|x) and the MoG prior p(z).
    encoder_means: Encoder means, shape (batch_size, T, K, D)
    encoder_logvars: Encoder log-variances, shape (batch_size, T, K, D)
    encoder_mogpreds: Encoder component logits, shape (batch_size, T, K)
    mog_prior: Instance of MoGPrior
    weight: Weight of the KL loss
    gumbel_softmax_temperature: Temperature for Gumbel-Softmax
    """
    batch_size, T, K, D = encoder_means.shape

    # Step 1: Use Gumbel-Softmax to sample component weights DIFFERENTIABLY
    # Generate Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(encoder_mogpreds)))  # Shape: (batch_size, T, K)

    # Apply Gumbel-Softmax
    logits = encoder_mogpreds + gumbel_noise  # Add Gumbel noise to raw logits
    logits = torch.clamp(logits, min=-10, max=10)  # Clamp logits to avoid extreme values
    component_weights = torch.softmax(logits / gumbel_softmax_temperature, dim=-1)  # Shape: (batch_size, T, K)
    component_weights = torch.clamp(component_weights, min=1e-10, max=1.0)  # Avoid log(0)

    # Step 2: Compute weighted means and log-variances for each token
    selected_means = torch.sum(encoder_means * component_weights.unsqueeze(-1), dim=2)  # Shape: (batch_size, T, D)
    selected_logvars = torch.sum(encoder_logvars * component_weights.unsqueeze(-1), dim=2)  # Shape: (batch_size, T, D)
    selected_logvars = torch.clamp(selected_logvars, min=-10, max=10)  # Clamp logvars to avoid extreme values

    # Step 3: Reparameterization trick to sample z at the TOKEN level
    eps = torch.randn_like(selected_means)  # Shape: (batch_size, T, D)
    z_token = selected_means + eps * torch.exp(0.5 * selected_logvars)  # Shape: (batch_size, T, D)

    # Step 4: Compute log q(z|x) at the TOKEN level
    log_qzx = -0.5 * (
        selected_logvars +
        torch.pow(z_token - selected_means, 2) / torch.exp(selected_logvars) +
        D * torch.log(2 * torch.tensor(torch.pi))
    ).sum(dim=-1)  # Shape: (batch_size, T)

    # Step 5: Compute log p(z) under the MoG prior at the TOKEN level
    z_flat = z_token.view(-1, D)  # Shape: (batch_size * T, D)
    log_pz = mog_prior(z_flat)  # Shape: (batch_size * T,)
    log_pz = log_pz.view(batch_size, T)  # Shape: (batch_size, T)

    # Step 6: Compute KL divergence at the TOKEN level
    kl_div = log_qzx - log_pz  # Shape: (batch_size, T)

    # Step 7: Average KL divergence over tokens and batch for global regularization
    kl_div = kl_div.mean()  # Scalar

    # Step 8: Normalize the loss by the number of tokens and latent dimensions
    normalized_loss = weight * kl_div / (batch_size * T * D)

    # Step 9: Return the loss, token-level z, and other values
    return normalized_loss, z_token

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


