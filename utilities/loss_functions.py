import torch
from torch import nn
import torch.nn.functional as F

'''
@author: grahamwjohnson
'''

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
        prior_weights (Tensor): Mixture weights of the prior distribution (mog_component).

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

def mogpreds_intersequence_diversity_loss(mogpreds, mogpreds_diversity_weight, **kwargs):
    """
    Compute the diversity loss for MoG predictions, promoting sequences to be far apart in the latent space.
    mogpreds: MoG component probabilities, shape (batch_size, T, K)
    mogpreds_diversity_weight: Weight for the diversity loss
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

    # Diversity loss: minimize average pairwise similarity
    diversity_loss = avg_pairwise_sim

    # Return the diversity loss (weighted)
    return mogpreds_diversity_weight * diversity_loss

def mogpreds_intrasequence_consistency_loss(mogpreds, mogpreds_consistency_weight, **kwargs):
    """
    Compute the consistency loss for MoG predictions, rewarding similar mogpreds WITHIN sequences.
    mogpreds: MoG component probabilities, shape (batch_size, T, K)
    mogpreds_consistency_weight: Weight for the consistency loss
    """
    # Ensure mogpreds is a valid probability distribution
    assert torch.all(mogpreds >= 0), "mogpreds contains negative values"
    assert torch.allclose(mogpreds.sum(dim=-1), torch.ones_like(mogpreds.sum(dim=-1))), "mogpreds does not sum to 1"

    # Clamp mogpreds to avoid log(0)
    mogpreds = torch.clamp(mogpreds, min=1e-10, max=1.0)

    # Compute the variance of mogpreds within each sequence
    # Shape: (batch_size, T, K) -> (batch_size, K)
    variance = torch.var(mogpreds, dim=1)

    # Average the variance across components and sequences
    consistency_loss = variance.mean()

    # Return the consistency loss (weighted)
    return mogpreds_consistency_weight * consistency_loss

def mogpreds_entropy_loss(mogpreds, mogpreds_entropy_weight, **kwargs):
    """
    Compute the entropy loss for MoG predictions, promoting entropy across the entire dataset.
    mogpreds: MoG component probabilities, shape (batch_size, T, K)
    mogpreds_entropy_weight: Weight for the entropy loss
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
    entropy = -torch.sum(aggregated_probs * torch.log(aggregated_probs))

    # Return the negative entropy (maximize entropy to promote diverse component usage)
    return -mogpreds_entropy_weight * entropy

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

# def mog_loss(encoder_means, encoder_logvars, encoder_mogpreds, mog_prior, weight, gumbel_softmax_temperature, **kwargs):
#     """
#     Compute the KL divergence between q(z|x) and the MoG prior p(z).
#     Inputs:
#         encoder_means: [B, K, latent_dim] - Means of the posterior mixture components.
#         encoder_logvars: [B, K, latent_dim] - Log variances of the posterior mixture components.
#         encoder_mogpreds: [B, K] - Logits for the component weights.
#         mog_prior: Instance of MoGPrior.
#         weight: Weight for the KL divergence term.
#         gumbel_softmax_temperature: Temperature for Gumbel-Softmax.
#     Outputs:
#         normalized_loss: Scalar loss value.
#         z: [B, latent_dim] - Sampled latent variable.
#     """
#     B, K, latent_dim = encoder_means.shape

#     # Step 1: Gumbel-Softmax for differentiable component selection
#     gumbel_noise = -torch.log(-torch.log(torch.rand_like(encoder_mogpreds)))
#     logits = (encoder_mogpreds + gumbel_noise) / gumbel_softmax_temperature  # Scale by temperature
#     log_component_weights = F.log_softmax(logits, dim=-1)  # [B, K]
#     component_weights = torch.exp(log_component_weights)  # [B, K]

#     # Step 2: Compute posterior means and logvars using component weights
#     selected_means = torch.sum(encoder_means * component_weights.unsqueeze(-1), dim=1)  # [B, latent_dim]
#     selected_logvars = torch.sum(encoder_logvars * component_weights.unsqueeze(-1), dim=1)  # [B, latent_dim]
#     selected_logvars = torch.clamp(selected_logvars, min=-5, max=5)  # Clamp logvars for stability

#     # Step 3: Reparameterization trick to sample z
#     eps = torch.randn_like(selected_means)
#     z = selected_means + eps * torch.exp(0.5 * selected_logvars)  # [B, latent_dim]
#     z = torch.clamp(z, min=-10, max=10)  # Clamp z to prevent extreme values

#     # Step 4: Compute log q(z|x) (posterior)
#     log_qzx_per_component = -0.5 * (
#         encoder_logvars +
#         torch.pow(z.unsqueeze(1) - encoder_means, 2) / torch.exp(encoder_logvars) +
#         torch.log(2 * torch.tensor(torch.pi))  # Remove D from the constant term
#     ).mean(dim=-1)  # [B, K]

#     log_qzx = torch.logsumexp(log_component_weights + log_qzx_per_component, dim=-1) + 1e-6  # To avoid log(0) # [B]

#     # Step 5: Compute log p(z) (prior)
#     log_pz = mog_prior(z)  # [B]

#     # Step 6: Compute KL divergence
#     kl_div = log_qzx - log_pz  # [B]
#     kl_div = kl_div.mean() / (latent_dim * B) # Scalar

#     if kl_div < 0:
#         raise Exception("ERROR: DEBUGGING - negative values of kl")

#     # Step 7: Normalize the loss
#     normalized_loss = weight * kl_div

#     return normalized_loss, z

# def mog_loss(encoder_means, encoder_logvars, encoder_mogpreds, mog_prior, weight, gumbel_softmax_temperature, **kwargs):
#     """
#     Compute the KL divergence between q(z|x) and the MoG prior p(z).
#     encoder_means: Encoder means, shape (batch_size, T, K, D)
#     encoder_logvars: Encoder log-variances, shape (batch_size, T, K, D)
#     encoder_mogpreds: Encoder component logits, shape (batch_size, T, K)
#     mog_prior: Instance of MoGPrior
#     weight: Weight of the KL loss
#     gumbel_softmax_temperature: Temperature for Gumbel-Softmax
#     """
#     batch_size, T, K, D = encoder_means.shape

#     # Step 1: Use Gumbel-Softmax to sample component weights DIFFERENTIABLY
#     # Generate Gumbel noise
#     gumbel_noise = -torch.log(-torch.log(torch.rand_like(encoder_mogpreds)))  # Shape: (batch_size, T, K)

#     # Apply Gumbel-Softmax
#     logits = encoder_mogpreds + gumbel_noise  # Add Gumbel noise to raw logits
#     logits = torch.clamp(logits, min=-10, max=10)  # Clamp logits to avoid extreme values
#     component_weights = torch.softmax(logits / gumbel_softmax_temperature, dim=-1)  # Shape: (batch_size, T, K)
#     component_weights = torch.clamp(component_weights, min=1e-10, max=1.0)  # Avoid log(0)

#     # Step 2: Compute weighted means and log-variances for each token
#     selected_means = torch.sum(encoder_means * component_weights.unsqueeze(-1), dim=2)  # Shape: (batch_size, T, D)
#     selected_logvars = torch.sum(encoder_logvars * component_weights.unsqueeze(-1), dim=2)  # Shape: (batch_size, T, D)
#     selected_logvars = torch.clamp(selected_logvars, min=-10, max=10)  # Clamp logvars to avoid extreme values

#     # Step 3: Reparameterization trick to sample z at the TOKEN level
#     eps = torch.randn_like(selected_means)  # Shape: (batch_size, T, D)
#     z_token = selected_means + eps * torch.exp(0.5 * selected_logvars)  # Shape: (batch_size, T, D)

#     # Step 4: Compute log q(z|x) at the TOKEN level
#     log_qzx = -0.5 * (
#         selected_logvars +
#         torch.pow(z_token - selected_means, 2) / torch.exp(selected_logvars) +
#         D * torch.log(2 * torch.tensor(torch.pi))
#     ).sum(dim=-1)  # Shape: (batch_size, T)

#     # Step 5: Compute log p(z) under the MoG prior at the TOKEN level
#     z_flat = z_token.view(-1, D)  # Shape: (batch_size * T, D)
#     log_pz = mog_prior(z_flat)  # Shape: (batch_size * T,)
#     log_pz = log_pz.view(batch_size, T)  # Shape: (batch_size, T)

#     # Step 6: Compute KL divergence at the TOKEN level
#     kl_div = log_qzx - log_pz  # Shape: (batch_size, T)

#     # Step 7: Average KL divergence over tokens and batch for global regularization
#     kl_div = kl_div.mean()  # Scalar

#     # Step 8: Normalize the loss by the number of tokens and latent dimensions
#     normalized_loss = weight * kl_div / (batch_size * T * D)

#     # Step 9: Return the loss, token-level z, and other values
#     return normalized_loss, z_token

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


