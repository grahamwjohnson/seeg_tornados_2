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
    
def adversarial_loss_function(class_probs, file_class_label, classifier_weight):

    # Class probs comes in as [batch, seq, num_classes] softmax
    # Change to [batch * seq, num_classes] softmax
    # class_probs_batched = class_probs.reshape(class_probs.shape[0] * class_probs.shape[1], -1)

    # Must repeat the file labels for entire sequence
    # labels_repeated = file_class_label.unsqueeze(1).repeat(1, class_probs.shape[1])
    # labels_batched = torch.squeeze(labels_repeated.reshape(labels_repeated.shape[0] * labels_repeated.shape[1], -1))
    # adversarial_loss = nn.functional.cross_entropy(class_probs_batched, labels_batched)

    adversarial_loss = nn.functional.cross_entropy(class_probs, file_class_label) / torch.log(torch.tensor(class_probs.shape[1]))

    return classifier_weight * adversarial_loss

# Gaussian kernel for MMD
def gaussian_kernel(x, y, sigma=1.0):
    return torch.exp(-torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=2) ** 2 / (2 * sigma ** 2))

def mmd_loss_function(z, weight):

    z_batched = z.reshape(z.shape[0] * z.shape[1], z.shape[2])

    # Sample from standard Gaussian prior
    prior_samples = torch.randn_like(z_batched)

    # Compute MMD
    z_kernel = gaussian_kernel(z_batched, z_batched)
    prior_kernel = gaussian_kernel(prior_samples, prior_samples)
    cross_kernel = gaussian_kernel(z_batched, prior_samples)
    mmd = z_kernel.mean() + prior_kernel.mean() - 2 * cross_kernel.mean()

    return mmd * weight