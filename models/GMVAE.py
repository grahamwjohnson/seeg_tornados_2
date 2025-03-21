import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch import Tensor
from torchinfo import summary

# Local imports
from .Transformer import ModelArgs, Transformer, RMSNorm
from utilities.loss_functions import adversarial_loss_function
from utilities.loss_functions import gmvae_kl_loss
import torch.distributions as dist

'''
@author: grahamwjohnson
2023-2025 

'''

from geomloss import SamplesLoss  # Import GeomLoss for Sinkhorn computation

class MoGPrior(nn.Module):
    def __init__(self, K, latent_dim, prior_initial_mean_spread, prior_initial_logvar, mean_lims, Wasserstein_order, Wasserstein_Sinkhorn_eps, gumbel_softmax_temperature, **kwargs):
        """
        Mixture of Gaussians (MoG) prior with Wasserstein stabilization using Sinkhorn loss.
        Inputs:
            K: Number of mixture components.
            latent_dim: Dimensionality of the latent space.
            prior_initial_mean_spread: Spread for uniform initialization of prior means.
            prior_initial_logvar: Initial log variance for all components.
            sinkhorn_eps: Entropic regularization coefficient for Sinkhorn loss.
            sinkhorn_niter: Number of iterations for Sinkhorn solver.
        """
        super(MoGPrior, self).__init__()
        self.K = K
        self.latent_dim = latent_dim
        self.Wasserstein_order = Wasserstein_order
        self.sinkhorn_eps = Wasserstein_Sinkhorn_eps
        self.mean_lims = mean_lims
        self.gumbel_softmax_temperature = gumbel_softmax_temperature
        
        # Uniformly initialize means in the range (-initial_mean_spread, initial_mean_spread)
        prior_means = (torch.rand(K, latent_dim) * 2 * prior_initial_mean_spread) - prior_initial_mean_spread

        # Initialize means, logvars, and weights
        self.means = nn.Parameter(prior_means)  # [K, latent_dim]
        self.logvars = nn.Parameter(torch.ones(K, latent_dim) * prior_initial_logvar)  # [K, latent_dim]
        self.weights = nn.Parameter(torch.ones(K) / K)  # [K]

        # Define Sinkhorn loss function
        self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=self.Wasserstein_order, blur=self.sinkhorn_eps, scaling=0.9, debias=True, backend="tensorized")

    def sample_prior(self, batch_size):
        """
        Sample from the MoG prior using Gumbel-Softmax for differentiable component selection.
        
        Inputs:
            batch_size: Number of samples to generate.
            self.gumbel_softmax_temperature: Temperature parameter for Gumbel-Softmax.
        
        Returns:
            z_prior: Differentiable samples from the MoG prior [batch_size, latent_dim]
            component_probs: Soft assignments of each sample to the MoG components [batch_size, K]
        """
        # Sample Gumbel noise and apply softmax to get soft component selection
        gumbel_noise = -torch.log(-torch.log(torch.rand(batch_size, self.K, device=self.weights.device) + 1e-20) + 1e-20)
        logits = torch.log_softmax(self.weights, dim=0) + gumbel_noise
        component_probs = F.gumbel_softmax(logits, tau=self.gumbel_softmax_temperature, hard=False)  # [batch_size, K], fully differentiable

        # Compute mixture samples using soft probabilities
        chosen_means = torch.matmul(component_probs, self.means)  # [batch_size, latent_dim]
        chosen_logvars = torch.matmul(component_probs, self.logvars)  # [batch_size, latent_dim]

        # Sample from Gaussian with chosen means and logvars
        eps = torch.randn_like(chosen_means)
        z_prior = chosen_means + eps * torch.exp(0.5 * chosen_logvars)  # [batch_size, latent_dim]

        z_prior = torch.clamp(z_prior, min=self.mean_lims[0], max=self.mean_lims[1])

        return z_prior, component_probs

    def sinkhorn_loss_fraction(self, z, weight):
        """
        Compute the Sinkhorn loss between sampled latent vectors and the MoG prior.
        Inputs:
            z: Resampled latent embeddings [batch_size, latent_dim]
        Returns:
            sinkhorn_loss: Wasserstein distance between samples and prior.
        """
        batch_size = z.shape[0]

        # Sample from the prior using Gumbel-Softmax
        z_prior, _ = self.sample_prior(batch_size)

        # Compute Sinkhorn loss (Wasserstein distance between z_samples and z_prior)
        sinkhorn_loss = self.sinkhorn_loss(z, z_prior)

        return (sinkhorn_loss * weight) / (self.latent_dim * self.K)

class RMSNorm_Conv(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight.unsqueeze(1).repeat(1, x.shape[2])

class TimeSeriesCrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(TimeSeriesCrossAttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = RMSNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = RMSNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # Cross-attention: query comes from the target, key and value come from the source
        # query: (seq_len, batch_size, embed_dim)
        # key, value: (seq_len, batch_size, embed_dim)

        # Cross-attention layer
        attn_output, _ = self.attn(query, key, value)  # q=query, k=key, v=value
        query = self.norm1(query + self.dropout1(attn_output))  # Residual connection

        # Feed-forward network
        ffn_output = self.ffn(query)
        query = self.norm2(query + self.dropout2(ffn_output))  # Residual connection
        
        return query

class Encoder_TimeSeriesWithCrossAttention(nn.Module):
    def __init__(self, 
        # in_channels, 
        padded_channels,
        crattn_embed_dim,
        crattn_num_highdim_heads,
        crattn_num_highdim_layers,
        crattn_num_lowdim_heads,
        crattn_num_lowdim_layers,
        crattn_max_seq_len,
        crattn_dropout, 
        **kwargs):

        super(Encoder_TimeSeriesWithCrossAttention, self).__init__()
        
        # self.in_channels = in_channels
        self.padded_channels = padded_channels
        self.embed_dim = crattn_embed_dim
        self.num_highdim_heads = crattn_num_highdim_heads
        self.num_highdim_layers = crattn_num_highdim_layers
        self.num_lowdim_heads = crattn_num_lowdim_heads
        self.num_lowdim_layers = crattn_num_lowdim_layers
        self.max_seq_len = crattn_max_seq_len
        self.dropout = crattn_dropout

        # Input Cross-attention Layer 
        self.highdim_attention_layers = nn.ModuleList([
            TimeSeriesCrossAttentionLayer(self.padded_channels, self.num_highdim_heads, self.dropout)
            for _ in range(self.num_highdim_layers)
        ])

        # Convert to embed dims
        self.high_to_low_dims = nn.Sequential(
            nn.Linear(self.padded_channels, self.padded_channels * 2),
            nn.SiLU(),
            nn.Linear(self.padded_channels * 2, self.padded_channels),
            nn.SiLU(),
            nn.Linear(self.padded_channels, self.embed_dim),
            nn.SiLU()
        )

        # Embed-Dim Cross-attention layers
        self.lowdim_attention_layers = nn.ModuleList([
            TimeSeriesCrossAttentionLayer(self.embed_dim, self.num_lowdim_heads, self.dropout)
            for _ in range(self.num_lowdim_layers)
        ])
        
        # Positional encoding
        self.positional_encoding = self._get_positional_encoding(self.max_seq_len, self.padded_channels)
        
    def _get_positional_encoding(self, max_seq_len, dim):
        # Dummey even value to make code below work
        dim_even = math.ceil(dim / 2) * 2

        # Get positional encodings for input tokens
        pe = torch.zeros(max_seq_len, dim_even)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim_even, 2).float() * -(torch.log(torch.tensor(10000.0)) / dim_even))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe[:, :, :dim]

    def forward(self, x):
        # inputs: a single time series of shape (batch_size, num_channels, seq_len)        
        # in_channels = x.shape[1]

        # # Step 1: pad the channel dimension
        # padding = (0, 0, 0, self.padded_channels - in_channels, 0, 0) # (dim0 left padding, dim0 right padding... etc.)
        # x = F.pad(x, padding, mode='constant', value=0)

        # Step 2: Permute to (batch_size, seq_len, padded_channels)
        x = x.permute(0, 2, 1)  # Shape: (batch_size, seq_len, padded_channels)
        
        # Step 3: Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device)
        
        # Step 4: Reshape for multi-head attention (seq_len, batch_size, high_dim)
        x = x.permute(1, 0, 2)  # Shape: (seq_len, batch_size, padded_channels)

        # Step 5: Apply Input Cross-Attention at PADDED dimension 
        for layer in self.highdim_attention_layers:
            x = layer(x, x, x)  # Cross-attention (q=k=v)

        # Step 6: Convert from padded channel dims to embed dims
        x = self.high_to_low_dims(x)

        # Step 7: Apply Embed-Dim Cross-Attention Layers
        for layer in self.lowdim_attention_layers:
            x = layer(x, x, x)  # Cross-attention (q=k=v)

        x = x.permute(1, 0, 2)  # Return to shape (batch_size, seq_len, low_dim)

        return x.flatten(start_dim=1) # (batch, seq_len * low_dim)
        # return torch.mean(x, dim=1)
        # return x[:,-1,:] # Return last in sequence (should be embedded with meaning)

class Decoder_MLP(nn.Module):
    def __init__(self, gpu_id, latent_dim, decoder_base_dims, output_channels, decode_samples):
        super(Decoder_MLP, self).__init__()
        self.gpu_id = gpu_id
        self.latent_dim = latent_dim
        self.decoder_base_dims = decoder_base_dims
        self.output_channels = output_channels
        self.decode_samples = decode_samples

        # Non-autoregressive decoder 
        self.non_autoregressive_fc = nn.Sequential(
            nn.Linear(latent_dim, decoder_base_dims * decode_samples),
            nn.SiLU(),
            RMSNorm(decoder_base_dims * decode_samples),
            nn.Linear(decoder_base_dims * decode_samples, decoder_base_dims * decode_samples),
            nn.SiLU(),
            RMSNorm(decoder_base_dims * decode_samples))
                
        # Now FC without norms, after reshaping so that each token is seperated
        self.non_autoregressive_output = nn.Sequential(
            nn.Linear(decoder_base_dims, output_channels),
            nn.Tanh())
            
    def forward(self, z):
        batch_size = z.size(0)
        
        # Step 1: Non-autoregressive generation 
        h_na = self.non_autoregressive_fc(z).view(batch_size, -1, self.decode_samples, self.decoder_base_dims)
        x_na = self.non_autoregressive_output(h_na)  # (batch_size, trans_seq_len, decode_samples, output_channels)
        
        return x_na

# Gradient Reversal Layer
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None # None is for alpha 

# Wrapper for Gradient Reversal
class GradientReversal(nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()

    def forward(self, x, alpha):
        return GradientReversalLayer.apply(x, alpha)

class LinearWithDropout(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.1):
        super(LinearWithDropout, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.linear(x)  # Apply linear transformation
        x = self.dropout(x)  # Apply dropout internally
        return x

# Define the Adversarial Classifier with Gradient Reversal
class AdversarialClassifier(nn.Module):
    def __init__(self, latent_dim, classifier_hidden_dims, classifier_num_pats, classifier_dropout, **kwargs):
        super(AdversarialClassifier, self).__init__()
        self.gradient_reversal = GradientReversal()
        
        self.classifier_dropout = classifier_dropout
        self.mlp_layers = nn.ModuleList()

        # Input layer
        self.mlp_layers.append(nn.Linear(latent_dim, classifier_hidden_dims[0]))
        self.mlp_layers.append(nn.SiLU())
        # self.mlp_layers.append(RMSNorm(classifier_hidden_dims[0]))

        # Hidden layers
        for i in range(len(classifier_hidden_dims) - 1):
            self.mlp_layers.append(LinearWithDropout(classifier_hidden_dims[i], classifier_hidden_dims[i + 1], classifier_dropout))
            self.mlp_layers.append(nn.SiLU())
            # self.mlp_layers.append(RMSNorm(classifier_hidden_dims[i + 1]))

        # Output layer
        self.mlp_layers.append(nn.Linear(classifier_hidden_dims[-1], classifier_num_pats)) # No activation and no norm

        # Softmax the output
        self.softmax = nn.Softmax(dim=1)

    def forward(self, mu, alpha):
        mu = self.gradient_reversal(mu, alpha)
        for layer in self.mlp_layers:
            mu = layer(mu)
        return self.softmax(mu)

# Define the MoG Predictor 
class MoGPredictor(nn.Module):
    def __init__(self, top_dim, mogpred_hidden_dim_list, num_mog_components, mog_predictor_dropout, **kwargs):
        super(MoGPredictor, self).__init__()

        self.dropout = mog_predictor_dropout
        self.mlp_layers = nn.ModuleList()

        # Input layer
        self.mlp_layers.append(nn.Linear(top_dim, mogpred_hidden_dim_list[0]))
        self.mlp_layers.append(nn.SiLU())
        self.mlp_layers.append(RMSNorm(mogpred_hidden_dim_list[0]))

        # Hidden layers
        for i in range(len(mogpred_hidden_dim_list) - 1):
            self.mlp_layers.append(LinearWithDropout(mogpred_hidden_dim_list[i], mogpred_hidden_dim_list[i + 1], self.dropout))
            self.mlp_layers.append(nn.SiLU())
            self.mlp_layers.append(RMSNorm(mogpred_hidden_dim_list[i + 1]))

        # Output layer
        self.mlp_layers.append(nn.Linear(mogpred_hidden_dim_list[-1], num_mog_components)) # No activation and no norm

    def forward(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
        return x # Do NOT softmax here, Gumbel-Softmax trick needs logits 

class GMVAE(nn.Module):
    '''
    GMVAE Reverseable Encoder/Decoder 
    '''
    def __init__(
        self, 
        encode_token_samples,
        padded_channels,
        crattn_embed_dim,
        transformer_seq_length,
        transformer_start_pos,
        transformer_dim,
        encoder_transformer_activation,
        top_dims,
        hidden_dims,
        latent_dim, 
        decoder_base_dims,
        mog_components,
        mean_lims,
        logvar_lims,
        gumbel_softmax_temperature,
        gpu_id=None,  
        **kwargs):

        super(GMVAE, self).__init__()

        self.gpu_id = gpu_id
        self.encode_token_samples = encode_token_samples
        self.padded_channels = padded_channels
        self.crattn_embed_dim = crattn_embed_dim
        self.transformer_seq_length = transformer_seq_length
        self.transformer_start_pos = transformer_start_pos
        self.transformer_dim = transformer_dim
        self.encoder_transformer_activation = encoder_transformer_activation
        self.top_dims = top_dims
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim 
        self.decoder_base_dims = decoder_base_dims
        self.mog_components = mog_components
        self.mean_lims = mean_lims
        self.logvar_lims = logvar_lims
        self.temperature = gumbel_softmax_temperature

        # Prior
        self.prior = MoGPrior(
            K = self.mog_components,
            latent_dim = self.latent_dim,
            mean_lims = self.mean_lims,
            gumbel_softmax_temperature=gumbel_softmax_temperature,
            **kwargs)

        # Raw CrossAttention Head
        self.encoder_head = Encoder_TimeSeriesWithCrossAttention(
            padded_channels=self.padded_channels, 
            crattn_embed_dim=self.crattn_embed_dim, 
            **kwargs)

        # Transformer - dimension is same as output of cross attention
        self.transformer_encoder = Transformer(ModelArgs(
            device=self.gpu_id, 
            dim=self.transformer_dim, 
            activation=self.encoder_transformer_activation, 
            **kwargs))

        # Core Encoder
        self.top_to_hidden = nn.Linear(self.top_dims, self.hidden_dims, bias=True)
        self.norm_hidden = RMSNorm(dim=self.hidden_dims)

        # Right before latent space
        self.mean_encode_layer = nn.Linear(self.hidden_dims, self.mog_components * self.latent_dim, bias=True)  
        self.logvar_encode_layer = nn.Linear(self.hidden_dims, self.mog_components * self.latent_dim, bias=True) 
        self.mogpreds_layer = MoGPredictor(
            top_dim=self.hidden_dims,
            num_mog_components=self.mog_components,
            **kwargs)

        # Decoder
        self.decoder = Decoder_MLP(
            gpu_id = self.gpu_id,
            latent_dim = self.latent_dim,
            decoder_base_dims = self.decoder_base_dims,
            output_channels = self.padded_channels,
            decode_samples = self.encode_token_samples)

        # Adversarial Classifier
        self.adversarial_classifier = AdversarialClassifier(latent_dim=self.latent_dim, **kwargs) # the name 'adversarial_classifier' is tied to model parameter search to create seperate optimizer for classifier

        # Non-linearity as needed
        self.silu = nn.SiLU()

    def forward(self, x, reverse=False, hash_pat_embedding=-1):

        if reverse == False:

            # CROSS-ATTENTION HEAD across channels on raw data
            y = x.reshape([x.shape[0]*x.shape[1], x.shape[2], x.shape[3]]) # [batch, token, channel, waveform] --> [batch x token, channel, waveform]
            y = self.encoder_head(y)
            y = torch.split(y, x.shape[1], dim=0) # [batch x token, latent_dim] --> [batch, token, latent_dim]
            y = torch.stack(y, dim=0)

            # TRANSFORMER
            y, attW = self.transformer_encoder(y, start_pos=self.transformer_start_pos, return_attW = True)

            # GMVAE CORE
            y = self.top_to_hidden(y)
            y = self.silu(y)
            y = self.norm_hidden(y)
            mean, logvar, mogpreds = self.mean_encode_layer(y), self.logvar_encode_layer(y), self.mogpreds_layer(y)
            mean = mean.view(-1, self.transformer_seq_length, self.mog_components, self.latent_dim)
            logvar = logvar.view(-1,self.transformer_seq_length, self.mog_components, self.latent_dim)

            # # Apply constrtains to mean/logvar
            # mean = torch.clamp(mean, min=self.mean_lims[0], max=self.mean_lims[1])
            # logvar = torch.clamp(logvar, min=self.logvar_lims[0], max=self.logvar_lims[1])
            mean_scale = (self.mean_lims[1] - self.mean_lims[0]) / 2
            mean_shift = (self.mean_lims[0] + self.mean_lims[1]) / 2
            mean = F.tanh(mean) * mean_scale + mean_shift

            logvar_scale = (self.logvar_lims[1] - self.logvar_lims[0]) / 2
            logvar_shift = (self.logvar_lims[0] + self.logvar_lims[1]) / 2
            logvar = F.tanh(logvar) * logvar_scale + logvar_shift

            # Now sample z
            mean_pseudobatch = mean.reshape(mean.shape[0]*mean.shape[1], mean.shape[2], mean.shape[3])
            logvar_pseudobatch = logvar.reshape(logvar.shape[0]*logvar.shape[1], logvar.shape[2], logvar.shape[3])
            mogpreds_pseudobatch = mogpreds.reshape(mogpreds.shape[0]*mogpreds.shape[1], mogpreds.shape[2])

            # Softmax the mogpreds for later use
            mogpreds_pseudobatch_softmax = torch.softmax(mogpreds_pseudobatch, dim=-1) 

            # Z is not currently constrained 
            z_pseudobatch, _ = self.sample_z(mean_pseudobatch, logvar_pseudobatch, mogpreds_pseudobatch, gumbel_softmax_temperature=self.temperature)
            
            return z_pseudobatch, mean_pseudobatch, logvar_pseudobatch, mogpreds_pseudobatch_softmax, attW

        elif reverse == True:

            # Add the hash_pat_embedding to latent vector
            y = x + hash_pat_embedding.unsqueeze(dim=1).repeat(1, x.shape[1], 1)

            # Transformer Decoder
            y = self.decoder(y).transpose(2,3)  # Comes out as [batch, token, waveform, num_channels] --> [batch, token, num_channels, waveform]

            return y

    # def sample_z(self, encoder_means, encoder_logvars, encoder_mogpreds, gumbel_softmax_temperature):
    #     """
    #     Sample z from the posterior using the Gumbel-Softmax trick and reparameterization.
        
    #     encoder_means: [batch, components, latent_dim] - Posterior means for each MoG component.
    #     encoder_logvars: [batch, components, latent_dim] - Posterior logvars for each MoG component.
    #     encoder_mogpreds: [batch, components] - Component logits for the MoG posterior.
    #     gumbel_softmax_temperature: Temperature for Gumbel-Softmax.
        
    #     Returns:
    #         z: [batch, latent_dim] - Sampled latent variable.
    #     """
    #     # Step 1: Gumbel-Softmax for differentiable component selection
    #     gumbel_noise = -torch.log(-torch.log(torch.rand_like(encoder_mogpreds)))
    #     logits = (encoder_mogpreds + gumbel_noise) / gumbel_softmax_temperature  # Scale by temperature
    #     log_component_weights = F.log_softmax(logits, dim=-1)  # [batch, components]
    #     component_weights = torch.exp(log_component_weights)  # [batch, components]

    #     # Step 2: Select the means and logvars for the selected component
    #     selected_means = torch.sum(encoder_means * component_weights.unsqueeze(-1), dim=1)  # [batch, latent_dim]
    #     selected_logvars = torch.sum(encoder_logvars * component_weights.unsqueeze(-1), dim=1)  # [batch, latent_dim]

    #     # Step 3: Reparameterization trick to sample z from the selected Gaussian component
    #     eps = torch.randn_like(selected_means)
    #     z = selected_means + eps * torch.exp(0.5 * selected_logvars)  # [batch, latent_dim]

    #     # Step 4: Clamp to ensure stability
    #     z = torch.clamp(z, min=self.mean_lims[0], max=self.mean_lims[1])

    #     return z

    def sample_z(self, encoder_means, encoder_logvars, encoder_mogpreds, gumbel_softmax_temperature):
        """
        Sample z from the posterior using the Gumbel-Softmax trick and reparameterization.

        encoder_means: [batch, components, latent_dim] - Posterior means for each MoG component.
        encoder_logvars: [batch, components, latent_dim] - Posterior logvars for each MoG component.
        encoder_mogpreds: [batch, components] - Component logits for the MoG posterior.
        gumbel_softmax_temperature: Temperature for Gumbel-Softmax.

        Returns:
            z: [batch, latent_dim] - Sampled latent variable.
            component_weights: [batch, components] - Soft assignments of each sample to the MoG components.
        """
        # Step 1: Gumbel-Softmax for differentiable component selection
        component_weights = F.gumbel_softmax(encoder_mogpreds, tau=gumbel_softmax_temperature, hard=False)  # [batch, components]

        # Step 2: Compute soft-selected means and logvars
        selected_means = torch.sum(encoder_means * component_weights.unsqueeze(-1), dim=1)  # [batch, latent_dim]
        selected_logvars = torch.sum(encoder_logvars * component_weights.unsqueeze(-1), dim=1)  # [batch, latent_dim]

        # Step 3: Reparameterization trick to sample z
        eps = torch.randn_like(selected_means)
        z = selected_means + eps * torch.exp(0.5 * selected_logvars)  # [batch, latent_dim]

        # # Step 4: Tanh for stability 
        # mean_scale = (self.mean_lims[1] - self.mean_lims[0]) / 2
        # mean_shift = (self.mean_lims[0] + self.mean_lims[1]) / 2
        # z = F.tanh(z) * mean_scale + mean_shift
        z = torch.clamp(z, min=-1, max=1)
        
        return z, component_weights

def print_models_flow(x, **kwargs):
    '''
    Builds models on CPU and prints sizes of forward passes with random data as inputs

    No losses are computed, just data flow through model
    
    '''

    # Build the WAE
    gmvae = GMVAE(**kwargs) 

    # Run through Encoder
    print(f"INPUT TO <ENC>\n"
    f"x:{x.shape}")
    z_pseudobatch, mean_pseudobatch, logvar_pseudobatch, mogpreds_pseudobatch, attW = gmvae(x, reverse=False)  
    summary(gmvae, input_size=(x.shape), depth=999, device="cpu")
    print(
    f"z_pseudobatch:{z_pseudobatch.shape}\n",
    f"mean_pseudobatch:{mean_pseudobatch.shape}\n"
    f"logvar_pseudobatch:{logvar_pseudobatch.shape}\n"
    f"mogpreds_pseudobatch:{mogpreds_pseudobatch.shape}\n"
    f"attW:{attW.shape}\n")
    
    z_token = z_pseudobatch.split(kwargs['transformer_seq_length'], dim=0)
    z_token = torch.stack(z_token, dim=0)

    # CLASSIFIER - on the mean of Z tokens
    z_meaned = torch.mean(z_token, dim=1)
    class_probs_mean_of_latent = gmvae.adversarial_classifier(z_meaned, alpha=1)

    # Run through WAE decoder
    hash_pat_embedding = torch.rand(x.shape[0], z_token.shape[2])
    hash_channel_order = np.arange(0, 199).tolist()
    print(f"\n\n\nINPUT TO <WAE - Decoder Mode> \n"
    f"z:{z_token.shape}\n"
    f"hash_pat_embedding:{hash_pat_embedding.shape}\n")
    summary(gmvae, input_data=[z_token, True, hash_pat_embedding], depth=999, device="cpu")
    core_out = gmvae(z_token, reverse=True, hash_pat_embedding=hash_pat_embedding)  
    print(f"decoder_out:{core_out.shape}\n")

    del gmvae

if __name__ == "__main__":

    kwargs = {
        'dummy': -1
    }
    
    in_channels = 1024
    kernel_sizes = [3,9,15]
    
    batchsize = 4
    data_length = 4
    time_change=True # will shrink/dilate time by 2 for every depth layer

    x = torch.rand(batchsize, in_channels, data_length, len(kernel_sizes))

    gmvae = GMVAE(
        encode_token_samples=data_length,
        in_channels=in_channels,
        kernel_sizes=kernel_sizes, 
        time_change=time_change,
        cnn_depth=2,
        cnn_resblock_layers=4,
        hidden_dims=2048,
        latent_dim=1024, 
        **kwargs
    )

    print(f"Are the weights of encoder and decoder tied? {torch.allclose(gmvae.top_to_hidden.weight.T, gmvae.hidden_to_top.weight)}")

