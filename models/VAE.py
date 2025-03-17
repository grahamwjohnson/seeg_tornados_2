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
from utilities.loss_functions import mog_loss
import torch.distributions as dist

'''
@author: grahamwjohnson
2023-2025 

'''

class MoGPrior(nn.Module):
    def __init__(self, K, D, **kwargs):
        """
        Mixture of Gaussians (MoG) prior.
        K: Number of components
        D: Latent space dimensionality
        """
        super(MoGPrior, self).__init__()
        self.K = K  # Number of mixture components
        self.D = D  # Latent dimension

        # Initialize means, logvars, and weights
        self.means = nn.Parameter(torch.randn(K, D))  # Shape: (K, D)
        self.logvars = nn.Parameter(torch.zeros(K, D))  # Shape: (K, D)
        self.weights = nn.Parameter(torch.ones(K) / K)  # Shape: (K,)

    def forward(self, z):
        """
        Compute the log probability of z under the MoG prior.
        z: Latent variable, shape (batch_size, D)
        """
        # Expand z to (batch_size, K, D) for broadcasting
        z = z.unsqueeze(1)  # Shape: (batch_size, 1, D)
        z = z.expand(-1, self.K, -1)  # Shape: (batch_size, K, D)

        # Compute log probability for each component
        log_probs = -0.5 * (
            self.logvars +
            torch.pow(z - self.means, 2) / torch.exp(self.logvars) +
            self.D * torch.log(2 * torch.tensor(torch.pi))
        )  # Shape: (batch_size, K, D)

        # Sum over dimensions to get log probability per component
        log_probs = log_probs.sum(dim=-1)  # Shape: (batch_size, K)

        # Add log weights and compute log mixture probability
        log_mixture_probs = torch.log_softmax(self.weights, dim=0)  # Shape: (K,)
        log_probs = log_probs + log_mixture_probs.unsqueeze(0)  # Shape: (batch_size, K)

        # Log-sum-exp to get the log probability of the mixture
        log_prob = torch.logsumexp(log_probs, dim=1)  # Shape: (batch_size,)

        return log_prob

# class MoGPrior(nn.Module):
#     def __init__(self, K, D, gumbel_softmax_temperature, **kwargs):
#         """
#         Mixture of Gaussians (MoG) prior with Gumbel-Softmax.
#         K: Number of components
#         D: Latent space dimensionality
#         temperature: Temperature for Gumbel-Softmax
#         """
#         super(MoGPrior, self).__init__()
#         self.K = K  # Number of mixture components
#         self.D = D  # Latent dimension
#         self.temperature = gumbel_softmax_temperature  # Temperature for Gumbel-Softmax

#         # Initialize means, logvars, and weights
#         self.means = nn.Parameter(torch.randn(K, D))  # Shape: (K, D)
#         self.logvars = nn.Parameter(torch.zeros(K, D))  # Shape: (K, D)
#         self.weights = nn.Parameter(torch.ones(K) / K)  # Shape: (K,)

#     def forward(self, z):
#         """
#         Compute the log probability of z under the MoG prior using Gumbel-Softmax.
#         z: Latent variable, shape (batch_size, D)
#         """
#         # Expand z to (batch_size, K, D) for broadcasting
#         z = z.unsqueeze(1)  # Shape: (batch_size, 1, D)
#         z = z.expand(-1, self.K, -1)  # Shape: (batch_size, K, D)

#         # Compute log probability for each component
#         log_probs = -0.5 * (
#             self.logvars +
#             torch.pow(z - self.means, 2) / torch.exp(self.logvars) +
#             self.D * torch.log(2 * torch.tensor(torch.pi))
#         )  # Shape: (batch_size, K, D)

#         # Sum over dimensions to get log probability per component
#         log_probs = log_probs.sum(dim=-1)  # Shape: (batch_size, K)

#         # Add log weights and compute log mixture probability
#         log_mixture_probs = torch.log_softmax(self.weights, dim=0)  # Shape: (K,)
#         log_probs = log_probs + log_mixture_probs.unsqueeze(0)  # Shape: (batch_size, K)

#         # Apply Gumbel-Softmax for discrete component selection
#         gumbel_sample = self.gumbel_softmax(log_probs)

#         # Log-sum-exp to get the log probability of the mixture
#         log_prob = torch.logsumexp(log_probs, dim=1)  # Shape: (batch_size,)

#         return log_prob, gumbel_sample

#     def gumbel_softmax(self, logits):
#         """
#         Gumbel-Softmax sampling for discrete clustering.
#         logits: Log probabilities for each MoG component.
#         """
#         # Gumbel noise
#         noise = torch.rand_like(logits).to(logits.device)
#         noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
        
#         # Apply Gumbel noise to logits and sample with softmax
#         gumbel_noise = logits + noise
#         return F.softmax(gumbel_noise / self.temperature, dim=-1)

# class MoGPrior(nn.Module):
#     def __init__(self, K, D):
#         """
#         Mixture of Gaussians (MoG) prior.
#         K: Number of components
#         D: Latent space dimensionality
#         """
#         super(MoGPrior, self).__init__()
#         self.K = K
#         self.D = D

#         # Initialize means, logvars, and weights
#         self.means = nn.Parameter(torch.randn(K, D))  # Shape: (K, D)
#         self.logvars = nn.Parameter(torch.zeros(K, D))  # Shape: (K, D)
#         self.weights = nn.Parameter(torch.ones(K) / K)  # Shape: (K,)

#     def forward(self, z):
#         """
#         Compute the log probability of z under the MoG prior.
#         z: Latent variable, shape (batch_size, D)
#         """

#         # Expand z to (batch_size, K, D) for broadcasting
#         z = z.unsqueeze(1)  # Shape: (batch_size, 1, D)
#         z = z.expand(-1, self.K, -1)  # Shape: (batch_size, K, D)

#         # Compute log probability for each component
#         log_probs = -0.5 * (
#             self.logvars +
#             torch.pow(z - self.means, 2) / torch.exp(self.logvars) +
#             self.D * torch.log(2 * torch.tensor(torch.pi))
#         )  # Shape: (batch_size, K, D)

#         # Sum over dimensions to get log probability per component
#         log_probs = log_probs.sum(dim=-1)  # Shape: (batch_size, K)

#         # Add log weights and compute log mixture probability
#         log_mixture_probs = torch.log_softmax(self.weights, dim=0)  # Shape: (K,)
#         log_probs = log_probs + log_mixture_probs.unsqueeze(0)  # Shape: (batch_size, K)

#         # Log-sum-exp to get the log probability of the mixture
#         log_prob = torch.logsumexp(log_probs, dim=1)  # Shape: (batch_size,)

#         return log_prob

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
            RMSNorm(decoder_base_dims * decode_samples),
            # nn.Linear(decoder_base_dims * decode_samples * 2, decoder_base_dims * decode_samples * 4),
            # nn.SiLU(),
            # RMSNorm(decoder_base_dims * decode_samples * 4),
            # nn.Linear(decoder_base_dims * decode_samples * 4, decoder_base_dims * decode_samples * 4),
            # nn.SiLU(),
            # RMSNorm(decoder_base_dims * decode_samples * 4),
            # nn.Linear(decoder_base_dims * decode_samples * 4, decoder_base_dims * decode_samples * 4),
            # nn.SiLU(),
            # RMSNorm(decoder_base_dims * decode_samples * 4)
            )        
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
    def __init__(self, input_dim, output_dim, dropout_prob=0.5):
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

class VAE(nn.Module):
    '''
    VAE Reverseable Encoder/Decoder 
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
        gpu_id=None,  
        **kwargs):

        super(VAE, self).__init__()

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

        # Prior
        self.prior = MoGPrior(
            K = self.mog_components,
            D = self.latent_dim,
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
        self.mogpreds_layer = nn.Linear(self.hidden_dims, self.mog_components, bias=True)

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

    # def reparameterize(self, mu, logvar):
    #     """
    #     Reparameterization trick to sample from N(mu, var) from N(0,1).
    #     :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    #     :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    #     :return: (Tensor) [B x D]
    #     """
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std

    # def reparameterize(self, mus, logvars, mogpreds):
    #     """
    #     Reparameterization trick for MoG-VAE with token-level encoding:
    #     - Selects a MoG component per token using mogpreds.
    #     - Samples token-level latents for decoder.
    #     - Computes the average latent for MoG regularization.

    #     Args:
    #         mus: (B, T, K, D) - Mean for each token and MoG component
    #         logvars: (B, T, K, D) - Log-variance for each token and MoG component
    #         mogpreds: (B, T, K) - MoG component probabilities per token (softmaxed)

    #     Returns:
    #         z_tokens: (B, T, D) - Sampled token-level latents
    #         z_avg: (B, D) - Averaged latent for MoG regularization
    #     """
    #     B, T, K, D = mus.shape

    #     # Sample one MoG component per token based on mogpreds
    #     component_indices = torch.multinomial(mogpreds.view(B * T, K), 1).view(B, T)  # (B, T)

    #     # Select corresponding mean and logvar for each token
    #     selected_means = mus[torch.arange(B).unsqueeze(1), torch.arange(T).unsqueeze(0), component_indices]  # (B, T, D)
    #     selected_logvars = logvars[torch.arange(B).unsqueeze(1), torch.arange(T).unsqueeze(0), component_indices]  # (B, T, D)

    #     # Reparameterization trick
    #     std = torch.exp(0.5 * selected_logvars)  # (B, T, D)
    #     eps = torch.randn_like(std)  # (B, T, D)
    #     z_tokens = selected_means + eps * std  # (B, T, D)

    #     # Compute the average latent across tokens for MoG regularization
    #     z_avg = z_tokens.mean(dim=1)  # (B, D)

    #     return z_tokens, z_avg, component_indices

    def forward(self, x, reverse=False, hash_pat_embedding=-1):

        if reverse == False:

            # CROSS-ATTENTION HEAD across channels on raw data
            y = x.reshape([x.shape[0]*x.shape[1], x.shape[2], x.shape[3]]) # [batch, token, channel, waveform] --> [batch x token, channel, waveform]
            y = self.encoder_head(y)
            y = torch.split(y, x.shape[1], dim=0) # [batch x token, latent_dim] --> [batch, token, latent_dim]
            y = torch.stack(y, dim=0)

            # TRANSFORMER
            y, attW = self.transformer_encoder(y, start_pos=self.transformer_start_pos, return_attW = True)

            # VAE CORE
            y = self.top_to_hidden(y)
            y = self.silu(y)
            y = self.norm_hidden(y)
            mean, logvar, mogpreds = self.mean_encode_layer(y), self.logvar_encode_layer(y), self.mogpreds_layer(y)
            mean = mean.view(-1, self.transformer_seq_length, self.mog_components, self.latent_dim)
            logvar = logvar.view(-1,self.transformer_seq_length, self.mog_components, self.latent_dim)

            # Reparametrization Trick
            mogpreds = torch.softmax(mogpreds, dim=2)
            # z_tokens, z_avg, component_indices = self.reparameterize(mean, logvar, mogpreds)
            
            return mean, logvar, mogpreds, attW

        elif reverse == True:

            # Add the hash_pat_embedding to latent vector
            y = x + hash_pat_embedding.unsqueeze(dim=1).repeat(1, x.shape[1], 1)

            # Transformer Decoder
            y = self.decoder(y).transpose(2,3)  # Comes out as [batch, token, waveform, num_channels] --> [batch, token, num_channels, waveform]

            return y

def print_models_flow(x, **kwargs):
    '''
    Builds models on CPU and prints sizes of forward passes with random data as inputs
    
    '''

    pat_num_channels = x.shape[2] 
    file_class_label = torch.tensor([0]*x.shape[0]) # dummy

    # Build the WAE
    vae = VAE(**kwargs) 

    # Run through Encoder
    print(f"INPUT TO <ENC>\n"
    f"x:{x.shape}")
    mean, logvar, mogpreds, attW = vae(x, reverse=False)  
    summary(vae, input_size=(x.shape), depth=999, device="cpu")
    print(
    f"mean:{mean.shape}\n"
    f"logvar:{logvar.shape}\n"
    f"mogpreds:{mogpreds.shape}\n"
    f"attW:{attW.shape}\n")

    reg_loss, z_token = mog_loss(
        encoder_means=mean, 
        encoder_logvars=logvar, 
        encoder_mogpreds=mogpreds,
        mog_prior=vae.prior, 
        weight=1,
        **kwargs)

    # CLASSIFIER - on the mean of Z tokens
    z_meaned = torch.mean(z_token, dim=1)
    class_probs_mean_of_latent = vae.adversarial_classifier(z_meaned, alpha=1)

    # Adversarial loss
    adversarial_loss = adversarial_loss_function(class_probs_mean_of_latent, file_class_label, classifier_weight = 1)
    print(f"Adversarial Loss: {adversarial_loss}")

    # Run through WAE decoder
    hash_pat_embedding = torch.rand(x.shape[0], z_token.shape[2])
    hash_channel_order = np.arange(0, 199).tolist()
    print(f"\n\n\nINPUT TO <WAE - Decoder Mode> \n"
    f"z:{z_token.shape}\n"
    f"hash_pat_embedding:{hash_pat_embedding.shape}\n")
    summary(vae, input_data=[z_token, True, hash_pat_embedding], depth=999, device="cpu")
    core_out = vae(z_token, reverse=True, hash_pat_embedding=hash_pat_embedding)  
    print(f"decoder_out:{core_out.shape}\n")

    del vae

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

    vae = VAE(
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

    print(f"Are the weights of encoder and decoder tied? {torch.allclose(vae.top_to_hidden.weight.T, vae.hidden_to_top.weight)}")

