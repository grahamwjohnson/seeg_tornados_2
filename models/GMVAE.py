import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torchinfo import summary

# Local imports
from .Transformer import ModelArgs, Transformer, RMSNorm

'''
@author: grahamwjohnson
Developed between 2023-2025

'''

class MoGPrior(nn.Module):
    """
    Mixture of Gaussians (MoG) Prior with Gumbel-Softmax Sampling.

    This class implements a prior distribution for a Gaussian Mixture Variational Autoencoder (GM-VAE).
    The prior consists of `K` Gaussian components in a `latent_dim`-dimensional space, with learnable 
    means and variances. To enable differentiable sampling, it uses the Gumbel-Softmax trick for soft 
    component selection.

    Parameters:
    -----------
    K : int
        Number of Gaussian mixture components in the prior.

    latent_dim : int
        Dimensionality of the latent space.

    prior_initial_mean_spread : float
        Spread for initializing the prior means. The means are sampled uniformly from 
        `[-prior_initial_mean_spread, prior_initial_mean_spread]` to encourage diversity.

    prior_initial_logvar : float
        Initial log-variance for all Gaussian components. This defines the starting 
        uncertainty for each mixture component.

    mean_lims : tuple (float, float)
        Clamping range `(min, max)` for sampled latent variables to ensure numerical stability.

    gumbel_softmax_temperature : float
        Temperature parameter for the Gumbel-Softmax trick, controlling the smoothness 
        of component selection. Lower values make selections more discrete.

    Attributes:
    -----------
    means : nn.Parameter (tensor of shape `[K, latent_dim]`)
        Learnable means of the Gaussian mixture components.

    logvars : nn.Parameter (tensor of shape `[K, latent_dim]`)
        Learnable log-variances of the Gaussian mixture components.

    weightlogits : nn.Parameter (tensor of shape `[K]`)
        Unnormalized logits for mixture component weights. These are softmaxed to obtain 
        valid mixture probabilities.

    Methods:
    --------
    sample_prior(batch_size)
        Samples `batch_size` points from the prior using Gumbel-Softmax for differentiable 
        mixture component selection, followed by Gaussian reparameterization.

        Parameters:
        batch_size : int
            Number of latent samples to generate.

        Returns:
        z_prior : torch.Tensor of shape `[batch_size, latent_dim]`
            Differentiable samples drawn from the MoG prior.

        component_probs : torch.Tensor of shape `[batch_size, K]`
            Soft assignment probabilities indicating the likelihood of each sample 
            belonging to each MoG component.

    Notes:
    ------
    - The prior mixture weights are parameterized as logits (`weightlogits`) to avoid numerical 
      issues with direct probability constraints.
    - The `sample_prior` method ensures differentiability using the Gumbel-Softmax trick, allowing 
      for gradient flow through discrete component assignments.
    - Latent samples are clamped within `mean_lims` to prevent extreme values from destabilizing training.
    - This MoG prior is designed to integrate with the GM-VAE framework.
    """

    def __init__(self, K, latent_dim, prior_initial_mean_spread, prior_initial_logvar, mean_lims, gumbel_softmax_temperature, **kwargs):
        """
        Mixture of Gaussians (MoG) prior with Wasserstein stabilization using Sinkhorn loss.
        Inputs:
            K: Number of mixture components.
            latent_dim: Dimensionality of the latent space.
            prior_initial_mean_spread: Spread for uniform initialization of prior means.
            prior_initial_logvar: Initial log variance for all components.
        """
        super(MoGPrior, self).__init__()
        self.K = K
        self.latent_dim = latent_dim
        self.mean_lims = mean_lims
        self.gumbel_softmax_temperature = gumbel_softmax_temperature
        
        # Uniformly initialize means in the range (-initial_mean_spread, initial_mean_spread)
        prior_means = (torch.rand(K, latent_dim) * 2 * prior_initial_mean_spread) - prior_initial_mean_spread

        # Initialize means, logvars, and weights
        self.means = nn.Parameter(prior_means)  # [K, latent_dim]
        self.logvars = nn.Parameter(torch.ones(K, latent_dim) * prior_initial_logvar)  # [K, latent_dim]
        self.weightlogits = nn.Parameter(torch.ones(K) / K)  # [K] # STore as logits, will softmax before use

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
        gumbel_noise = -torch.log(-torch.log(torch.rand(batch_size, self.K, device=self.weightlogits.device) + 1e-20) + 1e-20)
        logits = torch.log_softmax(self.weightlogits, dim=0) + gumbel_noise
        component_probs = F.gumbel_softmax(logits, tau=self.gumbel_softmax_temperature, hard=False)  # [batch_size, K], fully differentiable

        # Compute mixture samples using soft probabilities
        chosen_means = torch.matmul(component_probs, self.means)  # [batch_size, latent_dim]
        chosen_logvars = torch.matmul(component_probs, self.logvars)  # [batch_size, latent_dim]

        # Sample from Gaussian with chosen means and logvars
        eps = torch.randn_like(chosen_means)
        z_prior = chosen_means + eps * torch.exp(0.5 * chosen_logvars)  # [batch_size, latent_dim]

        # Same clamp as posterior
        z_prior = torch.clamp(z_prior, min=self.mean_lims[0], max=self.mean_lims[1])

        return z_prior, component_probs

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
    """
    Cross-Attention Layer for Time Series Data.

    This module applies cross-attention to capture dependencies across different time steps 
    and channels in a time series. It follows a transformer-like structure with residual 
    connections, layer normalization, and a feed-forward network (FFN).

    This layer is used in `Encoder_TimeSeriesWithCrossAttention` to process time series data 
    before passing it into the transformer-based encoder.

    Parameters:
    -----------
    embed_dim : int
        Dimensionality of the input embeddings (equivalent to `padded_channels` in the encoder).

    num_heads : int
        Number of attention heads for multi-head self-attention.

    dropout : float, optional (default=0.1)
        Dropout probability applied to attention outputs and the feed-forward network.

    Attributes:
    -----------
    attn : nn.MultiheadAttention
        Multi-head attention mechanism for computing attention over the input sequence.

    norm1 : RMSNorm
        Layer normalization applied after the cross-attention mechanism.

    dropout1 : nn.Dropout
        Dropout applied to the attention output before the residual connection.

    ffn : nn.Sequential
        A position-wise feed-forward network consisting of:
        - Linear layer expanding `embed_dim` to `embed_dim * 4`
        - SiLU activation function
        - Linear layer reducing back to `embed_dim`

    norm2 : RMSNorm
        Layer normalization applied after the feed-forward network.

    dropout2 : nn.Dropout
        Dropout applied to the FFN output before the residual connection.

    Methods:
    --------
    forward(query, key, value)
        Applies cross-attention, residual connections, and a feed-forward network.

        Parameters:
        query : torch.Tensor
            Query tensor of shape `[seq_len, batch_size, embed_dim]`.

        key : torch.Tensor
            Key tensor of shape `[seq_len, batch_size, embed_dim]`.

        value : torch.Tensor
            Value tensor of shape `[seq_len, batch_size, embed_dim]`.

        Returns:
        torch.Tensor
            Processed feature tensor of shape `[seq_len, batch_size, embed_dim]`.

    Notes:
    ------
    - The input sequence is first passed through a multi-head attention mechanism.
    - A residual connection is applied, followed by layer normalization.
    - The processed tensor is then passed through a feed-forward network.
    - Another residual connection and normalization are applied before returning the output.
    - This layer is designed to work within the cross-attention-based encoder 
      (`Encoder_TimeSeriesWithCrossAttention`), where queries, keys, and values are 
      derived from time series features.
    """

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

        # Cross-attention layer
        attn_output, _ = self.attn(query, key, value)  # q=query, k=key, v=value
        query = self.norm1(query + self.dropout1(attn_output))  # Residual connection

        # Feed-forward network
        ffn_output = self.ffn(query)
        query = self.norm2(query + self.dropout2(ffn_output))  # Residual connection
        
        return query

class Encoder_TimeSeriesWithCrossAttention(nn.Module):
    """
    Time Series Encoder with Cross-Attention Mechanism.

    This module encodes time series data using multiple layers of cross-attention, allowing
    it to capture complex dependencies across channels. It processes input sequences using 
    self-attention mechanisms and incorporates positional encodings to retain sequential 
    information.

    Parameters:
    -----------
    padded_channels : int
        Number of input channels after padding or transformation.

    crattn_num_heads : int
        Number of attention heads in each cross-attention layer.

    crattn_num_layers : int
        Number of stacked cross-attention layers in the encoder.

    crattn_max_seq_len : int
        Maximum sequence length supported by the positional encoding.

    crattn_dropout : float
        Dropout probability applied within the attention layers.

    Attributes:
    -----------
    attention_layers : nn.ModuleList
        A list of `TimeSeriesCrossAttentionLayer` modules, each applying cross-attention 
        to capture dependencies across channels.

    positional_encoding : torch.Tensor
        Precomputed positional encoding tensor with shape [1, max_seq_len, padded_channels],
        used to preserve sequence order information.

    Methods:
    --------
    _get_positional_encoding(max_seq_len, dim)
        Computes the sinusoidal positional encoding for the input sequence.

        Parameters:
        max_seq_len : int
            Maximum sequence length to support.

        dim : int
            Feature dimensionality per time step.

        Returns:
        torch.Tensor
            A tensor of shape [1, max_seq_len, dim] containing the positional encodings.

    forward(x)
        Forward pass through the cross-attention encoder.

        Parameters:
        x : torch.Tensor
            Input tensor of shape [batch_size, num_channels, seq_len], where `num_channels` 
            corresponds to `padded_channels`.

        Returns:
        torch.Tensor
            Encoded feature representation with shape [batch_size, seq_len * padded_channels].

    Notes:
    ------
    - The input is reshaped to `[batch_size, seq_len, padded_channels]` before applying 
      positional encodings.
    - Cross-attention layers process the data as `(seq_len, batch_size, padded_channels)`, 
      where queries, keys, and values are the same (`q = k = v`).
    - The final output is reshaped to `[batch_size, seq_len * padded_channels]` to ensure 
      compatibility with downstream tasks.
    """

    def __init__(self, 
        # in_channels, 
        padded_channels,
        crattn_num_heads,
        crattn_num_layers,
        crattn_max_seq_len,
        crattn_dropout, 
        **kwargs):

        super(Encoder_TimeSeriesWithCrossAttention, self).__init__()
        
        # self.in_channels = in_channels
        self.padded_channels = padded_channels
        self.num_heads = crattn_num_heads
        self.num_layers = crattn_num_layers
        self.max_seq_len = crattn_max_seq_len
        self.dropout = crattn_dropout

        # Input Cross-attention Layer 
        self.attention_layers = nn.ModuleList([
            TimeSeriesCrossAttentionLayer(self.padded_channels, self.num_heads, self.dropout)
            for _ in range(self.num_layers)])
        
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

        # Permute to (batch_size, seq_len, padded_channels)
        x = x.permute(0, 2, 1)  # Shape: (batch_size, seq_len, padded_channels)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device)
        
        # Reshape for multi-head attention (seq_len, batch_size, high_dim)
        x = x.permute(1, 0, 2)  # Shape: (seq_len, batch_size, padded_channels)

        # Apply Input Cross-Attention 
        for layer in self.attention_layers:
            x = layer(x, x, x)  # Cross-attention (q=k=v)

        x = x.permute(1, 0, 2)  # Return to shape (batch_size, seq_len, low_dim)

        return x.flatten(start_dim=1) # (batch, seq_len * low_dim)

class Decoder_MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Decoder for GMVAE.

    This decoder is a non-autoregressive MLP-based architecture designed to reconstruct 
    sequences from the latent representation. It transforms latent variables into waveform 
    reconstructions using fully connected layers with SiLU activations and normalization.

    Parameters:
    -----------
    gpu_id : int or None
        ID of the GPU device for computation. If None, uses CPU.

    latent_dim : int
        Dimensionality of the latent space input to the decoder.

    decoder_base_dims : int
        Base hidden dimensionality for fully connected layers in the decoder.

    output_channels : int
        Number of output channels in the reconstructed signal.

    decode_samples : int
        Number of time samples to generate per transformer sequence token.

    Attributes:
    -----------
    non_autoregressive_fc : nn.Sequential
        Fully connected layers mapping the latent space to a high-dimensional feature 
        space, followed by SiLU activations and RMS normalization.

    non_autoregressive_output : nn.Sequential
        Final fully connected layer mapping features to the output waveform with a 
        tanh activation to ensure bounded output values.

    Methods:
    --------
    forward(z)
        Forward pass through the decoder.
        
        Parameters:
        z : torch.Tensor
            Latent representation from the encoder. Shape: [batch_size, latent_dim].

        Returns:
        torch.Tensor
            Reconstructed waveform. Shape: [batch_size, transformer_seq_length, decode_samples, output_channels].

    Notes:
    ------
    - The decoder is **non-autoregressive**, meaning it predicts the entire sequence 
      in parallel rather than step-by-step.
    - RMSNorm is used after intermediate layers for stabilization.
    - The final tanh activation ensures output values remain within a bounded range, 
      which may help match expected signal characteristics.
    - Unlike autoregressive models, this decoder does not use past outputs to generate 
      future values, making it computationally efficient.
    """

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

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None # None is for alpha 

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

class AdversarialClassifier(nn.Module):
    """
    Adversarial Classifier for GMVAE.

    This classifier is designed to predict patient labels from the latent space 
    while using a Gradient Reversal Layer (GRL) to enable adversarial training. 
    The goal is to remove patient-specific biases from the latent representation 
    by making the encoder produce embeddings that are uninformative for classification.

    The classifier is a multi-layer perceptron (MLP) with SiLU activation functions, 
    optional dropout for regularization, and a final softmax activation to output 
    class probabilities.

    Parameters:
    -----------
    latent_dim : int
        Dimensionality of the latent space input to the classifier.

    classifier_hidden_dims : list of int
        List of hidden layer sizes for the MLP classifier.

    classifier_num_pats : int
        Number of unique patient classes (i.e., the number of output units).

    classifier_dropout : float
        Dropout probability applied to hidden layers for regularization.

    **kwargs : dict
        Additional keyword arguments for flexibility.

    Attributes:
    -----------
    gradient_reversal : GradientReversal
        A module that reverses gradients during backpropagation for adversarial training.

    classifier_dropout : float
        Dropout probability for regularization.

    mlp_layers : nn.ModuleList
        List of layers including linear transformations and SiLU activations.

    softmax : nn.Softmax
        Softmax activation applied to the final output to produce class probabilities.

    Methods:
    --------
    forward(mu, alpha)
        Forward pass through the adversarial classifier.
        
        Parameters:
        mu : torch.Tensor
            The latent representation from the encoder. Shape: [batch_size, latent_dim].

        alpha : float
            Scaling factor for the gradient reversal layer. Determines the strength 
            of the adversarial effect.

        Returns:
        torch.Tensor
            Class probabilities after the softmax activation. Shape: [batch_size, classifier_num_pats].

    Notes:
    ------
    - The `GradientReversal` layer ensures adversarial learning by inverting gradients.
    - The softmax function is applied at the output layer to produce probabilities.
    - The classifier is optimized separately from the main VAE loss to ensure effective 
      adversarial training.
    """
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

class MoGPredictor(nn.Module):
    """
    Mixture of Gaussians (MoG) Component Predictor.

    The `MoGPredictor` is a fully connected neural network that predicts the logits for 
    selecting a Gaussian mixture component in the latent space. It takes an encoded 
    representation from the GMVAE and outputs unnormalized logits for each MoG component. 
    The logits are later processed using the Gumbel-Softmax trick to enable differentiable 
    sampling.

    This module consists of an MLP with multiple hidden layers, SiLU activation functions, 
    RMS normalization, and optional dropout for regularization.

    Parameters:
    -----------
    top_dim : int
        The dimensionality of the input features coming from the encoder.

    posterior_mogpredictor_hidden_dim_list : list of int
        List defining the number of hidden units in each hidden layer.

    num_prior_mog_components : int
        Number of Gaussian components in the prior mixture model.

    posterior_mogpredictor_dropout : float
        Dropout probability applied in the hidden layers for regularization.

    **kwargs : dict
        Additional keyword arguments for flexibility.

    Attributes:
    -----------
    dropout : float
        Dropout probability for hidden layers.

    mlp_layers : nn.ModuleList
        List of layers including linear transformations, SiLU activations, and RMS normalization.

    Methods:
    --------
    forward(x)
        Performs a forward pass through the MLP network and returns logits for MoG component selection.
        The output should not be passed through a softmax function, as the Gumbel-Softmax trick requires raw logits.

    Notes:
    ------
    - The final layer does NOT apply softmax to the logits; this must be done externally.
    - The network is designed to maintain differentiability for component selection.
    """

    def __init__(self, top_dim, posterior_mogpredictor_hidden_dim_list, num_prior_mog_components, posterior_mogpredictor_dropout, **kwargs):
        super(MoGPredictor, self).__init__()

        self.dropout = posterior_mogpredictor_dropout
        self.mlp_layers = nn.ModuleList()

        # Input layer
        self.mlp_layers.append(nn.Linear(top_dim, posterior_mogpredictor_hidden_dim_list[0]))
        self.mlp_layers.append(nn.SiLU())
        self.mlp_layers.append(RMSNorm(posterior_mogpredictor_hidden_dim_list[0]))

        # Hidden layers
        for i in range(len(posterior_mogpredictor_hidden_dim_list) - 1):
            self.mlp_layers.append(LinearWithDropout(posterior_mogpredictor_hidden_dim_list[i], posterior_mogpredictor_hidden_dim_list[i + 1], self.dropout))
            self.mlp_layers.append(nn.SiLU())
            self.mlp_layers.append(RMSNorm(posterior_mogpredictor_hidden_dim_list[i + 1]))

        # Output layer
        self.mlp_layers.append(nn.Linear(posterior_mogpredictor_hidden_dim_list[-1], num_prior_mog_components)) # No activation and no norm

    def forward(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
        return x # Do NOT softmax here, Gumbel-Softmax trick needs logits 

class GMVAE(nn.Module):
    """
    Gaussian Mixture Variational Autoencoder (GMVAE) with Reversible Encoder/Decoder.

    This GMVAE model incorporates a mixture of Gaussians (MoG) prior with a transformer-based 
    encoder for time-series data. The encoder leverages cross-attention mechanisms to process 
    raw input data before passing it through a transformer for further feature extraction.
    The latent space is modeled as a mixture of Gaussians, where posterior sampling is done 
    using the Gumbel-Softmax trick for differentiable selection of mixture components.

    The decoder reconstructs the original input data from the latent space, and an adversarial 
    classifier is optionally included for additional regularization.

    Parameters:
    -----------
    encode_token_samples : int
        Number of tokens per input sample in the encoding stage.

    padded_channels : int
        Number of padded channels in the input waveform.

    transformer_seq_length : int
        Number of sequential time steps input into the transformer.

    transformer_start_pos : int
        Starting position for the transformer’s positional encoding.

    transformer_dim : int
        Dimensionality of the transformer model's hidden representation.

    encoder_transformer_activation : str
        Activation function used in the transformer encoder.

    top_dims : int
        Dimension of the output of the cross-attention encoder head.

    hidden_dims : int
        Number of hidden dimensions before the latent space transformation.

    latent_dim : int
        Dimensionality of the latent space.

    decoder_base_dims : int
        Dimensionality of the decoder’s input space.

    prior_mog_components : int
        Number of Gaussian components in the MoG prior.

    mean_lims : tuple (float, float)
        Limits for mean values in the latent space.

    logvar_lims : tuple (float, float)
        Limits for log-variance values in the latent space.

    gumbel_softmax_temperature : float
        Temperature parameter for Gumbel-Softmax sampling in the posterior.

    gpu_id : int, optional
        GPU identifier for model computation.

    **kwargs : dict
        Additional keyword arguments for submodules.

    Attributes:
    -----------
    prior : MoGPrior
        Mixture of Gaussians prior used for regularization in the latent space.

    encoder_head : Encoder_TimeSeriesWithCrossAttention
        Cross-attention encoder to extract features from time-series data.

    transformer_encoder : Transformer
        Transformer module applied after the cross-attention head.

    top_to_hidden : nn.Linear
        Linear layer to map from the encoder’s output to the hidden space.

    norm_hidden : RMSNorm
        Normalization layer for the hidden representation.

    mean_encode_layer : nn.Linear
        Linear layer mapping hidden representation to mean values of the posterior.

    logvar_encode_layer : nn.Linear
        Linear layer mapping hidden representation to log-variance values of the posterior.

    mogpreds_layer : MoGPredictor
        Predictor layer for MoG mixture component selection.

    decoder : Decoder_MLP
        Decoder network to reconstruct input data from latent representations.

    adversarial_classifier : AdversarialClassifier
        Optional classifier for adversarial training in the latent space.

    silu : nn.SiLU
        Activation function applied in the encoder.

    Methods:
    --------
    forward(x, reverse=False, hash_pat_embedding=-1)
        Forward pass through the GMVAE. If `reverse=False`, encodes input `x` into a latent 
        representation. If `reverse=True`, decodes a given latent `x` back into waveform data.

    sample_posterior(encoder_means, encoder_logvars, encoder_mogpreds, gumbel_softmax_temperature)
        Samples `z` from the posterior distribution using the Gumbel-Softmax trick.

    """
    def __init__(
        self, 
        encode_token_samples,
        padded_channels,
        transformer_seq_length,
        transformer_start_pos,
        transformer_dim,
        encoder_transformer_activation,
        top_dims,
        hidden_dims,
        latent_dim, 
        decoder_base_dims,
        prior_mog_components,
        mean_lims,
        logvar_lims,
        gumbel_softmax_temperature,
        gpu_id=None,  
        **kwargs):

        super(GMVAE, self).__init__()

        self.gpu_id = gpu_id
        self.encode_token_samples = encode_token_samples
        self.padded_channels = padded_channels
        self.transformer_seq_length = transformer_seq_length
        self.transformer_start_pos = transformer_start_pos
        self.transformer_dim = transformer_dim
        self.encoder_transformer_activation = encoder_transformer_activation
        self.top_dims = top_dims
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim 
        self.decoder_base_dims = decoder_base_dims
        self.prior_mog_components = prior_mog_components
        self.mean_lims = mean_lims
        self.logvar_lims = logvar_lims
        self.temperature = gumbel_softmax_temperature

        # Prior
        self.prior = MoGPrior(
            K = self.prior_mog_components,
            latent_dim = self.latent_dim,
            mean_lims = self.mean_lims,
            gumbel_softmax_temperature=gumbel_softmax_temperature,
            **kwargs)

        # Raw CrossAttention Head
        self.encoder_head = Encoder_TimeSeriesWithCrossAttention(
            padded_channels=self.padded_channels,
            **kwargs)

        # Transformer - dimension is same as output of cross attention
        self.transformer_encoder = Transformer(ModelArgs(
            device=self.gpu_id, 
            dim=self.transformer_dim, 
            activation=self.encoder_transformer_activation, 
            **kwargs))

        # Encoder/Posterior
        self.top_to_hidden = nn.Linear(self.top_dims, self.hidden_dims, bias=True)
        self.norm_hidden = RMSNorm(dim=self.hidden_dims)

        # Right before latent space
        self.mean_encode_layer = nn.Linear(self.hidden_dims, self.prior_mog_components * self.latent_dim, bias=True)  
        self.logvar_encode_layer = nn.Linear(self.hidden_dims, self.prior_mog_components * self.latent_dim, bias=True) 
        self.mogpreds_layer = MoGPredictor(
            top_dim=self.hidden_dims,
            num_prior_mog_components=self.prior_mog_components,
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
            mean = mean.view(-1, self.transformer_seq_length, self.prior_mog_components, self.latent_dim)
            logvar = logvar.view(-1,self.transformer_seq_length, self.prior_mog_components, self.latent_dim)

            # Apply constrtains to mean/logvar
            mean_scale = (self.mean_lims[1] - self.mean_lims[0]) / 2
            mean_shift = (self.mean_lims[0] + self.mean_lims[1]) / 2
            mean = F.tanh(mean) * mean_scale + mean_shift

            logvar_scale = (self.logvar_lims[1] - self.logvar_lims[0]) / 2
            logvar_shift = (self.logvar_lims[0] + self.logvar_lims[1]) / 2
            logvar = F.tanh(logvar) * logvar_scale + logvar_shift

            # Now sample z posterior
            mean_pseudobatch = mean.reshape(mean.shape[0]*mean.shape[1], mean.shape[2], mean.shape[3])
            logvar_pseudobatch = logvar.reshape(logvar.shape[0]*logvar.shape[1], logvar.shape[2], logvar.shape[3])
            mogpreds_pseudobatch = mogpreds.reshape(mogpreds.shape[0]*mogpreds.shape[1], mogpreds.shape[2])

            # Softmax the mogpreds 
            mogpreds_pseudobatch_softmax = torch.softmax(mogpreds_pseudobatch, dim=-1) 

            # Z is not currently constrained 
            z_pseudobatch, _ = self.sample_posterior(mean_pseudobatch, logvar_pseudobatch, mogpreds_pseudobatch, gumbel_softmax_temperature=self.temperature)
            
            return z_pseudobatch, mean_pseudobatch, logvar_pseudobatch, mogpreds_pseudobatch_softmax, attW

        elif reverse == True:

            # Add the hash_pat_embedding to latent vector
            y = x + hash_pat_embedding.unsqueeze(dim=1).repeat(1, x.shape[1], 1)

            # Transformer Decoder
            y = self.decoder(y).transpose(2,3)  # Comes out as [batch, token, waveform, num_channels] --> [batch, token, num_channels, waveform]

            return y

    def sample_posterior(self, encoder_means, encoder_logvars, encoder_mogpreds, gumbel_softmax_temperature):
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

        # # Step 4: Clamp to the the same limits as mean for consistency (not TanH to not distort latent space during sampling of posterior)
        z = torch.clamp(z, min=self.mean_lims[0], max=self.mean_lims[1])
        
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
    posterior_out = gmvae(z_token, reverse=True, hash_pat_embedding=hash_pat_embedding)  
    print(f"decoder_out:{posterior_out.shape}\n")

    del gmvae
