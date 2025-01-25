import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from torchinfo import summary

# Local imports
from .Transformer import ModelArgs, Transformer, RMSNorm

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

class TimeSeriesDecoder(nn.Module):
    def __init__(self, in_dim, padded_channels, seq_len, num_layers, kernel_size):
        super(TimeSeriesDecoder, self).__init__()

        self.in_dim = in_dim
        self.padded_channels = padded_channels
        self.seq_len = seq_len
        self.num_time_dilation_layers = num_layers
        self.num_channel_reduction_layers = int(math.sqrt(in_dim / padded_channels))
        self.kernel_size = kernel_size

        assert math.sqrt(in_dim/padded_channels) % 1 == 0

        layers = []

        # First we will expand the time dimension to raw time step size
        for i in range(self.num_time_dilation_layers):
            # Expand the time dimension with stride of 2 and output padding 1
            layers.append(nn.ConvTranspose1d(in_channels=self.in_dim, out_channels=self.in_dim, 
                                    kernel_size=kernel_size, stride=2, padding=(kernel_size // 2), output_padding=1))
            layers.append(nn.SiLU())
            layers.append(RMSNorm_Conv(self.in_dim))

        # Keep track of stepping down the channel dimension
        in_channel_count_curr = self.in_dim
        for i in range(self.num_channel_reduction_layers):
            # Now we will step the channel count down to padded channels, no output padding, stride is 1
            layers.append(nn.ConvTranspose1d(in_channels=in_channel_count_curr, out_channels=int(in_channel_count_curr/2), kernel_size=kernel_size, stride=1, padding=(kernel_size // 2), output_padding=0))

            if i < self.num_channel_reduction_layers -1:
                layers.append(nn.SiLU())
                layers.append(RMSNorm_Conv(int(in_channel_count_curr/2)))
                in_channel_count_curr = int(in_channel_count_curr / 2)

            # Final output no normalization and smooth with tanh
            elif i == self.num_channel_reduction_layers -1:
                # layers.append(nn.Tanh())
                layers.append(nn.Hardtanh())

        # Combine the layers into a Sequential container
        self.cnn = nn.Sequential(*layers)

    def forward(self, x, out_channels):
        # x: (batch_size, embed_dim)
        
        # Unsqueeze to add the sequence dimension
        x = x.unsqueeze(2)  # Shape: (batch_size, embed_dim, 1)
        
        # Pass through the convolutional layers
        x = self.cnn(x)  # Shape: (batch_size, num_channels, seq_len)

        # Index only desired output channels
        x = x[:, :out_channels, :]
        
        return x

class Decoder(nn.Module):
    '''
    Decoder
    '''
    def __init__(
        self, 
        autoencode_samples,
        padded_channels,
        crattn_embed_dim,
        top_dims,
        hidden_dims,
        latent_dim, 
        decoder_hidden_dims,
        decoder_top_dims,
        num_decode_layers,
        decode_kernel_size,
        gpu_id=None,  
        **kwargs):

        super(Decoder, self).__init__()

        self.gpu_id = gpu_id
        self.autoencode_samples = autoencode_samples
        self.padded_channels = padded_channels
        self.crattn_embed_dim = crattn_embed_dim
        self.top_dims = top_dims
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim 

        self.decoder_hidden_dims = decoder_hidden_dims
        self.decoder_top_dims = decoder_top_dims
        self.num_decode_layers = num_decode_layers
        self.decode_kernel_size = decode_kernel_size

        # Core Decoder
        self.latent_to_hidden = nn.Linear(self.latent_dim, self.decoder_hidden_dims, bias=True) # bias=False
        self.norm_hidden_rev = RMSNorm(dim=self.decoder_hidden_dims)
        self.hidden_to_top = nn.Linear(self.decoder_hidden_dims,  self.decoder_top_dims, bias=True)
        self.norm_top_rev = RMSNorm(dim=self.decoder_top_dims)

        # Deocder head
        self.decoder_head = TimeSeriesDecoder(in_dim=self.decoder_top_dims, padded_channels = self.padded_channels, seq_len=autoencode_samples, num_layers=num_decode_layers, kernel_size=decode_kernel_size)

        self.silu = nn.SiLU()

    def forward(self, x, hash_pat_embedding, out_channels):

        # Add the hash_pat_embedding to latent vector
        y = x + hash_pat_embedding

        # VAE CORE
        y = self.latent_to_hidden(y)
        y = self.silu(y)
        y = self.norm_hidden_rev(y)
        y = self.hidden_to_top(y)
        y = self.silu(y)
        y = self.norm_top_rev(y)

        # FEATURE HEAD
        y = self.decoder_head(y, out_channels)
        return y

