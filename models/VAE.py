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

# class TimeSeriesCrossAttentionLayer_ShapeDilationNoResidual(nn.Module):
#     def __init__(self, in_dim, out_dim, dropout=0.1):
#         super(TimeSeriesCrossAttentionLayer_ShapeDilationNoResidual, self).__init__()
#         self.attn = nn.MultiheadAttention(in_dim, num_heads=1, dropout=dropout)
#         self.norm1 = RMSNorm(in_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.ffn = nn.Sequential(
#             nn.Linear(in_dim, in_dim * 4),
#             nn.SiLU(),
#             nn.Linear(in_dim * 4, out_dim)
#         )
#         # self.norm2 = RMSNorm(out_dim)
#         # self.dropout2 = nn.Dropout(dropout)

#     def forward(self, query, key, value):
#         # Cross-attention: query comes from the target, key and value come from the source
#         # query: (seq_len, batch_size, embed_dim)
#         # key, value: (seq_len, batch_size, embed_dim)

#         # Cross-attention layer
#         attn_output, _ = self.attn(query, key, value)  # q=query, k=key, v=value
#         query = self.norm1(query + self.dropout1(attn_output))  # Residual connection

#         # Feed-forward network, no residual
#         ffn_output = self.ffn(query)
#         # query = self.norm2(query + self.dropout2(ffn_output))  # Residual connection
        
#         return ffn_output

class Encoder_TimeSeriesCNNWithCrossAttention(nn.Module):
    def __init__(self, 
        # in_channels, 
        padded_channels,
        crattn_embed_dim,
        crattn_num_highdim_heads,
        crattn_num_highdim_layers,
        crattn_num_lowdim_heads,
        crattn_num_lowdim_layers,
        crattn_max_seq_len,
        crattn_cnn_kernel_size, 
        crattn_dropout, 
        **kwargs):

        super(Encoder_TimeSeriesCNNWithCrossAttention, self).__init__()
        
        # self.in_channels = in_channels
        self.padded_channels = padded_channels
        self.embed_dim = crattn_embed_dim
        self.num_highdim_heads = crattn_num_highdim_heads
        self.num_highdim_layers = crattn_num_highdim_layers
        self.num_lowdim_heads = crattn_num_lowdim_heads
        self.num_lowdim_layers = crattn_num_lowdim_layers
        self.max_seq_len = crattn_max_seq_len
        # self.cnn_kernel_size = crattn_cnn_kernel_size
        self.dropout = crattn_dropout

        # Depthwise 1D CNN (groups=in_channels)
        # self.cnn = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=self.cnn_kernel_size, 
                            #  padding=(self.cnn_kernel_size // 2), groups=self.in_channels)  # Depthwise convolution
        
        # Input Cross-attention Layer 
        self.highdim_attention_layers = nn.ModuleList([
            TimeSeriesCrossAttentionLayer(self.padded_channels, self.num_highdim_heads, self.dropout)
            for _ in range(self.num_highdim_layers)
        ])

        # Convert to low dims
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
        # self.positional_encoding_embed_dim = self._get_positional_encoding(self.max_seq_len, self.embed_dim)
        
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
        
        # Step 1: Apply Depthwise CNN (all channels processed in parallel)
        # Directly pass input to the CNN (input shape: batch_size, num_channels, seq_len)
        # x = self.cnn(x)  # Shape: (batch_size, cnn_out_channels, seq_len)
        
        in_channels = x.shape[1]

        # Step 1: pad the channel dimension
        padding = (0, 0, 0, self.padded_channels - in_channels, 0, 0) # (dim0 left padding, dim0 right padding... etc.)
        x = F.pad(x, padding, mode='constant', value=0)

        # Step 2: Permute CNN output to (batch_size, seq_len, cnn_out_channels)
        x = x.permute(0, 2, 1)  # Shape: (batch_size, seq_len, cnn_out_channels)
        
        # Step 3: Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device)
        
        # Step 4: Reshape for multi-head attention (seq_len, batch_size, high_dim)
        x = x.permute(1, 0, 2)  # Shape: (seq_len, batch_size, cnn_out_channels)

        # Step 5: Apply Input Cross-Attention at PADDED dimension 
        for layer in self.highdim_attention_layers:
            x = layer(x, x, x)  # Cross-attention (q=k=v)

        # Step 6: Conver from high dims to low dims
        x = self.high_to_low_dims(x)

        # Step 7: Apply Embed-Dim Cross-Attention Layers
        for layer in self.lowdim_attention_layers:
            x = layer(x, x, x)  # Cross-attention (q=k=v)

        x = x.permute(1, 0, 2)  # Return to shape (batch_size, seq_len, low_dim)

        return x.flatten(start_dim=1) # (batch, seq_len * low_dim)
        # return x[:,-1,:] # Return last in sequence (should be embedded with meaning)

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
                layers.append(nn.Tanh())

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

class VAE(nn.Module):
    '''
    The Reverseable Encoder/Decoder 
    Shares weights between Conv/TransConv layers, in addition to FC layers (except Mean/Logvar layers)
    '''
    def __init__(
        self, 
        autoencode_samples,
        padded_channels,
        crattn_embed_dim,
        top_dims,
        hidden_dims,
        latent_dim, 
        num_decode_layers,
        decode_kernel_size,
        gpu_id=None,  
        **kwargs):

        super(VAE, self).__init__()

        self.gpu_id = gpu_id
        self.autoencode_samples = autoencode_samples
        self.padded_channels = padded_channels
        self.crattn_embed_dim = crattn_embed_dim
        self.top_dims = top_dims
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim 
        self.num_decode_layers = num_decode_layers
        self.decode_kernel_size = decode_kernel_size

        # Encoder Head
        self.encoder_head = Encoder_TimeSeriesCNNWithCrossAttention(padded_channels = self.padded_channels, crattn_embed_dim=crattn_embed_dim, **kwargs)

        # Core Encoder
        self.head_to_top = nn.Linear(self.crattn_embed_dim * self.autoencode_samples, self.top_dims, bias=True)
        self.norm_top = RMSNorm(dim=self.top_dims)
        self.top_to_hidden = nn.Linear(self.top_dims, self.hidden_dims, bias=True)
        self.norm_hidden = RMSNorm(dim=self.hidden_dims)
        
        # Encoder variational layers (not shared between enc/dec)
        self.mean_fc_layer = nn.Linear(self.hidden_dims, self.latent_dim, bias=True)  # bias=False
        self.logvar_fc_layer = nn.Linear(self.hidden_dims, self.latent_dim, bias=True) # bias=False

        # Core Decoder
        self.latent_to_hidden = nn.Linear(self.latent_dim, self.hidden_dims, bias=True) # bias=False
        self.norm_hidden_rev = RMSNorm(dim=self.hidden_dims)
        self.hidden_to_top = nn.Linear(self.hidden_dims, self.top_dims, bias=True)
        self.norm_top_rev = RMSNorm(dim=self.top_dims)

        # Deocder head
        self.decoder_head = TimeSeriesDecoder(in_dim=top_dims, padded_channels = self.padded_channels, seq_len=autoencode_samples, num_layers=num_decode_layers, kernel_size=decode_kernel_size)

        self.silu = nn.SiLU()

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)  
        epsilon = torch.randn_like(std).to(self.gpu_id) 
        z = mean + std * epsilon
        return z

    def forward(self, x, reverse=False, hash_pat_embedding=-1, out_channels=-1):

        if reverse == False:

            # FEATURE HEAD
            y = self.encoder_head(x)

            # VAE CORE
            y = self.head_to_top(y)
            y = self.silu(y)
            y = self.norm_top(y)
            y = self.top_to_hidden(y)
            y = self.silu(y)
            y = self.norm_hidden(y)
            mean, logvar = self.mean_fc_layer(y), self.logvar_fc_layer(y)
            y = self.reparameterization(mean, logvar)
            return mean, logvar, y

        elif reverse == True:

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

def print_models_flow(x, transformer_seq_length, **kwargs):
    '''
    Builds models on CPU and prints sizes of forward passes

    '''

    batch_size = x.shape[0]
    pat_num_channels = x.shape[1] 
    data_length = x.shape[2]

    # Build the VAE
    vae = VAE(**kwargs) 

    # Build the Transformer
    transformer = Transformer(ModelArgs(**kwargs))

    # Run through Enc Head
    print(f"INPUT TO <ENC>\n"
    f"x:{x.shape}")
    mean, logvar, latent = vae(x, reverse=False)  
    summary(vae, input_size=(x.shape), depth=999, device="cpu")
    print(
    f"mean:{mean.shape}\n"
    f"logvar:{logvar.shape}\n"
    f"latent:{latent.shape}\n")

    # Run through Transformer
    # Generate fake seuqential latents and shift the latent
    latent_shifted = torch.rand(latent.shape[0], transformer_seq_length-1, latent.shape[1])
    print(f"\n\n\nINPUT TO <Transformer>\n"
    f"Multiple enoder passes to get sequential latents: latent_shifted:{latent_shifted.shape}\n")
    trans_out = transformer(latent_shifted)  
    summary(transformer, input_size=(latent_shifted.shape), depth=999, device="cpu")
    print(f"trans_out:{trans_out.shape}\n")

    # Run through VAE decoder
    hash_pat_embedding = torch.rand(latent.shape[1])
    print(f"\n\n\nINPUT TO <VAE - Decoder Mode> \n"
    f"z:{latent.shape}\n"
    f"hash_pat_embedding:{hash_pat_embedding.shape}\n")
    core_out = vae(latent, reverse=True, hash_pat_embedding=hash_pat_embedding, out_channels=pat_num_channels)  
    print(f"core_out:{core_out.shape}\n")

    del vae, transformer

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
        autoencode_samples=data_length,
        in_channels=in_channels,
        kernel_sizes=kernel_sizes, 
        time_change=time_change,
        cnn_depth=2,
        cnn_resblock_layers=4,
        hidden_dims=2048,
        latent_dim=1024, 
        **kwargs
    )

    # mean,logvar,z = vae(x, reverse=False)
    # x_hat = vae(z, reverse=True)
    # loss_fn = nn.MSELoss(reduction='mean')
    # recon_loss = loss_fn(x, x_hat) 
    # recon_loss.backward()

    print(f"Are the weights of encoder and decoder tied? {torch.allclose(vae.top_to_hidden.weight.T, vae.hidden_to_top.weight)}")

