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
        crattn_cnn_kernel_size, 
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
        # self.cnn_kernel_size = crattn_cnn_kernel_size
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
        in_channels = x.shape[1]

        # Step 1: pad the channel dimension
        padding = (0, 0, 0, self.padded_channels - in_channels, 0, 0) # (dim0 left padding, dim0 right padding... etc.)
        x = F.pad(x, padding, mode='constant', value=0)

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

class HybridDecoder(nn.Module):
    def __init__(self, gpu_id, latent_dim, hidden_dim, output_channels, seq_length, num_transformer_layers, num_heads, ffm, max_bs, max_seq_len, activation):
        super(HybridDecoder, self).__init__()
        self.gpu_id = gpu_id
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.seq_length = seq_length
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.ffm = ffm
        self.max_bs = max_bs
        self.max_seq_len = max_seq_len

        # Non-autoregressive decoder (rough sketch generator)
        self.non_autoregressive_fc = nn.Sequential(
            nn.Linear(latent_dim, int(hidden_dim * seq_length/2)),
            # nn.Tanh(),
            nn.SiLU(),
            RMSNorm(int(hidden_dim * seq_length/2)),
            nn.Linear(int(hidden_dim * seq_length/2), hidden_dim * seq_length),
            # nn.Tanh(),
            nn.SiLU(),
            RMSNorm(hidden_dim * seq_length)
            )

        self.non_autoregressive_transformer = Transformer(ModelArgs(
            device=self.gpu_id, 
            dim=self.hidden_dim, 
            n_layers=self.num_transformer_layers,
            n_heads=self.num_heads,
            ffn_dim_multiplier=self.ffm,
            max_batch_size=self.max_bs,
            max_seq_len=self.max_seq_len,
            activation=activation)) 
        
        self.non_autoregressive_output = nn.Sequential(
            nn.Linear(hidden_dim, output_channels),
            nn.Tanh())
            
    def forward(self, z):
        batch_size = z.size(0)
        
        # Step 1: Non-autoregressive generation (rough sketch)
        h_na = self.non_autoregressive_fc(z).view(batch_size, self.seq_length, self.hidden_dim)
        h_na = self.non_autoregressive_transformer(h_na, start_pos=0, causal_mask_bool=False)  # Self-attention with no causal mask
        x_na = self.non_autoregressive_output(h_na)  # (batch_size, seq_length, output_channels)
        
        return x_na

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
        num_encode_concat_transformer_tokens,
        transformer_start_pos,
        transformer_dim,
        encoder_transformer_activation,
        top_dims,
        hidden_dims,
        latent_dim, 
        decoder_hidden_dims,
        decoder_num_heads,
        decoder_num_transformer_layers,
        decoder_ffm,
        decoder_max_batch_size,
        decoder_max_seq_len,
        decoder_transformer_activation,
        gpu_id=None,  
        **kwargs):

        super(VAE, self).__init__()

        self.gpu_id = gpu_id
        self.autoencode_samples = autoencode_samples
        self.padded_channels = padded_channels
        self.crattn_embed_dim = crattn_embed_dim
        self.num_encode_concat_transformer_tokens = num_encode_concat_transformer_tokens
        self.transformer_start_pos = transformer_start_pos
        self.transformer_dim = transformer_dim
        self.encoder_transformer_activation = encoder_transformer_activation
        self.top_dims = top_dims
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim 

        self.decoder_hidden_dims = decoder_hidden_dims
        self.decoder_num_transformer_layers = decoder_num_transformer_layers
        self.decoder_num_heads = decoder_num_heads
        self.decoder_ffm = decoder_ffm
        self.decoder_max_batch_size = decoder_max_batch_size
        self.decoder_max_seq_len = decoder_max_seq_len
        self.decoder_transformer_activation = decoder_transformer_activation

        # Raw CrossAttention Head
        self.encoder_head = Encoder_TimeSeriesWithCrossAttention(padded_channels = self.padded_channels, crattn_embed_dim=self.crattn_embed_dim, **kwargs)

        # Transformer - dimension is same as output of cross attention
        self.transformer_encoder = Transformer(ModelArgs(device=self.gpu_id, dim=self.transformer_dim, activation=self.encoder_transformer_activation, **kwargs))

        # Core Encoder
        # self.head_to_top = nn.Linear(self.crattn_embed_dim * self.autoencode_samples, self.top_dims, bias=True)
        # self.norm_top = RMSNorm(dim=self.top_dims)
        self.top_to_hidden = nn.Linear(self.top_dims, self.hidden_dims, bias=True)
        self.norm_hidden = RMSNorm(dim=self.hidden_dims)
        
        # Encoder variational layers (not shared between enc/dec)
        self.mean_fc_layer = nn.Linear(self.hidden_dims, self.latent_dim, bias=True)  # bias=False
        self.logvar_fc_layer = nn.Linear(self.hidden_dims, self.latent_dim, bias=True) # bias=False

        # self.decoder = SimpleDecoder(
        #     gpu_id = self.gpu_id,
        #     latent_dim = self.latent_dim,
        #     hidden_dim = self.decoder_hidden_dims,
        #     padded_channels = self.padded_channels,
        #     seq_length = self.autoencode_samples
        # )

        self.decoder = HybridDecoder(
            gpu_id = self.gpu_id,
            latent_dim = self.latent_dim,
            hidden_dim = self.decoder_hidden_dims,
            output_channels = self.padded_channels,
            seq_length = self.autoencode_samples,
            num_heads = self.decoder_num_heads,
            num_transformer_layers = self.decoder_num_transformer_layers,
            ffm = self.decoder_ffm,
            max_bs = self.decoder_max_batch_size,
            max_seq_len = self.decoder_max_seq_len,
            activation=self.decoder_transformer_activation
            )

        self.silu = nn.SiLU()

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)  
        epsilon = torch.randn_like(std).to(self.gpu_id) 
        z = mean + std * epsilon
        return z

    def concat_past_tokens(self, x):
        
        num_pulls = x.shape[1] - self.num_encode_concat_transformer_tokens - self.transformer_start_pos
        y = torch.zeros([x.shape[0], num_pulls, self.top_dims]).to(x)

        for i in range(num_pulls):
            y[:, i, :] = x[:, i:i+self.num_encode_concat_transformer_tokens, :].reshape(x.shape[0], self.num_encode_concat_transformer_tokens * x.shape[2])

        return y

    def forward(self, x, reverse=False, hash_pat_embedding=-1, out_channels=-1):

        if reverse == False:

            # RAW CROSS-ATTENTION HEAD
            # [batch, token, channel, waveform]
            y = x.reshape([x.shape[0]*x.shape[1], x.shape[2], x.shape[3]])
            y = self.encoder_head(y)
            y = torch.split(y, x.shape[1], dim=0)
            y = torch.stack(y, dim=0)

            # TRANSFORMER
            y = self.transformer_encoder(y, start_pos=self.transformer_start_pos)

            # VAE CORE
            y = self.concat_past_tokens(y) # Sliding window over transformer output
            y = y.reshape([y.shape[0]*y.shape[1], y.shape[2]]) # Batch the sliding windows for efficient decoding
            y = self.top_to_hidden(y)
            y = self.silu(y)
            y = self.norm_hidden(y)
            mean_batched, logvar_batched = self.mean_fc_layer(y), self.logvar_fc_layer(y)
            y = self.reparameterization(mean_batched, logvar_batched)
            return mean_batched, logvar_batched, y

        elif reverse == True:

            # Add the hash_pat_embedding to latent vector
            y = x + hash_pat_embedding

            y = self.decoder(y).transpose(1,2)
            # y = self.decoder(y)

            y = y[:, 0:out_channels, :]
            return y

def print_models_flow(x, **kwargs):
    '''
    Builds models on CPU and prints sizes of forward passes

    '''

    pat_num_channels = x.shape[2] 

    # Build the VAE
    vae = VAE(**kwargs) 

    # Run through Enc Head
    print(f"INPUT TO <ENC>\n"
    f"x:{x.shape}")
    mean, logvar, latent = vae(x, reverse=False)  
    summary(vae, input_size=(x.shape), depth=999, device="cpu")
    print(
    f"mean:{mean.shape}\n"
    f"logvar:{logvar.shape}\n"
    f"latent:{latent.shape}\n")

    # Run through VAE decoder
    hash_pat_embedding = torch.rand(latent.shape[1])
    print(f"\n\n\nINPUT TO <VAE - Decoder Mode> \n"
    f"z:{latent.shape}\n"
    f"hash_pat_embedding:{hash_pat_embedding.shape}\n")
    core_out = vae(latent, reverse=True, hash_pat_embedding=hash_pat_embedding, out_channels=pat_num_channels)  
    print(f"core_out:{core_out.shape}\n")

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

