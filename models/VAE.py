import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from torchinfo import summary

# Local imports
from .Transformer import ModelArgs, Transformer, RMSNorm

class Head_Optimizers():
    def __init__(self, heads, wd, betas, lr):
        super(Head_Optimizers, self).__init__()

        self.head_opts = [-1]*len(heads)
        for i in range(0, len(heads)):
            self.head_opts[i] = torch.optim.AdamW(heads[i].parameters(), lr=lr, weight_decay=wd, betas=betas)

    def set_all_lr(self, lr):
        for i in range(0, len(self.head_opts)):
            for g in self.head_opts[i].param_groups:
                g['lr'] = lr

    def get_lr(self):
        lrs = []
        for i in range(0, len(self.head_opts)):
            for g in self.head_opts[i].param_groups:
                lrs.append(g['lr'])

        if len(set(lrs)) <= 1:
            return lrs[0]
        else:
            raise Exception(f"Head LRs not all the same [{lrs}], not currently the same. Not coded to handle heads having different learning rates.")

    def zero_grad(self):
        for i in range(0, len(self.head_opts)):
            self.head_opts[i].zero_grad()
        
    def step(self):
        for i in range(0, len(self.head_opts)):
            self.head_opts[i].step()

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
        in_channels, 
        crattn_in_channel_padding,
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
        
        self.in_channels = in_channels
        self.in_channel_padded_dims = crattn_in_channel_padding
        self.embed_dim = crattn_embed_dim
        self.num_highdim_heads = crattn_num_highdim_heads
        self.num_highdim_layers = crattn_num_highdim_layers
        self.num_lowdim_heads = crattn_num_lowdim_heads
        self.num_lowdim_layers = crattn_num_lowdim_layers
        self.max_seq_len = crattn_max_seq_len
        # self.cnn_kernel_size = crattn_cnn_kernel_size
        self.dropout = crattn_dropout

        self.crattn_in_channel_padding = crattn_in_channel_padding

        # Depthwise 1D CNN (groups=in_channels)
        # self.cnn = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=self.cnn_kernel_size, 
                            #  padding=(self.cnn_kernel_size // 2), groups=self.in_channels)  # Depthwise convolution
        
        # Input Cross-attention Layer (also converts from in_channels to embed_dim in feedforward layer)
        self.highdim_attention_layers = nn.ModuleList([
            TimeSeriesCrossAttentionLayer(self.in_channel_padded_dims, self.num_highdim_heads, self.dropout)
            for _ in range(self.num_highdim_layers)
        ])

        # Convert to low dims
        self.high_to_low_dims = nn.Sequential(
            nn.Linear(self.in_channel_padded_dims, self.in_channel_padded_dims * 2),
            nn.SiLU(),
            nn.Linear(self.in_channel_padded_dims * 2, self.in_channel_padded_dims),
            nn.SiLU(),
            nn.Linear(self.in_channel_padded_dims, self.embed_dim),
            nn.SiLU()
        )

        # Embed-Dim Cross-attention layers
        self.lowdim_attention_layers = nn.ModuleList([
            TimeSeriesCrossAttentionLayer(self.embed_dim, self.num_lowdim_heads, self.dropout)
            for _ in range(self.num_lowdim_layers)
        ])
        
        # Positional encoding
        self.positional_encoding = self._get_positional_encoding(self.max_seq_len, self.in_channel_padded_dims)
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
        
        # Step 1: pad the channel dimension
        padding = (0, 0, 0, self.in_channel_padded_dims - self.in_channels, 0, 0) # (dim0 left padding, dim0 right padding... etc.)
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

        return torch.mean(x, dim=1) # Mean along the sequence direction
        # return x[:,-1,:] # Return last in sequence (should be embedded with meaning)

class TimeSeriesDecoder(nn.Module):
    def __init__(self, in_dim, num_channels, seq_len, num_layers, kernel_size, dropout=0.0):
        super(TimeSeriesDecoder, self).__init__()

        self.in_dim = in_dim
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout

        layers = []

        # First, we will apply a convolution to map to num_channels
        layers.append(nn.ConvTranspose1d(in_channels=self.in_dim, out_channels=num_channels, kernel_size=kernel_size, stride=2, padding=(kernel_size // 2), output_padding=1))
        layers.append(nn.SiLU())
        layers.append(RMSNorm_Conv(num_channels))

        # First, we need to expand the input (batch_size, embed_dim) to (batch_size, embed_dim, 1)
        # The output will be passed through N dilated 1D convolutional layers.

        # dilation_rate = 1  # We will increase the dilation rate with each layer
        output_length = seq_len  # Initial sequence length

        for i in range(num_layers):
            # Dilated convolution
            # layers.append(nn.Conv1d(in_channels=in_channels, out_channels=in_channels, 
            #                         kernel_size=kernel_size, 
            #                         dilation=dilation_rate, 
            #                         padding=(kernel_size // 2) * dilation_rate))
            layers.append(nn.ConvTranspose1d(in_channels=num_channels, out_channels=num_channels, 
                                    kernel_size=kernel_size, stride=2, padding=(kernel_size // 2), output_padding=1))
            

            if i < (num_layers -1):
                layers.append(nn.SiLU())
                layers.append(RMSNorm_Conv(num_channels))
                layers.append(nn.Dropout(self.dropout))

            elif i == num_layers -1:
                layers.append(nn.Tanh())
            
            # Increase the sequence length progressively
            # layers.append(nn.Upsample(scale_factor=2, mode='linear'))  # This doubles the sequence length after each layer
            
            # dilation_rate *= 2  # Increase the dilation rate for each layer (progressively dilates the sequence)

        # Combine the layers into a Sequential container
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch_size, embed_dim)
        
        # Unsqueeze to add the sequence dimension
        x = x.unsqueeze(2)  # Shape: (batch_size, embed_dim, 1)
        
        # Pass through the convolutional layers
        x = self.cnn(x)  # Shape: (batch_size, num_channels, seq_len)
        
        return x

class VAEHead_TiedEncDec(nn.Module):
    '''
    Interface from a single patient's raw data channels to core VAE
    Reversible with tied weights
    '''
    def __init__(self, pat_id, in_channels, autoencode_samples, crattn_embed_dim, num_decode_layers, decode_kernel_size, top_dims, **kwargs):
        super(VAEHead_TiedEncDec, self).__init__()

        self.pat_id = pat_id
        self.in_channels = in_channels
        self.autoencode_samples = autoencode_samples
        self.crattn_embed_dim = crattn_embed_dim
        self.num_decode_layers = num_decode_layers
        self.decode_kernel_size = decode_kernel_size
        
        ### ENCODER
        self.encoder = Encoder_TimeSeriesCNNWithCrossAttention(in_channels = self.in_channels, crattn_embed_dim=crattn_embed_dim, **kwargs)

        ### DECODER
        self.decoder = TimeSeriesDecoder(in_dim=top_dims, num_channels = self.in_channels, seq_len=autoencode_samples, num_layers=num_decode_layers, kernel_size=decode_kernel_size)

    def forward(self, x, reverse=False):

        # Encoder part
        if reverse == False: 
            return self.encoder(x)
        
        # Decoder part
        if reverse == True:
            return self.decoder(x)


class VAE(nn.Module):
    '''
    The Reverseable Encoder/Decoder 
    Shares weights between Conv/TransConv layers, in addition to FC layers (except Mean/Logvar layers)
    '''
    def __init__(
        self, 
        autoencode_samples,
        crattn_embed_dim,
        top_dims,
        hidden_dims,
        latent_dim, 
        gpu_id=None, 
        **kwargs):

        super(VAE, self).__init__()

        self.gpu_id = gpu_id
        self.autoencode_samples = autoencode_samples
        self.crattn_embed_dim = crattn_embed_dim
        self.top_dims = top_dims
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        self.head_to_top = nn.Linear(self.crattn_embed_dim, self.top_dims, bias=True)
        # self.top_to_head = nn.Linear(self.top_dims, self.crattn_embed_dim, bias=False)
        # self.top_to_head.weight = nn.Parameter(self.head_to_top.weight.T)
        # self.hidden_to_top.bias = nn.Parameter(self.top_to_hidden.bias.T.detach()) # Shapes do not match to share biases (could duplicate it??? naaa)

        self.top_to_hidden = nn.Linear(self.top_dims, self.hidden_dims, bias=True)
        self.hidden_to_top = nn.Linear(self.hidden_dims, self.top_dims, bias=True)
        # self.hidden_to_top.weight = nn.Parameter(self.top_to_hidden.weight.T)
        # self.hidden_to_top.bias = nn.Parameter(self.top_to_hidden.bias.T.detach()) # Shapes do not match to share biases (could duplicate it??? naaa)

        # Variational layers (not shared between enc/dec)
        self.mean_fc_layer = nn.Linear(self.hidden_dims, self.latent_dim, bias=True)  # bias=False
        self.logvar_fc_layer = nn.Linear(self.hidden_dims, self.latent_dim, bias=True) # bias=False

        # Hidden to latent
        # self.hidden_to_latent = nn.Linear(self.hidden_dims, self.latent_dim, bias=True)  # bias=False

        # Latent to hidden (not shared between enc/dec)
        self.latent_to_hidden = nn.Linear(self.latent_dim, self.hidden_dims, bias=True) # bias=False
        # self.latent_to_hidden.weight = nn.Parameter(self.mean_fc_layer.weight.T.detach()) # Tie the "mean" layer weights
        # self.latent_to_hidden.weight = nn.Parameter(self.hidden_to_latent.weight.T) # Tie weights

        # self.leaky_relu = nn.LeakyReLU(0.2)
        # self.tanh = nn.Tanh()
        self.silu = nn.SiLU()

        self.norm_top = RMSNorm(dim=self.top_dims)
        self.norm_top_rev = RMSNorm(dim=self.top_dims)

        self.norm_hidden = RMSNorm(dim=self.hidden_dims)
        self.norm_hidden_rev = RMSNorm(dim=self.hidden_dims)

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)  
        epsilon = torch.randn_like(std).to(self.gpu_id) 
        z = mean + std * epsilon
        return z

    def forward(self, x, reverse=False):

        if reverse == False:
            y = self.head_to_top(x)
            # y = self.tanh(y)
            y = self.silu(y)
            y = self.norm_top(y)
            y = self.top_to_hidden(y)
            # y = self.leaky_relu(y)
            # y = self.tanh(y)
            y = self.silu(y)
            y = self.norm_hidden(y)
            mean, logvar = self.mean_fc_layer(y), self.logvar_fc_layer(y)
            y = self.reparameterization(mean, logvar)
            # return mean, logvar, z
            # y = self.hidden_to_latent(y)
            # y = self.tanh(y)
            return mean, logvar, y

        elif reverse == True:
            y = self.latent_to_hidden(x)
            # y = self.leaky_relu(y)
            # y = self.tanh(y)
            y = self.silu(y)
            y = self.norm_hidden_rev(y)
            y = self.hidden_to_top(y)
            # y = self.leaky_relu(y)
            # y = self.tanh(y)
            y = self.silu(y)
            y = self.norm_top_rev(y)
            # y = self.top_to_head(y)
            # y = self.tanh(y)
            return y

def print_models_flow(x, transformer_seq_length, **kwargs):
    '''
    Builds models on CPU and prints sizes of forward passes

    '''

    batch_size = x.shape[0]
    pat_num_channels = x.shape[1] 
    data_length = x.shape[2]

    train_head = VAEHead_TiedEncDec(pat_id="na", in_channels=pat_num_channels, **kwargs)

    # Build the core models
    vae = VAE(**kwargs) 

    # Build the Transformer
    transformer = Transformer(ModelArgs(**kwargs))

    # Run through Enc Head
    print(f"INPUT TO <ENC HEAD>\n"
    f"x:{x.shape}")
    x_posthead = train_head(x, reverse=False)
    summary(train_head, input_size=x.shape, depth=999, device="cpu")
    print(f"x_posthead:{x_posthead.shape}\n")

    # Run through VAE Enc
    print(f"\n\n\nINPUT TO <VAE TIED ENC/DEC - Encoder Mode>\n"
    f"x_posthead:{x_posthead.shape}\n")
    # mean, logvar, latent = vae(x_posthead, reverse=False)  
    mean, logvar, latent = vae(x_posthead, reverse=False)  
    summary(vae, input_size=(x_posthead.shape), depth=999, device="cpu")
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
    print(f"\n\n\nINPUT TO <VAE TIED ENC/DEC - Deocder Mode> mostly same weights as core encoder\n"
    f"z:{latent.shape}\n")
    core_out = vae(latent, reverse=True)  
    print(f"core_out:{core_out.shape}\n")

    # Run through Dec Head
    print(f"\nINPUT TO <TIED HEAD - Decoder Mode> not the same as head encoder\n"
    f"core_out:{core_out.shape}")
    x_hat = train_head(core_out, reverse=True)
    print(f"\n<FINAL OUTPUT>\n"
    f"x_hat:{x_hat.shape}\n")

    del train_head, vae, transformer

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

