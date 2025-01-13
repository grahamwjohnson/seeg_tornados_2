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


# class CausalConv1D(nn.Module):
#     """Causal 1D convolution with dilation."""
#     def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
#         super(CausalConv1D, self).__init__()
#         self.pad = (kernel_size - 1) * dilation  # Padding to ensure causality
#         self.conv = nn.Conv1d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             dilation=dilation,
#             padding=self.pad
#         )
    
#     def forward(self, x):
#         x = self.conv(x)
#         return x[:, :, :-self.pad]  # Remove the extra padding

# Residual block with dilated convolutions
class WaveNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size):
        super(WaveNetBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.kernel_size = kernel_size
        
        # Causal convolution to ensure that the model doesn't violate causality
        self.conv = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, stride=1, 
                              padding=int(dilation * (kernel_size - 1) /2), dilation=dilation)
        
        # Skip connection
        # self.skip_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        
        # Residual connection
        # self.residual_conv = nn.Conv1d(self.out_channels, self.out_channels, kernel_size=1)

        self.silu = nn.SiLU()

        # RMS norm
        self.norm = RMSNorm_Conv(dim=self.out_channels)
        
    def forward(self, x):
        # Apply the dilated convolution
        # out = F.tanh(self.conv(x))
        out = self.silu(self.conv(x))
        out = self.norm(out)
        
        # Skip connection (output directly)
        # skip = self.skip_conv(out)
        
        # Residual connection (added back to input)
        # residual = self.residual_conv(out)

        
        # return skip, residual
        return out

class VAEHead_TiedEncDec(nn.Module):
    '''
    Interface from a single patient's raw data channels to core VAE
    Reversible with tied weights
    '''
    def __init__(self, pat_id, in_channels, autoencode_samples, head_interface_dims,
                cnn_channels, num_cnn_blocks, num_cnn_layers_per_block, kernel_size, dilation_base, **kwargs):
        super(VAEHead_TiedEncDec, self).__init__()

        self.pat_id = pat_id
        self.in_channels = in_channels
        self.cnn_channels = cnn_channels
        self.autoencode_samples = autoencode_samples
        # self.in_features = in_channels * autoencode_samples # int((self.cnn_channels *  autoencode_samples )/ 1 ) #
        self.head_interface_dims = head_interface_dims
        
        # self.skip_connection_channels = 256
        self.num_cnn_blocks = num_cnn_blocks
        self.num_cnn_layers_per_block = num_cnn_layers_per_block
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        

        ### ENCODER

        # Encoder: Initial convolution
        self.input_conv = nn.Conv1d(self.in_channels, self.cnn_channels, kernel_size=1)
        
        # Create the residual blocks for the encoder
        self.encoder_blocks = nn.ModuleList()
        for b in range(num_cnn_blocks):
            for i in range(num_cnn_layers_per_block):
                dilation = dilation_base ** (b * num_cnn_layers_per_block + i)
                self.encoder_blocks.append(WaveNetBlock(self.cnn_channels, self.cnn_channels, dilation, kernel_size=kernel_size))
        
        # Hidden space representation
        self.hidden_space = nn.Linear(self.cnn_channels, self.head_interface_dims)
        

        ### DECODER

        # Decoder: Initial linear layer
        self.decoder_input = nn.Linear(self.head_interface_dims, self.cnn_channels)
        
        # Create the residual blocks for the decoder (mirrored architecture)
        self.decoder_blocks = nn.ModuleList()
        for b in range(num_cnn_blocks):
            for i in range(num_cnn_layers_per_block):
                dilation = dilation_base ** (b * num_cnn_layers_per_block + i)
                self.decoder_blocks.append(WaveNetBlock(self.cnn_channels, self.cnn_channels, dilation, kernel_size=kernel_size))
        
        # Final output layer
        self.output_conv = nn.Conv1d(self.cnn_channels,  self.in_channels, kernel_size=1)
        
        # Skip connections
        # self.skip_conv = nn.Conv1d(self.cnn_channels, self.skip_connection_channels, kernel_size=1)
        # self.final_skip_conv = nn.Conv1d(self.skip_connection_channels,  self.in_channels, kernel_size=1)

        self.tanh = nn.Tanh()
        # self.silu = nn.SiLU()
        
    def forward(self, x, reverse=False):


        # Encoder part
        if reverse == False:
            x = self.input_conv(x)
            
            # skip_connections = []
            
            # Apply encoder residual blocks
            for block in self.encoder_blocks:
                # skip, residual = block(x)
                residual = block(x)
                # skip_connections.append(skip)
                x = x + residual  # Residual connection
            
            # Global pooling for hidden space
            x = x.mean(dim=-1)  # Global average pooling along the time dimension
            # x = x.view(x.size(0), -1)
            x = self.hidden_space(x)

            return x
        
        # Decoder part
        if reverse == True:
            x = self.decoder_input(x)
            x = x.view(x.size(0), self.cnn_channels, -1)
            x = x.repeat(1,1,self.autoencode_samples)
            
            # Apply decoder residual blocks
            for block in self.decoder_blocks:
                # skip, residual = block(x)
                # residual = block(x)
                # skip_connections.append(skip)
                x = x + block(x)  # Residual connection
            
            # # Combine skip connections
            # skip_out = sum(skip_connections)
            # skip_out = F.relu(skip_out)
            # skip_out = self.skip_conv(skip_out)
            # skip_out = F.relu(skip_out)
            
            # Final output layer
            # output = self.final_skip_conv(skip_out)
            x = self.output_conv(x)

            x = self.tanh(x)
            
            return x


class VAE(nn.Module):
    '''
    The Reverseable Encoder/Decoder 
    Shares weights between Conv/TransConv layers, in addition to FC layers (except Mean/Logvar layers)
    '''
    def __init__(
        self, 
        autoencode_samples,
        head_interface_dims,
        top_dims,
        hidden_dims,
        latent_dim, 
        gpu_id=None, 
        **kwargs):

        super(VAE, self).__init__()

        self.gpu_id = gpu_id
        self.autoencode_samples = autoencode_samples
        self.head_interface_dims = head_interface_dims
        self.top_dims = top_dims
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        self.head_to_top = nn.Linear(self.head_interface_dims, self.top_dims, bias=False)
        self.top_to_head = nn.Linear(self.top_dims, self.head_interface_dims, bias=False)
        self.top_to_head.weight = nn.Parameter(self.head_to_top.weight.T)
        # self.hidden_to_top.bias = nn.Parameter(self.top_to_hidden.bias.T.detach()) # Shapes do not match to share biases (could duplicate it??? naaa)

        self.top_to_hidden = nn.Linear(self.top_dims, self.hidden_dims, bias=False)
        self.hidden_to_top = nn.Linear(self.hidden_dims, self.top_dims, bias=False)
        self.hidden_to_top.weight = nn.Parameter(self.top_to_hidden.weight.T)
        # self.hidden_to_top.bias = nn.Parameter(self.top_to_hidden.bias.T.detach()) # Shapes do not match to share biases (could duplicate it??? naaa)

        # # Variational layers (not shared between enc/dec)
        # self.mean_fc_layer = nn.Linear(self.hidden_dims, self.latent_dim, bias=False)  # bias=False
        # self.logvar_fc_layer = nn.Linear(self.hidden_dims, self.latent_dim, bias=False) # bias=False

        # Hidden to latent
        self.hidden_to_latent = nn.Linear(self.hidden_dims, self.latent_dim, bias=False)  # bias=False

        # Latent to hidden (not shared between enc/dec)
        self.latent_to_hidden = nn.Linear(self.latent_dim, self.hidden_dims, bias=False) # bias=False
        # self.latent_to_hidden.weight = nn.Parameter(self.mean_fc_layer.weight.T.detach()) # Tie the "mean" layer weights
        self.latent_to_hidden.weight = nn.Parameter(self.hidden_to_latent.weight.T) # Tie weights

        # self.leaky_relu = nn.LeakyReLU(0.2)
        # self.tanh = nn.Tanh()
        self.silu = nn.SiLU()

        self.norm_top = RMSNorm(dim=self.top_dims)
        self.norm_top_rev = RMSNorm(dim=self.top_dims)

        self.norm_hidden = RMSNorm(dim=self.hidden_dims)
        self.norm_hidden_rev = RMSNorm(dim=self.hidden_dims)

    # def reparameterization(self, mean, logvar):
    #     std = torch.exp(0.5 * logvar)  
    #     epsilon = torch.randn_like(std).to(self.gpu_id) 
    #     z = mean + std * epsilon
    #     return z

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
            # mean, logvar = self.mean_fc_layer(y), self.logvar_fc_layer(y)
            # z = self.reparameterization(mean, logvar)
            # return mean, logvar, z
            y = self.hidden_to_latent(y)
            # y = self.tanh(y)
            return y

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
            y = self.top_to_head(y)
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
    latent = vae(x_posthead, reverse=False)  
    summary(vae, input_size=(x_posthead.shape), depth=999, device="cpu")
    print(
    # f"mean:{mean.shape}\n"
    # f"logvar:{logvar.shape}\n"
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
    print(f"\nINPUT TO <TIED HEAD - Decoder Mode> mostly same weights as head encoder\n"
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

