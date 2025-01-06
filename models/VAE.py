import torch
import torch.nn as nn
import math
from torch import Tensor
from torchinfo import summary

# Local imports
from .Transformer import ModelArgs, Transformer

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

class Tied_ConvTransConv1D_Block(nn.Module):
    # A single Conv and TransConv tied layer
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(Tied_ConvTransConv1D_Block, self).__init__()
        
        # Define the Conv1D layer
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        
        # Define the TransConv1D (ConvTranspose1D) layer
        self.trans_conv1d = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False)
        
        # Assign the shared weight to both Conv1D and TransConv1D layers
        self.trans_conv1d.weight = nn.Parameter(self.conv1d.weight.detach())
        # self.trans_conv1d.bias = nn.Parameter(self.conv1d.bias.detach())   # Shapes do not match to share biases (could duplicate it??? naaa)

        # self.leakyrelu = nn.LeakyReLU(0.2)
        self.hardtanh = nn.Hardtanh()

    def forward(self, x, reverse=False):
        if reverse == False:
            y = self.conv1d(x)

        elif reverse == True:
            y = self.trans_conv1d(x)
            
        # x = self.leakyrelu(x)
        y = self.hardtanh(y)

        return y

class VAEHead_TiedEncDec(nn.Module):
    '''
    Interface from a single patient's raw data channels to core VAE
    Reversible with tied weights
    '''
    def __init__(self, pat_id, in_channels, common_cnn_channels, kernel_sizes, stride=1, output_padding=0, **kwargs):
        super(VAEHead_TiedEncDec, self).__init__()

        self.pat_id = pat_id
        self.in_channels = in_channels
        self.common_cnn_channels = common_cnn_channels
        self.stride = stride
        self.output_padding = output_padding

        self.kernel_columns = nn.ModuleList()
        for k in kernel_sizes:
            unit = Tied_ConvTransConv1D_Block(in_channels=self.in_channels, out_channels=self.common_cnn_channels, kernel_size=k, stride=self.stride, padding=int((k-1)/2),output_padding=output_padding)
            self.kernel_columns.append(unit)
        
        # self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x, reverse=False):
        
        if reverse == False:
            # Send *** DOWN *** the layers seperately, so iterate kernel in top for loop
            kernel_outs = []
            for k in range(len(self.kernel_columns)):
                unit = self.kernel_columns[k]
                x_k = unit(x, reverse=reverse)
                # x_k = self.leaky_relu(x_k)
                x_k = self.tanh(x_k)
                kernel_outs.append(x_k)
            y = torch.stack(kernel_outs, dim=3)

        elif reverse == True:
            kernel_outs = []
            # Send *** UP *** the layers seperately, so iterate kernel in top for loop
            for k in reversed(range(len(self.kernel_columns))):
                unit = self.kernel_columns[k]
                x_k = x[:, :, :, k]
                # x_k = self.leaky_relu(x_k)
                x_k = self.tanh(x_k)
                x_k = unit(x_k, reverse=reverse)
                kernel_outs.append(x_k)
            
            y = torch.mean(torch.stack(kernel_outs, dim=3), dim=3) # Mean for output to encourage large values (smooshed by tanh anyway)
            # y = self.tanh(y)
            
        return y

class VAECore_TiedEncDec(nn.Module):
    # Tied VAE Core Enc/Dec 
    # A single encoder decoder that share weights on either side of transformer

    def __init__(self, in_channels, kernel_sizes, stride, depth, resblock_size, time_change):
        super(VAECore_TiedEncDec, self).__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.depth = depth
        self.resblock_size = resblock_size
        self.time_change = time_change

        # Shared Encoder/Decoder
        self.kernel_columns = nn.ModuleList()
        for k in kernel_sizes:
            column = nn.ModuleList()
            for l in range(0, self.depth):
                resblock = nn.ModuleList()
                for res_layer in range(0, self.resblock_size):

                    if (res_layer == 0) & (self.time_change == True):
                        # Time reduction with stride
                        unit = Tied_ConvTransConv1D_Block(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=k, stride=self.stride, padding=int((k-1)/2),output_padding=1)

                    else:
                        # No time reduction in subsequent resblock layers
                        unit = Tied_ConvTransConv1D_Block(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=k, stride = 1, padding=int((k-1)/2),output_padding=0)

                    resblock.append(unit)
                column.append(resblock)
            self.kernel_columns.append(column)


    def forward(self, x_stack, reverse=False):
        
        # Encoder Mode
        if reverse == False:
            # Send *** DOWN *** the layers seperately by kernel size, so iterate kernel in top 'for' loop
            # TODO: blend kernels at each layer
            kernel_outs = []
            for k in range(len(self.kernel_columns)):
                x_k = x_stack[:, :, :, k]
                for l in range(0, self.depth):
                    x0 = [] # rest the residual start data
                    for r in range(0, self.resblock_size):
                        unit = self.kernel_columns[k][l][r]
                        x_k = unit(x_k, reverse=reverse)
                        if r == 0: x0 = x_k #save first layer outputs to be added on (skip connection)
                    
                    # Add the skip connection
                    x_k += x0
                
                # Save the full depth of network for this kernel
                kernel_outs.append(x_k)
            y = torch.stack(kernel_outs, dim=3)

        # Decoder Mode
        elif reverse == True:
            # Send *** UP *** the layers seperately by kernel size, so iterate kernel in top 'for' loop
            # TODO: blend kernels at each layer
            kernel_outs = []
            for k in reversed(range(len(self.kernel_columns))):
                x_k = x_stack[:, :, :, k]
                for l in reversed(range(0, self.depth)):
                    xR = [] # rest the residual start data
                    for r in reversed(range(0, self.resblock_size)):
                        unit = self.kernel_columns[k][l][r]
                        if r == self.resblock_size-1: xR = x_k #save first layer outputs to be added on (skip connection)
                        x_k = unit(x_k, reverse=reverse)
                        
                    # Add the skip connection from first layer in resblock
                    x_k += self.kernel_columns[k][l][0](xR, reverse=reverse)
                
                # Save the full depth of network for this kernel
                kernel_outs.append(x_k)
            y = torch.stack(kernel_outs, dim=3)

        return y

class VAE(nn.Module):
    '''
    The Reverseable Encoder/Decoder 
    Shares weights between Conv/TransConv layers, in addition to FC layers (except Mean/Logvar layers)
    '''
    def __init__(
        self, 
        autoencode_samples,
        common_cnn_channels,
        kernel_sizes, 
        time_change,
        cnn_depth,
        cnn_resblock_layers,
        hidden_dims,
        latent_dim, 
        gpu_id=None, 
        **kwargs):

        super(VAE, self).__init__()

        self.gpu_id = gpu_id
        self.autoencode_samples = autoencode_samples
        self.in_channels = common_cnn_channels
        self.kernel_sizes = kernel_sizes
        self.time_change = time_change
        self.cnn_depth = cnn_depth
        self.cnn_resblock_layers = cnn_resblock_layers
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        self.stride = 2

        self.vae_tied_core = VAECore_TiedEncDec(
            in_channels=self.in_channels,
            kernel_sizes=self.kernel_sizes, 
            stride=self.stride, 
            depth=self.cnn_depth, 
            resblock_size=self.cnn_resblock_layers,
            time_change=self.time_change
        )

        # Calculate the size of the bottom "time" dimension
        if time_change: 
            self.bottom_time_samples = int((self.autoencode_samples) / (2 ** self.cnn_depth))
        else: self.bottom_time_samples = int((self.autoencode_samples))
        
        # For the top of the FC layer
        self.top_enc_dims = (self.in_channels) * (self.bottom_time_samples) * len(self.kernel_sizes)

        # Shared between enc/dec
        self.top_to_hidden = nn.Linear(self.top_enc_dims, self.hidden_dims, bias=False)
        self.hidden_to_top = nn.Linear(self.hidden_dims, self.top_enc_dims, bias=False)
        self.hidden_to_top.weight = nn.Parameter(self.top_to_hidden.weight.T.detach())
        # self.hidden_to_top.bias = nn.Parameter(self.top_to_hidden.bias.T.detach()) # Shapes do not match to share biases (could duplicate it??? naaa)

        # Variational layers (not shared between enc/dec)
        self.mean_fc_layer = nn.Linear(self.hidden_dims, self.latent_dim, bias=False)  # bias=False
        self.logvar_fc_layer = nn.Linear(self.hidden_dims, self.latent_dim, bias=False) # bias=False

        # Latent to hidden (not shared between enc/dec)
        self.latent_to_hidden = nn.Linear(self.latent_dim, self.hidden_dims, bias=False) # bias=False
        self.latent_to_hidden.weight = nn.Parameter(self.mean_fc_layer.weight.T.detach()) # Tie the "mean" layer weights

        # self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)  
        epsilon = torch.randn_like(std).to(self.gpu_id) 
        z = mean + std * epsilon
        return z

    def forward(self, x, reverse=False):

        if reverse == False:
            y = self.vae_tied_core(x, reverse) # Encoder mode
            # y = torch.sum(y, dim = 3) # Sum across results of all kernels
            # y = self.leaky_relu(y)
            y = self.tanh(y)
            y = y.flatten(start_dim=1)
            y = self.top_to_hidden(y)
            # y = self.leaky_relu(y)
            # y = self.tanh(y)
            mean, logvar = self.mean_fc_layer(y), self.logvar_fc_layer(y)
            z = self.reparameterization(mean, logvar)
            return mean, logvar, z

        elif reverse == True:
            y = self.latent_to_hidden(x)
            # y = self.leaky_relu(y)
            # y = self.tanh(y)
            y = self.hidden_to_top(y)
            # y = self.leaky_relu(y)
            y = self.tanh(y)
            y = y.reshape(y.shape[0], self.in_channels, self.bottom_time_samples, len(self.kernel_sizes))
            y = self.vae_tied_core(y, reverse=reverse) # Decoder mode
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

    mean,logvar,z = vae(x, reverse=False)
    x_hat = vae(z, reverse=True)
    loss_fn = nn.MSELoss(reduction='mean')
    recon_loss = loss_fn(x, x_hat) 
    recon_loss.backward()

    print(f"Are the weights of encoder and decoder tied? {torch.allclose(vae.top_to_hidden.weight.T, vae.hidden_to_top.weight)}")

