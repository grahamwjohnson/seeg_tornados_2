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

class VAEHead_TiedEncDec(nn.Module):
    '''
    Interface from a single patient's raw data channels to core VAE
    Reversible with tied weights
    '''
    def __init__(self, pat_id, in_channels, autoencode_samples, top_dims, **kwargs):
        super(VAEHead_TiedEncDec, self).__init__()

        self.pat_id = pat_id
        self.in_channels = in_channels
        self.autoencode_samples = autoencode_samples
        self.in_features = in_channels * autoencode_samples
        self.top_dims = top_dims

        # Shared between enc/dec head
        self.subject_to_top = nn.Linear(self.in_features, self.top_dims, bias=False)
        self.top_to_subject = nn.Linear(self.top_dims, self.in_features, bias=False)
        self.top_to_subject.weight = nn.Parameter(self.subject_to_top.weight.T.detach())

        # self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x, reverse=False):
        
        if reverse == False:
            y = x.flatten(start_dim=1)
            y = self.subject_to_top(y)

        elif reverse == True:
            y = self.top_to_subject(x)
            y = y.reshape(y.shape[0], self.in_channels, self.autoencode_samples)
            
        y = self.tanh(y)

        return y

class VAE(nn.Module):
    '''
    The Reverseable Encoder/Decoder 
    Shares weights between Conv/TransConv layers, in addition to FC layers (except Mean/Logvar layers)
    '''
    def __init__(
        self, 
        autoencode_samples,
        top_dims,
        hidden_dims,
        latent_dim, 
        gpu_id=None, 
        **kwargs):

        super(VAE, self).__init__()

        self.gpu_id = gpu_id
        self.autoencode_samples = autoencode_samples
        self.top_dims = top_dims
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Shared between enc/dec
        self.top_to_hidden = nn.Linear(self.top_dims, self.hidden_dims, bias=False)
        self.hidden_to_top = nn.Linear(self.hidden_dims, self.top_dims, bias=False)
        self.hidden_to_top.weight = nn.Parameter(self.top_to_hidden.weight.T.detach())
        # self.hidden_to_top.bias = nn.Parameter(self.top_to_hidden.bias.T.detach()) # Shapes do not match to share biases (could duplicate it??? naaa)

        # # Variational layers (not shared between enc/dec)
        # self.mean_fc_layer = nn.Linear(self.hidden_dims, self.latent_dim, bias=False)  # bias=False
        # self.logvar_fc_layer = nn.Linear(self.hidden_dims, self.latent_dim, bias=False) # bias=False

        # Hidden to latent
        self.hidden_to_latent = nn.Linear(self.hidden_dims, self.latent_dim, bias=False)  # bias=False

        # Latent to hidden (not shared between enc/dec)
        self.latent_to_hidden = nn.Linear(self.latent_dim, self.hidden_dims, bias=False) # bias=False
        # self.latent_to_hidden.weight = nn.Parameter(self.mean_fc_layer.weight.T.detach()) # Tie the "mean" layer weights
        self.latent_to_hidden.weight = nn.Parameter(self.hidden_to_latent.weight.T.detach()) # Tie weights

        # self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    # def reparameterization(self, mean, logvar):
    #     std = torch.exp(0.5 * logvar)  
    #     epsilon = torch.randn_like(std).to(self.gpu_id) 
    #     z = mean + std * epsilon
    #     return z

    def forward(self, x, reverse=False):

        if reverse == False:
            y = self.top_to_hidden(x)
            # y = self.leaky_relu(y)
            y = self.tanh(y)
            # mean, logvar = self.mean_fc_layer(y), self.logvar_fc_layer(y)
            # z = self.reparameterization(mean, logvar)
            # return mean, logvar, z
            y = self.hidden_to_latent(y)
            return y

        elif reverse == True:
            y = self.latent_to_hidden(x)
            # y = self.leaky_relu(y)
            y = self.tanh(y)
            y = self.hidden_to_top(y)
            # y = self.leaky_relu(y)
            y = self.tanh(y)
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

