import torch
import torch.nn as nn
import math
from torch import Tensor
from torchinfo import summary

# Local imports
from .VAE_heads import BSE_Enc_Head, BSE_Dec_Head, BSE_Dec_Hint_Prep
from .Transformer import ModelArgs, Transformer

# Takes input from Swappable_Enc_Head 
class Enc_CNN_TimeReducer(nn.Module):
    def __init__(self, in_channels, enc_kernel_sizes, stride, depth, resblock_size, poolsize, poolstride):
        super(Enc_CNN_TimeReducer, self).__init__()

        self.in_channels = in_channels

        self.stride = stride
        self.depth = depth
        self.resblock_size = resblock_size
        self.poolsize = poolsize
        self.poolstride = poolstride

        self.kernel_columns = nn.ModuleList()
        for k in enc_kernel_sizes:
            column = nn.ModuleList()
            for l in range(0, self.depth):
                resblock = nn.ModuleList()
                for res_layer in range(0, self.resblock_size):

                    if res_layer == 0:
                        # Time reduction with maxpool
                        unit = nn.Sequential(
                            # nn.Conv1d(in_channels=int(self.in_channels/(2 ** l)), out_channels=int(self.in_channels/(2 ** (l+1))), kernel_size=k, stride=self.stride, padding=int((k-1)/2)),
                            nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=k, stride=self.stride, padding=int((k-1)/2)),
                            nn.MaxPool1d(self.poolsize, stride=poolstride),
                            nn.LeakyReLU(0.2)
                        )

                    else:
                        # No time reduction
                        unit = nn.Sequential(
                            # nn.Conv1d(in_channels=int(self.in_channels/(2 ** (l+1))), out_channels=int(self.in_channels/(2 ** (l+1))), kernel_size=k, stride=self.stride, padding=int((k-1)/2)),
                            nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=k, stride=self.stride, padding=int((k-1)/2)),
                            nn.LeakyReLU(0.2)
                        )

                    resblock.append(unit)
                column.append(resblock)
            self.kernel_columns.append(column)

    def forward(self, x_stack):
        
        # Send down the layers seperately, so iterate kernel in top 'for' loop
        kernel_outs = []
        for k in range(len(self.kernel_columns)):
            x_k = x_stack[:, :, :, k]
            for l in range(0, self.depth):
                x0 = []
                for r in range(0, self.resblock_size):
                    unit = self.kernel_columns[k][l][r]
                    x_k = unit(x_k)
                    if r == 0: x0 = x_k #save first layer outputs to be added on (skip connection)
                
                # Add the skip connection
                x_k += x0
            
            # Save the full depth of network for this kernel
            kernel_outs.append(x_k)
        # y = torch.stack(kernel_outs, dim=3)
        y = torch.sum(torch.stack(kernel_outs,dim=3), dim = 3)

        return y


class Dec_CNN_TimeDilator(nn.Module):
    def __init__(self, in_channels, dec_kernel_sizes, depth, resblock_size):
        super(Dec_CNN_TimeDilator, self).__init__()

        # self.stride = 2 # Will grow Seq length by 2
        self.depth = depth
        self.resblock_size = resblock_size
        self.in_channels = in_channels
        self.dec_kernel_sizes = dec_kernel_sizes

        self.kernel_columns = nn.ModuleList()
        for k in dec_kernel_sizes:
            column = nn.ModuleList()
            for l in range(0, self.depth):
                resblock = nn.ModuleList()
                for res_layer in range(0, self.resblock_size):
                    
                    # time expansion
                    if res_layer == 0:
                        unit = nn.Sequential(
                            # nn.ConvTranspose1d(in_channels=int(self.in_channels*(2**l)), out_channels=int(self.in_channels*(2**(l+1))), kernel_size=k, stride=2, padding=int((k-1)/2), output_padding=1),
                            nn.ConvTranspose1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=k, stride=2, padding=int((k-1)/2), output_padding=1),
                            nn.LeakyReLU(0.2)
                        )

                    # No time or channel expansion
                    else:
                        unit = nn.Sequential(
                            nn.ConvTranspose1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=k, stride=1, padding=int((k-1)/2), output_padding=0),
                            # nn.ConvTranspose1d(in_channels=int(self.in_channels*(2**(l+1))), out_channels=int(self.in_channels*(2**(l+1))), kernel_size=k, stride=1, padding=int((k-1)/2), output_padding=0),
                            nn.LeakyReLU(0.2)
                        )
                    resblock.append(unit)
                column.append(resblock)
            self.kernel_columns.append(column)

    def forward(self, x):
        
        x_stack = x.reshape(x.shape[0], self.in_channels, -1, len(self.dec_kernel_sizes))

        # Send up the layers seperately, so iterate kernel in top 'for' loop
        kernel_outs = []
        for k in range(len(self.kernel_columns)):
            x_k = x_stack[:, :, :, k]
            for l in range(0, self.depth):
                x0 = []
                for r in range(0, self.resblock_size):
                    unit = self.kernel_columns[k][l][r]
                    x_k = unit(x_k)
                    if r == 0: x0 = x_k # save for skip connection
                
                # Add the skip connection
                x_k += x0

            kernel_outs.append(x_k)
        y = torch.stack(kernel_outs, dim=3)

        return y


class Dec_CNN_FlatTimeFlatChannel(nn.Module):
    def __init__(self, in_channels, dec_kernel_sizes, depth, resblock_size):
        super(Dec_CNN_FlatTimeFlatChannel, self).__init__()

        self.stride = 1 # Time stable
        self.depth = depth
        self.resblock_size = resblock_size
        self.in_channels = in_channels
        self.dec_kernel_sizes = dec_kernel_sizes

        self.kernel_columns = nn.ModuleList()
        for k in dec_kernel_sizes:
            column = nn.ModuleList()
            for l in range(0, self.depth):
                resblock = nn.ModuleList()
                for reslayer in range(0, self.resblock_size):
                    # No special shape cdhanges between reslayers
                    unit = nn.Sequential(
                        nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels, kernel_size=k, stride=self.stride, padding=int((k-1)/2), output_padding=0),
                        nn.LeakyReLU(0.2)
                    )
                    resblock.append(unit)
                column.append(resblock)
            self.kernel_columns.append(column)

    def forward(self, x_stack):
        
        # Send up the layers seperately, so iterate kernel in top 'for' loop
        kernel_outs = []
        for k in range(len(self.kernel_columns)):
            x_k = x_stack[:, :, :, k]
            for l in range(0, self.depth):
                x0 = []
                for r in range(0, self.resblock_size):
                    unit = self.kernel_columns[k][l][r]
                    x_k = unit(x_k)
                    if r ==0: x0 = x_k # save the skip connection

                # Add the skip connection
                x_k += x0

            kernel_outs.append(x_k)
        y = torch.stack(kernel_outs, dim=3)

        return y


class VAE_Enc(nn.Module):
    def __init__(
            self, 
            common_ENC_cnn_channels,
            precode_samples, 
            latent_dim, 
            decode_samples, 
            hidden_encode_dims,
            dropout_dec, 
            dropout_enc, 
            enc_conv_depth, # Dictates seq length compression
            enc_kernel_sizes, 
            dec_conv_dilator_depth,
            dec_conv_flat_depth,
            transconv_kernel_sizes, 
            enc_conv_resblock_size,
            dec_conv_dilator_resblock_size,
            dec_conv_flat_resblock_size,
            hint_size_factor,
            gpu_id=None, 
            **kwargs):
        
        super(VAE_Enc, self).__init__()

        self.gpu_id = gpu_id
        self.hidden_encode_dims = hidden_encode_dims

        self.common_ENC_cnn_channels = common_ENC_cnn_channels # all subjects will go to these channels in middle of autoencoder

        self.encoder_drop_val = dropout_enc
        print(f"Dropout Encoder: {self.encoder_drop_val}")
        self.dropout_encoder = nn.Dropout(self.encoder_drop_val) 

        self.enc_conv_depth = enc_conv_depth
        self.enc_kernel_sizes = enc_kernel_sizes
        self.dec_conv_dilator_depth = dec_conv_dilator_depth
        self.transconv_kernel_sizes = transconv_kernel_sizes

        self.enc_conv_resblock_size = enc_conv_resblock_size
        self.dec_conv_dilator_resblock_size = dec_conv_dilator_resblock_size
        self.dec_conv_flat_resblock_size = dec_conv_flat_resblock_size

        self.precode_samples = precode_samples 
        self.decode_samples = decode_samples

        self.latent_dim = latent_dim
            
        self.enc_conv_stride = 1
        self.enc_conv_poolsize = 2
        self.enc_conv_poolstride = 2
        self.enc_cnn_PRE = Enc_CNN_TimeReducer(
            in_channels=self.common_ENC_cnn_channels, 
            enc_kernel_sizes=enc_kernel_sizes, 
            stride=self.enc_conv_stride, 
            depth=self.enc_conv_depth, 
            resblock_size=self.enc_conv_resblock_size,
            poolsize=self.enc_conv_poolsize, 
            poolstride=self.enc_conv_poolstride)
        
        self.top_enc_dims = int((self.common_ENC_cnn_channels) * ((self.precode_samples) / (2 ** self.enc_conv_depth)))

        self.top_to_hidden = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(self.top_enc_dims, self.hidden_encode_dims),
            nn.LeakyReLU(0.2)
        )
        
        # Variational layers
        self.mean_fc_layer = nn.Linear(self.hidden_encode_dims, self.latent_dim)
        self.logvar_fc_layer = nn.Linear(self.hidden_encode_dims, self.latent_dim)

    def get_script_filename(self):
        return __file__

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)  
        epsilon = torch.randn_like(std).to(self.gpu_id) 
        z = mean + std * epsilon
        return z
      
    def encode(self, x_pre_posthead):
        y = self.enc_cnn_PRE(x_pre_posthead)
        y = y.flatten(start_dim=1)
        y = self.dropout_encoder(self.top_to_hidden(y))    
        mean, logvar = self.mean_fc_layer(y), self.logvar_fc_layer(y)
        return mean, logvar

    def forward(self, x_pre_posthead):
        
        # Encode
        mean, logvar = self.encode(x_pre_posthead)

        # Reparameterize
        z = self.reparameterization(mean, logvar)

        return mean, logvar, z
        
        
class VAE_Dec(nn.Module):
    def __init__(
        self, 
        latent_dim, 
        decode_samples, 
        hidden_encode_dims,
        dropout_dec, 
        dec_conv_dilator_depth,
        dec_conv_flat_depth,
        transconv_kernel_sizes, 
        dec_conv_dilator_resblock_size,
        dec_conv_flat_resblock_size,
        hint_size_factor,
        gpu_id=None, 
        **kwargs):
        
        super(VAE_Dec, self).__init__()
        
        self.gpu_id = gpu_id
        self.hidden_encode_dims = hidden_encode_dims

        self.decoder_drop_val = dropout_dec
        print(f"Dropout Decoder: {self.decoder_drop_val}")

        self.dec_conv_dilator_depth = dec_conv_dilator_depth
        self.transconv_kernel_sizes = transconv_kernel_sizes

        self.dec_conv_dilator_resblock_size = dec_conv_dilator_resblock_size
        self.dec_conv_flat_resblock_size = dec_conv_flat_resblock_size

        self.decode_samples = decode_samples
        self.latent_dim = latent_dim
        self.latent_hint_size = int(self.latent_dim/hint_size_factor)

        self.dec_hidden_1_size = self.hidden_encode_dims
        self.dec_hidden_2_size = self.hidden_encode_dims * 2
        self.length_dec_bottom = 1
        self.num_cnn_chans_start_dec = int(self.dec_hidden_2_size / len(self.transconv_kernel_sizes) / self.length_dec_bottom)

        self.latent_to_top = nn.Sequential(
            nn.Linear(self.latent_dim + self.latent_hint_size, self.dec_hidden_1_size),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dec_hidden_1_size, self.dec_hidden_2_size)
        )

        self.dec_cnn_dilator = Dec_CNN_TimeDilator(
            in_channels=self.num_cnn_chans_start_dec, 
            dec_kernel_sizes=self.transconv_kernel_sizes, 
            depth=dec_conv_dilator_depth,
            resblock_size=self.dec_conv_dilator_resblock_size
        )

        self.dec_cnn_flat = Dec_CNN_FlatTimeFlatChannel(
            in_channels=self.num_cnn_chans_start_dec, 
            dec_kernel_sizes=self.transconv_kernel_sizes, 
            depth=dec_conv_flat_depth,
            resblock_size=self.dec_conv_flat_resblock_size
        )
        
    def get_script_filename(self):
        return __file__
    
    def decode(self, z_with_hints):
        # y = self.latent_to_parallel_hidden(z)
        y = self.latent_to_top(z_with_hints)
        y = self.dec_cnn_dilator(y)
        y = self.dec_cnn_flat(y)

        return y


    def forward(self, z, x_pre_hint_flat_prepped):
        
        z_with_hints = torch.cat([z, x_pre_hint_flat_prepped], dim=1)

        # Decode TODO: consider having parallel architecture to encoder
        out_prehead = self.decode(z_with_hints)

        return out_prehead
        
# Builds models on CPU and prints sizes of forward passes
def print_models_flow(x_pre, feedforward_hint_samples, transformer_seq_length, **kwargs):
    
    batch_size = x_pre.shape[0]
    pat_num_channels = x_pre.shape[1]
    data_length = x_pre.shape[2]

    train_enc_head = BSE_Enc_Head(pat_id="na", num_channels=pat_num_channels, **kwargs)
    train_dec_head = BSE_Dec_Head(pat_id="na", num_channels=pat_num_channels, **kwargs)
    train_hint_prepper = BSE_Dec_Hint_Prep(pat_id="na", feedforward_hint_samples=feedforward_hint_samples, num_channels=pat_num_channels, **kwargs)

    # Break off the last samples of pre and first samples of post to provide hint to decoder for phase alignment
    x_pre_hint = x_pre[:, :, -feedforward_hint_samples:]

    # Build the core models
    vae_enc = VAE_Enc(**kwargs) 
    vae_dec = VAE_Dec(**kwargs) 

    # Build the Transformer
    transformer = Transformer(ModelArgs(**kwargs))

    # Run through Enc Head
    print(f"INPUT TO <ENC HEAD>\n"
    f"x_pre:{x_pre.shape}")
    x_pre_posthead = train_enc_head(x_pre)
    summary(train_enc_head, input_size=x_pre.shape, depth=999, device="cpu")
    print(f"x_pre_posthead:{x_pre_posthead.shape}\n")

    # Prep the phase hints
    print(f"\n\n\nINPUT TO <HINT PREPPER>\n"
    f"x_pre_hint:{x_pre_hint.shape}")
    x_pre_hint_flat_prepped = train_hint_prepper(x_pre_hint)
    summary(train_hint_prepper, input_size=x_pre_hint.shape, depth=999, device="cpu")
    print(f"x_pre_hint_flat_prepped:{x_pre_hint_flat_prepped.shape}\n")

    # Run through VAE Enc
    print(f"\n\n\nINPUT TO <VAE ENC>\n"
    f"x_pre_posthead:{x_pre_posthead.shape}\n")
    mean, logvar, latent = vae_enc(x_pre_posthead)  
    summary(vae_enc, input_size=(x_pre_posthead.shape), depth=999, device="cpu")
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
    print(f"\n\n\nINPUT TO <VAE DEC>\n"
    f"z:{latent.shape}\n"
    f"x_pre_hint_flat_prepped:{x_pre_hint_flat_prepped.shape}")
    core_out = vae_dec(latent, x_pre_hint_flat_prepped )  
    summary(vae_dec, input_size=(latent.shape, x_pre_hint_flat_prepped.shape), depth=999, device="cpu")
    print(f"core_out:{core_out.shape}\n")

    # Run through Dec Head
    print(f"\n\n\nINPUT TO <DEC HEAD>\n"
    f"core_out:{core_out.shape}")
    x_hat = train_dec_head(core_out)
    summary(train_dec_head, input_size=core_out.shape, depth=999, device="cpu")
    print(f"\n<FINAL OUTPUT>\n"
    f"x_pre_posthead:{x_hat.shape}\n")

    del train_enc_head, vae_enc, vae_dec, train_hint_prepper, train_dec_head, transformer

if __name__ == "__main__":
    
    print("Nothing here")