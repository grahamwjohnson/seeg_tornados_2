import torch
import torch.nn as nn
import math
from torch import Tensor

# Local imports
from VAE_heads import BSE_Enc_Head, BSE_Dec_Head, BSE_Dec_Hint_Prep

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


class BSE_Middle_VAE(nn.Module):
    def __init__(
            self, 
            gpu_id, 
            common_cnn_channels,
            past_sequence_length, 
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
            **kwargs):
        
        super(BSE_Middle_VAE, self).__init__()

        self.gpu_id = gpu_id
        self.hidden_encode_dims = hidden_encode_dims

        self.common_cnn_channels = common_cnn_channels # all subjects will go to these channels in middle of autoencoder

        self.decoder_drop_val = dropout_dec
        self.encoder_drop_val = dropout_enc
        print(f"Dropout Decoder: {self.decoder_drop_val}")
        print(f"Dropout Encoder: {self.encoder_drop_val}")
        # self.dropout_decoder = nn.Dropout(self.decoder_drop_val) # Only cripple the decoder
        self.dropout_encoder = nn.Dropout(self.encoder_drop_val) 

        self.enc_conv_depth = enc_conv_depth
        self.enc_kernel_sizes = enc_kernel_sizes
        self.dec_conv_dilator_depth = dec_conv_dilator_depth
        self.transconv_kernel_sizes = transconv_kernel_sizes

        self.enc_conv_resblock_size = enc_conv_resblock_size
        self.dec_conv_dilator_resblock_size = dec_conv_dilator_resblock_size
        self.dec_conv_flat_resblock_size = dec_conv_flat_resblock_size

        self.past_sequence_length = past_sequence_length 
        self.decode_samples = decode_samples

        self.latent_dim = latent_dim
            
        self.enc_conv_stride = 1
        self.enc_conv_poolsize = 2
        self.enc_conv_poolstride = 2
        self.enc_cnn_PRE = Enc_CNN_TimeReducer(
            in_channels=self.common_cnn_channels, 
            enc_kernel_sizes=enc_kernel_sizes, 
            stride=self.enc_conv_stride, 
            depth=self.enc_conv_depth, 
            resblock_size=self.enc_conv_resblock_size,
            poolsize=self.enc_conv_poolsize, 
            poolstride=self.enc_conv_poolstride)
        
        # self.top_enc_dims = 2 * len(enc_kernel_sizes) * int((self.common_cnn_channels/2) / (self.enc_conv_depth + 1)) * int(self.past_sequence_length/((self.enc_conv_stride * self.enc_conv_poolstride) ** (self.enc_conv_depth + 1)))
        self.top_enc_dims = int((self.common_cnn_channels) * ((self.past_sequence_length) / (2 ** self.enc_conv_depth)))

        self.top_to_hidden = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(self.top_enc_dims, self.hidden_encode_dims),
            nn.LeakyReLU(0.2)
        )
        
        # Variational layers
        self.mean_fc_layer = nn.Linear(self.hidden_encode_dims, self.latent_dim)
        self.logvar_fc_layer = nn.Linear(self.hidden_encode_dims, self.latent_dim)

        # Decoder 
        self.latent_hint_size = int(self.latent_dim/hint_size_factor)

        self.dec_hidden_1_size = self.hidden_encode_dims
        self.dec_hidden_2_size = self.top_enc_dims
        self.length_dec_bottom = 8
        self.num_cnn_chans_start_dec = int(self.dec_hidden_2_size / len(self.transconv_kernel_sizes) / self.length_dec_bottom)

        self.latent_to_top = nn.Sequential(
            nn.Linear(self.latent_dim + 2 * self.latent_hint_size, self.dec_hidden_1_size),
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
            in_channels=self.common_cnn_channels, 
            dec_kernel_sizes=self.transconv_kernel_sizes, 
            depth=dec_conv_flat_depth,
            resblock_size=self.dec_conv_flat_resblock_size
        )
        
    def print_model_size_flow(self):
        print(''
            # f"Input: [{self.num_channels}x{self.past_sequence_length}: {self.num_channels*self.past_sequence_length}]-->"
            # f"GRU-->[{self.gru_dim_mult * self.num_channels * self.bidir_mult}x{self.past_sequence_length}: {self.gru_dim_mult * self.num_channels * self.past_sequence_length * self.bidir_mult}]"
            # f"-->Subsample hidden spacing: {self.past_hidden_samp_spacing}-->"
            # f"[{int((self.gru_dim_mult * self.num_channels * self.past_sequence_length * self.bidir_mult))/(self.past_hidden_samp_spacing)}]-->FC-->"
            # f"DROPOUT({self.encoder_drop_val}[1x{self.hidden_encode_dims_1}])-->"
            # f"mean(GRU)/logvar FC-->"
            # f"Latent: 1x{self.latent_dim}-->"
            # f"DROPOUT({self.decoder_drop_val}({self.num_channels}x{self.hidden_decode_dims_1/self.num_channels}[{self.hidden_decode_dims_1}])-->" +
            # f"TransConv Block-->"
            # f"Output: [{self.num_channels}x{self.decode_samples}: {self.num_channels*self.decode_samples}]"
              )

    def get_script_filename(self):
        return __file__
    
    def encode(self, x_pre_posthead):
        y = self.enc_cnn_PRE(x_pre_posthead)
        y = y.flatten(start_dim=1)
        y = self.dropout_encoder(self.top_to_hidden(y))    
        mean, logvar = self.mean_fc_layer(y), self.logvar_fc_layer(y)
        return mean, logvar

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)  
        epsilon = torch.randn_like(std).to(self.gpu_id) 
        z = mean + std * epsilon
        return z
    
    def decode(self, z_with_hints):
        # y = self.latent_to_parallel_hidden(z)
        y = self.latent_to_top(z_with_hints)
        y = self.dec_cnn_dilator(y)
        y = self.dec_cnn_flat(y)

        return y

    def get_latent(self, x_pre_posthead, x_post_posthead):
        # Encode
        mean, logvar = self.encode(x_pre_posthead, x_post_posthead)

        # Reparameterize
        z = self.reparameterization(mean, logvar)

        return mean, logvar, z

    def forward(self, x_pre_posthead, x_pre_hint_flat_prepped):
        
        # Encode
        mean, logvar = self.encode(x_pre_posthead)

        # Reparameterize
        z = self.reparameterization(mean, logvar)
        z_with_hints = torch.cat([z, x_pre_hint_flat_prepped], dim=1)

        # Decode TODO: consider having parallel architecture to encoder
        out_prehead = self.decode(z_with_hints)

        return out_prehead, mean, logvar, z
    

if __name__ == "__main__":
    train_pats_list = ['EpatX', 'Epaty']
    pat_num_channels = [135, 182]

    kwargs={
        'common_cnn_channels':256,
        'past_sequence_length':1024, 
        'latent_dim':2048,
        'decode_samples':256, 
        'hidden_encode_dims':4096,
        'dropout_dec':0.1,
        'dropout_enc':0.0,
        'enc_conv_depth': 6, 
        'enc_conv_resblock_size': 4,
        'enc_kernel_sizes': [3,5,7,9], # [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33] # [3,7,15,31,63]
        'dec_conv_dilator_depth': 5,
        'dec_conv_dilator_resblock_size': 4,
        'dec_conv_flat_depth': 2,
        'dec_conv_flat_resblock_size': 4,
        'transconv_kernel_sizes': [3,5,7,9], # [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33] # [2,4,8,16,32,64]
        'hint_size_factor':8,
        'feedforward_hint_samples':4
    }

    gpu_id = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_enc_heads = [-1]*len(train_pats_list)
    train_dec_heads = [-1]*len(train_pats_list)
    train_hint_preppers = [-1]*len(train_pats_list)
    for i in range(0, len(train_pats_list)):
        train_enc_heads[i] = BSE_Enc_Head(pat_id=train_pats_list[i], num_channels=pat_num_channels[i], **kwargs).to(gpu_id)
        train_dec_heads[i] = BSE_Dec_Head(pat_id=train_pats_list[i], num_channels=pat_num_channels[i], **kwargs).to(gpu_id)
        train_hint_preppers[i] = BSE_Dec_Hint_Prep(pat_id=train_pats_list[i], num_channels=pat_num_channels[i], **kwargs).to(gpu_id)

    # Build the core model
    vae_core = BSE_Middle_VAE(gpu_id=gpu_id, **kwargs) 
    vae_core = vae_core.to(gpu_id) # move to GPU here to avoid opt_core problems when loading states

    # # Build the optimizers, one for core and individual opts for swappable heads
    # opt_core = torch.optim.AdamW(vae_core.parameters(), lr=kwargs['LR_min_core'], weight_decay=core_weight_decay)
    # opts_train = Head_Optimizers(heads=train_heads, wd=head_weight_decay, lr=kwargs['LR_min_heads'])
    # opts_val = Head_Optimizers(heads=val_heads, wd=head_weight_decay, lr=kwargs['LR_min_heads'])




    # FORWARD EpatX

    fake_pat = 0
    batch_size = 4
    data_length = 1024
    feedforward_hint_samples = 4

    enc_head = train_enc_heads[fake_pat]
    hint_prepper = train_hint_preppers[fake_pat]
    dec_head = train_dec_heads[fake_pat]

    x_pre = torch.rand(batch_size, pat_num_channels[fake_pat], data_length).to(gpu_id)

    # Break off the last samples of pre and first samples of post to provide hint to decoder for phase alignment
    x_pre_hint = x_pre[:, :, -feedforward_hint_samples:]
    
    # Run through Enc Head
    x_pre_posthead = enc_head(x_pre)

    # Prep the phase hints
    x_pre_hint_flat_prepped = hint_prepper(x_pre_hint)

    # Run through full VAE Core
    core_out, mean, logvar, latent = vae_core(x_pre_posthead, x_pre_hint_flat_prepped)  

    # Run through Dec Head
    x_hat = dec_head(core_out)
    
    print(x_hat.shape)