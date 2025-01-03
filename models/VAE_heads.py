import torch
import torch.nn as nn
import math
from torch import Tensor

class Head_Optimizers():
    def __init__(self, heads, wd, lr):
        super(Head_Optimizers, self).__init__()

        enc_heads = heads[0]
        dec_heads = heads[1]

        self.enc_head_opts = [-1]*len(enc_heads)
        self.dec_head_opts = [-1]*len(dec_heads)

        for i in range(0, len(enc_heads)):
            self.enc_head_opts[i] = torch.optim.AdamW(enc_heads[i].parameters(), lr=lr, weight_decay=wd)
            self.dec_head_opts[i] = torch.optim.AdamW(dec_heads[i].parameters(), lr=lr, weight_decay=wd)

    def set_all_lr(self, lr):
        for i in range(0, len(self.enc_head_opts)):
            for g in self.enc_head_opts[i].param_groups:
                g['lr'] = lr
        
        for i in range(0, len(self.dec_head_opts)):
            for g in self.dec_head_opts[i].param_groups:
                g['lr'] = lr

    def get_lr(self):
        lrs = []
        for i in range(0, len(self.enc_head_opts)):
            for g in self.enc_head_opts[i].param_groups:
                lrs.append(g['lr'])
        
        for i in range(0, len(self.dec_head_opts)):
            for g in self.dec_head_opts[i].param_groups:
                lrs.append(g['lr'])

        if len(set(lrs)) <= 1:
            return lrs[0]
        else:
            raise Exception(f"Head LRs not all the same [{lrs}], not currently the same. Not coded to handle heads having different learning rates.")

    def zero_grad(self):
        for i in range(0, len(self.enc_head_opts)):
            self.enc_head_opts[i].zero_grad()
        
        for i in range(0, len(self.dec_head_opts)):
            self.dec_head_opts[i].zero_grad()
        
    def step(self, idx):
        self.enc_head_opts[idx].step()
        self.dec_head_opts[idx].step()

# Single depth apadpter from raw to VAE
class Swappable_Enc_Head(nn.Module):
    def __init__(self, in_channels, out_channels, enc_kernel_sizes, stride):
        super(Swappable_Enc_Head, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = stride

        self.kernel_columns = nn.ModuleList()
        for k in enc_kernel_sizes:
            unit = nn.Sequential(
                nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=k, stride=self.stride, padding=int((k-1)/2)),
                nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=k, stride=self.stride, padding=int((k-1)/2)),
                nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=k, stride=self.stride, padding=int((k-1)/2)),
                nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=k, stride=self.stride, padding=int((k-1)/2)),
                nn.LeakyReLU(0.2)
            )
            self.kernel_columns.append(unit)

    def forward(self, x):
        
        # Send down the layers seperately, so iterate kernel in top for loop
        kernel_outs = []
        for k in range(len(self.kernel_columns)):
            unit = self.kernel_columns[k]
            x_k = unit(x)
            kernel_outs.append(x_k)
        y = torch.stack(kernel_outs, dim=3)

        return y

# Takes outputs from Dec_CNN_TimeDilator
class Swappable_Dec_Head(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(Swappable_Dec_Head, self).__init__()

        stride=1 

        self.in_channels = in_channels
        self.num_channels = out_channels

        self.dec_cnn_level = nn.ModuleList()
        for k in kernel_sizes:
            # Generate the trans conv layers so that data is upsampled without overlap
            unit = nn.Sequential(
                nn.ConvTranspose1d(self.in_channels, self.in_channels, kernel_size=k, stride=stride, padding=int((k-1)/2), output_padding=0),
                nn.ConvTranspose1d(self.in_channels, self.in_channels, kernel_size=k, stride=stride, padding=int((k-1)/2), output_padding=0),
                nn.ConvTranspose1d(self.in_channels, self.in_channels, kernel_size=k, stride=stride, padding=int((k-1)/2), output_padding=0),
                nn.ConvTranspose1d(self.in_channels, self.num_channels, kernel_size=k, stride=stride, padding=int((k-1)/2), output_padding=0),
            )
            self.dec_cnn_level.append(unit)

            self.final_tanh = nn.Tanh()

    def forward(self, x_stack):

        outs = []
        for k in range(len(self.dec_cnn_level)):
            x_k = x_stack[:, :, :, k] # Expects dim2 to be a stack of parallel outputs from K linear layers
            # x_in = x_in.reshape(x_in.shape[0], self.in_channels, -1)
            unit = self.dec_cnn_level[k]
            o = unit(x_k)
            outs.append(o)
        x = torch.sum(torch.stack(outs, dim=0), dim=0)

        x = self.final_tanh(x)
        
        return x

class BSE_Enc_Head(nn.Module):
    def __init__(
            self,
            pat_id,
            enc_kernel_sizes,
            num_channels,
            common_ENC_cnn_channels,
            **kwargs):
        
        super(BSE_Enc_Head, self).__init__()  

        self.pat_id = pat_id

        self.enc_conv_stride = 1
        self.common_ENC_cnn_channels = common_ENC_cnn_channels
        self.enc_kernel_sizes = enc_kernel_sizes
        self.num_channels = num_channels

        self.swappable_enc_cnn_head_PRE = Swappable_Enc_Head(in_channels=self.num_channels, out_channels=self.common_ENC_cnn_channels, enc_kernel_sizes=enc_kernel_sizes, stride=self.enc_conv_stride)
            
    def forward(self, x_pre):
        x_pre_posthead = self.swappable_enc_cnn_head_PRE(x_pre)

        return x_pre_posthead

class BSE_Dec_Head(nn.Module):
    def __init__(
            self,
            pat_id,
            num_channels,
            common_DEC_cnn_channels,
            transconv_kernel_sizes,
            hidden_encode_dims,
            **kwargs):
        
        super(BSE_Dec_Head, self).__init__() 

        self.pat_id = pat_id

        self.num_channels = num_channels
        self.common_DEC_cnn_channels = common_DEC_cnn_channels
        self.transconv_kernel_sizes = transconv_kernel_sizes
        self.hidden_encode_dims = hidden_encode_dims

        self.trans_conv_block = Swappable_Dec_Head(
            in_channels=self.common_DEC_cnn_channels,
            out_channels=self.num_channels, 
            kernel_sizes=self.transconv_kernel_sizes)

    def forward(self, x):
        return self.trans_conv_block(x)

