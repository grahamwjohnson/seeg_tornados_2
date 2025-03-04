import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch import Tensor
from torchinfo import summary

# Local imports
from .Transformer import ModelArgs, Transformer, RMSNorm
from utilities.loss_functions import adversarial_loss_function

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
        # in_channels = x.shape[1]

        # # Step 1: pad the channel dimension
        # padding = (0, 0, 0, self.padded_channels - in_channels, 0, 0) # (dim0 left padding, dim0 right padding... etc.)
        # x = F.pad(x, padding, mode='constant', value=0)

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

class Decoder_MLP(nn.Module):
    def __init__(self, gpu_id, latent_dim, decoder_base_dims, output_channels, decode_samples):
        super(Decoder_MLP, self).__init__()
        self.gpu_id = gpu_id
        self.latent_dim = latent_dim
        self.decoder_base_dims = decoder_base_dims
        self.output_channels = output_channels
        self.decode_samples = decode_samples

        # Non-autoregressive decoder 
        self.non_autoregressive_fc = nn.Sequential(
            nn.Linear(latent_dim, decoder_base_dims * decode_samples),
            nn.SiLU(),
            RMSNorm(decoder_base_dims * decode_samples),
            nn.Linear(decoder_base_dims * decode_samples, decoder_base_dims * decode_samples),
            nn.SiLU(),
            RMSNorm(decoder_base_dims * decode_samples),
            # nn.Linear(decoder_base_dims * decode_samples * 2, decoder_base_dims * decode_samples * 4),
            # nn.SiLU(),
            # RMSNorm(decoder_base_dims * decode_samples * 4),
            # nn.Linear(decoder_base_dims * decode_samples * 4, decoder_base_dims * decode_samples * 4),
            # nn.SiLU(),
            # RMSNorm(decoder_base_dims * decode_samples * 4),
            # nn.Linear(decoder_base_dims * decode_samples * 4, decoder_base_dims * decode_samples * 4),
            # nn.SiLU(),
            # RMSNorm(decoder_base_dims * decode_samples * 4)
            )        
        # Now FC without norms, after reshaping so that each token is seperated
        self.non_autoregressive_output = nn.Sequential(
            nn.Linear(decoder_base_dims, output_channels),
            nn.Tanh())
            
    def forward(self, z):
        batch_size = z.size(0)
        
        # Step 1: Non-autoregressive generation 
        h_na = self.non_autoregressive_fc(z).view(batch_size, self.decode_samples, self.decoder_base_dims)
        x_na = self.non_autoregressive_output(h_na)  # (batch_size, seq_length, output_channels)
        
        return x_na

# Gradient Reversal Layer
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None # None is for alpha 

# Wrapper for Gradient Reversal
class GradientReversal(nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()

    def forward(self, x, alpha):
        return GradientReversalLayer.apply(x, alpha)

class LinearWithDropout(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.5):
        super(LinearWithDropout, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.linear(x)  # Apply linear transformation
        x = self.dropout(x)  # Apply dropout internally
        return x

# Define the Adversarial Classifier with Gradient Reversal
class AdversarialClassifier(nn.Module):
    def __init__(self, latent_dim, classifier_hidden_dims, classifier_num_pats, classifier_dropout, **kwargs):
        super(AdversarialClassifier, self).__init__()
        self.gradient_reversal = GradientReversal()
        
        self.classifier_dropout = classifier_dropout
        self.mlp_layers = nn.ModuleList()

        # Input layer
        self.mlp_layers.append(nn.Linear(latent_dim, classifier_hidden_dims[0]))
        self.mlp_layers.append(nn.SiLU())
        # self.mlp_layers.append(RMSNorm(classifier_hidden_dims[0]))

        # Hidden layers
        for i in range(len(classifier_hidden_dims) - 1):
            self.mlp_layers.append(LinearWithDropout(classifier_hidden_dims[i], classifier_hidden_dims[i + 1], classifier_dropout))
            self.mlp_layers.append(nn.SiLU())
            # self.mlp_layers.append(RMSNorm(classifier_hidden_dims[i + 1]))

        # Output layer
        self.mlp_layers.append(nn.Linear(classifier_hidden_dims[-1], classifier_num_pats)) # No activation and no norm

        # Softmax the output
        self.softmax = nn.Softmax(dim=1)

    def forward(self, mu, alpha):
        mu = self.gradient_reversal(mu, alpha)
        for layer in self.mlp_layers:
            mu = layer(mu)
        return self.softmax(mu)

class WAE(nn.Module):
    '''
    The Reverseable Encoder/Decoder 
    Shares weights between Conv/TransConv layers, in addition to FC layers (except Mean/Logvar layers)
    '''
    def __init__(
        self, 
        autoencode_samples,
        padded_channels,
        crattn_embed_dim,
        transformer_seq_length,
        num_encode_concat_transformer_tokens,
        transformer_start_pos,
        transformer_dim,
        encoder_transformer_activation,
        top_dims,
        hidden_dims,
        latent_dim, 
        decoder_base_dims,
        gpu_id=None,  
        **kwargs):

        super(WAE, self).__init__()

        self.gpu_id = gpu_id
        self.autoencode_samples = autoencode_samples
        self.padded_channels = padded_channels
        self.crattn_embed_dim = crattn_embed_dim
        self.num_encode_concat_transformer_tokens = num_encode_concat_transformer_tokens
        self.transformer_seq_length = transformer_seq_length
        self.transformer_start_pos = transformer_start_pos
        self.transformer_dim = transformer_dim
        self.encoder_transformer_activation = encoder_transformer_activation
        self.top_dims = top_dims
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim 
        self.decoder_base_dims = decoder_base_dims

        # Raw CrossAttention Head
        self.encoder_head = Encoder_TimeSeriesWithCrossAttention(
            padded_channels=self.padded_channels, 
            crattn_embed_dim=self.crattn_embed_dim, 
            **kwargs)

        # Transformer - dimension is same as output of cross attention
        self.transformer_encoder = Transformer(ModelArgs(
            device=self.gpu_id, 
            dim=self.transformer_dim, 
            activation=self.encoder_transformer_activation, 
            **kwargs))

        # Core Encoder
        self.top_to_hidden = nn.Linear(self.top_dims, self.hidden_dims, bias=True)
        self.norm_hidden = RMSNorm(dim=self.hidden_dims)

        # Right before latent space
        self.final_encode_layer = nn.Linear(self.hidden_dims, self.latent_dim, bias=True)  # bias=False

        # Decoder
        self.decoder = Decoder_MLP(
            gpu_id = self.gpu_id,
            latent_dim = self.latent_dim,
            decoder_base_dims = self.decoder_base_dims,
            output_channels = self.padded_channels,
            decode_samples = self.autoencode_samples)

        # Adversarial Classifier
        self.adversarial_classifier = AdversarialClassifier(latent_dim=self.latent_dim, **kwargs) # the name 'adversarial_classifier' is tied to model parameter search to create seperate optimizer for classifier

        # Non-linearity as needed
        self.silu = nn.SiLU()

    def concat_past_tokens(self, x):
        '''
        Sliding window with stride of 1 that collects N concat tokens at a time
        '''
        num_pulls = x.shape[1] - self.num_encode_concat_transformer_tokens - self.transformer_start_pos
        y = torch.zeros([x.shape[0], num_pulls, self.top_dims]).to(x)

        for i in range(num_pulls):
            y[:, i, :] = x[:, i:i+self.num_encode_concat_transformer_tokens, :].reshape(x.shape[0], self.num_encode_concat_transformer_tokens * x.shape[2])

        return y

    def forward(self, x, reverse=False, hash_pat_embedding=-1, alpha=None):

        if reverse == False:

            # RAW CROSS-ATTENTION HEAD
            y = x.reshape([x.shape[0]*x.shape[1], x.shape[2], x.shape[3]]) # [batch, token, channel, waveform] --> [batch x token, channel, waveform]
            y = self.encoder_head(y)
            y = torch.split(y, x.shape[1], dim=0) # [batch x token, latent_dim] --> [batch, token, latent_dim]
            y = torch.stack(y, dim=0)

            # TRANSFORMER
            y = self.transformer_encoder(y, start_pos=self.transformer_start_pos)

            # WAE CORE
            y = self.concat_past_tokens(y) # Sliding window over transformer output: [batch, token, latent_dim] --> [batch, token_prime, latent_dim * num_encode_concat_transformer_tokens]
            y = y.reshape([y.shape[0]*y.shape[1], y.shape[2]]) # Batch the sliding windows for efficient decoding: [batch, token_prime, latent_dim * num_encode_concat_transformer_tokens] --> [batch x token_prime, latent_dim * num_encode_concat_transformer_tokens]
            y = self.top_to_hidden(y)
            y = self.silu(y)
            y = self.norm_hidden(y)
            latent_batched = self.final_encode_layer(y)

            # Split the batched dimension and stack into sequence dimension [batch, seq, latent_dims]
            # NOTE: you lose the priming tokens needed by transformer
            latent = torch.stack(torch.split(latent_batched, self.transformer_seq_length - self.num_encode_concat_transformer_tokens - 1, dim=0), dim=0)

            # CLASSIFIER - on the mean of the means
            mean_of_latent = torch.mean(latent, dim=1)
            class_probs_mean_of_latent = self.adversarial_classifier(mean_of_latent, alpha)
            
            return latent, class_probs_mean_of_latent

        elif reverse == True:

            # Add the hash_pat_embedding to latent vector
            y = x + hash_pat_embedding.unsqueeze(dim=1).repeat(1, x.shape[1], 1)

            # Stack the sequences into batch dimension for faster decoding
            y = y.reshape([y.shape[0]*y.shape[1], y.shape[2]]) # [batch, token, latent_dim] --> [batch x token, latent_dim]

            # Transformer Decoder
            # Goes in as [batch * seq, ]
            y = self.decoder(y).transpose(1,2)  # Comes out as [batch, waveform, num_channels] --> [batch, num_channels, waveform]

            # Index the correct output channels for each batch index
            y = torch.split(y, self.transformer_seq_length - self.num_encode_concat_transformer_tokens - 1, dim=0)
            y = torch.stack(y, dim=0)

            return y

def print_models_flow(x, **kwargs):
    '''
    Builds models on CPU and prints sizes of forward passes

    '''

    pat_num_channels = x.shape[2] 
    file_class_label = torch.tensor([0]*x.shape[0]) # dummy

    # Build the WAE
    wae = WAE(**kwargs) 

    # Run through Encoder
    print(f"INPUT TO <ENC>\n"
    f"x:{x.shape}")
    latent, class_probs = wae(x, reverse=False, alpha=1)  
    summary(wae, input_size=(x.shape), depth=999, device="cpu")
    print(
    f"latent:{latent.shape}\n"
    f"class_probs:{class_probs.shape}\n")

    # Adversarial loss
    adversarial_loss = adversarial_loss_function(class_probs, file_class_label, classifier_weight = 1)
    print(f"Adversarial Loss: {adversarial_loss}")

    # Run through WAE decoder
    hash_pat_embedding = torch.rand(x.shape[0], latent.shape[2])
    hash_channel_order = np.arange(0, 199).tolist()
    print(f"\n\n\nINPUT TO <WAE - Decoder Mode> \n"
    f"z:{latent.shape}\n"
    f"hash_pat_embedding:{hash_pat_embedding.shape}\n")
    summary(wae, input_data=[latent, True, hash_pat_embedding, hash_channel_order], depth=999, device="cpu")
    core_out = wae(latent, reverse=True, hash_pat_embedding=hash_pat_embedding)  
    print(f"decoder_out:{core_out.shape}\n")

    del wae

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

    wae = WAE(
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

    # mean,logvar,z = wae(x, reverse=False)
    # x_hat = wae(z, reverse=True)
    # loss_fn = nn.MSELoss(reduction='mean')
    # recon_loss = loss_fn(x, x_hat) 
    # recon_loss.backward()

    print(f"Are the weights of encoder and decoder tied? {torch.allclose(wae.top_to_hidden.weight.T, wae.hidden_to_top.weight)}")

