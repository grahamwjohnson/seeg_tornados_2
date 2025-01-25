# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

# import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
# from fairscale.nn.model_parallel.layers import (
#     ColumnParallelLinear,
#     RowParallelLinear,
#     VocabParallelEmbedding)

from torch import nn


@dataclass
class ModelArgs:
    def __init__(
        self, 
        dim: int = None,
        n_layers: int = 16,
        n_heads: int = 16,
        n_kv_heads: Optional[int] = None,
        vocab_size: int = -1,
        multiple_of: int = 256, # make SwiGLU hidden layer size multiple of large power of 2
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        rope_theta: float = 500000,
        # Hardware limitations
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        device: int = None,
        **kwargs):

        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        # Hardware limitations
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(device: int, dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs = freqs.to(device)
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def attention_mask_dropout(mask, inf_percentage):

    for row in range(1, mask.shape[0]):
        random_indices = torch.randperm(row+1).tolist()
        rand_inf_indices = random_indices[:int((row+1) * inf_percentage)]
        mask[row, rand_inf_indices] = float('-inf')

    return mask

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads # // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads # // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.device = args.device

        # self.wq = ColumnParallelLinear(
        #     args.dim,
        #     args.n_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)


        # self.wk = ColumnParallelLinear(
        #     args.dim,
        #     self.n_kv_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)


        # self.wv = ColumnParallelLinear(
        #     args.dim,
        #     self.n_kv_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        # self.wo = RowParallelLinear(
        #     args.n_heads * self.head_dim,
        #     args.dim,
        #     bias=False,
        #     input_is_parallel=True,
        #     init_method=lambda x: x,
        # )
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).to(self.device)

        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        return_attW: bool=False
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq).detach()  ##################### GWJ added the detach()
        self.cache_v = self.cache_v.to(xq).detach()  ##################### GWJ added the detach()

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        if return_attW:
            scores_meanHeads_lastRow = torch.mean(scores[:, :, -1, :], dim=1) # How much does i attend to j, so last row is how much does last token attend to all previous tokens... right?
            return self.wo(output), scores_meanHeads_lastRow
        else:
            return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # self.w1 = ColumnParallelLinear(
        #     dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        # )
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)


        # self.w2 = RowParallelLinear(
        #     hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        # )
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)


        # self.w3 = ColumnParallelLinear(
        #     dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        # )
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
        # return self.w2(F.tanh(self.w1(x)) * self.w3(x))           ############################ SWITCHED TO TANH ######################


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.multiple_of = args.multiple_of
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=self.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        return_attW: bool=False
    ):

        if return_attW:
            h, scores_meanHeads_lastRow = self.attention(self.attention_norm(x), start_pos, freqs_cis, mask, return_attW=True)
            h = x + h
            out = h + self.feed_forward(self.ffn_norm(h))
            return out, scores_meanHeads_lastRow

        else:
            h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask, return_attW=False)
            out = h + self.feed_forward(self.ffn_norm(h))
            return out            


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.multiple_of = params.multiple_of
        self.device = params.device

        # self.tok_embeddings = VocabParallelEmbedding(
        #     params.vocab_size, params.dim, init_method=lambda x: x
        # )

        # self.input_mlp = nn.Sequential(
        #     nn.Linear(params.vae_dim, int(params.vae_dim)),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(int(params.vae_dim), params.dim)
        # )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # self.output = ColumnParallelLinear(
        #     params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        # )

        # self.output_mlp = nn.Sequential(
        #     nn.Linear(params.dim, int(params.vae_dim)),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(int(params.vae_dim), params.vae_dim)
        # )

        self.freqs_cis = precompute_freqs_cis(
            params.device,
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

        self.tanh = nn.Tanh()

    # @torch.inference_mode()
    def forward(
        self, 
        h_in_vae: torch.Tensor, 
        start_pos: int=0, 
        return_attW: bool=False,
        attention_dropout: float=0.0
        ):
        # _bsz, seqlen = tokens.shape
        # h = self.tok_embeddings(tokens)

        # h = self.input_mlp(h_in_vae)
        h = h_in_vae

        # h is brainstate embeddings: expected shape of [batch, seq, embed_dim]
        seqlen = h.shape[1] 
        self.freqs_cis = self.freqs_cis.to(self.device).detach()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=self.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=self.device), mask]
            ).type_as(h)

            # Apply attention dropout
            if attention_dropout > 0:
                mask = attention_mask_dropout(mask, attention_dropout)
            

        if return_attW:
            layer_count = 0
            for layer in self.layers:
                if layer_count == 0:
                    h, scores_firstLayer_meanHeads_lastRow = layer(h, start_pos, freqs_cis, mask, return_attW=True)
                else:
                    h = layer(h, start_pos, freqs_cis, mask, return_attW=False)
                layer_count = layer_count + 1
            h = self.norm(h)

            # output = self.output_mlp(h)
            output = h
            # output = self.tanh(h)          ##################################### Added tanh #########################################

            return output, scores_firstLayer_meanHeads_lastRow

        else:
            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, mask, return_attW=False)
            h = self.norm(h)

            # output = self.output_mlp(h)
            output = h
            # output = self.tanh(h)          ##################################### Added tanh #########################################

            return output

            
