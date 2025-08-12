# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from torch import nn
import os

# import custom kernels
from .kernels import minimal
from .kernels import v1
from .kernels import minimal_v2
from .kernels import v2
from .kernels import fdm
from .kernels import fdm_splitkv
from clusterfusion import llama_decoder_layer

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

# import from llama.py
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def rotate_every_two(x):
    # 相邻两维成对旋转，匹配 view_as_complex(..., 2) 的语义
    x_even = x[..., ::2]
    x_odd  = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_every_two(q) * sin)
    k_embed = (k * cos) + (rotate_every_two(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
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
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
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

class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

        def original_attention(q, k, v, mask):
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            return torch.matmul(scores, v)

        self.use_cluster_fusion = os.getenv("USE_CLUSTER_FUSION", 'false').lower() == 'true'

        # Kernel choice
        kernel_map = {
            "USE_FLASH_DECODE_MINIMAL": (fdm.forward, "flash-decode-minimal"),
            "USE_FLASH_V2": (v2.forward, "flash-attn-v2"),
            "USE_FLASH_MINIMAL_V2": (minimal_v2.forward, "flash-attn-minimal-v2"),
            "USE_FLASH_V1": (v1.forward, "flash-attn-v1"),
            "USE_FLASH_MINIMAL": (minimal.forward, "flash-attn-minimal"),
            "USE_FDM_SPLIT_KV": (fdm_splitkv.forward, "flash-decode-minimal-split-kv")
        }
        self.attention_kernel = original_attention
        self.method_name = "Original"
        for env_var, (kernel_fn, name) in kernel_map.items():
            if os.getenv(env_var, 'false').lower() == 'true':
                self.attention_kernel = kernel_fn
                self.method_name = name
                break

        # Profiler
        self.prefill_duration_ms = 0.0
        self.prefill_call_count = 0
        self.decode_duration_ms = 0.0
        self.decode_call_count = 0
        self.total_tokens_processed = 0
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.total_duration_ms = 0.0
        self.wordy = False # set to true will make it output time for every attention call, slowing down overall generation.

        if self.use_cluster_fusion:
            # 创建一个钩子函数，在模型第一次前向传播后获取权重
            def get_weights_hook(self):
                normal_qkv = nn.Linear(args.dim, 3 * args.dim, bias=False)
                normal_qkv = normal_qkv.to(device='cuda', dtype=torch.float16)
                
                # 正确提取权重
                with torch.no_grad():
                    # 打印权重形状和信息
                    # for name, param in self.wq.named_parameters():
                        # print(f"wq.{name} shape: {param.shape}, non-zero: {(param != 0).sum().item()}")
                    
                    # 将QKV权重合并到一个矩阵中
                    if hasattr(self.wq, 'weight') and self.wq.weight is not None:
                        # 获取正确的维度
                        qw_shape = self.wq.weight.shape
                        # print(f"Final weight shape: {qw_shape}")
                        
                        # 合并权重
                        normal_qkv.weight.data[:args.dim] = self.wq.weight.data.T
                        normal_qkv.weight.data[args.dim:2*args.dim] = self.wk.weight.data.T
                        normal_qkv.weight.data[2*args.dim:] = self.wv.weight.data.T
                        
                        # 检查结果
                        # print(f"Combined QKV weight non-zeros: {(normal_qkv.weight != 0).sum().item()}")
                    wo_weight = None
                    if hasattr(self.wo, 'weight') and self.wo.weight is not None:
                        # 需要转置并确保内存连续，与 test_llama.py 中的处理方式一致
                        wo_weight = self.wo.weight.data.T.contiguous()
                        # print(f"wo_weight shape: {wo_weight.shape}, non-zero: {(wo_weight != 0).sum().item()}")
                        
                return normal_qkv.weight.data, wo_weight
            
            # 存储钩子函数以便在前向传播后调用
            self._get_weights_hook = get_weights_hook
            self.weights_initialized = False

    def forward(
        self,
        unnormed_x: torch.Tensor,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        rms_input_weight: torch.Tensor,  # 新增参数
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        if self.use_cluster_fusion and not self.weights_initialized:
            self.weight_qkv, self.weight_o = self._get_weights_hook(self)
            self.weights_initialized = True
        if self.use_cluster_fusion and mask is None:
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
            
            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

            input_tensor = unnormed_x.view(1, -1)
            kv_cache_k = self.cache_k[:bsz, :start_pos].contiguous().view(-1, self.n_local_kv_heads * self.head_dim)
            kv_cache_v = self.cache_v[:bsz, :start_pos].contiguous().view(-1, self.n_local_kv_heads * self.head_dim)
            rms_input_weight = rms_input_weight.reshape(1, 4096)
            rms_attn_weight = torch.zeros(self.head_dim, device=x.device)
            gate_up_proj_weight_fuse = torch.zeros(self.head_dim, device=x.device)
            down_proj_weight_fuse = torch.zeros(self.head_dim, device=x.device)
            cos_full = torch.repeat_interleave(freqs_cis.real, 2, dim=-1)  # (seqlen, head_dim)
            sin_full = torch.repeat_interleave(freqs_cis.imag, 2, dim=-1)  # (seqlen, head_dim)
        
            # dump 参数
            # dump_params = {
                # 'input_tensor': input_tensor.clone().detach().cpu(),
                # 'weight_qkv': self.weight_qkv.clone().detach().cpu(),
                # 'weight_o': self.weight_o.clone().detach().cpu(),
                # 'kv_cache_k': kv_cache_k.clone().detach().cpu(),
                # 'kv_cache_v': kv_cache_v.clone().detach().cpu(),
                # 'gate_up_proj_weight_fuse': gate_up_proj_weight_fuse.clone().detach().cpu(),
                # 'down_proj_weight_fuse': down_proj_weight_fuse.clone().detach().cpu(),
                # 'rms_input_weight': rms_input_weight.clone().detach().cpu(),
                # 'rms_attn_weight': rms_attn_weight.clone().detach().cpu(),
                # 'cos_full': cos_full.clone().detach().cpu(),
                # 'sin_full': sin_full.clone().detach().cpu()
            # }
            # # 保存到文件
            # torch.save(dump_params, '/home/ylhuang/sandbox/llama_decoder_params.pt')
            # print("Parameters dumped to /home/ylhuang/sandbox/llama_decoder_params.pt")
            # exit()
            
            # print(f"input_tensor.shape: {input_tensor.shape}", input_tensor.is_contiguous(), input_tensor.dtype)
            # print(f"weight_qkv.shape: {self.weight_qkv.shape}", self.weight_qkv.is_contiguous(), self.weight_qkv.dtype)
            # print(f"weight_o.shape: {self.weight_o.shape}", self.weight_o.is_contiguous(), self.weight_o.dtype)
            # print(f"kv_cache_k.shape: {kv_cache_k.shape}", kv_cache_k.is_contiguous(), kv_cache_k.dtype)
            # print(f"kv_cache_v.shape: {kv_cache_v.shape}", kv_cache_v.is_contiguous(), kv_cache_v.dtype)
            # print(f"rms_input_weight.shape: {rms_input_weight.shape}", rms_input_weight.is_contiguous(), rms_input_weight.dtype)
            # print(f"cos_full.shape: {cos_full.shape}", cos_full.is_contiguous(), cos_full.dtype)
            # print(f"sin_cull.shape: {sin_full.shape}", sin_full.is_contiguous(), sin_full.dtype)
            # exit()

            # Compare RoPE
            # my_xq, my_xk = apply_rotary_pos_emb(xq, xk, cos_full, sin_full)
            # print("--- Result from apply_rotary_pos_emb (Corrected) ---")
            # print(my_xq.shape, my_xq)
            # --- 标准的 apply_rotary_emb 作为对比 ---
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
            # print("\n--- Result from apply_rotary_emb (Reference) ---")
            # print(xq.shape, xq)
            # print(f"\n--- Comparison ---")
            # print(f"Are results close? {torch.allclose(my_xq, xq, atol=1e-6)}")
            # print(f"Are results close? {torch.allclose(my_xk, xk, atol=1e-6)}")
            # exit()
# 
            # Update KV cache
            # xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)
            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            output = llama_decoder_layer(
                input_tensor,          
                self.weight_qkv,                          
                self.weight_o,              
                kv_cache_k,
                kv_cache_v,           
                gate_up_proj_weight_fuse,      
                down_proj_weight_fuse,      
                rms_input_weight,      
                rms_attn_weight,       
                cos_full,                   
                sin_full               
            )
            output = output.view(bsz, seqlen, 4096)
            # print(output.shape, output)
            # print(self.weight_o.data[..., 0: 128])
            # exit()
            return output
        else:
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
            # print(seqlen)
            # print(xq[..., 0: 128])
            # print(xk[..., 0: 128])
            # print(xv[..., 0: 128])

            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            # xq, xk, xv are contiguous in memory

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]

            # keys, values are not contiguous in memory. xq remains contiguous

            # repeat k/v heads if n_kv_heads < n_heads
            keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
            values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

            xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
            keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
            values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)

            self.start_event.record()
            if self.method_name != "Original" and mask is None:
                mask = torch.empty(0, dtype=torch.float16, device=xq.device)
            output = self.attention_kernel(xq, keys, values, mask)
            self.end_event.record()
            torch.cuda.synchronize()
            
            duration_ms = self.start_event.elapsed_time(self.end_event)
            self.total_duration_ms += duration_ms

            if self.wordy:
                print(f"[{self.method_name}] attention - seqlen: {xq.size(2)}, cache_len: {keys.size(2)-xq.size(2)}, time: {duration_ms:.3f}ms")

            self.total_tokens_processed += xq.size(0) * xq.size(2) # bsz * seqlen
            if xq.size(2) > 1: # prefill
                self.prefill_duration_ms += duration_ms
                self.prefill_call_count += 1
            else: # decode
                self.decode_duration_ms += duration_ms
                self.decode_call_count += 1
            # print(output.shape)

            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            output = self.wo(output)
            # print(output.shape, output[0, 0, 0:128])
            # print(self.wo.weight.data[..., 0: 128])
            return output

    # profile summary
    def print_summary(self, layer_id):
        print(f"--- layer {layer_id} attention summary ({self.method_name}) ---")
        if self.prefill_call_count > 0:
            avg_prefill_time = self.prefill_duration_ms / self.prefill_call_count
            print(f"  prefill : {self.prefill_duration_ms:.2f} ms total, {self.prefill_call_count} calls, {avg_prefill_time:.2f} ms/call")
        
        if self.decode_call_count > 0:
            avg_decode_time = self.decode_duration_ms / self.decode_call_count
            avg_decode_token_time = self.decode_duration_ms / self.total_tokens_processed if self.total_tokens_processed > 0 else 0
            print(f"  decode  : {self.decode_duration_ms:.2f} ms total, {self.decode_call_count} calls, {avg_decode_time:.2f} ms/call")
            print(f"  tokens processed: {self.total_tokens_processed}, avg time/token: {avg_decode_token_time:.3f} ms")
        print("-" * (30 + len(self.method_name)))



class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
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
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(
            x, self.attention_norm(x), start_pos, freqs_cis, mask, self.attention_norm.weight
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.use_cluster_fusion = os.getenv("USE_CLUSTER_FUSION", 'false').lower() == 'true'
        if self.use_cluster_fusion == True:
            print("Using clusterfusion kernel")

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i c, float* maskorresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
