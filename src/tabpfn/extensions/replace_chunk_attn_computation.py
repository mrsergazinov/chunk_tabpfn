"""
Monkey-patch attention using efficient attn, e.g., flash. 
"""

from __future__ import annotations

from functools import wraps
import importlib
import sys
from typing import Any, Callable

import torch
import torch.nn.functional as F

from tabpfn.model.multi_head_attention import MultiHeadAttention

def _make_chunked_compute_attention_heads(original_call: Callable[..., torch.Tensor], chunk_size:int = 5000, batch_size:int = 20_000)-> Callable[..., torch.Tensor]:
    '''
    Replace attn computation 

    ---
    Parameters:
    original_call
    chunk_size
    batch_size
    '''
    def chunked_compute_attention_heads(
        q: torch.Tensor | None,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        kv: torch.Tensor | None,
        qkv: torch.Tensor | None,
        dropout_p: float | None = None,
        softmax_scale: float | None = None,
    ) -> torch.Tensor:
        assert (k is None) == (v is None)
        assert sum([qkv is None, kv is None, k is None and v is None]) == 2
        assert (qkv is None) != (q is None)

        if qkv is not None:
            q, k, v = qkv.unbind(dim=-3)
        elif kv is not None:
            k, v = kv.unbind(dim=-3)

        assert q is not None
        assert k is not None
        assert v is not None

        batch_size_curr, seqlen_q, nhead, d_k = q.shape
        _, seqlen_kv, nhead_kv, d_v = v.shape
        share_kv_across_n_heads = nhead // nhead_kv

        if dropout_p is None:
            dropout_p = 0.0  # TODO: necessary?

        if seqlen_q > chunk_size:
            k = MultiHeadAttention.broadcast_kv_across_heads(k, share_kv_across_n_heads)
            v = MultiHeadAttention.broadcast_kv_across_heads(v, share_kv_across_n_heads)

            # SDPA expects (B, H, L, D); current tensors are (B, L, H, D)
            K = k.permute(0, 2, 1, 3).contiguous()  # (B, H, K, D)
            V = v.permute(0, 2, 1, 3).contiguous()  # (B, H, K, D)

            outputs_q = []
            with torch.backends.cuda.sdp_kernel(enable_flash=True,
                                                enable_mem_efficient=True,
                                                enable_math=False):
                for chunk_q in torch.split(q, chunk_size, dim=1):
                    Q = chunk_q.permute(0, 2, 1, 3).contiguous()  # (B, H, qc, D)

                    # softmax_scale: pass through if given; else SDPA will default to 1/sqrt(D)
                    out = F.scaled_dot_product_attention(
                        Q, K, V,
                        attn_mask=None,
                        dropout_p=dropout_p if dropout_p is not None else 0.0,
                        is_causal=False,                 # set True if you need causal attention
                        scale=softmax_scale
                    )  # -> (B, H, qc, D)

                    outputs_q.append(out.permute(0, 2, 1, 3).contiguous())  # back to (B, qc, H, D)

            attention_head_outputs = torch.cat(outputs_q, dim=1)
        elif batch_size_curr > batch_size:
            k_b = MultiHeadAttention.broadcast_kv_across_heads(k, share_kv_across_n_heads)
            v_b = MultiHeadAttention.broadcast_kv_across_heads(v, share_kv_across_n_heads)

            outputs = []
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_mem_efficient=True, enable_math=False
            ):
                for q_chunk, k_chunk, v_chunk in zip(
                    torch.split(q,     batch_size, dim=0),
                    torch.split(k_b,   batch_size, dim=0),
                    torch.split(v_b,   batch_size, dim=0),
                ):
                    Q = q_chunk.permute(0, 2, 1, 3).contiguous()
                    K = k_chunk.permute(0, 2, 1, 3).contiguous()
                    V = v_chunk.permute(0, 2, 1, 3).contiguous()

                    out = F.scaled_dot_product_attention(
                        Q, K, V,
                        attn_mask=None,
                        dropout_p=dropout_p if dropout_p is not None else 0.0,
                        is_causal=False,              
                        scale=softmax_scale           
                    )  # (B, H, Lq, D)

                    outputs.append(out.permute(0, 2, 1, 3).contiguous())  # -> (B, Lq, H, D)
            attention_head_outputs = torch.cat(outputs, dim=0)
        else: 
            attention_head_outputs = original_call(q, k, v, None, None, dropout_p, softmax_scale)

        return attention_head_outputs.reshape(
            batch_size_curr,
            seqlen_q,
            nhead,
            d_v,
        )
    return chunked_compute_attention_heads

def enable_chunked_attention(*, chunk_size:int = 5000, batch_size:int = 20_000) -> None:
    """

    Parameters
    ----------
    chunk_size : int, default 2048
        Maximum number of rows per chunk.  Must be > 0.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    cls = MultiHeadAttention
    cls._original_compute_attention_heads = cls.compute_attention_heads  # type: ignore[attr-defined]
    cls.compute_attention_heads = _make_chunked_compute_attention_heads(  # type: ignore[method-assignment]
        cls._original_compute_attention_heads,
        chunk_size=chunk_size,
        batch_size=batch_size,
    )
    cls._is_chunk_patched = True  # type: ignore[attr-defined]

    sys.stderr.write(f"[TabPFN] ▶ Enabled chunked self‑attention (chunk_size={chunk_size}, batch_size={batch_size})\n")
