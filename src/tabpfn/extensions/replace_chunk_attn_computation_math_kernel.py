"""
Monkey-patch attention. This is slow, it's direct sdpa implementation in pure torch. 
Upside: works on old GPUs.
"""

from __future__ import annotations

from functools import wraps
import importlib
import sys
from typing import Any, Callable

import torch

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
            outputs_q = []
            scale = (
                torch.tensor(1.0 / d_k, dtype=q.dtype, device=q.device)
                .sqrt()
                if softmax_scale is None
                else softmax_scale
            )
            for chunk_q in torch.split(q, chunk_size, dim=1):
                qc = chunk_q.shape[1]                      # current q-chunk length
                offset_max = torch.full(
                    (batch_size_curr, qc, 1, nhead),
                    -float("inf"),
                    dtype=chunk_q.dtype,
                    device=chunk_q.device,
                )
                exp_sum = torch.zeros_like(offset_max)
                acc = torch.zeros(
                    (batch_size_curr, qc, nhead, d_v),
                    dtype=chunk_q.dtype,
                    device=chunk_q.device,
                )

                for chunk_k, chunk_v in zip(torch.split(k, chunk_size, dim=1), torch.split(v, chunk_size, dim=1)):
                    logits = torch.einsum("b q h d, b k h d -> b q k h", chunk_q, chunk_k)
                    logits *= scale

                    offset = torch.maximum(offset_max, logits.max(dim=2, keepdim=True).values)
                    scale_fac = torch.exp(offset_max - offset)
                    exp_sum = exp_sum * scale_fac
                    acc = acc * scale_fac.transpose(2, 3)
                    
                    chunk_exp = torch.exp(logits - offset)
                    exp_sum += chunk_exp.sum(dim=2, keepdim=True)
                    acc += torch.einsum("b q k h, b k h d -> b q h d", chunk_exp, chunk_v)
                    offset_max = offset
                
                probs = acc / exp_sum.transpose(2, 3)
                probs = torch.dropout(probs, p=dropout_p, train=True)
                outputs_q.append(probs)
            attention_head_outputs = torch.cat(outputs_q, dim=1)
        elif batch_size_curr > batch_size: # chunk q only
            outputs = []
            for q_chunk, k_chunk, v_chunk in zip(
                torch.split(q, batch_size, dim=0),
                torch.split(k, batch_size, dim=0),
                torch.split(v, batch_size, dim=0),
            ):
                out = original_call(
                    q_chunk, k_chunk, v_chunk, None, None, dropout_p, softmax_scale
                )
                outputs.append(out)
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
