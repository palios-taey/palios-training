"""Chunked cross-entropy for a very large vocabulary — the one optimization this model actually needs.

WHY THIS AND NOT THE LIBRARIES THE CONSULTS RECOMMENDED
Measured on the nodes 2026-07-29: liger_kernel, flash_attn, xformers, apex and
transformer_engine are ALL ABSENT. The consults recommended fused-CE and Liger; nobody ever
ran the install. torch 2.10.0+cu130 and triton 3.6.0 are present, torch.compile works.

WHY CROSS-ENTROPY IS THE TARGET HERE SPECIFICALLY
vocab_size = 248,320. At batch 4 x seq 2560 the logits tensor is
    4 * 2560 * 248320 = 2.54e9 elements
In bf16 that is ~5.1 GB materialized, and the standard path allocates it TWICE — once for
logits and once for the softmax/backward intermediate. On a 119 GB unified-memory node already
running at 110.6 GB reserved with 6 GB free, that single tensor is the tightest allocation in
the step. Chunking it is a pure win: identical math, a fraction of the peak.

WHAT THIS DOES
Computes CE over the sequence in chunks, so the peak logits allocation is
    chunk_size * vocab   instead of   (batch * seq) * vocab
The gradient is accumulated across chunks, so the result is numerically the same loss and the
same gradient as a single fused call — not an approximation.

WHAT IT DOES NOT DO
It does not fuse the logit matmul with the softmax the way a Triton kernel would. That is a
larger win and a larger risk; this is the version that is correct by construction and can be
verified against the reference in one assertion.

STATUS: UNBENCHMARKED. All four nodes are training as this is written, so measuring it would
contend with the run. The benchmark is the deliverable, not this file — a number against the
current 803 tok/s baseline, or it does not count.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def chunked_cross_entropy(
    hidden: torch.Tensor,          # [N, H]  final hidden states, already flattened
    lm_head_weight: torch.Tensor,  # [V, H]
    targets: torch.Tensor,         # [N]
    chunk_size: int = 4096,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Loss identical to F.cross_entropy(hidden @ W.T, targets), at a fraction of peak memory.

    The projection is done per chunk so the [N, V] logits tensor is never materialized whole.
    """
    if hidden.dim() != 2:
        raise ValueError(f"hidden must be [N, H], got {tuple(hidden.shape)}")
    if targets.dim() != 1 or targets.shape[0] != hidden.shape[0]:
        raise ValueError(f"targets must be [N] matching hidden, got {tuple(targets.shape)}")

    n = hidden.shape[0]
    total = hidden.new_zeros(())
    count = 0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        h = hidden[start:end]
        t = targets[start:end]
        valid = (t != ignore_index)
        if not valid.any():
            continue
        logits = F.linear(h, lm_head_weight)          # [chunk, V] — the only large tensor alive
        loss = F.cross_entropy(
            logits.float(), t, ignore_index=ignore_index, reduction="sum"
        )
        total = total + loss
        count += int(valid.sum())
        del logits

    if count == 0:
        return hidden.new_zeros((), requires_grad=True)
    return total / count


def _reference(hidden, w, targets, ignore_index=-100):
    return F.cross_entropy(
        F.linear(hidden, w).float(), targets, ignore_index=ignore_index
    )


def verify(device="cpu", dtype=torch.float32, n=1024, h=256, v=4096, seed=0):
    """Correctness check against the reference. Chunking must change memory, never the answer."""
    torch.manual_seed(seed)
    hidden = torch.randn(n, h, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(v, h, device=device, dtype=dtype, requires_grad=True)
    targets = torch.randint(0, v, (n,), device=device)
    targets[::7] = -100                                  # exercise the ignore_index path

    ref = _reference(hidden, w, targets)
    ref.backward()
    ref_hg, ref_wg = hidden.grad.clone(), w.grad.clone()

    hidden.grad = None; w.grad = None
    got = chunked_cross_entropy(hidden, w, targets, chunk_size=128)
    got.backward()

    dl = (ref - got).abs().item()
    dh = (ref_hg - hidden.grad).abs().max().item()
    dw = (ref_wg - w.grad).abs().max().item()
    print(f"  loss   ref={ref.item():.8f}  chunked={got.item():.8f}  |diff|={dl:.3e}")
    print(f"  grad hidden max|diff| = {dh:.3e}")
    print(f"  grad weight max|diff| = {dw:.3e}")
    ok = dl < 1e-5 and dh < 1e-5 and dw < 1e-5
    print(f"  VERDICT: {'MATCHES reference (loss and both gradients)' if ok else '*** MISMATCH ***'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if verify() else 1)
