"""Benchmark the three cross-entropy paths at THIS model's real dimensions.

WHY A BENCHMARK AND NOT A DECISION
Every optimization recommendation in this operation's history is 0-occurrence in the trainer.
The consults produced advice; nobody produced a number. This file exists so the next change to
the loss path lands on evidence: a tok/s and a peak-memory figure against the measured 803 tok/s
baseline, or it does not land.

WHAT IS COMPARED, at vocab=248,320 / hidden=5,120 / tokens=10,240 (batch 4 x seq 2560):

  baseline    F.cross_entropy(F.linear(h, W), t)
              materialises [10240, 248320] = 2.54e9 elements = 5.09 GB bf16, twice over
              (forward logits + softmax backward). This is what the trainer does today.

  chunked     careers-qwen/chunked_ce.py — same math, peak logits = chunk*V instead of N*V.
              Correctness already verified against the reference (loss identical to 8dp,
              hidden grad exact, weight grad 9.3e-10). Needs no external library.

  liger       LigerFusedLinearCrossEntropyLoss — fuses the lm_head projection INTO the CE, so
              the full logits tensor is never materialised at all. Strictly better in principle
              than chunking; the question is whether it is faster in practice on sm_121.

MEASURED FIRST, THEN JUDGED. A candidate that is faster but wrong is not a candidate, so every
path is checked against the reference loss before its timing is reported.

RUN ON AN IDLE NODE. All four are training when this is written; running it under load would
measure contention, not the kernels.
"""
from __future__ import annotations

import argparse, sys, time

import torch
import torch.nn.functional as F


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_gb():
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def time_path(fn, hidden, w, targets, iters, warmup):
    """Return (mean_seconds_per_iter, peak_gb, loss_value). Includes backward — that is the real cost."""
    for _ in range(warmup):
        loss = fn(hidden, w, targets)
        loss.backward()
        hidden.grad = None; w.grad = None
    _sync(); _reset_peak()
    t0 = time.perf_counter()
    for _ in range(iters):
        loss = fn(hidden, w, targets)
        loss.backward()
        hidden.grad = None; w.grad = None
    _sync()
    dt = (time.perf_counter() - t0) / iters
    return dt, _peak_gb(), float(loss.detach())


def baseline(h, w, t):
    return F.cross_entropy(F.linear(h, w).float(), t, ignore_index=-100)


def make_chunked(chunk):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chunked_ce import chunked_cross_entropy
    def f(h, w, t):
        return chunked_cross_entropy(h, w, t, chunk_size=chunk)
    return f


def make_liger():
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    crit = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
    def f(h, w, t):
        return crit(w, h, t)          # (weight, input, target)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=248320)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--tokens", type=int, default=10240, help="batch*seq; run default is 4*2560")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = getattr(torch, args.dtype)
    print(f"  device={dev} dtype={args.dtype} vocab={args.vocab} hidden={args.hidden} tokens={args.tokens}")
    print(f"  full logits would be {args.tokens*args.vocab*2/1e9:.2f} GB in bf16\n")

    torch.manual_seed(0)
    hidden = torch.randn(args.tokens, args.hidden, device=dev, dtype=dt, requires_grad=True)
    w = torch.randn(args.vocab, args.hidden, device=dev, dtype=dt, requires_grad=True)
    targets = torch.randint(0, args.vocab, (args.tokens,), device=dev)

    cands = [("baseline", baseline), (f"chunked({args.chunk})", make_chunked(args.chunk))]
    try:
        cands.append(("liger_fused", make_liger()))
    except Exception as e:
        print(f"  liger unavailable: {type(e).__name__}: {e}\n")

    ref_loss = None
    results = []
    for name, fn in cands:
        try:
            sec, peak, loss = time_path(fn, hidden, w, targets, args.iters, args.warmup)
        except torch.cuda.OutOfMemoryError:
            print(f"  {name:<16} OOM — cannot run at this shape"); continue
        except Exception as e:
            print(f"  {name:<16} FAILED {type(e).__name__}: {e}"); continue
        if ref_loss is None:
            ref_loss = loss
            ok = "reference"
        else:
            ok = "MATCHES" if abs(loss - ref_loss) < 2e-2 else f"*** LOSS DIFFERS ({loss:.4f} vs {ref_loss:.4f}) ***"
        results.append((name, sec, peak, ok))
        print(f"  {name:<16} {sec*1000:>8.1f} ms/iter   peak {peak:>6.2f} GB   loss {loss:.4f}  {ok}")

    if len(results) > 1:
        base = results[0][1]
        print("\n  speedup vs baseline (correctness-gated — a wrong result is not a win):")
        for name, sec, peak, ok in results[1:]:
            verdict = f"{base/sec:.2f}x" if ok.startswith(("MATCHES", "reference")) else "DISQUALIFIED"
            print(f"    {name:<16} {verdict}   peak {results[0][2]:.2f} -> {peak:.2f} GB")
    print("\n  NOTE: this measures the loss path only. The end-to-end number that counts is")
    print("  tok/s on a real 4-node run against the 803 tok/s baseline.")


if __name__ == "__main__":
    main()
