#!/usr/bin/env python3
# bench.py — High-throughput GEMM benchmark for NVIDIA GPUs (WSL-friendly)
# Modes: TF32, FP16, BF16. Uses CUDA events for timing and optionally CUDA Graphs.

import argparse
import csv
import os
import platform
import sys
from typing import Iterable, List, Sequence, Tuple
import torch

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

def fmt_bytes(b: int | float) -> str:
    """Format a byte count into a human-readable string.

    Accepts int or float. Internally uses a float accumulator so we don't
    reassign a different type back to an int parameter (silences Pylance).
    """
    value: float = float(b)
    for u in ("B", "KB", "MB", "GB", "TB", "PB"):
        if value < 1024:
            return f"{value:.1f}{u}"
        value /= 1024
    return f"{value:.1f}EB"

def device_info():
    assert torch.cuda.is_available(), "CUDA not available."
    dev = 0
    name = torch.cuda.get_device_name(dev)
    cap = ".".join(map(str, torch.cuda.get_device_capability(dev)))
    # Access torch.version.cuda defensively (Pylance stubs sometimes omit 'version').
    try:
        drv = getattr(torch.version, "cuda", None) or "unknown"  # type: ignore[attr-defined]
    except Exception:
        drv = "unknown"
    try:
        free, total = torch.cuda.mem_get_info()
        mem = f"{fmt_bytes(total - free)} used / {fmt_bytes(total)} total"
    except Exception:
        mem = "n/a"
    return name, cap, drv, mem

def run_gemm(m, n, k, iters, mode, use_graphs=False, warmup=None):
    """Run z = x @ y repeatedly and return (avg_ms_per_iter, tflops).
       Uses CUDA Graphs when stable; falls back automatically on capture errors."""
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    # Mode → dtype + TF32 toggle
    if mode == "tf32":
        dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    elif mode == "fp16":
        dtype = torch.float16
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
    elif mode == "bf16":
        dtype = torch.bfloat16
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
    else:
        raise ValueError(mode)

    # Static allocations (no new allocs during capture)
    x = torch.randn(m, k, device=device, dtype=dtype)
    y = torch.randn(k, n, device=device, dtype=dtype)
    out = torch.empty(m, n, device=device, dtype=dtype)

    # Warmup outside capture (build kernels & caches)
    if warmup is None:
        warmup = max(10, iters // 5)
    torch.cuda.synchronize()
    for _ in range(warmup):
        out.copy_(x @ y)
    torch.cuda.synchronize()

    # Optional CUDA Graph (pool-aware, safe fallback)
    graph = None
    if use_graphs:
        try:
            pool = torch.cuda.graphs.graph_pool_handle()
            g = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            stream = torch.cuda.current_stream()
            # Provide explicit stream argument to satisfy type checker.
            with torch.cuda.graph(g, stream=stream, pool=pool):  # type: ignore[arg-type]
                out.copy_(x @ y)
            torch.cuda.synchronize()
            graph = g
        except Exception as e:  # noqa: BLE001 (broad for fallback safety)
            print(f"[warn] disabling CUDA Graphs for {mode}: {e}")

    # Timed run (CUDA events)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream()
    start.record(stream=stream)  # type: ignore[arg-type]
    if graph:
        for _ in range(iters):
            # Some older stubs require a stream argument; runtime accepts none.
            graph.replay()  # type: ignore[call-arg]
    else:
        for _ in range(iters):
            out.copy_(x @ y)
    end.record(stream=stream)  # type: ignore[arg-type]
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / iters

    # GEMM FLOPs per iter = 2 * m * n * k
    flops_total = 2.0 * m * n * k * iters
    tflops = (flops_total / (total_ms / 1e3)) / 1e12

    # Cleanup
    del x, y, out
    torch.cuda.empty_cache()
    return avg_ms, tflops

def parse_sizes(args):
    if args.size:
        return [(args.size,)*3]
    if args.sizes:
        return [(s,)*3 for s in args.sizes]
    if args.sweep:
        a, b, c = args.sweep
        return [(s,)*3 for s in range(a, b + 1, c)]
    return [(4096,)*3]  # default

def main():
    p = argparse.ArgumentParser("Max-throughput GEMM benchmark (TF32/FP16/BF16)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--size", type=int, help="Run one cubic size m=n=k")
    g.add_argument("--sizes", type=int, nargs="+", help="List of cubic sizes (e.g., 2048 4096 8192)")
    g.add_argument("--sweep", type=int, nargs=3, metavar=("START", "STOP", "STEP"),
                   help="Sweep cubic sizes from START..STOP by STEP")
    p.add_argument("--iters", type=int, default=30, help="Iterations per measurement")
    p.add_argument("--warmup", type=int, default=None, help="Warmup iterations (default auto)")
    p.add_argument("--modes", type=str, default="tf32,fp16,bf16", help="Comma list: tf32,fp16,bf16")
    p.add_argument("--graphs", action="store_true", help="Use CUDA Graphs")
    p.add_argument("--csv", type=str, help="Write results to CSV")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.", file=sys.stderr)
        sys.exit(1)

    sizes = parse_sizes(args)
    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    # Capability-based pruning: TF32/BF16 only if supported. Avoids user confusion on older GPUs.
    cap_major, cap_minor = torch.cuda.get_device_capability(0)
    is_ampere_plus = cap_major >= 8  # Ampere (SM 8.x) or newer
    if not is_ampere_plus:
        filtered = []
        for m in modes:
            if m in ("bf16", "tf32"):
                print(f"[info] Skipping mode '{m}' (requires NVIDIA Ampere+ compute capability 8.0+)")
            else:
                filtered.append(m)
        modes = filtered or ["fp16"]  # ensure at least one mode remains

    name, cap, drv, mem = device_info()
    print("="*72)
    print(f"PyTorch {torch.__version__} | Python {platform.python_version()} | CUDA {drv}")
    print(f"GPU: {name} (cc {cap}) | Mem: {mem}")
    print(f"Sizes: {', '.join(str(m) for (m,_,_) in sizes)} | iters={args.iters} warmup={args.warmup or 'auto'} | graphs={args.graphs}")
    print("-"*72)

    rows = []
    for (m, n, k) in sizes:
        print(f"[m=n=k={m}]")
        for mode in modes:
            avg_ms, tflops = run_gemm(m, n, k, args.iters, mode, use_graphs=args.graphs, warmup=args.warmup)
            print(f"  {mode.upper():<5}  {avg_ms:8.3f} ms/iter   {tflops:8.2f} TFLOP/s")
            rows.append({
                "mode": mode, "m": m, "n": n, "k": k,
                "iters": args.iters, "avg_ms": round(avg_ms, 6), "tflops": round(tflops, 6),
                "graphs": args.graphs, "torch": torch.__version__, "python": platform.python_version(),
                "cuda": drv, "gpu": name, "cc": cap
            })
        print("-"*72)
    print("="*72)

    if args.csv:
        fields = ["mode","m","n","k","iters","avg_ms","tflops","graphs","torch","python","cuda","gpu","cc"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote CSV: {args.csv}")

if __name__ == "__main__":
    main()