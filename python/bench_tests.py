#!/usr/bin/env python3
"""Automated stress / validation matrix for bench.py

This script executes an expanded matrix of GEMM benchmarks spanning very small
to large square sizes, multiple data modes (TF32/FP16/BF16), and CUDA Graphs
enabled/disabled. The goal is to ensure the benchmark path is resilient across
allocator regimes, warmup behavior, graph capture, and timing logic. Output is
structured and information-rich for quick visual inspection.

Approx default test count: ~108 (18 sizes * 3 modes * 2 graph settings)

Environment Variables:
    STRESS=1       Add even larger sizes (adds 6144, 8192) increasing coverage.
    TEST_CSV=path  Write per-test raw results to a CSV file.
    VERBOSE=1      Print full tracebacks for any failures.
"""
from __future__ import annotations
import os
import math
import statistics as stats
import traceback
import time
import csv
import torch
import bench  # type: ignore

# Make sure CUDA is present before proceeding.
assert torch.cuda.is_available(), "CUDA not available for tests."

def run_matrix():
    torch.manual_seed(0)
    # Progressive size ladder (roughly geometric + some in-betweens) to exercise allocator patterns.
    base_sizes = [
        1, 8, 16, 32, 64, 96, 128, 160, 192, 256,
        384, 512, 768, 1024, 1536, 2048, 3072, 4096,
    ]
    if os.getenv("STRESS"):
        base_sizes += [6144, 8192]

    modes = ["tf32", "fp16", "bf16"]
    graphs_opts = [False, True]

    total_tests = len(base_sizes) * len(modes) * len(graphs_opts)
    results: list[dict] = []
    failures: list = []

    # Device / HW info for additional computed metrics.
    props = torch.cuda.get_device_properties(0)
    sm_count = getattr(props, 'multi_processor_count', None)
    clock_ghz = getattr(props, 'clock_rate', 0) / 1e6 if getattr(props, 'clock_rate', 0) else None

    start_time = time.time()
    print("=" * 96)
    print("INDEX SIZE    MODE  GRAPHS ITERS WARMUP  AVG_MS    TFLOP/S   GFLOP/S/SM  BYTES   FLOPS/Byte")
    print("-" * 96)

    test_index = 0
    for size in base_sizes:
        for mode in modes:
            for graphs in graphs_opts:
                test_index += 1
                # Dynamic iteration count: more reps for tiny sizes for timing stability.
                if size <= 64:
                    iters = 6
                elif size <= 256:
                    iters = 5
                elif size <= 1024:
                    iters = 4
                else:
                    iters = 2
                warmup = 1
                try:
                    ms, tflops = bench.run_gemm(size, size, size, iters, mode, use_graphs=graphs, warmup=warmup)
                    assert ms > 0 and math.isfinite(ms), "avg ms invalid"
                    assert tflops > 0 and math.isfinite(tflops), "tflops invalid"
                    if size == 1:
                        # Guard unrealistic TFLOPs from extremely tiny timing noise.
                        tflops = min(tflops, 0.5)
                    # Memory traffic heuristic (bytes read + written). Very rough: A,B read once, C written once per iter.
                    bytes_per_iter = (size*size*2 + size*size) * torch.tensor([], dtype=torch.float32).element_size()  # placeholder dtype size=4
                    # FP16/BF16 differ in element size; override.
                    if mode in ("fp16", "bf16"):
                        # element size 2 bytes
                        bytes_per_iter = (size*size*2 + size*size) * 2
                    # Operational intensity.
                    flops_per_iter = 2 * size * size * size
                    intensity = flops_per_iter / bytes_per_iter if bytes_per_iter else float('nan')
                    gflops_per_sm = (tflops * 1e3 / sm_count) if sm_count else float('nan')
                    progress = f"{test_index:3d}/{total_tests:<3d} ({(test_index/total_tests)*100:5.1f}%)"
                    print(f"{progress} {size:5d}  {mode.upper():5}    {int(graphs)}     {iters:3d}    {warmup:2d}  {ms:7.3f}   {tflops:8.2f}    {gflops_per_sm:9.2f}  {bytes_per_iter/1e6:6.1f}MB  {intensity:10.2f}")
                    results.append({
                        "index": test_index,
                        "size": size,
                        "mode": mode,
                        "graphs": graphs,
                        "iters": iters,
                        "warmup": warmup,
                        "avg_ms": ms,
                        "tflops": tflops,
                        "gflops_per_sm": gflops_per_sm,
                        "bytes_per_iter": bytes_per_iter,
                        "intensity": intensity,
                    })
                except Exception as e:  # noqa: BLE001
                    failures.append((test_index, size, mode, graphs, e, traceback.format_exc()))
                    print(f"ERR {test_index:3d}/{total_tests:<3d} size={size} mode={mode} graphs={graphs} error={e}")

    elapsed = time.time() - start_time
    print("-" * 96)
    print(f"Completed {test_index} tests in {elapsed:.2f}s; failures={len(failures)}")
    return results, failures


def test_invalid_mode():
    try:
        bench.run_gemm(16, 16, 16, 1, "invalid_mode", use_graphs=False, warmup=0)
    except ValueError:
        return True
    raise AssertionError("Expected ValueError for invalid mode")


def percentile(values, p):
    if not values:
        return float('nan')
    k = (len(values)-1) * (p/100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] + (values[c]-values[f]) * (k - f)


def summarize(results, failures):
    # Aggregate by (mode, graphs)
    by_key: dict[tuple[str,bool], list[float]] = {}
    for r in results:
        by_key.setdefault((r['mode'], r['graphs']), []).append(r['tflops'])
    lines: list[str] = []
    lines.append("Per-Mode Summary (TFLOP/s):")
    lines.append("  MODE  GRAPHS   COUNT   MEDIAN     MEAN     P90     MAX     MIN    STDEV")
    for (mode, graphs), vals in sorted(by_key.items()):
        vals_sorted = sorted(vals)
        median = stats.median(vals_sorted)
        mean = stats.fmean(vals_sorted)
        p90 = percentile(vals_sorted, 90)
        stdev = stats.pstdev(vals_sorted) if len(vals_sorted) > 1 else 0.0
        lines.append(f"  {mode.upper():5}   {int(graphs):5d}   {len(vals_sorted):5d}  {median:8.2f} {mean:8.2f} {p90:8.2f} {max(vals_sorted):8.2f} {min(vals_sorted):8.2f} {stdev:8.2f}")

    # Best per size (pick fastest across modes/graphs).
    best_by_size: dict[int, dict] = {}
    for r in results:
        size = r['size']
        cur = best_by_size.get(size)
        if not cur or r['tflops'] > cur['tflops']:
            best_by_size[size] = r
    lines.append("\nBest Achieved Per Size (fastest config):")
    lines.append("  SIZE   MODE  GRAPHS  TFLOP/S   AVG_MS  INTENSITY  GFLOP/S/SM")
    for size in sorted(best_by_size):
        r = best_by_size[size]
        lines.append(f"  {size:5d}  {r['mode'].upper():5}    {int(r['graphs'])}    {r['tflops']:7.2f}  {r['avg_ms']:7.3f}  {r['intensity']:9.2f}   {r['gflops_per_sm']:9.2f}")

    if failures:
        lines.append("\nFailures (truncated):")
        for idx, size, mode, graphs, exc, _tb in failures[:10]:
            lines.append(f"  idx={idx} size={size} mode={mode} graphs={graphs} error={exc}")
        if len(failures) > 10:
            lines.append(f"  ... {len(failures)-10} more failures omitted ...")
    return "\n".join(lines)


if __name__ == "__main__":
    # Device / runtime header.
    import platform
    name = torch.cuda.get_device_name(0)
    cap = '.'.join(map(str, torch.cuda.get_device_capability(0)))
    try:
        drv = getattr(torch.version, 'cuda', 'unknown')  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        drv = 'unknown'
    print("GPU Device:", name, f"(cc {cap}) | CUDA {drv} | PyTorch {torch.__version__} | Python {platform.python_version()}")
    results, failures = run_matrix()
    test_invalid_mode()
    print(summarize(results, failures))
    # Optional CSV export
    csv_path = os.getenv("TEST_CSV")
    if csv_path:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote CSV: {csv_path}")
    if failures:
        if os.getenv("VERBOSE"):
            for (_, _, _, _, _, tb) in failures:
                print(tb)
        raise SystemExit(1)
