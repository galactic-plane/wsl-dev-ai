# WSL AI & Desktop Environment Setup + GPU Benchmark Suite

A concise, end‑to‑end reference for:

1. Standing up a modern **WSL2 Ubuntu 24.04** environment on Windows
2. (Optional) Installing a full **KDE Plasma desktop reachable via XRDP**
3. Enabling **GPU acceleration (CUDA + PyTorch)** inside WSL for local AI workloads
4. Installing **Docker Engine + NVIDIA Container Toolkit** for GPU containers
5. Running and validating **high‑throughput GEMM benchmarks** (`bench.py`, `bench_tests.py`)

This master README consolidates and cross‑links the two detailed guides and the Python benchmarking utilities contained in the repo.

---
## Repository Layout

| Path | Purpose |
|------|---------|
| `wsl-kde-xrdp.md` | Step‑by‑step KDE Plasma + XRDP desktop enablement (optional GUI path) |
| `wsl2-gpu-ai-docker-setup.md` | Core WSL GPU + CUDA + Docker + PyTorch environment bootstrap with benchmark usage notes |
| `python/bench.py` | Stand‑alone high‑throughput GEMM (matrix multiply) benchmark (TF32 / FP16 / BF16 where supported, optional CUDA Graphs) |
| `python/bench_tests.py` | Automated stress & validation matrix across sizes/modes/graphs; produces summaries & optional CSV |
| `README.md` | (This file) Unified overview and quick navigation |

---
## Quick Start (Minimal Path)

1. **Install / Update WSL2** (Admin PowerShell):
   ```powershell
   wsl --install   # if first time
   wsl --update
   wsl --status
   ```
2. **Install Ubuntu 24.04** (if not already):
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
3. **Enable systemd inside WSL (once)** inside Ubuntu shell:
   ```bash
   ps -p 1 -o comm=
   # If not 'systemd':
   echo -e "[boot]\nsystemd=true" | sudo tee /etc/wsl.conf
   wsl --shutdown  # run from Windows side or just exit and `wsl --shutdown`
   ```
4. **Install CUDA toolkit (driver already handled by Windows NVIDIA driver)** — follow the repo script in `wsl2-gpu-ai-docker-setup.md` Section 3.
5. **Install Docker Engine + NVIDIA Container Toolkit** — Section 5 & 6 of the same guide.
6. **Create Python venv + Install PyTorch CUDA wheels** — Section 7.
7. **Run a benchmark**:
   ```bash
   source ~/.venvs/ai/bin/activate
   python python/bench.py --size 4096 --iters 30
   ```
8. **(Optional)** Run validation matrix:
   ```bash
   python python/bench_tests.py
   ```

For richer explanations and rationale, read the detailed guide: `wsl2-gpu-ai-docker-setup.md`.

---
## When to Use the Optional KDE + XRDP Guide
If you need a *remoteable* full Linux desktop (GUI IDEs, visualization tools) accessible via Windows’ Remote Desktop Client, use `wsl-kde-xrdp.md`. If you only need terminals & VS Code (WSLg already gives basic GUI support), you can skip it.

---
## Detailed Topics (Consolidated)

### 1. WSL2 + Ubuntu 24.04
- Install / verify with `wsl --list --verbose` and `lsb_release -a`.
- Keep WSL updated (`wsl --update`).
- Enable systemd for smooth service management (Docker, etc.).

### 2. GPU Enablement Strategy
- **Windows NVIDIA Driver** is the single authoritative driver; do *not* install a Linux kernel driver inside WSL.
- Use NVIDIA’s **WSL CUDA repository** to get user‑space CUDA toolkit binaries (e.g., `nvcc`).
- Avoid globally forcing `LD_LIBRARY_PATH` to CUDA—preserves WSLg’s D3D12 stack for GUI acceleration.
- The scripts detect available capabilities: TF32/BF16 modes are only attempted on Ampere (SM 8.0) or newer.

### 3. Docker + GPU Containers
- Install Docker CE packages; enable and start the service under systemd.
- Install `nvidia-container-toolkit` & run `sudo nvidia-ctk runtime configure --runtime=docker`.
- Validate with:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
  ```

### 4. Python Environment & PyTorch
- Create an isolated venv: `python3 -m venv ~/.venvs/ai`.
- Activate: `source ~/.venvs/ai/bin/activate`.
- Install CUDA‑enabled PyTorch wheels (example uses CUDA 12.1 index):
  ```bash
  pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- Sanity check inside Python:
  ```python
  import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
  ```

### 5. GPU Monitoring (Windows Side)
Use `nvidia-smi` (PowerShell) for live telemetry, e.g.:
```powershell
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,power.draw --format=csv -l 1
```

### 6. (Optional) KDE Plasma + XRDP
- Install via `tasksel` selecting KDE.
- Install `xrdp` and connect with Windows `mstsc`, choosing session type **Xorg**.
- Useful if you want a full Linux desktop vs. WSLg’s per‑app windows.

---
## Benchmarking Suite

### `bench.py` Overview
High‑throughput GEMM benchmark focusing on TF32 / FP16 / BF16 performance (automatically skipping unavailable precisions) and optional CUDA Graphs. Key characteristics:
- Uses CUDA events for precise timing.
- Auto warmup phase (customizable via `--warmup`).
- Static allocations to accommodate CUDA Graph capture.
- Reports average ms/iter + achieved TFLOP/s per mode & size.

#### Common Arguments
| Flag | Meaning |
|------|---------|
| `--size N` | Single cubic matrix (m=n=k=N) |
| `--sizes N1 N2 ...` | Multiple explicit sizes |
| `--sweep START STOP STEP` | Generate a size range |
| `--iters K` | Timed iterations (default 30) |
| `--warmup K` | Override warmup iteration count |
| `--modes tf32,fp16,bf16` | Comma‑delimited subset |
| `--graphs` | Enable CUDA Graph capture/replay |
| `--csv file.csv` | Export results to CSV |

#### Example Invocations
```bash
# Default (4096, all modes):
python bench.py

# Large size with CUDA Graphs:
python bench.py --size 8192 --iters 50 --graphs

# Multiple sizes + CSV:
python bench.py --sizes 2048 4096 6144 8192 --graphs --csv results.csv
```

### `bench_tests.py` Overview
Automated matrix for functional + performance regression style coverage.

- Iterates a progressive ladder of sizes (tiny → large) + modes + graphs (on/off).
- Dynamically adjusts iteration counts for timing stability vs. runtime.
- Computes operational intensity heuristic & GFLOP/s per SM.
- Prints a per‑test table and summary statistics (median / mean / P90 / min / max / stdev) per (mode, graphs) combo.
- Contains an embedded negative test (`test_invalid_mode`).

#### Environment Variables
| Variable | Effect |
|----------|--------|
| `STRESS=1` | Adds very large sizes (6144, 8192) |
| `TEST_CSV=path.csv` | Writes raw per‑test rows to CSV |
| `VERBOSE=1` | Emits full tracebacks for failures |

#### Examples
```bash
# Standard run
python bench_tests.py

# Include stress sizes + export CSV
STRESS=1 TEST_CSV=matrix.csv python bench_tests.py

# Verbose errors if something fails
VERBOSE=1 python bench_tests.py
```

### Interpreting Output
- `TFLOP/s` gives aggregate throughput; compare across modes to understand precision tradeoffs.
- `AVG_MS` is latency per iteration for the given GEMM and mode.
- `GFLOP/S/SM` provides rough per‑SM scaling sanity (depends on accurate SM count inference).
- If CUDA Graphs provide a noticeable improvement, you will see consistent TFLOP/s uplift and/or lower ms.

---
## Recommended Workflow
1. Stand up baseline WSL + CUDA + PyTorch (no desktop). Validate `torch.cuda.is_available()`.
2. Run `bench.py` at a modest size (4096) to establish baseline TF32/FP16/BF16 numbers.
3. Enable `--graphs` and compare. Retain results (CSV) for future regressions.
4. Periodically run `bench_tests.py` (possibly with `STRESS=1`) after driver / PyTorch updates.
5. (Optional) Add KDE + XRDP later if a full desktop is required.

---
## Troubleshooting Cheat Sheet
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `CUDA not available` in Python | venv created before installing driver / CUDA, or running in wrong environment | Activate correct venv; verify Windows NVIDIA driver; reinstall PyTorch with CUDA wheels |
| BF16/TF32 rows missing | GPU does not support those precisions (pre‑Ampere) | Expected; upgrade GPU if needed |
| `docker: Error response from daemon: could not select device driver` | NVIDIA Container Toolkit not configured | Re-run `sudo nvidia-ctk runtime configure --runtime=docker` then restart Docker |
| `nvidia-smi` works on Windows but not in container | Missing `--gpus all` flag | Add `--gpus all` to `docker run` |
| Bench graphs warn & disable | Capture unsafe due to allocations or older driver | Accept fallback; ensure static allocations not modified |
| Unrealistic TFLOP/s for size=1 | Timing noise | Script caps tiny-size outliers; ignore tiny-size metrics |

---
## Extending the Benchmarks
- Add new dtypes (e.g., FP8) by extending mode handling in `bench.py`.
- Integrate additional kernels (convolution, attention) following the same timing & graph pattern.
- Feed CSV outputs into a dashboard (Prometheus / Grafana or lightweight HTML) for historical tracking.

---
## Design Notes
- **CUDA Graphs**: Only captured once per (size, mode) with static tensors to avoid illegal memory ops during replay.
- **Warmup Strategy**: Larger relative warmup for high iteration counts ensures kernel autotuning caches populate.
- **Memory Intensity Heuristic** in tests is intentionally approximate; refine with precise element sizes / reads if needed.

---
## Security & Safe Practices
- Do not install conflicting CUDA drivers inside WSL; rely on Windows host driver.
- Avoid running untrusted containers with `--gpus all` unless you understand the security implications.
- Keep your Python environment isolated (venv) to prevent accidental system package pollution.
- Restrict benchmark modes to what the GPU supports (script already performs capability checks).

---
## Updating / Syncing Scripts
If you copy `bench.py` / `bench_tests.py` to your home folder (as recommended in the setup guide) and later pull repo changes, just recopy them. They are self‑contained, no relative imports beyond `bench` used by `bench_tests.py`.

---
## Contributing / Future Ideas
- Add CI (GitHub Actions) to lint Python, maybe run a reduced CPU‑only logic test when CUDA is absent.
- Provide a containerized benchmark image (`Dockerfile`) with pinned PyTorch + CUDA toolkit versions.
- Add JSON output option for easier machine ingestion.
- Collect and visualize performance deltas across driver / PyTorch updates.

---
## License
Released under the MIT License. See the `LICENSE` file for full text.

SPDX-License-Identifier: MIT

---
## Source Integrity
No external network actions or secret material are stored here—scripts are self‑contained. Run them locally under your own environment.

---
Happy benchmarking & productive hacking inside WSL! 🚀
