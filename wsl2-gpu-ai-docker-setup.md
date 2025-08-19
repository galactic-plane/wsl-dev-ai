# WSL2 GPU-Accelerated AI & Docker Setup (Single Script, WSL-Only)

## Table of Contents
1. [Windows Host Prerequisites](#1-windows-host-prerequisites)
2. [Enable systemd in WSL](#2-enable-systemd-in-wsl)
3. [CUDA Toolkit in WSL (Driverless NVIDIA Repo)](#3-cuda-toolkit-in-wsl-driverless-nvidia-repo)
4. [Make WSLg’s D3D12 GL Stack the Default (GPU-Accelerated Linux GUI)](#4-make-wslgs-d3d12-gl-stack-the-default-gpu-accelerated-linux-gui)
5. [Docker Engine (Inside WSL)](#5-docker-engine-inside-wsl)
6. [NVIDIA Container Toolkit (GPU Runtime for Docker)](#6-nvidia-container-toolkit-gpu-runtime-for-docker)
7. [Minimum & Recommended Hardware / Software](#minimum--recommended-hardware--software)
8. [Python venv + PyTorch CUDA Wheels (No Containers)](#7-python-venv-pytorch-cuda-wheels-no-containers)
9. [VS Code in WSL (Benchmark Scripts)](#8-vs-code-in-wsl-benchmark-scripts)
  - [Install VS Code on Windows + WSL Extension](#81-install-vs-code-on-windows--wsl-extension)
  - [Open your WSL home in VS Code](#82-open-your-wsl-home-in-vs-code)
  - [Copy benchmark scripts into your WSL home](#83-copy-benchmark-scripts-into-your-wsl-home)
  - [Script purposes & arguments](#84-script-purposes--arguments)
  - [Run the benchmarks (from any directory)](#85-run-the-benchmarks-from-any-directory)
10. [Monitor GPU Stats (PowerShell, Windows)](#9-monitor-gpu-stats-powershell-windows)
11. [Venv Quick Ops](#venv-quick-ops)
12. [Notes](#notes)


This guide provides a single, copy-pasteable setup for WSL2 with GPU acceleration, CUDA, Docker Engine + NVIDIA Container Toolkit, and GPU monitoring. It uses VS Code in WSL to create and run `bench.py` for benchmarking.

---

## 1) Windows Host Prerequisites

```powershell
wsl --update
wsl --status
wsl --shutdown
```

- Install the latest NVIDIA Game Ready or Studio driver, reboot, then:

```powershell
nvidia-smi   # should list your NVIDIA GPU model + driver version
```

- Launch Ubuntu from the Windows Start menu (this is WSLg, not XRDP).

---

## 2) Enable systemd in WSL

```bash
ps -p 1 -o comm=
# If not 'systemd', enable it:
echo -e "[boot]\nsystemd=true" | sudo tee /etc/wsl.conf
wsl --shutdown
```

- Reopen Ubuntu from Start.

---

## 3) CUDA Toolkit in WSL (Driverless NVIDIA Repo)

```bash
sudo apt-get update
sudo mkdir -p /usr/share/keyrings
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-wsl.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-wsl.gpg] https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /" \
  | sudo tee /etc/apt/sources.list.d/cuda-wsl-ubuntu.list
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin \
  | sudo tee /etc/apt/preferences.d/cuda-repository-pin-600 >/dev/null
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
# Add nvcc to PATH (do not add a global CUDA LD_LIBRARY_PATH):
grep -q '/usr/local/cuda-12.8/bin' ~/.bashrc || \
  echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

---

## 4) Make WSLg’s D3D12 GL Stack the Default (GPU-Accelerated Linux GUI)

```bash
# Remove any old CUDA LD_LIBRARY_PATH lines
sed -i '/LD_LIBRARY_PATH=.*cuda/d' ~/.bashrc
# Ensure WSLg libs are visible to the linker
echo "/usr/lib/wsl/lib" | sudo tee /etc/ld.so.conf.d/wslg.conf
sudo ldconfig
# Prefer D3D12-backed Mesa for interactive shells
{
  echo 'export LIBGL_ALWAYS_SOFTWARE=0'
  echo 'export MESA_LOADER_DRIVER_OVERRIDE=d3d12'
  echo 'export GALLIUM_DRIVER=d3d12'
  echo 'export MESA_D3D12_DEFAULT_ADAPTER_NAME="NVIDIA"'
} >> ~/.bashrc
source ~/.bashrc
# Verify acceleration:
sudo apt-get install -y mesa-utils glmark2
echo "$WAYLAND_DISPLAY"                  # expect: wayland-0
glxinfo -B | egrep 'OpenGL renderer|OpenGL version'  # Expect: OpenGL renderer string: D3D12 (NVIDIA <Your GPU Model>)
glmark2   # optional FPS sanity (should be high)
```

---

## 5) Docker Engine (Inside WSL)

```bash
# Clean conflicting packages (safe if none installed)
sudo snap remove docker 2>/dev/null || true
sudo apt-get remove -y docker.io docker-doc podman-docker containerd runc docker-compose-plugin 2>/dev/null || true
# Docker CE repo
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
 https://download.docker.com/linux/ubuntu $(. /etc/os-release; echo $VERSION_CODENAME) stable" \
 | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
sudo systemctl enable --now docker
docker info | grep -i runtime
```

---

## 6) NVIDIA Container Toolkit (GPU Runtime for Docker)

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
# Container GPU test:
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
# Should show your NVIDIA GPU from inside the container

```

## Minimum & Recommended Hardware / Software

| Category | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| Windows Version | Windows 11 (22H2) or Windows 10 21H2+ | Latest Windows 11 | Newer builds ship newer WSL kernel & GPU stack |
| WSL Distro | Ubuntu 22.04 / 24.04 | Ubuntu 24.04 | Guide written/tested against 24.04 |
| NVIDIA GPU | Compute Capability 7.0 (Volta) | Ampere (SM 8.0+) or newer | TF32 & BF16 acceleration require Ampere+; older GPUs run FP16/FP32 only |
| CUDA Toolkit | 12.x user‑space | Latest 12.x | Driver supplied by Windows host driver |
| VRAM | 4 GB | 8+ GB | Largest example (8192^3 GEMM) fits well below 1 GB; extra headroom helps multitasking |
| System RAM | 8 GB | 16+ GB | More RAM helps with large Docker builds & multitasking |
| Driver | Current Game Ready / Studio | Latest | Use `nvidia-smi` to confirm visibility in WSL & containers |

If your GPU lacks BF16/TF32 (pre‑Ampere), the benchmark scripts will skip those modes automatically.

---


## 7) Python venv + PyTorch CUDA Wheels (No Containers)

```bash
sudo add-apt-repository -y universe
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv
python3 --version
python3 -m pip --version
# Create and activate venv
python3 -m venv ~/.venvs/ai
source ~/.venvs/ai/bin/activate
python -m pip install -U pip setuptools wheel
# Install PyTorch CUDA 12.1 wheels + NumPy (matching build)
pip install -U numpy
pip install -U torch --index-url https://download.pytorch.org/whl/cu121
pip install -U torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Sanity
python - <<'PY'
import torch, numpy
print("NumPy:", numpy.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

---

## 8) VS Code in WSL (Benchmark Scripts)

### 8.1 Install VS Code on Windows + WSL Extension

```powershell
winget install --id Microsoft.VisualStudioCode -e
```
- Open VS Code → Extensions → install Remote – WSL (ms-vscode-remote.remote-wsl).

### 8.2 Open your WSL home in VS Code

```bash
code ~
```
- VS Code will install its server in WSL and open your home folder.

### 8.3 Copy benchmark scripts into your WSL home

The repository contains the reference versions under `python/bench.py` and `python/bench_tests.py`. To keep your home directory clean and runnable regardless of where this repo lives, copy their contents into new files in `~`:

1. In VS Code (opened on `~`), use File → New File, name it `bench.py`.
2. In another VS Code window or tab open the repo file `python/bench.py` (e.g. `code /path/to/repo/python/bench.py`). Select all, copy, and paste into `~/bench.py`. Save.
3. Repeat for `bench_tests.py`: create `~/bench_tests.py`, copy the contents of `python/bench_tests.py`, paste, save.

(Alternative CLI copy if the repo is already cloned inside your home:)
```bash
cp /path/to/repo/python/bench.py ~/bench.py
cp /path/to/repo/python/bench_tests.py ~/bench_tests.py
```

Now you can run them directly as `~/bench.py` and `~/bench_tests.py` regardless of current working directory.

### 8.4 Script purposes & arguments

- `bench.py` – High‑throughput GEMM (matrix multiply) benchmark for NVIDIA GPUs. Measures TF32 / FP16 / BF16, can optionally use CUDA Graphs. Key arguments:
  - `--size N` single cubic size (m=n=k=N)
  - `--sizes S1 S2 ...` list of cubic sizes
  - `--sweep START STOP STEP` generate a size range
  - `--iters` iterations per measurement (default 30)
  - `--warmup` override warmup iteration count
  - `--modes tf32,fp16,bf16` subset selection
  - `--graphs` enable CUDA Graph capture/replay
  - `--csv results.csv` write table to CSV

- `bench_tests.py` – Automated stress / validation matrix. Runs many size / mode / graph combinations to validate stability and performance. Environment variables:
  - `STRESS=1` include larger sizes (6144, 8192)
  - `TEST_CSV=path` export per‑test rows
  - `VERBOSE=1` show full tracebacks on failures

### 8.5 Run the benchmarks (from any directory)

```bash
# 4k³ baseline (all modes):
python ~/bench.py --size 4096 --iters 30

# Push the card with CUDA Graphs:
python ~/bench.py --size 8192 --iters 50 --graphs

# Multi-size sweep + CSV output:
python ~/bench.py --sizes 2048 4096 6144 8192 --iters 30 --graphs --csv results.csv

# Run validation matrix (default sizes):
python ~/bench_tests.py

# Add large stress sizes & write CSV:
STRESS=1 TEST_CSV=matrix.csv python ~/bench_tests.py
```

The repository versions remain the authoritative source; you can recopy if you update or pull changes.

---

## 9) Monitor GPU Stats (PowerShell, Windows)

**Live 1-sec updates:**
```powershell
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,clocks.sm,clocks.gr,memory.used,memory.total,power.draw,pstate `
  --format=csv -l 1
```
**Minimal:**
```powershell
nvidia-smi --query-gpu=utilization.gpu,power.draw,clocks.sm,clocks.gr,temperature.gpu `
  --format=csv,noheader -l 1
```
**Snapshot:**
```powershell
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.gr,pstate --format=csv
```

---

## Venv Quick Ops

```bash
# enter
source ~/.venvs/ai/bin/activate
# exit
deactivate
# where is python
which python
# am I in a venv?
echo $VIRTUAL_ENV   # empty = not in a venv
```

---

**Notes:**
- Don’t install Ubuntu’s `nvidia-cuda-toolkit` inside WSL (it pulls a Linux driver).
- Don’t set a global `LD_LIBRARY_PATH` to CUDA—keep only the PATH export; set LD_LIBRARY_PATH inline per build if you truly need it.
