# Cuda launch benchmark

A tiny CUDA micro-benchmark to compare two launch strategies for a simple 4-stage pipeline of kernels `A → B → C → D`:

1) **Host launch** — the CPU launches kernels A, B, C, D sequentially on a CUDA stream.  
2) **Dynamic parallelism (DP)** — a small parent kernel launches A, B, C, D from the device using `cudaStreamTailLaunch`.

The kernels perform deterministic, moderately compute-heavy math so we can focus on launch overhead and scheduling rather than memory I/O.

---

## Requirements

- **CUDA Toolkit** (e.g., 12.x) with `nvcc` in PATH.
- **CMake 3.24+**.
- A C++17 compiler


## Build & run

1) Configure

```powershell
# from repo root
py -3 scripts\setup.py configure --export-compile-commands
````


2) Build

```bash
python3 scripts/setup.py build --config Release
```

3) Run

```bash
# Run with custom problem size / work and more iterations (Release)
python3 scripts/setup.py run --config Release -- -n 2000000 --iters 128 --runs 20 --block 256
```

---

## Command-line options

The executable accepts:

```
--n <int>         # number of elements (default: 1<<20)
--iters <int>     # per-thread ALU work scaling (default: 64)
--runs <int>      # measured repetitions for each strategy (default: 10)
--warmup <int>    # warmup iterations per strategy (default: 3)
--block <int>     # threads per block (default: 256)
--device <int>    # CUDA device index (default: 0)
```

## Results

In practice, **host-sequenced launches are often slightly more efficient** for this chained, strictly-serialized workload. 
