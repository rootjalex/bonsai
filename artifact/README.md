# Artifact: Data Layout Polymorphism for Bounding Volume Hierarchies

**Paper**: *Data Layout Polymorphism for Bounding Volume Hierarchies*
**Authors**: **redacted**
**Venue**: PLDI 2026

## Overview

This artifact contains Scion, a domain-specific language and compiler for
recursive geometric queries on bounding volume hierarchies (BVHs). Scion
enables independent specification and optimization of data layouts separately
from traversal algorithms, allowing systematic design space exploration across
layout choices.

The artifact reproduces the main experimental results from the paper:

1. **Design space exploration** across 26 BVH layouts for ray tracing and
   closest point queries.
2. **Comparison against Embree** on ray tracing performance vs. memory
   trade-offs.
3. **Comparison against FCL** on collision detection.
4. **Comparison against FCPW** on closest point queries.

### Hardware Requirements

The full evaluation was conducted on three platforms:

| Machine | CPU / GPU | Arch | RAM | OS |
|---------|----------|------|-----|-----|
| x86 Desktop | Intel Core i9-14900K (24 cores) | x86_64 | 64 GB | Ubuntu 22.04 |
| MBP M2 | Apple M2 Pro (10 cores) | arm64 | 16 GB | macOS 14+ |
| GPU | NVIDIA GeForce RTX 4090 | CUDA | 24 GB GDDR6X | Ubuntu 22.04 |

Any deviation from this may affect the results. In fact, that is the entire thesis of this work.

### Software Requirements

The run script installs all dependencies automatically. For reference:
- CMake >= 3.30
- LLVM 19.1.6 (Clang/LLVM)
- Embree 4.x (ray tracing comparison; installed via `brew` / `conda`, `find_package(embree 4.0)`)
- OpenMP (parallel execution)
- Eigen3 (linear algebra)
- Python 3 with matplotlib, numpy, pandas (analysis scripts)
- FCL 0.7.0 and libccd 2.1 (collision detection; auto-fetched by CMake via FetchContent,
  [`fcl@0.7.0`](https://github.com/flexible-collision-library/fcl/tree/0.7.0),
  [`libccd@v2.1`](https://github.com/danfis/libccd/tree/v2.1))
- FCPW (closest point queries; auto-fetched by CMake via FetchContent,
  [`fcpw@e36bc9b`](https://github.com/rohan-sawhney/fcpw/tree/e36bc9b34af6088fb78ddbb6a93e26686779678a))
- CUDA driver 12.6 (GPU evaluation only; `--gpu` flag)

---

## Getting Started (~30 minutes)

### Step 1: Enter the repository

```bash
cd bonsai
```

### Step 2: Run the short smoke test

The `--short` flag runs a minimal configuration (1 scene, 1 run, small ray
counts) to verify that everything compiles and executes correctly:

```bash
./artifact/run.sh --short
```

This takes approximately **10 minutes** and exercises the full pipeline:
- Installs dependencies (if needed)
- Builds the Scion compiler
- Compiles and runs 4 ray tracing layouts (2 BVH2 + 2 BVH8) on 1 scene
- Compiles and runs 2 Embree configurations on 1 scene
- Compiles and runs collision detection with 2 layouts on 1 scene pair
- Compiles and runs closest point queries with 1 layout on 1 scene

### Step 3: Verify output

After the short run completes, you should see (amongst other noise):

```
[  OK  ] Scion compiler built successfully
[  OK  ] DSE RT complete. Results in artifact/results/rt/
[  OK  ] Embree comparison complete. Results in artifact/results/embree/
[  OK  ] FCL comparison complete. Results in artifact/results/cd/
```

The `artifact/results/` directory will contain text files with timing data.
Each line reports hits and trace times in milliseconds, confirming correct
execution.

---

## Step-by-Step Instructions

### Running Benchmarks

Run benchmarks on each machine independently. Results are written to
`artifact/results/` with architecture-tagged filenames so they do not conflict.

```bash
./artifact/run.sh benchmark                # CPU benchmarks (several hours)
./artifact/run.sh benchmark --short        # Quick smoke test (~10 min)
./artifact/run.sh benchmark --gpu          # GPU (CUDA) benchmarks only
./artifact/run.sh benchmark --gpu --short  # Quick GPU smoke test
```

**NOTE**: Running all benchmarks across x86, ARM, and GPU requires > 24 hours as we performance a large fan across 42 different 
evaluation contexts, and additionally require testing different ray-triangle intersections for fair comparison with Embree.

The default mode (no subcommand) is `benchmark`:

```bash
./artifact/run.sh                        # Same as: ./artifact/run.sh benchmark
./artifact/run.sh --short                # Same as: ./artifact/run.sh benchmark --short
```

GPU mode requires an NVIDIA GPU with the CUDA toolkit (`nvcc`) installed. On
HPC systems, you may need to `module load cuda` first. The `--gpu` flag runs
steps 2-3 on the GPU; steps 4-6 (Embree, FCL, FCPW) are CPU-only and are
skipped.

### Generating Figures

After collecting results from all machines into one `artifact/results/`
directory, generate figures:

```bash
./artifact/run.sh figures
```

This converts raw `.txt` results to CSV, then produces PDF figures and LaTeX
tables matching the paper.

### All-in-One

To run CPU benchmarks, GPU benchmarks, and generate figures in sequence:

```bash
./artifact/run.sh all                    # Full CPU + GPU + figures
./artifact/run.sh all --short            # Quick smoke test of everything
```

### Multi-Machine Workflow

The paper evaluates on three platforms (x86, arm, CUDA). To reproduce:

1. Run benchmarks on each machine:
   ```bash
   arm-machine$  ./artifact/run.sh benchmark
   x86-machine$  ./artifact/run.sh benchmark
   gpu-machine$  ./artifact/run.sh benchmark --gpu
   ```

2. Collect all `artifact/results/` directories into one location (e.g., via
   `rsync` or `scp`):
   ```bash
   rsync -av x86-machine:bonsai/artifact/results/ artifact/results/
   rsync -av arm-machine:bonsai/artifact/results/ artifact/results/
   rsync -av gpu-machine:bonsai/artifact/results/ artifact/results/
   ```
   All raw data files are architecture-tagged, so they do not overwrite each
   other:
   - RT: `rt/arm-camera.txt`, `rt/x86-camera.txt`, `rt/cuda-camera.txt`, ...
   - CPQ: `cpq/arm-dse.txt`, `cpq/x86-dse.txt`, `cpq/cuda-dse.txt`
   - CD: `cd/arm-fcl_comparison.txt`, `cd/x86-fcl_comparison.txt`
   - Embree: `embree/arm-camera.txt`, `embree/x86-camera.txt`, ...

3. Generate figures from the combined data:
   ```bash
   ./artifact/run.sh figures
   ```

**Note on data safety:** The script never deletes the `artifact/results/`
directory. Each benchmark step truncates only its own output file (e.g.,
running on arm only overwrites `arm-*.txt` files, leaving `x86-*.txt` and
`cuda-*.txt` untouched). The `figures` mode regenerates the intermediate CSV
files from the raw `.txt` data each time, so CSVs are always consistent with
the current contents of `artifact/results/`.

### Running Individual Steps

Each step can be run independently. Use `--help` for full documentation:

```bash
./artifact/run.sh --help                    # Show all options and step descriptions
./artifact/run.sh benchmark --step 0        # Install dependencies only
./artifact/run.sh benchmark --step 1        # Build the Scion compiler
./artifact/run.sh benchmark --step 2        # DSE: Ray tracing (Table 1, Figure 15{a,b}, 16)
./artifact/run.sh benchmark --step 3        # DSE: Closest point queries (Figure 16)
./artifact/run.sh benchmark --step 4        # Embree comparison (Figure 18, Appendix D)
./artifact/run.sh benchmark --step 5        # FCL comparison (Figure 19a)
./artifact/run.sh benchmark --step 6        # FCPW comparison (Figure 19b)
./artifact/run.sh benchmark --step 2 --gpu  # DSE: Ray tracing on GPU (CUDA) (Figure 15{a,b}, 16)
./artifact/run.sh benchmark --step 3 --gpu  # DSE: Closest point queries on GPU (CUDA) (Figure 16)
```

## Claims Supported by the Artifact

### Supported Claims

1. **Pareto-optimal layouts vary across algorithms, architectures, and workload characteristics** 
   (Section 8.2, Table 1, Figures 15 and 16). The design space exploration
   demonstrates that no single layout dominates across all scenarios. The
   Pareto frontier analysis shows different layouts are optimal for different
   (scene, ray type, architecture) combinations. Additionally we provide a 
   script at `artifacts/collect_dse.py` (and part of the Supplemental Materials) 
   to explore the results of the entire evaluation space.

2. **Scion-generated code is competitive with hand-optimized libraries**
   (Section 8.3, Figures 18 and 19). Steps 4-6 compare Scion against:
   - **Embree** (Intel's production ray tracing library): Scion matches or
     exceeds Embree performance on several layout/scene combinations while
     exploring a broader design space. In other cases, Scion performances much 
     worse due to the lack of scheduling optimizations performed by Embree, 
     e.g., vectorization.
   - **FCL** (Flexible Collision Library): Scion-generated collision detection
     outperforms FCL across tested configurations.
   - **FCPW** (Fast Closest Point in the West): Scion-generated closest
     point queries are competitive with FCPW.

### Note on CPQ Traversal Optimization

The closest-point query (CPQ) benchmarks apply a manual patch to the
Scion-generated C++ code (`artifact/cpq_traverse_patch.cpp`). This patch
replaces the generated `_traverse_tree0` function with a version that uses
priority-queue child ordering (visiting the nearer child first), which is a
standard scheduling optimization for nearest-neighbor traversals.

The current version of Scion does not yet support this scheduling directive
natively. However, the traversal algorithm is orthogonal to the data layout
contributions of this paper: Scion's layout polymorphism operates independently
of algorithmic choices, and this manual edit does not affect any layout-related
results. Future versions of Scion will support this scheduling optimization
as a built-in directive.

## Troubleshooting

### LLVM not found

If CMake cannot find LLVM 19, set the path explicitly:

```bash
export LLVM_ROOT=/path/to/llvm-19
# Or, on macOS with Homebrew:
export LLVM_ROOT=$(brew --prefix llvm@19)
```

### Embree not found

On macOS: `brew install embree`
On Linux: Install from conda-forge (`conda install -c conda-forge embree`) or
build from source (https://github.com/RenderKit/embree/releases).

### Python plotting errors

Install the required Python packages:

```bash
pip3 install matplotlib numpy pandas
```