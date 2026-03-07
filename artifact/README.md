# Artifact: Data Layout Polymorphism for Bounding Volume Hierarchies

**Paper**: *Data Layout Polymorphism for Bounding Volume Hierarchies*
**Authors**: Christophe Gyurgyik, Alexander J Root, Fredrik Kjolstad
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

The artifact can be evaluated on any machine with:
- macOS (Apple Silicon or Intel) or Linux (x86_64 or aarch64)
- At least 8 GB RAM (16 GB recommended)
- At least 5 GB free disk space

Any deviation from this may affect the results. In fact, that is the entire thesis of this work.

### Software Requirements

The run script installs all dependencies automatically. For reference:
- CMake >= 3.30
- LLVM 19.1.6 (Clang/LLVM)
- Embree 4.x (ray tracing comparison; installed via brew/conda, `find_package(embree 4.0)`)
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

## Getting Started Guide (~30 minutes)

### Step 1: Clone and enter the repository

```bash
git clone <repository-url> bonsai
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
./artifact/run.sh benchmark              # CPU benchmarks (several hours)
./artifact/run.sh benchmark --short      # Quick smoke test (~10 min)
./artifact/run.sh benchmark --gpu        # GPU (CUDA) benchmarks only
./artifact/run.sh benchmark --gpu --short  # Quick GPU smoke test
```

**NOTE**: Running all benchmarks across x86, ARM, and GPU requires > 24 hours as we performance a large fan across 42 different evaluation contexts, and additionally require testing different ray-triangle intersections for fair comparison with Embree.

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
./artifact/run.sh --help                 # Show all options and step descriptions
./artifact/run.sh benchmark --step 0     # Install dependencies only
./artifact/run.sh benchmark --step 1     # Build the Scion compiler
./artifact/run.sh benchmark --step 2     # DSE: Ray tracing (Table 1, Figure 7)
./artifact/run.sh benchmark --step 3     # DSE: Closest point queries (Table 1, Figure 7)
./artifact/run.sh benchmark --step 4     # Embree comparison (Figure 8)
./artifact/run.sh benchmark --step 5     # FCL comparison (Figure 9a)
./artifact/run.sh benchmark --step 6     # FCPW comparison (Figure 9b)
./artifact/run.sh benchmark --step 2 --gpu  # DSE: Ray tracing on GPU (CUDA)
./artifact/run.sh benchmark --step 3 --gpu  # DSE: Closest point queries on GPU (CUDA)
```

### Output Format

All benchmark results are written as plain text to `artifact/results/`:

```
artifact/results/
├── rt/                     # Ray tracing DSE raw data
│   ├── <arch>-<ray_type>.txt  # CPU: e.g. arm-camera.txt, x86-secondary.txt
│   └── cuda-<ray_type>.txt   # GPU: e.g. cuda-camera.txt (with --gpu)
├── cpq/                    # Closest point query raw data
│   ├── <arch>-dse.txt          # CPU: e.g. arm-dse.txt, x86-dse.txt
│   └── cuda-dse.txt           # GPU results (with --gpu)
├── embree/                 # Embree comparison raw data
│   └── <arch>-<ray_type>.txt
├── cd/                     # Collision detection raw data
│   └── <arch>-fcl_comparison.txt  # e.g. arm-fcl_comparison.txt
├── rt-results.csv          # Hygiened RT + Embree CSV (input to plotting)
├── cpq-results.csv         # Hygiened CPQ CSV
├── cd-results.csv          # Hygiened CD CSV
└── figures/                # Generated PDF figures
    ├── dse.pdf                         # Figure 7: DSE Pareto frontiers
    ├── data-dependent1.pdf             # Figure 6a: Data-dependent Pareto
    ├── data-dependent2.pdf             # Figure 6b: Data-dependent speedup
    ├── machine-dependent1.pdf          # Figure 6c: Machine-dependent Pareto
    ├── algorithm-dependent.pdf         # Figure 6d: Algorithm-dependent
    ├── embree-comparison-rt.pdf        # Figure 8: Embree comparison
    ├── fcl-comparison-cd_*.pdf         # Figure 9a: FCL comparison
    ├── fcpw-comparison-cpq.pdf         # Figure 9b: FCPW comparison
    └── novel-layout-table.tex          # Table: Novel layout domination
```

### Mapping Outputs to Paper Figures and Tables

| Step | Script | Output | Paper Reference |
|------|--------|--------|-----------------|
| 2 | `hygiene_rt.py` → `collect_dse.py` | `dse.pdf` | Figure 7: DSE Pareto frontiers |
| 2 | `collect_dse.py` | `data-dependent1.pdf` | Figure 6a: Data-dependent Pareto |
| 2 | `speedup_bar_chart.py` | `data-dependent2.pdf` | Figure 6b: Data-dependent speedup |
| 2 | `collect_dse.py` | `machine-dependent1.pdf` | Figure 6c: Machine-dependent Pareto |
| 2,3 | `collect_dse.py --compare-cpq` | `algorithm-dependent.pdf` | Figure 6d: Algorithm-dependent |
| 4 | `embree_comparison_rt.py` | `embree-comparison-rt.pdf` | Figure 8: Embree comparison |
| 5 | `fcl_comparison_cd.py` | `fcl-comparison-cd_*.pdf` | Figure 9a: FCL comparison |
| 6 | `fcpw_comparison_cpq.py` | `fcpw-comparison-cpq.pdf` | Figure 9b: FCPW comparison |
| 2 | `is_pareto_optimal.py` | `novel-layout-table.tex` | Table 1: Novel layout domination |

---

## Claims Supported by the Artifact

### Supported Claims

1. **Pareto-optimal layouts vary across algorithms, architectures, and workload
   characteristics** (Section 6.1, Table 1, Figure 7). The DSE (steps 2-3)
   demonstrates that no single layout dominates across all scenarios. The
   Pareto frontier analysis shows different layouts are optimal for different
   (scene, ray type, architecture) combinations.

2. **Scion-generated code is competitive with hand-optimized libraries**
   (Section 6.2-6.3, Figures 8-9). Steps 4-6 compare Scion against:
   - **Embree** (Intel's production ray tracing library): Scion matches or
     exceeds Embree performance on several layout/scene combinations while
     exploring a broader design space. In other cases, Scion performances much 
     worse due to a lack of scheduling
   - **FCL** (Flexible Collision Library): Scion-generated collision detection
     outperforms FCL across tested configurations.
   - **FCPW** (Fast Closest Point in the West): Scion-generated closest
     point queries are competitive with FCPW.

3. **The design space is large and non-trivial** (Section 6.1). The 26 layouts
   (9 BVH2 variants + 10 BVH8 variants + 7 mixed variants) each differ in
   quantization, alignment, data layout (AOS/SOA), and compression strategy.
   Scion enables exploring this space by decoupling layout from algorithm.

4. **Scion compiles BVH traversal algorithms from a high-level specification**
   (Section 4-5). The compiler build (step 1) and subsequent benchmark
   compilation demonstrate end-to-end code generation from `.bonsai` source
   to optimized native code.

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

---

## Artifact Layout

```
bonsai/
├── artifact/
│   ├── README.md          # This file
│   └── run.sh             # Push-button evaluation script
├── compiler.cpp           # Scion compiler entry point
├── include/               # Compiler headers
├── src/                   # Compiler source (parser, IR, lowering, codegen)
│   ├── Parser/            # Lexer and parser
│   ├── IR/                # Intermediate representation
│   ├── Lower/             # Lowering passes (layouts, trees, loops, ...)
│   ├── Opt/               # Optimization passes (CSE, DCE, fusion, ...)
│   └── CodeGen/           # Code generation (LLVM, C++, CUDA)
├── stdlib/                # Bonsai standard library (.bonsai files)
│   ├── aabb.bonsai        # Axis-aligned bounding boxes
│   ├── triangle.bonsai    # Triangle primitives
│   ├── ray.bonsai         # Ray definitions
│   ├── distance.bonsai    # Distance/closest point functions
│   └── intersects.bonsai  # Intersection tests
├── runtime/CPP/           # C++ runtime support headers
├── runtime/CUDA/          # CUDA runtime support headers
├── apps/
│   ├── rt/                # Ray tracing application
│   │   ├── cpu/           # CPU benchmark (Scion-generated)
│   │   ├── cuda/          # GPU benchmark (Scion-generated, CUDA)
│   │   ├── embree/        # Embree baseline
│   │   ├── layouts/{2,8}/ # BVH layout definitions
│   │   └── data/          # OBJ mesh files (7 scenes)
│   ├── cd/                # Collision detection application
│   │   ├── cpu/           # CPU benchmark (Scion vs. FCL)
│   │   └── data/          # OBJ mesh files
│   ├── wos/               # Walk-on-Spheres / closest point queries
│   │   ├── cuda/          # GPU benchmark (CUDA)
│   │   └── fcpw/          # FCPW comparison (CPU)
│   └── scripts/           # Analysis and plotting scripts
│       ├── collect_dse.py
│       ├── embree_comparison_rt.py
│       ├── fcl_comparison_cd.py
│       └── collect_comparison_rt_cpq.py
├── tests/                 # Compiler test suite
├── cmake/                 # CMake modules
└── CMakeLists.txt         # Top-level build configuration
```

---

## Troubleshooting

### LLVM not found

If CMake cannot find LLVM 19, set the path explicitly:

```bash
export LLVM_ROOT=/path/to/llvm-19
# On macOS with Homebrew:
export LLVM_ROOT=$(brew --prefix llvm@19)
```

### Embree not found

On macOS: `brew install embree`
On Linux: Install from conda-forge (`conda install -c conda-forge embree`) or
build from source (https://github.com/RenderKit/embree/releases).

### OpenMP not found (macOS)

```bash
brew install libomp
```

### Mesh data files missing

The OBJ mesh files (lucy, hairball, dragon, etc.) are included in the
repository under `apps/rt/data/` and `apps/cd/data/`. If missing, the
benchmarks will fail with a "file not found" error.

### Python plotting errors

Install the required Python packages:

```bash
pip3 install matplotlib numpy pandas
```