# Parallel Sparse Mixture-of-Experts Inference

A C++17 implementation of one Top-2 Mixture-of-Experts (MoE) inference layer,
with CPU serial, MPI data parallelism (DP), MPI expert parallelism (EP),
OpenMP threading, and an optional single-GPU CUDA Expert FFN.

Each Token selects two Experts. DP replicates all Expert weights and partitions
Tokens; EP partitions Expert weights and exchanges Token assignments and results.
Inputs, weights and controlled routes are generated deterministically: no external
model or dataset download is required. 

We study how parallel execution speeds up MoE inference and how communication
and uneven Expert workloads limit its performance. We compare CPU parallel
strategies using MPI and OpenMP, and evaluate Expert computation on a single GPU.

## 1. Requirements

- Linux/Bash for the commands below; CMake 3.20 or newer and a C++17 compiler.
- MPI with C++ support and OpenMP. Both are required by the CMake project,
  including when building only the serial executable.
- Optional GPU: NVIDIA CUDA toolkit and a CUDA-capable GPU. The current CMake
  target sets architecture **86**, tested on NVIDIA A40. For another architecture,
  change the `CUDA_ARCHITECTURES` target property in `CMakeLists.txt` before building.
- Analysis only: Python 3.10+ and Matplotlib (`python3 -m pip install matplotlib`).
- Formal Vanda submissions additionally need Git, PBS, and a committed checkout.

Run commands from the repository root. On Vanda, compile and run programs inside
an allocated compute job; use the login node to submit jobs.

## 2. Build and verify the CPU code

On a Linux machine with MPI/OpenMP available:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

The full build includes the three unit-test executables (protocol, serial and
placement). Expected CTest output: `100% tests passed`.

For Vanda, first request an interactive allocation from the login node
(replace `YOUR_PROJECT` with your own project name), then load the modules:

```bash
qsub -I -P YOUR_PROJECT \
  -l select=1:ncpus=4:mpiprocs=2:ompthreads=2:mem=16gb \
  -l walltime=00:30:00
# After the compute-node prompt appears:
cd "$HOME/parallel-sparse-moe"
module load gompi/2024a
module load CMake/3.29.3-GCCcore-13.3.0
export LD_LIBRARY_PATH="/opt/pbs/lib:${LD_LIBRARY_PATH:-}"
```

Adjust the project directory if necessary, then run the build commands above.
A fresh allocation needs its modules loaded again. If MPI reports insufficient
slots, check the requested `mpiprocs`; the examples below need two MPI slots
and four CPU cores in total.

Set the thread layout and verify the MPI/OpenMP launch:

```bash
export OMP_NUM_THREADS=2 OMP_PLACES=cores OMP_PROC_BIND=close OMP_DYNAMIC=FALSE
mpirun --map-by slot:PE=2 --bind-to core -np 2 \
  ./build/moe_launch_probe --expect-threads 2
```

These binding flags are for OpenMPI. Adapt launcher flags for another MPI
implementation. The launch probe should print `RESULT: PASS`.

Check serial and parallel numerical correctness:

```bash
./build/moe_serial --router-mode trace --skew uniform --tokens 2048

mpirun --map-by slot:PE=2 --bind-to core -np 2 \
  ./build/moe_parallel_check --parallelism ep \
  --router-mode trace --skew uniform --tokens 2048

mpirun --map-by slot:PE=2 --bind-to core -np 2 \
  ./build/moe_parallel_check --parallelism dp \
  --router-mode trace --skew uniform --tokens 2048
```

Expected results are serial `RESULT: PASS` and parallel `CPU ep/dp correctness:
PASS`. The parallel correctness executable is limited to 8,192 Tokens.
It is not the performance executable.

## 3. Run CPU benchmarks

The following programs each perform three warmups and five measured trials:

```bash
# One-core serial baseline (OpenMPI binds the single process).
OMP_NUM_THREADS=1 mpirun --bind-to core -np 1 \
  ./build/moe_serial_bench --router-mode trace --tokens 32768

# EP: two MPI ranks, two threads each; load-aware placement and P2P.
mpirun --map-by slot:PE=2 --bind-to core -np 2 \
  ./build/moe_cpu_bench --parallelism ep --communication p2p --placement la \
  --router-mode trace --skew moderate --tokens 32768 --experts 16

# DP: replicated weights, no Token exchange inside the timed forward.
mpirun --map-by slot:PE=2 --bind-to core -np 2 \
  ./build/moe_cpu_bench --parallelism dp \
  --router-mode trace --skew uniform --tokens 32768
```

Use `--placement rr` for round robin and `--communication alltoallv` for the
collective EP implementation. DP requires `rr` and the default `alltoallv`
option, although DP does not actually perform an all-to-all in its forward.
Record stdout and stderr separately, for example `> run.txt 2> run.stderr`.
Only trials with `bitwise_ok=1` are eligible for performance analysis.

### Common options

| Option | Default | Meaning / supported values |
|---|---|---|
| `--tokens` | `2048` | Global Token count |
| `--dim`, `--hidden`, `--experts` | `128`, `256`, `16` | Model dimensions and Expert count |
| `--topk` | `2` | Must remain 2 |
| `--router-mode` | `router` | Computed routing (`router`) or controlled synthetic routes (`trace`) |
| `--skew` | `uniform` | `uniform`, `moderate`, `extreme`; nonuniform modes require Trace and 16 Experts |
| `--hot-layout` | `A` | `A`, `B`, `R1`, `R2`, `R3` hotspot layouts |
| `--parallelism` | `ep` | CPU `ep` or `dp` |
| `--communication` | `alltoallv` | EP `alltoallv` or `p2p` |
| `--placement` | `rr` | EP `rr` or calibrated load-aware `la` |
| `--omp-schedule` | `static` | CPU task scheduling: `static` or `dynamic` |
| `--backend` | `cpu` | Use `cuda` with `moe_gpu` |

Default seeds: model `42`, input `314159`, routing `271828`, calibration `161803`.
Override with `--model-seed`, `--input-seed`, `--routing-seed`, and
`--calibration-seed`. Keep model parameters and seeds fixed within comparisons.

## 4. Run the single-GPU benchmark

Allocate one GPU before running. On Vanda, request it from the login node:

```bash
qsub -I -P YOUR_PROJECT \
  -l select=1:ncpus=4:ngpus=1:mpiprocs=1:mem=32gb \
  -l walltime=00:30:00
# After entering the compute node:
cd "$HOME/parallel-sparse-moe"
module load gompi/2024a
module load CMake/3.29.3-GCCcore-13.3.0
module load cuda12.4/toolkit/12.4.1
export OMP_NUM_THREADS=1

cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DMOE_ENABLE_CUDA=ON
cmake --build build-gpu --target moe_gpu -j 4
./build-gpu/moe_gpu --backend cuda --router-mode trace --skew uniform --tokens 8192
```

`moe_gpu` supports at most 32,768 Tokens and runs directly, without `mpirun`.
It prints five GPU trials, five CPU serial trials, errors and `GPU correctness:
PASS`. Routing, packing and output combination run on the CPU; Expert FFNs run
on one GPU. Output includes host packing, H2D, kernels, D2H, host combination,
and total elapsed time. One-time weight upload is reported separately.

## 5. Reproduce formal Vanda experiments

From a **committed, clean Git checkout** on the login node, choose a CPU job:

```bash
bash scripts/submit_vanda_cpu_formal.sh scripts/vanda_cpu_formal.pbs
```

| Experiment | CPU PBS script | Raw output directory |
|---|---|---|
| Strong/weak scaling to 256 cores | `scripts/vanda_cpu_formal.pbs` | `private/formal-cpu-JOBID/` |
| Serial size and serial/EP/DP comparison | `scripts/vanda_cpu_compare.pbs` | `private/compare-cpu-JOBID/` |
| Fixed 256-core MPI/OpenMP comparison | `scripts/vanda_cpu_hybrid.pbs` | `private/hybrid-cpu-JOBID/` |
| Communication and skew | `scripts/vanda_cpu_comm_skew.pbs` | `private/comm-skew-JOBID/` |
| Expert placement | `scripts/vanda_cpu_placement.pbs` | `private/placement-JOBID/` |

For the formal single-GPU matrix:

```bash
bash scripts/submit_vanda_gpu_formal.sh
```

Its output is `private/gpu-formal-JOBID/`. The wrappers archive committed source
and print its commit and PBS job ID; the job verifies the source checksum before
building. `JOBID` in directory names is the numeric portion of the PBS ID.
Check resource requests and your site's project/queue requirements before
submission. If a project directive is required, add `#PBS -P YOUR_PROJECT` to
the chosen script and commit it before using the wrapper. A source ZIP without
Git metadata can use the manual build/run steps above; these wrappers need Git.

Monitor with `qstat -fx FULL_JOB_ID`. Successful jobs write `exit_status=0` and
`stage=complete` to `job_status.txt`. Retain the complete raw directory for
analysis. Do not merge diagnostic stderr into benchmark stdout or enable
`MOE_DEBUG_PHASES` for formal timing.

### Analyze data and regenerate figures

For example, after a placement job finishes:

```bash
python3 scripts/analyze_cpu_placement.py private/placement-JOBID \
  --output results/processed/vanda-cpu-placement-JOBID.csv
python3 scripts/plot_cpu_placement.py results/processed/vanda-cpu-placement-JOBID.csv
```

The corresponding analyzer/plot script suffixes are `cpu_formal`, `cpu_compare`,
`cpu_hybrid`, `cpu_comm_skew`, `cpu_placement`, and `gpu_formal`. Each analyzer
accepts the raw directory plus `--output`; each plot script accepts the processed
CSV. Use the result filename conventions shown in the
[experiment inventory](docs/experiment-status.md). Raw CPU/GPU formal logs are
local artifacts under `private/`; they are not included in the source package.
Processed CSVs are in `results/processed/`, and PNG/PDF plots are in `figures/`.
To regenerate a plot alone, use its existing processed CSV.

## 6. Source layout and results

| Location | Contents |
|---|---|
| `include/moe/`, `src/` | Configuration, synthetic data, routing, placement, FFN kernels and executable entry points |
| `tests/` | Protocol, serial numerical and placement unit tests |
| `scripts/` | PBS submission, environment probes, data validation and plotting |
| `docs/` | Experiment methods, validated results and limitations |
| `results/`, `figures/` | Recorded data summaries and report figures |

Additional diagnostics are `moe_contract_check` (configuration/partition),
`moe_mpi_loopback` (dispatch/return without FFNs), and `moe_cpu_pilot` (exploratory
measurements). Use `moe_cpu_bench` for formal CPU timings.

See the [experiment inventory](docs/experiment-status.md) for recorded CPU and
GPU studies and their results.
