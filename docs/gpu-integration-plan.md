# Revised execution plan: CPU MPI+OpenMP plus CUDA Expert FFN

The primary course implementation remains MPI+OpenMP on CPUs. It supplies the
full EP/DP, communication, placement, strong-scaling, weak-scaling, and hybrid
layout experiments. CUDA is an additional Expert-FFN backend, not a claim that
one GPU demonstrates MPI scaling.

For a GPU run, one MPI rank owns each allocated GPU. A rank never shares a GPU
with another rank. With one GPU, GPU experiments are `P=1`; with two allocated
GPUs, the optional MPI+CUDA experiment uses two ranks, each with an exclusive
GPU. GPU results are labelled separately from CPU multi-node scaling results.

## 0. Scope and experiment contract

Keep the frozen one-layer Top-2 MoE definition, deterministic input generation,
and CPU `serial` reference. Add `--backend cpu|cuda`; the default is `cpu`.

The initial CUDA path offloads only batched Expert FFN evaluation. Router,
assignment packing, MPI dispatch/return, and final weighted combine remain on
the host. This avoids assuming GPU-aware MPI. GPU timings must be split into
host-to-device transfer, GPU kernel, and device-to-host transfer.

## 1. Vanda account and resource confirmation

The confirmed Vanda CPU toolchain is:

```bash
module purge
module load gompi/2024a
module load CMake/3.29.3-GCCcore-13.3.0
```

Before CUDA implementation, use `hpc project` to obtain the permitted project
name. On an allocated GPU node, confirm the CUDA toolkit and GPU model with:

```bash
module load cuda12.4/toolkit/12.4.1
which nvcc
nvcc --version
nvidia-smi -L
```

Do not assume that a Python/PyTorch Singularity image is needed for this C++
project. First prefer the host CUDA toolkit. A container is only a fallback if
the host toolchain cannot compile a CUDA C++ program.

## 2. CPU and GPU launch checks

Run the existing CPU launch probe on compute nodes for `2 ranks x 1 thread`,
`2 ranks x 2 threads`, and two CPU nodes.

Add a CUDA probe before any MoE GPU work. It must report visible device count,
device name, CUDA runtime version, and rank-to-device mapping. On a one-GPU
allocation it must reject more than one MPI rank. On a two-GPU allocation it
must reject two ranks mapped to the same visible device.

## 3. Shared protocol layer

The protocol now contains `--backend cpu|cuda`. Token IDs, weights, routes,
assignment headers, source ranks, slots, placement, and message layout stay
identical for both backends. The assignment header is transferred on MPI as the
fixed wire representation; only its token/output float payload changes location
between host and device.

## 4. Deterministic data and routing workloads

No change to logical data generation. Generate inputs and weights on the host
from the same global IDs and seeds, then copy only the local Expert weights and
received assignment batches to the GPU. GPU runs must record the same realized
routing histogram as their CPU counterpart.

## 5. CPU reference and numerical baseline

Keep `naive_ref` and `serial` CPU implementations as the correctness oracle and
speedup denominator. Do not use a GPU result as a correctness reference.

## 6. CPU communication loopback

Keep the MPI loopback test entirely on host memory. It validates protocol,
packing, zero-count transfers, token/slot restoration, and rank ownership before
GPU work is added.

## 7. CPU EP and DP implementations

Finish correct CPU EP and DP paths before enabling CUDA. CPU EP remains the
main MPI research implementation. CPU DP remains the replicated,
communication-free reference.

## 8. CPU OpenMP hybrid implementation

Use OpenMP for per-rank Router, packing, Expert task execution, and/or combine
only after each is race-free. The primary hybrid-layout experiment varies MPI
ranks and OpenMP threads on CPU nodes.

## 9. CPU communication and placement strategies

Implement and validate Alltoallv versus batched nonblocking P2P, then RR versus
load-aware Expert placement, on the CPU backend first. This preserves a stable
baseline when GPU staging changes the compute/communication balance.

## 10. CUDA Expert backend

Add `expert_cuda.cu` and a C++ header with the same logical interface as the CPU
Expert kernel. For each received Expert batch:

1. Host packs a contiguous Token batch.
2. Host copies Tokens and local Expert weights to device memory.
3. CUDA computes `W2 * ReLU(W1 * x)` in FP32.
4. Device copies outputs to host memory.
5. Host performs the existing MPI return and fixed-order combine.

Device buffers and local Expert weights may be retained across warm-up and timed
repetitions. Per-batch allocation, H2D copy, kernel, D2H copy, and host-side
packing are separately timed. Never time only the kernel and present it as MoE
end-to-end latency.

## 11. GPU correctness gates

CPU Trace comparisons retain the existing bitwise L2 requirement. CUDA does not
require bitwise equality because GPU FMA and reduction order may differ.

For CUDA, require bitwise-identical routes, Expert IDs, assignment counts, and
token/slot restoration, then compare outputs to CPU `serial` using the L1
elementwise and relative-L2 tolerance already defined in the README. Record the
observed maximum error before changing a threshold.

## 12. Revised experiment matrix

| Experiment | Backend/resources | Purpose |
|---|---|---|
| CPU strong and weak scaling | CPU MPI+OpenMP, one/multiple nodes | Main scalability evidence |
| CPU hybrid layout | Fixed CPU cores, varied `P x T_omp` | MPI/OpenMP trade-off |
| CPU communication and placement | CPU MPI+OpenMP | Alltoallv/P2P and RR/LA study |
| GPU compute breakdown | 1 GPU, 1 MPI rank | H2D, FFN kernel, D2H, and end-to-end comparison with CPU `P=1` |
| Optional GPU EP | 2 exclusive GPUs, 2 MPI ranks | MPI+CUDA dispatch/return; separate from CPU scaling plots |

A one-GPU result is a compute-backend comparison, not a parallel-efficiency or
MPI-scaling point. If two-GPU allocation is unavailable, omit the optional GPU
EP experiment and state that limitation.

## 13. Vanda GPU jobs and report

Vanda's documented interactive GPU request is:

```bash
qsub -I -P <project-name> -l select=1:ngpus=1 -l walltime=01:00:00
```

Use it only for short debug work. The project name must come from `hpc project`.
For a batch GPU job, request one chunk with one GPU and one MPI process; choose
OpenMP threads only after CPU/core affinity is verified. Do not copy the
Python-only Singularity sample command into the C++ workflow without confirming
that its CUDA compiler and MPI libraries match the host allocation.

The report must distinguish CPU MPI+OpenMP scaling from one-GPU acceleration,
state whether GPU-to-GPU MPI was tested, and disclose that the main data are
synthetic.
