# Parallel Sparse Mixture-of-Experts Inference

This project studies a single-layer Top-2 sparse Mixture-of-Experts (MoE)
forward pass using C++, MPI, and OpenMP. The main subject is Expert Parallelism
(EP): tokens are dispatched to the MPI ranks that own their selected experts,
processed there, and returned to their source ranks. A replicated Data
Parallel (DP) implementation provides a communication-free reference.
A CUDA Expert-FFN backend is an additional, separately reported extension; its
staged integration plan is in [docs/gpu-integration-plan.md](docs/gpu-integration-plan.md).

> **Project status:** design and implementation, protocol version 0.2.
> This README is the team's frozen experimental contract and supersedes the
> earlier pilot parameters and routing definitions in version 0.1. Calibration
> choices may change only before the formal experiment matrix is frozen.

## Scope and research questions

The project implements one MoE inference layer, not a complete language model
and not model training. It does not implement hot-expert splitting, expert
replication within EP, or dynamic expert migration.

The study asks:

1. What communication and load-imbalance costs does EP introduce relative to a
   replicated DP baseline?
2. How do `MPI_Alltoallv` and batched nonblocking point-to-point communication
   compare under the same input, placement, and data volume?
3. When does load-aware expert placement improve over round-robin placement?
4. How do different MPI-rank/OpenMP-thread splits affect performance?
5. How do controlled routing distributions affect scaling and efficiency?

The project does **not** assume that EP will beat DP at the selected teaching
scale. Instead, it quantifies the cost of EP and explains when EP becomes
necessary because model weights can no longer be replicated on every rank.

## Symbols

| Symbol | Meaning |
|---|---|
| `N` | Global number of input tokens |
| `d` | Token/model dimension |
| `h` | Expert hidden dimension |
| `E` | Number of experts |
| `k` | Experts selected per token (`k = 2`) |
| `P` | Number of MPI ranks |
| `T_omp` | OpenMP threads per rank |

Each token selects two distinct experts, so every forward pass contains exactly
`2N` token-expert assignments.

## Frozen algorithm definition

| Parameter | Frozen value |
|---|---:|
| Experts `E` | 16 |
| Model dimension `d` | 128 |
| Hidden dimension `h` | 256 |
| Top-k | 2, without replacement |
| Activation | ReLU |
| Numeric type | `float` / FP32 |
| Token dropping | None |
| Expert biases | Zero |

For expert `e`,

```text
Expert_e(x) = W2_e * ReLU(W1_e * x)
```

with deterministic initialization

```text
W1_e ~ Normal(0, 1/d)
W2_e ~ Normal(0, 1/h)
```

For a token whose selected experts are `e0` and `e1`, the two selected router
scores are converted to weights with a numerically stable softmax. The output is

```text
y = a0 * Expert_e0(x) + a1 * Expert_e1(x)
```

The following conventions are fixed:

- score ties are broken in favour of the smaller expert ID;
- partial outputs are stored as slot 0 and slot 1;
- the final expression is evaluated in the fixed order
  `y = a0 * o0 + a1 * o1`;
- expert weights and token values do not change with `P` or `T_omp`; and
- formal comparisons use `-O3 -fno-fast-math -ffp-contract=off`.

With the frozen dimensions, one expert's two matrices occupy approximately
256 KiB and all 16 experts occupy approximately 4 MiB in FP32. Full replication
is therefore practical in this experiment and motivates the DP reference line.

## Implementations

| Name/interface | Description | Purpose |
|---|---|---|
| `naive_ref` | One assignment at a time, double-precision accumulation, independent code path | Correctness oracle only |
| `serial` | Serial implementation using the shared production kernel | Speedup denominator |
| `--parallelism dp` | Every rank generates all experts and processes only its local tokens | Replicated, communication-free reference |
| `--parallelism ep` | Every rank stores only its assigned experts; tokens require dispatch and return | Main research implementation |
| `--backend cpu` | OpenMP Expert kernel on CPU | Default and primary experimental backend |
| `--backend cuda` | One exclusive GPU per MPI rank | CUDA extension; implementation follows the CPU correctness gates |

Both DP and EP support OpenMP on the CPU backend. MPI is initialized with
`MPI_Init_thread(..., MPI_THREAD_FUNNELED, ...)`; the provided level is checked,
and MPI calls occur only on the main thread outside OpenMP parallel regions.

## EP execution model

1. Tokens are distributed as contiguous global-ID ranges across MPI ranks, with
   remainders assigned to the first ranks.
2. Each local token produces two assignments.
3. Assignments are packed by destination rank.
4. Counts are exchanged, followed by assignment payloads.
5. Receivers stably group assignments by local expert.
6. Local expert tasks execute through the shared production kernel.
7. Results are restored to arrival order and returned to source ranks.
8. Source ranks restore `[token][slot][d]` and combine slot 0 then slot 1.

The baseline placement is round-robin:

```text
owner(expert_id) = expert_id % P
```

The return is an all-to-all exchange, not a gather to rank 0.

## Deterministic synthetic data

The project uses synthetic data rather than a real text dataset. No large input
or weight file is required.

- Tokens are functions of global token IDs.
- Expert weights are functions of expert IDs.
- A counter-based generator is used, for example SplitMix64 followed by a
  deterministic Box-Muller transform where normal values are required.
- Model, input, routing, and calibration use separate recorded seeds.
- Every rank generates only the data it needs; DP ranks generate all experts.
- The same case produces identical logical inputs, weights, and routes for
  every process/thread configuration.

Do not define the global workload with `seed + rank`. Generation parameters and
seeds are saved with every result.

## Routing modes

### Router mode

Router scores are computed and Top-2 experts are selected. No artificial skew
is injected. The resulting expert histogram and rank `max/mean` load are
recorded and must not be described as uniform unless the measurements show it.

### Controlled trace mode

Trace mode controls expert-assignment shares independently of a learned router.
Let `s_e` be expert `e`'s share of all `2N` assignments. Then

```text
sum_e(s_e) = 1
q_e = 2 * s_e
```

where `q_e` is the fraction of tokens selecting expert `e`. Because a token
cannot select the same expert twice, `s_e <= 0.5`.

| Workload/layout | Target assignment shares | Hot expert IDs |
|---|---|---|
| `uniform` | 6.25% for each of 16 experts | None |
| `moderate-A` | Four hot experts at 15% each; other 12 at about 3.33% each | `{0,4,8,12}` |
| `moderate-B` | Same shares as moderate-A | `{0,1,2,3}` |
| `moderate-R1/R2/R3` | Same shares as moderate-A | Three fixed-seed random four-expert sets |
| `extreme` | Expert 0 at 40%; every other expert at 4% | `{0}` |

For a trace, target shares are converted to integer per-expert quotas using
largest-remainder allocation over exactly `2N` assignments. The implementation
then chooses two distinct experts with the largest remaining quotas for each
global token ID; `hash(routing_seed, global_token_id, expert_id)` resolves
quota ties. This gives the requested histogram exactly, subject only to integer
rounding, and requires no calibration file or timed preprocessing.

For uniform traces with at least 64 and an even number of Experts (including
the 256 Expert scaling case), a seeded Expert permutation and per-cycle
rotation generate routes in O(N) time. Every complete cycle assigns each
Expert once; a partial cycle changes any two Expert counts by at most one.
This route family is fixed across process counts for the same seed and N.

The realized share of each expert must be within:

- 1 percentage point of the target for `N >= 8192`; and
- 2 percentage points for `N = 2048`.

Every `(workload, layout, N, seed)` records its realized histogram. Analysis
uses this recorded histogram when calculating load and remote-byte metrics.

Trace gate weights are deterministic:

```text
a0 ~ Uniform(0.5, 1.0)
a1 = 1 - a0
```

Slot 0 is the larger weight. Tests and timed runs derive their routes from the
same frozen quota rule and record their routing seed.

The command-line interface includes:

```text
--parallelism ep|dp
--backend cpu|cuda
--router-mode router|trace
--skew uniform|moderate|extreme
--hot-layout A|B|R1|R2|R3
```

## Placement strategies and expected load

Two fixed expert-placement methods are compared:

- **RR:** round-robin `expert_id % P`;
- **LA:** a calibration histogram followed by deterministic LPT greedy
  assignment. Calibration and mapping costs are reported separately.

The table gives constructed `max rank load / mean rank load`. `Lower` is the
unavoidable bound when experts cannot be split.

| Workload | `P=4`: RR / LA / Lower | `P=8`: RR / LA / Lower |
|---|---:|---:|
| moderate-A | 2.40 / 1.00 / 1.00 | 2.40 / 1.20 / 1.20 |
| moderate-B | 1.00 / 1.00 / 1.00 | 1.47 / 1.20 / 1.20 |
| extreme | 2.08 / 1.60 / 1.60 | 3.52 / 3.20 / 3.20 |

Measured loads must agree with this construction before performance results are
accepted. Runs record predicted and actual rank load, prediction error, actual
`max/mean`, remote-assignment ratio, and remote bytes.

## Shared production kernel

Serial, DP, and EP use the same production kernel:

```cpp
expert_forward(const ExpertWeights& w,
               const float* X, // [m][d]
               int m,
               float* Y,       // [m][d]
               float* scratch);
```

The fixed layout is row-major `W1[d][h]` and `W2[h][d]`. Assignments are grouped
with a stable counting sort. Each expert list is cut into tasks containing at
most `B` tokens, and every task calls `expert_forward`. Partial outputs are then
restored to `[token][slot][d]` before the fixed-order combine.

The kernel processes `MB` tokens together so that a loaded weight row serves
multiple tokens. Calibration candidates are:

```text
MB in {4, 8}
B  in {32, 64, 128}
```

A single-core compute-node microbenchmark records GFLOP/s and compares against
one-assignment-at-a-time GEMV. The selected `(MB, B)` is then frozen. External
BLAS is not used so that production implementations and OpenMP behaviour remain
controlled.

OpenMP flattens all `(expert, token block)` tasks into one array. Each thread
has private scratch storage and each task writes a disjoint output region. The
first version uses `schedule(static)`; skew experiments add
`schedule(dynamic)`. Scratch buffers are first-touched inside the OpenMP region.

## Communication strategies

EP compares identical packed messages using:

1. `MPI_Alltoall` for counts followed by `MPI_Alltoallv` for payloads; and
2. batched P2P: exchange counts, post all `MPI_Irecv`/`MPI_Isend`, then
   `MPI_Waitall`.

Dispatch and return use distinct message tags. Before adding the expert kernel,
a communication-loopback test returns assignments unchanged and verifies every
ID, slot, and vector. Communication/computation overlap is optional and must be
labelled separately.

## Correctness gates

| Level | Comparison | Passing rule |
|---|---|---|
| L0 | Hand-computable micro cases | Relative error `<= 1e-6` |
| L1 | `serial` vs double-accumulation `naive_ref` | Elementwise `abs(delta) <= 1e-4 + 1e-4*abs(ref)` and relative L2 `<= 1e-5` |
| L2-CPU | `serial` vs all Trace CPU DP/EP process/thread/communication/placement variants | Bitwise equality (`memcmp`) |
| L2-CUDA | `serial` vs CUDA variants | Bitwise-identical routes and assignment metadata; output follows the L1 numerical tolerance |
| L3 | Router mode across implementations | Top-2 IDs and routing weights are bitwise identical before output comparison |
| L4 | `serial` vs `naive_ref` routing | Any differing token must have second/third score gap `< 1e-4`; report and exclude it from L1 output comparison |

L1 thresholds are initial values. First observed errors are recorded; a
threshold may change only with a written numerical justification.

Required invariants are:

- total assignments equal `2N`;
- expert execution counts equal the realized trace histogram;
- each token receives both slots exactly once;
- no assignment is lost, duplicated, or run by the wrong expert;
- no output contains NaN or Inf; and
- repeated forward calls do not retain stale output or counters.

Bitwise L2 comparison depends on the shared kernel, fixed per-output accumulation
order, disabled fast-math/FMA contraction, and fixed slot-combine order. Failures
must be diagnosed before relaxing the requirement.

For `N <= 8192`, validation compares complete outputs. For larger cases, each
token output receives a 64-bit hash and an order-independent unsigned 64-bit
`MPI_Allreduce` sum is compared with serial.

The micro test uses `N=7, d=4, h=8, E=4, k=2` and includes a hand-computable
FFN, a tied-router case, and a fixed two-output combine. Communication tests also
cover `N < P`, non-divisible token counts, zero sends/receives, two experts on
the same rank, and two experts on different ranks.

## Timing and result contract

Input/weight generation, calibration, file I/O, allocation, and one-time setup
are outside the primary timed region. Grouping, packing, communication, expert
computation, return, unpacking, and combine are included. Router mode labels
whether router computation is included.

- Synchronize before the end-to-end measurement.
- Record elapsed time on every rank and reduce the maximum outside the timed
  region.
- Retain phase diagnostics, but do not sum independent phase maxima and call the
  result end-to-end time.
- Warm up before at least five measured repetitions.
- Interleave variants in the same job where possible.
- Run comparable variants on the same allocated nodes.
- Record single-node and cross-node results separately.
- A CPU result enters a performance figure only when `bitwise_ok = true`; CUDA results require the L2-CUDA gate.

Every raw record includes at least:

```text
Git commit and dirty state
system, node list, compiler, MPI version, compile flags
model/input/routing/calibration seeds
N, E, d, h, k, MB, B
parallelism, backend, router mode, skew, hot layout
placement, communication strategy, OpenMP schedule
MPI ranks, OpenMP threads, affinity settings
GPU model, CUDA runtime/compiler version, visible-device mapping (CUDA runs)
per-phase and end-to-end timings, including H2D/kernel/D2H (CUDA runs)
predicted and actual rank load and max/mean
remote assignment ratio and bytes
per-process expert-weight memory
output hash and bitwise_ok
```

Failed configurations are retained and marked but excluded from performance
statistics.

## Scaling and formal experiment matrix

Speedup and efficiency use the true `serial` production implementation as the
main denominator. One-rank parallel-program time is also retained to expose
MPI/OpenMP framework overhead.

| Experiment | Fixed | Varied |
|---|---|---|
| Strong scaling | Router mode, model, global input | Total cores; EP and DP on the same plot |
| Controlled weak scaling | Uniform trace, tokens per core | Cores and global `N`; EP and DP on the same plot |
| Hybrid layout | One node, fixed physical cores | `P x T_omp`; EP and DP |
| Communication | Trace, placement, resources, bytes | Alltoallv vs P2P |
| Load balance | Trace, communication, resources; `P=4` primary, `P=8` supplementary | RR vs LA across moderate-A/B/R1-R3 and extreme; DP and theoretical lower bound shown |
| Skew vs hybrid layout (optional) | Extreme trace, fixed total cores | `P x T_omp` |
| GPU compute breakdown | One GPU, one MPI rank | CPU/CUDA Expert FFN, H2D, kernel, D2H, and end-to-end time |
| GPU EP (optional) | Two exclusive GPUs, one MPI rank per GPU | Dispatch/return with CUDA Experts; reported separately from CPU scaling |

A compute/communication sensitivity study may vary `h` as a supplementary
experiment. With fixed `E=16`, ordinary weak scaling is limited to `P <= 16`.
If serial is too short or a rank has fewer than about four task blocks per
OpenMP thread, increase `N` or repeat forward passes before freezing the matrix.

## Memory interpretation and limitations

The analytical per-process expert-weight model is

```text
EP weight memory per process ~= (E / P) * one-expert weights
DP weight memory per process ~= E * one-expert weights
```

The report extrapolates this model to realistic MoE dimensions (`d` in the
thousands and `h` in the tens of thousands) to explain why replication becomes
infeasible. It must state these limitations:

- routing skew is synthetic and does not represent a measured real-model
  distribution;
- teaching-scale expert weights fit easily, so DP is expected to be faster;
- EP is studied for memory scalability and systems trade-offs, not because it
  is expected to win at this scale; and
- the project uses synthetic data, so there is no real-dataset conclusion.

Do not assume in advance that LA beats RR or that Alltoallv beats P2P. Report
the measured conditions under which each choice helps or hurts.

## Cluster and environment checklist

Before formal implementation and benchmarking, record:

- primary system (Atlas or Vanda), course account, and allocation/charging;
- node limits, physical cores, sockets, memory, and job-time limit;
- scheduler and recommended MPI launcher;
- compiler, MPI, and OpenMP module versions;
- CUDA toolkit, GPU model, driver/runtime versions, and GPU-rank mapping when CUDA is used;
- process and thread binding policy;
- persistent/scratch directory and cleanup rules; and
- whether login and compute nodes have the same CPU architecture.

If login and compute CPUs differ, any `-march=native` build must be performed on
a compute node. First validate 2 ranks x 1 thread, 2 ranks x 2 threads, and a
two-node run with at least one rank per node.

## Planned repository layout

```text
src/
include/
tests/
configs/
scripts/
results/
  raw/
  processed/
figures/
report/
```

Raw results are never overwritten. Processed tables and figures must be
reproducible from raw records and versioned scripts.

## Milestones

| Date | Required result |
|---|---|
| Oct 26-27 | Cluster confirmation, launch tests, definitions and validation gates (Steps 1-3) |
| Oct 28 | Send progress email; start deterministic generation and traces (Step 4) |
| Oct 28-29 | Micro tests, `naive_ref`, shared kernel, microbenchmark, `serial` (Steps 5-6) |
| Oct 30-Nov 1 | DP baseline, communication loopback, pure-MPI EP (Steps 7-9) |
| Nov 2-3 | Hybrid versions; Nov 3 checkpoint: EP and DP pass L2 |
| Nov 4-5 | P2P, load-aware placement, pilot matrix, allocation budget (Steps 11-13) |
| Nov 6-9 | Formal cluster experiments and reruns |
| Nov 10-12 | Analysis, report, clean-environment reproduction, submission preparation |

If the project falls behind, remove random layouts R1-R3 first, then the
optional skew-by-hybrid-layout experiment, then supplementary
compute/communication sensitivity. Do not remove the DP baseline, shared
kernel, or L2 bitwise validation.

## Build and run

The contract checker validates the frozen command-line settings and prints the
deterministic contiguous Token partition. `moe_serial` implements the CPU
serial path and compares it with an independent double-accumulation reference.
`moe_mpi_loopback` performs the host-memory dispatch/return protocol without
yet evaluating Experts. Build them only on an approved Vanda compute allocation
after loading its documented compiler and MPI environment:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target moe_contract_check moe_serial moe_mpi_loopback
./build/moe_contract_check --tokens 7 --inspect-ranks 4
ctest --test-dir build --output-on-failure
./build/moe_serial --router-mode trace --skew uniform --tokens 2048
OMP_NUM_THREADS=1 mpirun -np 2 ./build/moe_mpi_loopback \
  --router-mode trace --skew uniform --tokens 2048
```

These are correctness checks, not performance runs. The loopback is deliberately
backend-independent: `--backend cuda` is accepted as protocol metadata there,
but `moe_serial` currently requires `--backend cpu` until the CUDA Expert kernel
is added. The final README will add the scheduler, benchmark, and figure commands
needed for clean-directory reproduction.

### CPU DP/EP correctness checkpoint

`moe_parallel_check` now joins the existing Expert kernel to local DP
execution or EP dispatch, remote Expert evaluation, return, and fixed
slot-order combine. It checks every local output bitwise against the serial
implementation, including duplicate/missing return slots. Run on an allocated
compute node after loading the compiler, MPI, and CMake modules:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target moe_parallel_check -j 8
OMP_NUM_THREADS=1 mpirun -np 2 ./build/moe_parallel_check \
  --parallelism ep --router-mode trace --skew uniform --tokens 2048
OMP_NUM_THREADS=1 mpirun -np 2 ./build/moe_parallel_check \
  --parallelism dp --router-mode trace --skew uniform --tokens 2048
```

This correctness checker accepts CPU EP with `--placement rr|la` and
`N <= 8192`. EP can use `--communication alltoallv|p2p`; DP uses RR, has no exchange, and
uses the default `alltoallv` setting. The P2P variant exchanges counts, posts
nonblocking receives and sends for the two payload arrays, then waits for all
requests. Dispatch and return use distinct tags. EP now generates only the
Expert weights owned by its MPI rank; Router weights remain replicated. The
correctness program separately creates a complete model and global serial
reference after the parallel forward call, so it is still **not** a performance
benchmark. Do not use its runtime in scaling figures. Serial, DP, and EP now
group assignments by Expert
and call a shared micro-batch kernel (currently `MB=8`); the unit test compares
it bitwise with the per-Token scalar kernel. In DP/EP, OpenMP distributes
disjoint Expert micro-batch tasks across threads. A tuned task block size
remains to be implemented. The current `moe_serial` `forward_seconds` includes
allocation and is only a functional
diagnostic, not yet a formal benchmark result.

The checker prints `diagnostic_setup_seconds_max` (model, input, trace, route
generation) and `diagnostic_forward_seconds_max` (one synchronized forward
call). Each is the maximum elapsed time across ranks. The forward diagnostic
includes temporary allocations and validation collectives. The serial oracle
is executed after both measurements. These numbers help spot gross regressions
but do not meet the repeated-measurement contract for performance figures.

### CPU pilot measurements

`moe_cpu_pilot` runs three warmups and five measured forward calls. Rank 0 builds
the global serial oracle outside the timed loop, scatters expected local
outputs, and every rank checks bitwise equality after each call. Each printed
trial records the maximum forward time across ranks, per-phase rank maxima,
and `bitwise_ok`. Phase maxima may come from different ranks and must not be
added together to reconstruct end-to-end time. EP holds
only its owned Expert weights during the measured calls. Run it inside a Vanda
allocation after loading the same modules as the checker:

```bash
OMP_NUM_THREADS=2 mpirun --map-by slot:PE=2 --bind-to core -np 2 \
  ./build/moe_cpu_pilot --parallelism ep --communication p2p \
  --router-mode trace --skew uniform --tokens 2048
```

This is a pilot to choose workload size and check repeatability. It includes
temporary allocations and protocol validation collectives inside each forward
call; the final benchmark will separate phases and reuse buffers. Do not put
pilot times into the formal scaling figures.

### CPU benchmark checkpoint

`moe_cpu_bench` uses the same implementation and output check as the pilot,
with three warmups and five timed trials. It removes the two EP validation
collectives from each forward call; local packet and return-slot checks remain.
After timing each trial, every rank checks its output bitwise against a serial
oracle built outside the timed region. Only `bitwise_ok=1` trials are eligible
for analysis. The program prints `benchmark,trial=...` and the per-phase rank
maxima. Phase maxima may come from different ranks and are not additive.
Before timing, rank 0 also prints the exact configuration and seeds, the
realized per-Expert assignment histogram, and the assignment totals per rank
under the selected placement. Retain these lines with the raw trials. For
`--placement la`, rank 0 builds an independent 8,192-Token calibration sample
from `calibration_seed`, counts Expert assignments, and maps the heaviest
Experts to the currently least loaded MPI rank. The mapping is fixed for every
timed forward. The program separately reports calibration/mapping time,
predicted and realized rank loads, prediction error, and remote payload bytes.
Calibration and mapping are outside the measured forward; include them if
reporting full setup-to-output latency. Controlled traces prescribe exact
Expert shares, so their calibration prediction is deliberately easier than a
real workload with changing routes.

The timed region includes per-forward temporary allocations, packing,
communication, Expert computation, unpacking, and combining. Input/model/trace
generation, the serial oracle, MPI barriers before each trial, and output
checks after each trial are excluded. Use the same executable and options at
every point of a scaling series. Treat this as a candidate formal timing path
until it passes Vanda correctness and repeatability checks at the intended
resource counts. Record the Git commit, allocation, compiler modules, seeds,
rank/thread layout, and all five raw trials alongside the output.
`expert_compute_seconds` includes Expert grouping, task creation and temporary
buffers as well as FFN work. The communication-stage timers include message
packing and count exchange, so they are not pure network latency.

```bash
OMP_NUM_THREADS=2 mpirun --map-by slot:PE=2 --bind-to core -np 2 \
  ./build/moe_cpu_bench --parallelism ep --communication p2p \
  --router-mode trace --skew uniform --tokens 2048
```

### Single-GPU Expert FFN checkpoint

The optional `moe_gpu` target uses one CUDA device. The host generates the same
model and Tokens as `moe_serial`, determines routes, packs two assignments per
Token, and combines the returned Expert outputs. CUDA runs the two FFN layers
in FP32. Expert weights are uploaded once before the measured repetitions;
assignment H2D, two kernels, output D2H, host packing, host combine, and total
time are printed separately. Each measured GPU output is checked against the
CPU serial output with the documented elementwise and relative-L2 tolerances.

On an allocated Vanda A40 GPU node, load the CPU build modules and the site
CUDA toolkit, then configure a separate build directory:

```bash
module load gompi/2024a
module load CMake/3.29.3-GCCcore-13.3.0
module load cuda12.4/toolkit/12.4.1
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DMOE_ENABLE_CUDA=ON
cmake --build build-gpu --target moe_gpu -j 8
./build-gpu/moe_gpu --backend cuda --router-mode trace --skew uniform \
  --tokens 2048
```

The program reports the visible GPU count, device name, CUDA runtime, one-time
weight upload, five GPU trials, five CPU serial trials, numerical errors, and
correctness outcomes. Both backends receive three warmups before their five
timed trials. It supports up to 32768 Tokens. The GPU trials use
preallocated device buffers, whereas the CPU serial forward currently
allocates temporary buffers on each call; report this distinction. One GPU
result is a compute-backend comparison, not an MPI scaling point.

The checkpoint can also run its Expert micro-batch tasks with OpenMP. Set
`OMP_NUM_THREADS` and choose `--omp-schedule static|dynamic`; each thread has
private scratch, while MPI calls remain on the main thread. The automated
GitHub Actions CPU check compiles and exercises both one-thread and two-thread
cases on a generic Linux runner. Vanda launch, affinity, and performance must
still be tested on Vanda compute nodes.
