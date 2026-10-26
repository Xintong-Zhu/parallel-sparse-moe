# Parallel Sparse Mixture-of-Experts Inference

This project studies the parallelization of a sparse Mixture-of-Experts (MoE)
inference layer using MPI for distributed-memory communication and OpenMP for
shared-memory computation.

> **Project status:** experimental design, version 0.1. The definitions marked
> **fixed** below are the shared contract for all implementations. Numerical
> values marked **pilot** are starting points and must be calibrated on Atlas or
> Vanda before the final benchmark campaign.

## Research question

How do routing skew, communication strategy, and the MPI/OpenMP process layout
affect load balance, strong/weak scaling, and parallel efficiency in distributed
sparse MoE inference?

The project focuses on one MoE inference layer rather than a complete
Transformer or model-training pipeline. This keeps the study focused on token
routing, inter-process communication, expert computation, and straggler effects.

## Algorithm definition (fixed)

The input is a matrix of token embeddings

```text
X: [T, d_model]
```

For every token, a Top-2 router selects two distinct experts. Each expert is a
two-layer feed-forward network:

```text
Expert_e(x) = W2_e * ReLU(W1_e * x + b1_e) + b2_e
```

where the default hidden dimension is

```text
d_hidden = 4 * d_model
```

If the selected experts are `e1` and `e2`, the two routing weights are
normalized so that `g1 + g2 = 1`, and the output is

```text
y = g1 * Expert_e1(x) + g2 * Expert_e2(x)
```

The baseline uses FP32 values and ReLU. The correctness baseline does not drop
tokens and does not impose an expert-capacity limit. Capacity-limited or padded
routing may be added later as a separately labelled optimization.

## Distributed execution model (fixed)

1. Global input tokens are divided approximately evenly among MPI ranks.
2. Every rank routes its own local tokens. Routing is not centralized on rank 0.
3. Experts are assigned statically with round-robin ownership:
   `owner(expert_id) = expert_id % world_size`.
4. Each token is packed once for each of its two selected experts.
5. An `MPI_Alltoallv` dispatches token/expert assignments to the owning ranks.
6. Each rank evaluates its local expert subset. The hybrid version uses OpenMP
   inside the rank.
7. A second `MPI_Alltoallv` returns expert outputs to the tokens' original ranks.
8. Original ranks restore token order and perform the weighted Top-2 combine.

A rank may own multiple experts. “One rank equals one expert” is not assumed.
The return operation is an all-to-all exchange, not a gather to rank 0.

## Implementations to compare (fixed minimum)

| Name | MPI | OpenMP | Purpose |
|---|---:|---:|---|
| `serial` | No | No | Correctness and algorithmic speedup reference |
| `mpi` | Yes | No | Distributed communication baseline |
| `mpi_omp` | Yes | Yes | Hybrid distributed/shared-memory implementation |

At least one additional design comparison should be selected after the baseline
is working. Candidates include OpenMP static versus dynamic scheduling,
blocking versus nonblocking communication, communication/computation overlap,
or a load-aware expert-placement strategy. These are not yet fixed.

## Parameter policy

### Fixed parameters

| Parameter | Value |
|---|---:|
| Precision | FP32 |
| Top-k | 2 |
| Activation | ReLU |
| Hidden ratio | `d_hidden = 4 * d_model` |
| Default expert placement | Round-robin |
| Default token ownership | Even block distribution by global token ID |

### Pilot configuration

The following is an initial calibration workload, not a final benchmark claim:

```text
tokens       = 65,536
experts      = 32
d_model      = 256
d_hidden     = 1,024
top_k        = 2
precision    = FP32
seed         = 5,208
```

Under perfectly balanced routing, this gives an average of 4,096 token-expert
assignments per expert. It should be large enough for meaningful batching, but
its runtime and memory use must be measured on the target machine.

Candidate problem-size sweeps are:

```text
tokens:       2^14, 2^16, 2^18
d_model:      128, 256, 512
d_hidden:     4 * d_model
```

These candidates represent relatively communication-heavy, balanced, and
compute-heavy workloads. The final subset will be selected after a pilot run.
The selected main workload should:

- take long enough that timer and MPI-startup noise do not dominate;
- remain short enough to repeat reliably;
- provide at least tens to hundreds of assignments per active expert;
- remain below roughly half of the available node memory; and
- keep the same global problem and model when measuring strong scaling.

The main expert count must be at least the largest tested MPI process count.
Primary scaling configurations should preferably use process counts that divide
the expert count. Non-divisible cases belong in correctness and robustness tests.

## Data and reproducibility contract (fixed)

The primary experiments use synthetic data. No external text dataset or
pretrained model is required. Synthetic data is appropriate because the study
concerns communication and load balance, and it allows routing skew to be
controlled independently of an external model.

Input embeddings, router data, routing decisions, gate weights, and expert
weights must be deterministic. They are defined by global indices rather than
by MPI rank or random-number call order. Conceptually:

```text
X[token_id, dimension]       = f(seed, token_id, dimension)
W[expert_id, row, column]    = f(seed, expert_id, row, column)
```

The implementation should use a stateless indexed generator or hash (for
example, SplitMix64 followed by a deterministic conversion to FP32). It must
not use `seed + rank` as the only randomization rule. Consequently:

- changing the number of MPI ranks does not change the global strong-scaling
  workload;
- serial and parallel implementations receive identical logical inputs;
- different team members and cluster runs can reproduce the same case; and
- large input and weight files do not need to be copied between systems.

The global seed for the baseline is `5208`. Any additional seeds must be listed
explicitly in benchmark output.

## Routing workloads

Routing distribution is a primary experimental variable. At minimum, the study
will contain:

| Mode | Definition | Purpose |
|---|---|---|
| `balanced` | Exactly equal expert assignment counts where possible | Ideal load-balance reference |
| `uniform` | Deterministic uniform-random expert selection | Natural sampling variation |
| `zipf` | Expert probability proportional to `(expert_id + 1)^(-alpha)` | Controlled routing skew |

Candidate Zipf exponents are:

```text
alpha = 0.0, 0.5, 1.0, 1.5
```

Top-2 selections must contain two distinct experts. Gate weights must be
deterministically generated and normalized. The exact-balanced mode may use
fixed weights such as `0.6` and `0.4` for an easily checked baseline.

Every run must report the realized distribution, not only the requested
distribution parameter. Required load statistics are:

- assignments per expert and per rank;
- maximum load divided by mean load;
- standard deviation or coefficient of variation; and
- the identity and load of the slowest rank.

Two routing paths may be supported and must be labelled separately:

1. **End-to-end routing:** compute router logits and select Top-2 experts.
2. **Controlled routing:** generate Top-2 expert IDs from a specified
   distribution to isolate communication and load-balancing behaviour.

Results from these modes must not be mixed without clearly identifying the
mode. Controlled routing is the primary mode for skew experiments; end-to-end
routing demonstrates a complete MoE-layer execution path.

## Correctness requirements

All parallel outputs must be compared with the true serial implementation using
the same global input, weights, routing decisions, and gate weights.

Required numerical checks are:

- maximum absolute error;
- maximum relative error; and
- root-mean-square error (RMSE).

An initial FP32 tolerance of `1e-4` may be used for development, but the final
tolerance must be justified from the observed accumulation error.

Required structural checks are:

- every token is assigned to exactly two distinct experts;
- every dispatched token-expert assignment is returned exactly once;
- every token receives exactly two expert outputs;
- global token IDs and original order are restored correctly; and
- serial and parallel output checksums agree within the numerical tolerance.

Correctness tests must include one rank, multiple ranks, token and expert counts
that are not divisible by the rank count, highly skewed routing, two experts on
the same rank, and two experts on different ranks.

Large performance runs may use checksums and sampled element comparisons after
full element-by-element equivalence has been established on smaller cases.

## Scaling definitions (fixed)

### Strong scaling

Keep the global token count, number of experts, dimensions, seed, routing mode,
and all weights fixed while increasing the number of MPI ranks and/or OpenMP
threads.

### Weak scaling

Keep the number of input tokens per MPI rank fixed while increasing the rank
count. Keep the model definition and number of experts fixed where the selected
process range permits it. Any experiment that also increases the expert count
must be identified as model scaling rather than ordinary weak scaling.

### Hybrid layout comparison

For a fixed total number of CPU cores, compare layouts such as:

```text
1 rank  x 32 threads
2 ranks x 16 threads
4 ranks x  8 threads
8 ranks x  4 threads
```

The actual layouts will be adapted to the physical cores, sockets, and scheduler
configuration of Atlas or Vanda.

## Timing and performance metrics

Dataset/weight generation, file output, and one-time initialization are excluded
from the primary inference runtime. The timed region includes:

- routing or controlled route generation, as labelled;
- packing;
- MPI dispatch;
- local expert computation;
- MPI return;
- unpacking and weighted combine.

Each phase must also be timed separately. Timings use `MPI_Wtime()`. Global
phase time is the maximum elapsed time across ranks, because completion is
limited by the slowest rank.

The initial measurement protocol is two untimed warm-up iterations followed by
at least seven recorded iterations. The median is the primary reported value;
minimum and maximum or another variability measure should also be retained. The
same protocol must be used for every point in a comparison figure.

Report both:

```text
algorithmic speedup = true serial time / parallel time
MPI scaling speedup = one-rank MPI time / P-rank MPI time
parallel efficiency = speedup / P
```

For hybrid experiments, also report efficiency relative to the total number of
CPU cores where appropriate.

## Required run metadata

Every benchmark record must include:

```text
Git commit
host/system name
compiler and version
MPI implementation and version
compile flags
global seed
all algorithm parameters
routing mode and skew parameter
MPI rank count
OpenMP threads per rank
process/thread affinity settings
per-phase and total timings
per-rank/per-expert assignment counts
correctness result or checksum
```

The executable should eventually expose a shared command-line interface similar
to:

```text
--tokens 65536
--d-model 256
--d-hidden 1024
--experts 32
--topk 2
--routing zipf
--zipf-alpha 1.0
--seed 5208
```

MPI rank count is supplied by the MPI launcher and OpenMP thread count by
`OMP_NUM_THREADS`.

## Items still to determine before final benchmarking

- whether Atlas, Vanda, or both will be the primary experimental platform;
- physical cores, sockets, memory, and interconnect of the selected nodes;
- compiler and MPI module versions;
- maximum node and MPI-rank count available for the project;
- CPU, MPI-process, and OpenMP-thread binding policy;
- final main values of `tokens`, `experts`, and `d_model` after calibration;
- the additional optimization/design comparison beyond the three minimum
  implementations; and
- the final run matrix and job-time budget.

These choices must be frozen before collecting the results used in the report.
Changing them after measurements begin requires either rerunning the affected
configurations or clearly treating the new measurements as a separate study.

## Build and run instructions

Source code and build/run commands will be added after the experimental contract
is agreed and the initial implementation is created.
