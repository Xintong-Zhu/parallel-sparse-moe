# Formal single-A40 GPU experiment plan

## Research question and connection to the CPU studies

The CPU experiments show that EP performance is affected by communication and
rank-level load imbalance. This experiment asks how the same frozen Top-2 MoE
workloads behave when Expert FFNs run on one A40 GPU: how end-to-end time scales
with Token count, how much time is spent in host work and transfers rather than
CUDA kernels, and whether controlled routing skew changes single-device time.

This is a one-GPU compute-backend study, not GPU EP or MPI scaling. It shares the
E=16 model and controlled traces used by the CPU communication/skew and placement
studies, but results from different allocations are not used as paired speedups.

## Frozen matrix

- One allocated NVIDIA A40, one process, one visible GPU. Run a matched CPU
  serial reference on the same allocated node.
- One-layer Top-2 MoE; E=16, d=128, h=256, FP32, fixed seeds
  (model=42, input=314159, routing=271828, calibration=161803).
- Trace mode, with N in {2048, 8192, 32768} and four trace/layout cases:
  uniform-A, moderate-A, moderate-B, extreme-A. This is 12 conditions.
- For each condition, run two rounds. Round B reverses the complete condition
  order. The existing executable uses three GPU warmups, five GPU timed trials,
  three CPU warmups, and five CPU timed trials per invocation. Thus the complete
  matrix yields 24 condition logs, 120 GPU trials and 120 CPU serial trials.
- Use one frozen source snapshot and the same A40 allocation for the full
  matrix. Keep GPU clocks and power settings at site defaults; record the device,
  node, driver/CUDA runtime, compiler, modules, affinity and source commit.

The 32768-Token uniform case is the primary anchor. The three uniform sizes
measure size dependence. Within each size, the four matched traces measure skew
and hotspot-layout sensitivity. The moderate-A/B pair has the same shares but
different hot Expert IDs; it is a control for conclusions based on one layout.

The current `moe_gpu` executable rejects N > 32768, so this matrix deliberately
does not include the 131072-Token CPU EP configuration.

## Timing and correctness contract

For every GPU trial, retain host routing/packing, assignment H2D, two FFN
kernels, output D2H, host combine and end-to-end time. Record one-time model
weight H2D separately; it is excluded from repeated forward timings. Calculate
the non-kernel share from the measured end-to-end and kernel times, and show
the component times without claiming that independently measured components
sum exactly to the wall-clock total.

For the CPU serial reference, report its own measured forward time and identify
it as a comparison of distinct implementations. The CPU serial path allocates
temporary buffers per call, while the GPU path reuses device buffers. Do not
label their ratio as a hardware-only speedup. Do not compare it numerically with
MPI strong-scaling speedup.

Accept a condition only if all five trials in both rounds are present, all
routes/Expert assignments match the CPU reference, every GPU output meets the
existing elementwise and relative-L2 tolerance, every CPU repeat passes, and
all times are finite and positive. Verify the realized Expert histogram for
every condition. Save separate stdout and stderr, the job exit status and the
full trial-level logs. Failed or incomplete jobs contribute no report points.

## Analysis and report figures

1. Plot median GPU end-to-end and CPU serial time against N for uniform-A,
   showing the range over trials and both rounds. The CPU/GPU ratio, if shown,
   is explicitly an implementation-level comparison.
2. Plot GPU end-to-end breakdown against N for uniform-A: host pack, H2D,
   kernel, D2H and host combine. Mark the one-time weight upload separately.
3. At N=32768, compare the four traces for GPU end-to-end and kernel time;
   display the realized Expert histogram or imbalance alongside the times.
   Extend the supplementary table to all three N values.

Interpret the skew comparison beside the CPU E=16 communication/skew and
placement studies: one GPU has no inter-rank dispatch or rank-level imbalance,
while CPU EP does. Similarities or differences are descriptive because the
hardware, implementation and allocation differ. This study tests whether
offloading shifts the measured bottleneck toward host staging and transfers;
it does not assume that result in advance.

The formal batch runner should save `private/gpu-formal-<jobid>/` logs and
metadata. After a successful strict analysis, write a processed CSV under
`results/processed/`, a results note under `docs/`, and PNG/PDF figures under
`figures/`. Do not substitute the existing terminal-transcribed pilot CSVs for
this formal job.

## Run and analyze

The runner, strict analyzer and plotter are in `scripts/`. After the scripts are
committed and available in the Vanda login-node checkout, run:

```bash
git pull --ff-only
bash scripts/submit_vanda_gpu_formal.sh
```

The wrapper prints a frozen source path, Git commit and PBS job ID. Once the
job finishes, inspect `private/gpu-formal-<jobid>/job_status.txt`. Sync that
complete directory locally and run:

```bash
python scripts/analyze_gpu_formal.py private/gpu-formal-<jobid> \
  --output results/processed/vanda-gpu-formal-<jobid>.csv
python scripts/plot_gpu_formal.py results/processed/vanda-gpu-formal-<jobid>.csv
```

The analyzer rejects incomplete jobs or trials, incorrect routing loads,
incorrect GPU identity, non-finite times and correctness failures before
writing a processed CSV. The plotter produces three PNG/PDF figure pairs. No
new GPU inference kernel is needed for this experiment.
