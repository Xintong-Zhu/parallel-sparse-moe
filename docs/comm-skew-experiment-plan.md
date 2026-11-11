# CPU communication and routing-skew experiment

The next Vanda job compares EP Alltoallv and P2P under controlled Top-2 routing
traces. It tests the effect of message pattern, Token count, and Expert load
imbalance while holding the code revision and hardware allocation fixed.

## Frozen design

- Four same-CPU-model nodes; each requests four physical cores, two MPI ranks,
  two OpenMP threads per rank, 16 GB. Eight ranks and 16 active CPU cores total.
- 16 Experts, d=128, h=256, Top-2, round-robin placement, static OpenMP schedule.
- Token counts: 32768 and 131072. Router mode is `trace`; skew is uniform,
  moderate or extreme with frozen hot layout A.
- Communication: Alltoallv or P2P; both return the same bitwise-correct output.
- The usual seeds remain model=42, input=314159, routing=271828,
  calibration=161803. Trace and model creation occur outside the timed forward.
- Two rounds cover 2 sizes × 3 skews × 2 communication methods = 12 conditions
  each. Round B reverses all three orders. Each call warms up three times and
  records five trials, yielding 120 measured forwards.

The four-node allocation is used for the whole matrix. Before timing, the job
checks all node CPU models, eight-rank/two-thread MPI launch across four hosts,
and extreme-skew correctness separately for Alltoallv and P2P. All measured
trials must print `bitwise_ok=1`. The raw Expert and rank assignment histograms
in every file provide the realized load; do not infer it from skew labels alone.

This experiment can explain observed performance differences but cannot by
itself isolate network transfer from MPI synchronization, nor identify a
hardware-level cause of outliers. Compare methods within the same Token count,
skew and round; show both rounds and trial spread. The phase times are per-stage
maxima across ranks and need not add to elapsed time. Analyze communication
time alongside Expert compute time and load imbalance.

The job uses **round-robin placement only**. The existing `la` option is parsed
but rejected by the production benchmark, so no load-aware placement result
will be claimed. That method requires its own implementation, correctness
validation and matched experiment after this job.

Submit from the Vanda login-node checkout after pulling the published commit:

```bash
bash scripts/submit_vanda_cpu_formal.sh scripts/vanda_cpu_comm_skew.pbs
```

Frozen-source metadata, job status, per-condition logs, node list and PBS
allocation are saved under ignored `private/comm-skew-<jobid>/`.
