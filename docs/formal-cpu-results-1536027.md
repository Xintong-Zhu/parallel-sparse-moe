# Validated EP scaling: Vanda job 1536027 (2026-11-10)

The full original logs are retained locally under the ignored
`private/formal-cpu-1536027/` directory. Job status was `exit_status=0` and
`stage=complete`. The validated compact results are in
`results/processed/vanda-cpu-formal-1536027.csv`; figures can be regenerated
with `scripts/plot_cpu_formal.py`.

- Source commit: `9e762f8a7107a619ea00007038624649daf58ef1`.
- Nodes: CN-054, CN-067, CN-070, CN-073. Each reported Intel Xeon Platinum
  8452Y. PBS allocated 64 physical cores/MPI slots on each of four nodes.
- C++ compiler: GCC 13.3.0; MPI: OpenMPI 5.0.3; one OpenMP thread per rank.
- Workload: one-layer Top-2 MoE, d=128, h=256, E=256, uniform controlled
  trace, round-robin placement, Alltoallv, static OpenMP scheduling.
- Strong scaling fixes global N=131072; weak scaling fixes 4096 Tokens/rank.
  Both include P=1,2,4,8,16,32,64,128,256 ranks. Each point had three
  warmups and five measured trials.
- Four-node and 256-rank launch probes passed. The analysis program checked
  every raw configuration line, seed, realized Expert/rank histogram, trial
  count, finite timing and `bitwise_ok=1`: **18 points and 90 trials passed**.

| Series | P=1 median | P=128 median | P=256 median | P=256 efficiency |
|---|---:|---:|---:|---:|
| Strong, fixed N=131072 | 2.30227 s | 0.0407717 s | 0.0703977 s | 12.77% |
| Weak, 4096 Tokens/rank | 0.069415 s | 0.131475 s | 0.177394 s | 39.13% |

Strong speedup uses the **one-rank EP** forward as its baseline: 56.47x at
P=128 and 32.70x at P=256. Strong efficiency is `T1/(P*TP)`. Weak efficiency
is `T1/TP`. These are parallel-program baselines; a separate CPU serial
baseline is still needed for the report's serial-relative speedup.

The P=256 strong run is about 72.7% slower than P=128 despite twice as many
cores. Its median Expert compute-stage maximum fell from 23.23 to 12.42 ms,
while dispatch communication rose from 7.52 to 28.65 ms and return
communication from 12.40 to 31.97 ms. This supports communication/coordination
overhead as the likely cause of the turnover. The phase values are independent
maxima across ranks; they must **not** be added to reconstruct end-to-end
time. The uniform trace gave exactly 1024 assignments per Expert and rank at
P=256 in the strong case, so the observed turnover is not explained by the
constructed Expert load distribution.

In weak scaling, time grew from 69.4 to 177.4 ms although the work per rank
was fixed. At P=256 the median Expert compute-stage maximum was 91.39 ms,
dispatch communication 38.59 ms, and return communication 54.10 ms. These
numbers show that both the measured compute-stage maximum and exchange stages
grew at scale. A follow-up profile would be needed to attribute the causes.

This series establishes EP scaling to 256 cores. It does not provide a DP
curve, the true-serial speedup denominator, a hybrid MPI/OpenMP comparison,
or evidence for load-aware placement. Those require separately verified data.
