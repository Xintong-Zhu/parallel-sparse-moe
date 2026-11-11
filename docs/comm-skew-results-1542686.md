# EP communication and route-skew results: Vanda job 1542686

Job status: `exit_status=0`, `stage=complete`. The validation script accepted
all 24 conditions, 120 bitwise-correct trials, frozen parameters, four same-model
nodes, an eight-rank/two-thread launch probe, extreme-skew correctness preflights,
and identical realized Expert/rank histograms for matched Alltoallv/P2P cases.
The frozen source was `206d0705f5fcaeddc0753fb353d8a61fcf571cf0`.

The workload has one MoE layer, 16 Experts, Top-2 routing and RR placement.
Eight MPI ranks with two OpenMP threads each ran across four nodes. Input sizes
are 32768 and 131072 Tokens. All numbers below are five-trial median forward
times in milliseconds; rounds A and B reversed the order of all conditions.

| Tokens | Trace | Alltoallv A | P2P A | Alltoallv B | P2P B | Realized rank assignments (min–max) |
|---:|---|---:|---:|---:|---:|---:|
| 32768 | Uniform | 92.97 | 95.18 | 93.51 | 94.49 | 8192–8192 |
| 32768 | Moderate | 202.32 | 203.33 | 199.31 | 196.99 | 4369–19660 |
| 32768 | Extreme | 265.71 | 277.03 | 266.27 | 267.74 | 5243–28835 |
| 131072 | Uniform | 351.22 | 346.26 | 348.84 | 344.82 | 32768–32768 |
| 131072 | Moderate | 751.49 | 752.86 | 757.36 | 750.45 | 17476–78644 |
| 131072 | Extreme | 719.25 | 720.34 | 685.10 | 715.36 | 20971–115343 |

The two communication implementations are close in total time within matched
conditions. The largest difference between paired medians is about 4.4%, and
the faster method changes with condition/round. This job does not establish a
general P2P advantage. It does show that both methods preserve bitwise output
under the same controlled routes, including extreme skew.

Routing skew has a much larger effect. At N=32768, uniform Alltoallv is about
93 ms; moderate is about 200 ms (2.1x), extreme about 266 ms (2.9x). Under
extreme skew, one rank receives 28835 assignments, 3.52 times the uniform
per-rank load of 8192. At N=131072, moderate and extreme are both roughly twice
the uniform time, but moderate is slower than extreme despite a smaller peak
rank count. Assignment count alone does not predict all work: Expert identity,
locality, message pattern and synchronization can matter. This experiment does
not measure them separately, so no causal explanation for that reversal is
claimed.

The Expert histogram and rank histogram in the processed CSV preserve the
actual realized counts. For example, N=32768 extreme has Expert loads ranging
from 2621 to 26214 assignments and rank loads from 5243 to 28835. The rank
counts are the sums of Experts mapped to each rank by RR placement. All matched
method and round pairs use exactly the same histograms.

Return communication maxima rise sharply under skew; for N=32768 Alltoallv,
the round-A median maximum return phase is about 10.6 ms for uniform, 149 ms
for moderate and 205 ms for extreme. This phase includes waiting for slower
ranks to reach the exchange, so it is **not** a pure network transfer time.
All stage values are rankwise maxima and cannot be added to obtain an exact
fraction of end-to-end time.

The earlier job 1540626 timed out on its first Alltoallv point after 600 s.
Diagnostic job 1542661 completed the same N=32768 configuration in about 0.107 s
and passed all four diagnostic size/method cases. Job 1542686 completed all
conditions. The earlier stall was not reproduced; its cause remains unknown.
The failed run contributes no performance data to the table or plots.

The benchmark still supports RR placement only. Load-aware placement has not
been implemented or measured; report it as outstanding work, not as a result.

Recreate the validated summary and figures:

```bash
python3 scripts/analyze_cpu_comm_skew.py private/comm-skew-1542686 \
  --output results/processed/vanda-cpu-comm-skew-1542686.csv
python3 scripts/plot_cpu_comm_skew.py results/processed/vanda-cpu-comm-skew-1542686.csv
```

Raw data remain in ignored `private/comm-skew-1542686/`. The processed CSV and
this document are suitable for the report; PNG/PDF figures are in `figures/`.
