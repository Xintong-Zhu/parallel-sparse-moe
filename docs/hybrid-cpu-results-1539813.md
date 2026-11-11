# Fixed 256-core hybrid experiment: job 1539813

Validated source: `86540c669ac508a2e054c7f18a94ed041872410a`.
Job exit status is zero and stage is complete. Five affinity layouts and all
100 measured forwards (20 runs, five trials each, three warmups per run) passed.
All configurations use exactly the same 256 distinct physical cores: 64 per
node, 32 per socket. No threads share a physical core and no rank spans sockets.
The allocation reserves 288 CPU cores; 256 are used by the computation.

The allocation contains CN-043, CN-064, CN-068 and GN-A40-089. All report Intel
Xeon Platinum 8452Y. This is a CPU-only experiment; a GN node does not imply GPU
use. The mixed CN/GN node types must be disclosed: the same CPU model does not
prove every other hardware characteristic identical. All within-job comparisons
use the same nodes and physical core set. Do not combine timings with earlier
allocations as though they were one controlled series.

## Results

Each entry below is the median of five timed forwards, in milliseconds.
Input is the frozen uniform Top-2 trace with 131072 Tokens, 256 Experts,
d=128, h=256 and the usual four seeds. EP uses Alltoallv and RR placement.

| MPI x OpenMP | EP round A (ms) | EP round B (ms) | DP round A (ms) | DP round B (ms) |
|---|---:|---:|---:|---:|
| 256 x 1 | 69.073 | 67.245 | 14.281 | 14.434 |
| 128 x 2 | 26.109 | 27.810 | 20.913 | 13.424 |
| 64 x 4 | 39.430 | 38.558 | 13.819 | 13.211 |
| 32 x 8 | 36.634 | 36.759 | 13.719 | 14.361 |
| 16 x 16 | 45.761 | 46.217 | 14.271 | 15.029 |

EP 128x2 is best among the five measured configurations in both rounds. Relative
to the same-round EP 256x1 baseline, its gain is 2.646x in A and 2.418x in B.
Its two medians differ by about 6.5%, below the predeclared 10% drift threshold.
Thus the data support a repeatable benefit from this hybrid configuration for
this fixed workload; they do not establish a universal best MPI/thread ratio.

EP dispatch communication falls from roughly 27.6–28.2 ms at 256x1 to 5.20–5.24 ms
at 128x2, and return communication from 30.4–30.7 ms to 10.0–10.5 ms. The compute
phase rises from 11.5–11.7 ms to 14.6–15.8 ms, but overall time decreases. This
supports a communication/computation tradeoff. The phases are maxima across
ranks; they must not be summed to assign an exact percentage of end-to-end time.
Further reducing ranks does not monotonically improve runtime: larger messages,
packing/combination work and local threading costs may contribute. The data do
not directly measure cache misses, NUMA traffic, or network contention.

EP 32x8 has pronounced within-round slow trials despite similar round medians:
round A trial 4 takes 252.195 ms, and round B trial 0 takes 250.882 ms. Both are
roughly seven times the corresponding median. Dispatch and return phase maxima
in these trials rise to about 222–226 ms; the cause of the delay is not identified.
Retain these observations and min/max error bars. Similar round medians alone
are insufficient to call this configuration stable, and the 128x2 recommendation
concerns the best observed median, not a guaranteed latency bound.

DP 128x2 is unstable: its round-B median is 35.8% lower than round A, exceeding
the planned 10% threshold, and its relative performance reverses (0.683x vs
1.075x its same-round 256x1 baseline). Retain both rounds and flag this point;
do not select only the faster measurement or hide instability with a pooled
median. DP's fastest configuration also changes between rounds (32x8 in A,
64x4 in B), with most other configurations around 13–15 ms. No robust best DP
configuration is established by this job. If resolving the DP comparison is
needed, repeat 256x1, 128x2 and 64x4 together under one allocation in alternating
orders, preserving these original measurements.

## Interpretation and timing scope

Speedup here is T(mode, same round, 256x1) / T(mode, same round, P x T).
It measures a fixed-core configuration gain, not a conventional strong-scaling
efficiency, so do not divide it by 256. EP and DP each have their own baseline.
Weights, input generation, precomputed routes, serial oracle and correctness
verification are outside the timer. DP output remains distributed; no global
output gather is timed. EP includes its necessary dispatch/return exchanges.
The printed alltoallv option for DP does not mean DP performs an all-to-all
inside its forward. These results concern this synthetic one-layer workload.

Completed project evidence now includes serial size dependence, EP strong/weak
scaling through 256 cores, paired DP/EP strong scaling, single-GPU measurements,
and the five fixed-core MPI/OpenMP configurations. Communication/skew controls,
load-aware placement (implementation and correctness first), and the final
report/submission review remain separate outstanding work.

## Reproduction

```bash
python3 scripts/analyze_cpu_hybrid.py private/hybrid-cpu-1539813 \
  --output results/processed/vanda-cpu-hybrid-1539813.csv
python3 scripts/plot_cpu_hybrid.py results/processed/vanda-cpu-hybrid-1539813.csv
```

The CSV retains both rounds, timing ranges, phase medians, baseline gains,
drift flags and provenance. PNG/PDF figures are generated under `figures/`.
Raw cluster logs remain under ignored `private/hybrid-cpu-1539813/`.
