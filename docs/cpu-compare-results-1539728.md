# Vanda serial, EP, and DP comparison — job 1539728

The job finished successfully (`exit_status=0`, `stage=complete`). The analysis
script validated all 23 points and 115 bitwise-correct timed trials before
producing `results/processed/vanda-cpu-compare-1539728.csv`. Each point used
three warmups and five measured repetitions. The four nodes were CN-040,
CN-046, CN-053 and CN-058, all Intel Xeon Platinum 8452Y. The four-node and
256-rank probes passed. Source revision: `cbac4395a17627746aa88fc307120d2efcb9f56a`.

## Same-job results

All strong-scaling points use 131,072 Tokens, 256 Experts, Top-2 uniform trace,
one bound CPU core per MPI rank, and the same frozen source. Times below are
medians of five forwards in seconds. The serial baseline is 2.336019 s.

| CPU cores / ranks | EP (s) | DP (s) | EP speedup vs serial | DP speedup vs serial |
|---:|---:|---:|---:|---:|
| 1 | 2.237410 | 1.980180 | 1.04 | 1.18 |
| 2 | 1.123870 | 0.989278 | 2.08 | 2.36 |
| 4 | 0.572101 | 0.497932 | 4.08 | 4.69 |
| 8 | 0.296107 | 0.246675 | 7.89 | 9.47 |
| 16 | 0.156376 | 0.129383 | 14.94 | 18.06 |
| 32 | 0.099499 | 0.080802 | 23.48 | 28.91 |
| 64 | 0.052538 | 0.041430 | 44.46 | 56.38 |
| 128 | 0.033916 | 0.022225 | 68.88 | 105.11 |
| 256 | 0.068735 | 0.014842 | 33.99 | 157.39 |

Serial workload size (`N` Tokens → median forward time): 512 → 0.016665 s;
2,048 → 0.057831 s; 8,192 → 0.226807 s; 32,768 → 0.551503 s;
131,072 → 2.336019 s. The final fourfold size increase takes 4.24 times
as long, consistent with nearly linear growth in the large-input regime.

The best measured EP point is 128 cores. EP at 256 cores takes 2.03 times as
long as EP at 128. Its median maximum dispatch communication phase rises from
0.00629 to 0.02828 s; the return phase rises from 0.00772 to 0.03118 s.
The Expert compute phase at 256 cores is approximately 0.01168 s, so the
communication phases are individually larger. These phase measurements are
**maximum values across ranks per phase**. Their maxima can occur on different
ranks and must not be added to claim an exact end-to-end percentage. The
separate, earlier formal EP run (job 1536027) also shows a 128-to-256-core
regression, supporting this observed scaling limit.

DP continues to improve to 256 cores and is 4.63 times faster than EP at that
point for this implementation and workload. Its forward computation is local
because each rank owns a Token subset and has a full copy of the weights.
The timed DP forward does **not** gather the global output, and its setup,
model replication, route preparation, serial correctness oracle, and final
validation are outside the timer. EP's timed forward includes its assignment
dispatch and return communication. Thus these numbers compare the actual
measured forward paths, not identical full application pipelines. DP's weight
replication also uses more total memory than EP. The label `alltoallv` printed
for DP is a parsed option; DP does not perform Token all-to-all communication
inside its timed forward.

The serial benchmark is bound to one physical core. It generates weights,
inputs and the trace outside timing. Serial forward copies and validates the
precomputed routes inside timing, while EP/DP prepare their local route arrays
before timing. Serial-relative speedup and efficiency use the same-job serial
median: speedup = 2.336019 / parallel median; efficiency = speedup / cores.
At 256 cores those efficiencies are 13.28% for EP and 61.48% for DP. The
processed CSV also provides within-mode one-rank baselines. Values above 100%
at small core counts are possible because the serial and MPI paths have
different overheads; they are not evidence of superlinear scaling of one
identical implementation.

## Reproduce the analysis

```bash
python3 scripts/analyze_cpu_compare.py private/compare-cpu-1539728 \
  --output results/processed/vanda-cpu-compare-1539728.csv
python3 scripts/plot_cpu_compare.py results/processed/vanda-cpu-compare-1539728.csv
```

The plot is produced as PNG and vector PDF under the tracked `figures/`
directory. Raw logs stay in ignored `private/compare-cpu-1539728/`.
