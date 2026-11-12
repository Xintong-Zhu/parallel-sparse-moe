# Experiment inventory (2026-11-12)

The CPU formal jobs below completed with PBS exit status zero. Their analyzers
checked configuration, provenance, trial counts and correctness before writing
the processed CSV files. Full cluster logs remain under ignored `private/` and
are not part of the GitHub source or submission package. The GPU results are
correctness-checked single-A40 measurements transcribed from Vanda terminal
output; they were not collected by the CPU formal PBS harness.

| Study | Scale and completed evidence | Analysis | Data | Figure |
|---|---|---|---|---|
| Serial problem size | 512, 2,048, 8,192, 32,768, 131,072 Tokens; five trials each, same job as EP/DP (`1539728`) | [Serial/EP/DP results](cpu-compare-results-1539728.md) | [CPU comparison CSV](../results/processed/vanda-cpu-compare-1539728.csv) | [Comparison chart](../figures/cpu-compare-1539728.png) |
| EP strong and weak scaling | 1–256 MPI ranks × 1 thread on 4 nodes; 18 points, 90 correct trials (`1536027`); strong fixed 131,072 Tokens, weak 4,096 Tokens/rank | [Scaling results](formal-cpu-results-1536027.md) | [Scaling CSV](../results/processed/vanda-cpu-formal-1536027.csv) | [Strong time](../figures/cpu_strong_time.png), [speedup](../figures/cpu_strong_speedup.png), [efficiency](../figures/cpu_efficiency.png), [weak time](../figures/cpu_weak_time.png) |
| Same-job serial vs EP vs DP | 256 Experts, 131,072 Tokens; serial baseline and 1–256 MPI ranks; 23 points, 115 correct trials (`1539728`) | [CPU comparison results](cpu-compare-results-1539728.md) | [CPU comparison CSV](../results/processed/vanda-cpu-compare-1539728.csv) | [Comparison chart](../figures/cpu-compare-1539728.png) |
| MPI + OpenMP at fixed 256 cores | 256×1, 128×2, 64×4, 32×8, 16×16; EP and DP; two reversed rounds, 100 correct trials (`1539813`) | [Hybrid results](hybrid-cpu-results-1539813.md) | [Hybrid CSV](../results/processed/vanda-cpu-hybrid-1539813.csv) | [Hybrid chart](../figures/cpu-hybrid-1539813.png) |
| EP communication and route skew | Alltoallv vs P2P; uniform/moderate/extreme traces; 8 ranks × 2 threads; 24 points, 120 correct trials (`1542686`) | [Communication/skew results](comm-skew-results-1542686.md) | [Communication/skew CSV](../results/processed/vanda-cpu-comm-skew-1542686.csv) | [Communication/skew chart](../figures/cpu-comm-skew-1542686.png) |
| Expert placement | RR vs calibrated load-aware placement, 4/8 ranks × 2 threads, two sizes, two communication methods and reversed rounds; 80 logs, 400 correct trials (`1543859`) | [Placement results](placement-results-1543859.md) | [Placement CSV](../results/processed/vanda-cpu-placement-1543859.csv) | [Forward time](../figures/cpu-placement-times-1543859.png), [load and traffic](../figures/cpu-placement-load-1543859.png) |
| One-GPU Expert FFN | One NVIDIA A40; 2,048/8,192/32,768 Tokens, 16 Experts; five GPU and five CPU trials per size, all PASS | [GPU validation and timing notes](validation-log.md#2026-11-07-vanda-single-a40-gpu-checkpoint) | [2,048 raw CSV](../results/raw/vanda-gpu-pilot-2026-11-07-a40-n2048-warmed.csv), [8,192/32,768 raw CSV](../results/raw/vanda-gpu-pilot-2026-11-07-a40-n8192-n32768.csv) | No report-ready GPU figure yet; timing breakdown is in the raw CSV and notes |

## Main findings and boundaries

- The 256-core EP strong run is slower than the 128-core point: coordination
  and communication increase as per-rank work shrinks. The separate same-job
  comparison finds DP faster at 256 cores for this replicable teaching-size
  model. DP does not time a global output gather and duplicates all weights;
  the two forward paths have different end-to-end boundaries.
- At fixed 256 cores, EP 128×2 beats EP 256×1 in both hybrid rounds. DP 128×2
  drifts across rounds, so no stable best DP split is claimed.
- Alltoallv and P2P have similar total time in the controlled RR experiment.
  Route skew and the chosen Expert owner map have a much larger measured effect.
- Load-aware placement substantially helps when RR piles hot Experts on one
  rank. It adds little with uniform load or a hotspot layout already balanced
  by RR. The 8/16-core placement study is separate from the 256-core scaling
  study. Its exact-quota synthetic trace makes calibration unusually accurate.
- One GPU passes the requested correctness and timing test. The CPU/GPU speed
  ratios compare different implementations and exclude one-time GPU weight
  upload, so they are not MPI scaling results.

## Work remaining for final submission

1. Write and render the report PDF: explain Token, Router, Expert, DP and EP;
   define each timing boundary and baseline; include the figures, validation,
   synthetic-data generation, limitations and the specific uses of AI.
2. Reproduce a small set of key commands from a clean source checkout and
   prepare the final source/README package without generated logs.
3. Archive the full trial-level CPU raw records outside the source package so
   that every processed median can be audited. The current local `private/`
   directories hold those records.

## Optional extensions, in priority order

1. Collect the single-GPU comparison in one scripted, versioned PBS job and
   make a report-ready transfer/kernel/end-to-end figure. The existing A40
   measurements already meet the single-GPU test requirement.
2. Repeat the unstable DP 128×2 hybrid condition in alternating order and
   investigate the two large EP 32×8 outliers.
3. Measure DP with a timed global output gather and report replicated weight
   memory, so the DP/EP application-level tradeoff is clearer.
4. Test load-aware placement with stochastic or drifting routes and include
   calibration cost for one-shot inference. The current controlled trace uses
   exact Expert quotas.
5. Profile network waits, NUMA and CPU cache effects around the 128-to-256-core
   EP turnover. Multiple GPUs and CUDA-aware MPI remain optional extensions.
