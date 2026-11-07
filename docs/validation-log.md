# Correctness validation log

This log records observed correctness checks. It is not a performance dataset.

## 2026-11-03: Vanda single-node hybrid CPU check

- Target commit: `3abfc3a`. GitHub Actions passed; the compute-node `HEAD`
  output was not retained in the conversation.
- Allocation: one Vanda CPU node, two MPI processes, two OpenMP threads per process.
- Launch probe evidence: screenshot from node `CN-092` shows rank 0 bound to
  package 1 cores 1 and 3, rank 1 to package 1 cores 5 and 7. Both ranks
  observed two OpenMP threads and passed the ring check. MPI provided
  `MPI_THREAD_FUNNELED`; probe printed `RESULT: PASS`.
- User reported `PASS` for EP uniform trace with static schedule, EP moderate
  trace with dynamic schedule, and DP router mode with static schedule, all at
  `P=2`, `N=2048`, `OMP_NUM_THREADS=2`.
- This check validates launch and output correctness for those cases. No timed
  repetitions or scaling claims were made.

## 2026-11-06: Vanda P2P CPU check

- Target commit: `273e5cc`. The user confirmed its GitHub Actions run passed.
- On a Vanda two-process, two-thread allocation, the user reported `PASS` for
  EP uniform trace with Alltoallv, EP uniform trace with P2P, EP moderate trace
  with P2P and dynamic OpenMP schedule (all `N=2048`), and a P2P case with
  `N=1`, `d=4`, `h=8`, `E=4`. The exact compute-node `HEAD` and raw terminal log
  were not retained in the conversation.
- These are correctness checks. No timing or scaling data were collected.

## 2026-11-07: Vanda partitioned EP and CPU pilot

- On `CN-152`, the user reported EP P2P `PASS` at `P=2`, `T=2`, `N=2048`,
  uniform trace, with `diagnostic_setup_seconds_max=0.040928` and
  `diagnostic_forward_seconds_max=0.0260679`.
- The user supplied five EP P2P and five DP pilot measurements for the same
  synthetic input. All ten had `bitwise_ok=1`; the raw records are in
  `results/raw/vanda-pilot-2026-11-07-cn152-p2x2.csv`.
- The user subsequently confirmed the compute-node commit was `24b72ef`.
  Pilot measurements include temporary allocations and
  validation collectives and must not enter formal speedup figures.
- The same allocation later produced 15 additional pilot records at `N=32768`:
  EP Alltoallv, EP P2P, and DP, five trials each. All passed bitwise checks.
  Median elapsed times were 0.317387 s, 0.305473 s, and 0.259568 s,
  respectively. These are exploratory values and not formal comparisons.

## 2026-11-07: Vanda single-A40 GPU checkpoint

- The user supplied the full output of `moe_gpu` for `N=2048`, `E=16`,
  controlled uniform trace on one visible NVIDIA A40. The CUDA runtime was
  12.4; the GPU node hostname and `git HEAD` were not included in the pasted
  output. The requested target commit was `c3144fa`.
- The realized histogram was 256 assignments for each of 16 Experts. All five
  GPU trials passed against the CPU serial result with zero reported absolute
  and relative-L2 error. Five CPU serial repeats also passed. Raw values are
  preserved in `results/raw/vanda-gpu-pilot-2026-11-07-a40-n2048.csv`.
- Median GPU end-to-end was 0.0013191 s, including host routing/packing,
  assignment H2D, GPU FFN, output D2H and host combine; median CPU serial was
  0.0468374 s. The quotient is 35.5, an exploratory comparison only. The CPU
  repeats varied from 0.0302744 to 0.0478187 s, and the GPU path reuses device
  buffers while the CPU serial path allocates temporary buffers per call.
- One-time model-weight H2D was 0.00163943 s and excluded from the repeated
  GPU timings. No claim about MPI scaling follows from this one-GPU test.

## 2026-11-07: Vanda A40 GPU size pilots

- The user supplied five GPU and five CPU serial trials each at `N=8192` and
  `N=32768`, with `E=16`, controlled uniform trace and one visible NVIDIA A40.
  The user subsequently confirmed compute-node `HEAD=5e46f9e`, which includes
  three CPU warmups. The earlier screenshot of this GPU job showed node
  `GN-A40-034`; the hostname was not repeated in the timing output. The raw
  records are in `results/raw/vanda-gpu-pilot-2026-11-07-a40-n8192-n32768.csv`.
- All GPU trials matched CPU serial output with reported maximum absolute and
  relative-L2 errors of zero. All CPU serial repeats also passed. At `N=8192`
  each Expert received 1024 assignments; at `N=32768`, each received 4096.
- Median GPU end-to-end times were 0.00431691 s and 0.0180814 s at `N=8192`
  and `N=32768`; median CPU serial times were 0.148643 s and 0.569162 s.
  Median quotients were 34.4 and 31.5, respectively. These compare distinct
  CPU and GPU implementations and exclude the one-time GPU weight upload.
- At `N=32768`, the median GPU kernel time was about 0.00700 s and the median
  transfer times were about 0.00259 s H2D plus 0.00222 s D2H. CPU host packing
  and combining each took about 0.0031 s. These phases explain why the full
  GPU pass is materially longer than the kernel alone.
- The user then repeated `N=2048` with the same `5e46f9e` warmup protocol.
  Five GPU and five CPU serial trials all passed; the GPU reported zero error.
  Median GPU end-to-end was 0.00117653 s and median CPU serial was 0.0375016 s,
  a quotient of 31.9. One-time weight upload was 0.000420593 s, excluded from
  trials. Raw values are in
  `results/raw/vanda-gpu-pilot-2026-11-07-a40-n2048-warmed.csv`.
- For the three same-protocol sizes `N=2048,8192,32768`, the GPU end-to-end
  medians are 0.00117653, 0.00431691, and 0.0180814 s. All pass correctness;
  the original `N=2048` run on `c3144fa` is retained separately for provenance
  but should not be mixed into the same-protocol comparison.

## 2026-11-07: Vanda CPU benchmark checkpoint at E=16 and E=256

- On `CN-152`, the user ran compute-node commit `1036672` with 2 MPI ranks,
  2 OpenMP threads each, CPU EP, P2P, `N=32768`, controlled uniform trace,
  and both `E=16` and `E=256`. All ten measured trials had `bitwise_ok=1`.
  Raw phase timings are in
  `results/raw/vanda-cpu-bench-2026-11-07-cn152-p2x2-n32768.csv`.
- Median end-to-end times were 0.268495 s at E=16 and 0.272678 s at E=256,
  a 1.6% difference at this single resource layout. Expert compute-stage
  medians were 0.225531 s and 0.231224 s, respectively. This stage includes
  grouping/tasks and temporary allocations, not only FFN arithmetic.
- Return-communication stage values varied substantially (E=16:
  0.00986646–0.0362627 s; E=256: 0.00981214–0.0232019 s). All values are
  maxima across ranks for their respective phases; do not add phase maxima
  to explain the end-to-end maximum. The benchmark prints no realized Expert
  histogram yet, so these are size/implementation pilots, not the final
  placement or scaling dataset.
- In the same `CN-152` allocation and binary, the user then ran CPU EP with
  Alltoallv, `E=256`, `N=131072`, 2 MPI ranks and 2 OpenMP threads. All five
  trials passed bitwise checks. End-to-end values ranged from 0.743594 to
  0.763439 s, with median 0.755526 s; the median Expert compute stage was
  0.607352 s. Raw values are in
  `results/raw/vanda-cpu-bench-2026-11-07-cn152-p2x2-n131072-e256.csv`.
  This is a stable pilot for a possible strong-scaling global Token count;
  the one-rank baseline and larger-rank behavior still need checking.
- On the same `CN-152` allocation and binary, the user ran the matching
  `N=131072`, `E=256`, Alltoallv EP case at 1 MPI rank × 1 OpenMP thread. All
  five trials passed bitwise checks. The end-to-end median was 2.60025 s
  (range 2.5312–2.65165 s). Compared with the 2-rank × 2-thread median
  0.755526 s, the exploratory four-core speedup is 3.44 and efficiency is
  about 86.0%, using the one-core EP forward as the baseline. P=1 exchange
  stages are local copies/packing rather than network traffic. Raw values are
  in `results/raw/vanda-cpu-bench-2026-11-07-cn152-p1x1-n131072-e256.csv`.

## 2026-11-07: Vanda weak-scaling start on GN-A40-047

- The user confirmed compute-node commit `79234ab` and CPU model Intel Xeon
  Platinum 8452Y. PBS assigned `GN-A40-047` as a CPU-only job with 36 cores,
  2 MPI slots, and no GPU requested. This is a different node from the
  `CN-152` strong-scaling pilots; its CPU model has not been compared with
  `CN-152`.
- With CPU EP, Alltoallv, controlled uniform trace and E=256, the user ran
  1 rank × 1 thread at N=4096 and 2 ranks × 1 thread at N=8192: 4096 Tokens
  per rank. All ten measured trials had `bitwise_ok=1`.
- The program printed 32 assignments per Expert and 8192 per rank at P=1,
  then 64 assignments per Expert and 8192 per rank at P=2. This is the realized
  Top-2 load, not just the requested distribution.
- Median end-to-end times were 0.0790673 s at P=1 and 0.083977 s at P=2,
  giving exploratory weak-scaling efficiency 0.9415 when defined as T1/T2.
  The P=1 range was 0.0746002–0.0823059 s, so the short baseline has some
  variability. Raw data are in
  `results/raw/vanda-cpu-weak-pilot-2026-11-07-gn-a40-047.csv`.
- The pilot supports freezing 4096 Tokens per rank for the planned weak curve;
  the protocol and the fixed N=131072 strong curve are recorded in
  `docs/formal-cpu-experiment-config.md`. Formal measurements still require
  a homogeneous multi-node allocation and a 256-rank pilot.
