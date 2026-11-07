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
