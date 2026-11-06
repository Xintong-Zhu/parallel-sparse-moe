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
