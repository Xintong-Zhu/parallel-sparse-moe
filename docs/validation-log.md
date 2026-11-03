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
