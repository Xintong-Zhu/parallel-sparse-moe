# Frozen CPU scaling protocol (2026-11-07)

This file fixes the workload before the 256-core runs. If a resource or memory
limit forces a change, rerun the entire affected series under a new documented
configuration; do not mix points from different configurations in one curve.

## Common settings

- Backend: CPU. One-layer Top-2 MoE inference with `d=128`, `h=256`, `E=256`.
- Router input: controlled uniform trace, with the exact realized Expert and
  rank assignment histograms printed by `moe_cpu_bench`.
- Expert placement: round-robin. Communication: Alltoallv. OpenMP scheduling:
  static. Default seeds: model 42, input 314159, routing 271828, calibration
  161803.
- Three warmups and five measured forwards per point. A rank-0 CPU serial
  oracle is prepared outside the timer. Every trial must print `bitwise_ok=1`.
- Record every raw trial, the compute-node Git commit, compiler/MPI modules,
  PBS job ID, node names and CPU models, and allocated versus used cores.
- Use physical cores without oversubscription. Bind MPI processes to cores.
  Compare points only on the same CPU model; if PBS supplies mixed models,
  report and rerun on homogeneous resources rather than silently combining
  them.
- The Vanda user guide dated 05 Jun 2025 lists Intel Xeon Platinum 8452Y
  processors with two 36-core sockets in both CPU and A40 GPU nodes. GN-A40-047
  reported this model. Verify the model on every allocated node, since the
  guide does not prove the exact hardware of every individual allocation.

## Strong scaling

- Fix global `N=131072` Tokens.
- Run `P=1,2,4,8,16,32,64,128,256` MPI ranks with one OpenMP thread each.
- Baseline is the `P=1`, one-core EP/Alltoallv result from the same formal
  series. Speedup is `T1/TP`; parallel efficiency is `T1/(P*TP)`.
- Pilot evidence: on `CN-152`, `P=1,T=1` median 2.60025 s; `P=2,T=2` median
  0.755526 s. All trials passed bitwise checks. The second pilot uses four
  cores, so it is a sizing observation, not the formal `P=4,T=1` point.

## Weak scaling

- Fix 4096 Tokens per MPI rank; global `N=4096*P`.
- Run the same nine `P` values, each with one OpenMP thread. The largest
  problem has 1,048,576 Tokens.
- Baseline is the `P=1`, 4096-Token result from the same formal series. Weak
  scaling efficiency is `T1/TP`; show global `N` and core count on the figure.
- Pilot evidence: on `GN-A40-047`, `P=1` median 0.0790673 s and `P=2` median
  0.083977 s, with all trials bitwise correct. The measured assignment load
  was 8192 per rank at both points. The short `P=1` baseline varied by about
  10% from minimum to maximum, so keep all five raw trials and report spread.

## Before formal collection

1. Verify MPI launch across two CPU nodes and record actual hostnames and CPU
   models. Confirm the PBS nodefile provides enough MPI slots.
2. Build the current source on a compute node. The benchmark must print
   configuration, seeds, per-Expert load, and per-rank load before its trials.
3. Run a `P=256` correctness/performance pilot for both fixed-size and
   weak-size cases. Confirm memory and walltime are sufficient. If the frozen
   scale is infeasible, revise this file and rerun every point in that series.
4. Collect the complete nine-point formal curves under one code revision and
   consistent node type. Do not use earlier pilot timings as formal points.
