# Cluster setup record

Use this file with the environment checklist in the root README before
implementation and formal benchmarking. Do not record account identifiers,
passwords, tokens, or private keys in the repository.

## Information to confirm

| Item | Status / evidence |
|---|---|
| Primary system: Atlas or Vanda | Pending |
| Course/project allocation and charging rule | Pending |
| Quota-exhaustion behavior | Pending |
| Node, physical-core, socket, and memory limits | Pending |
| Multi-node MPI permission and limits | Pending |
| Scheduler and recommended MPI launcher | Pending |
| Interactive allocation procedure | Pending |
| Batch submission procedure | Pending |
| Compiler, MPI, CMake, and OpenMP module versions | Pending |
| Process/thread affinity policy | Pending |
| Persistent storage, scratch path, and cleanup policy | Pending |
| Whether login and compute CPUs have the same architecture | Pending |

Record authoritative documentation references or confirmation dates in a
private team record. The public repository should contain only reproducible
module-loading and build instructions once they are known.

## Lightweight environment inventory

After loading the course-recommended modules, run:

```bash
bash scripts/collect_environment.sh
```

It writes a private, ignored record under `private/environment/`. It does not
compile code, launch MPI ranks, request an allocation, or submit a job. Run it
again inside an approved compute allocation during the launch-test step: login
node hardware may differ from compute-node hardware.

The script only reports what is visible in its current shell. A missing tool may
mean that a module has not been loaded; a visible scheduler command does not
prove that the account may submit jobs.

## Step 1 exit criteria

- Select Atlas or Vanda and record the authoritative course guidance.
- Confirm allocation, quota, storage, and multi-node policies.
- Record the required compiler/MPI environment and launcher.
- Keep source in persistent storage; reserve scratch for regenerable artifacts.

The next step is a compute-node launch test: 2 MPI ranks x 1 thread, 2 ranks x
2 threads, then one MPI run spanning two nodes. Those tests are not included in
the environment-inventory script.

## Step 2 launch probe

After confirming the correct module commands on the selected system, configure
the small probe program:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target moe_launch_probe
```

Run it only through an approved interactive allocation or batch job. The
following are command shapes, not scheduler instructions:

```bash
# 2 ranks x 1 OpenMP thread
OMP_NUM_THREADS=1 scripts/run_launch_probe.sh -- <cluster-launcher> \
  build/moe_launch_probe --expect-threads 1

# 2 ranks x 2 OpenMP threads
OMP_NUM_THREADS=2 scripts/run_launch_probe.sh -- <cluster-launcher> \
  build/moe_launch_probe --expect-threads 2

# A run explicitly allocated on two nodes
OMP_NUM_THREADS=1 scripts/run_launch_probe.sh -- <cluster-launcher> \
  build/moe_launch_probe --expect-threads 1 --require-distinct-hosts
```

Replace `<cluster-launcher>` with the complete launcher command prescribed by
the cluster documentation, including its rank and node arguments. The probe
verifies MPI's `MPI_THREAD_FUNNELED` support, reports each rank's hostname and
OpenMP thread count, and uses a ring `MPI_Sendrecv` check. It fails if the
requested thread count is not observed or if the two-node check observes only
one hostname.
