# Load-aware Expert placement: Vanda experiment

This experiment completes the planned RR (round robin) versus LA (load aware)
comparison. The same Top-2 routes, input Tokens, model weights, CPU nodes,
MPI/OpenMP binding, and communication implementation are used for a paired
RR/LA point. Both variants must be bitwise identical to the serial oracle.

## Algorithm and timing boundary

For LA, rank 0 generates an **independent** 8,192-Token calibration workload
using `calibration_seed=161803`. In Trace mode, it retains the requested skew
and hotspot layout while changing the route seed. In Router mode, it generates
fresh calibration Tokens and routes them with the unchanged Router weights.
The calibration assignment histogram predicts each Expert's demand. LPT sorts
Experts by decreasing predicted count (Expert ID breaks ties) and repeatedly
assigns the next Expert to the rank with the smallest predicted total (rank ID
breaks ties). The owner map is broadcast and frozen for all measured forwards.

The benchmark reports calibration and LPT mapping time separately. Neither is
inside `end_to_end_seconds`; the forward timer includes dispatch, FFN, return,
unpack, and combine. The reported remote payload bytes count each remote
assignment's 16-byte header and 128 FP32 input values on dispatch and the
same sized response on return. Count exchanges, MPI protocol overhead, and
hardware traffic are outside that byte estimate. The synthetic Trace generator
sets exact Expert quotas, making its calibration prediction unusually easy;
this limitation must appear in the report.

## Controlled matrix

One four-node PBS allocation has two MPI slots and two OpenMP threads per slot
on every node. P=4 uses one rank per node (8 physical cores); P=8 uses two ranks
per node (16 physical cores). Each condition runs three warmups and five timed
forwards, then repeats in reverse condition order:

- P=4 and P=8, N=32,768: uniform-A, moderate-A, moderate-B, extreme-A;
  Alltoallv/P2P; RR/LA.
- P=8, N=131,072: moderate-A and extreme-A; Alltoallv/P2P; RR/LA.

The resulting 80 condition logs contain 400 correctness-checked trials.
Moderate-A puts the four hotspots on RR ranks 0 and 4 at P=8; moderate-B puts
them on ranks 0, 1, 2, and 3. This checks whether a placement conclusion depends
on the choice of hotspot IDs. The larger N also revisits the observed
moderate/extreme timing inversion. It may not identify its cause because the
benchmark records rankwise phase maxima, not individual-rank phase times.

For every point, compare forward medians and spread, realized rank max/mean,
predicted versus actual rank loads, remote assignment ratio and bytes, and
calibration/mapping overhead. LA may lower imbalance while increasing remote
traffic; neither a speedup nor a slowdown is assumed in advance.

## Submit and analyze

After the code commit passes CI, sync it to Vanda and run on the **login node**:

```bash
bash scripts/submit_vanda_cpu_formal.sh scripts/vanda_cpu_placement.pbs
```

The wrapper prints the frozen source path, Git commit and PBS job ID. The raw
logs will be in `private/placement-<job-id>/`. Once `qstat -fx <job-id>` reports
`job_state = F` and `Exit_status = 0`, sync that complete directory locally and
run:

```bash
python3 scripts/analyze_cpu_placement.py private/placement-<job-id> \
  --output results/processed/vanda-cpu-placement-<job-id>.csv
python3 scripts/plot_cpu_placement.py results/processed/vanda-cpu-placement-<job-id>.csv
```

The analyzer rejects failed or incomplete jobs, wrong configurations, mismatched
RR/LA routes, invalid mappings, inconsistent predicted and realized loads, and
any trial with `bitwise_ok != 1`. No failed point enters a plot.
