# Expert placement results: Vanda job 1543859

## Validation and setup

The PBS job finished with `exit_status=0` and `stage=complete`. The strict
analyzer accepted all 80 condition logs (40 matched RR/LA pairs) and all 400
timed trials with `bitwise_ok=1`. The protocol, serial and placement unit tests
also passed. The frozen source was
`b025392a8b74591a8167f3ecb5a84ad69f27ff2b`. Four CN nodes (148, 151,
152, 153) all reported Intel Xeon Platinum 8452Y.

This is one MoE layer with 16 Experts, Top-2 routes, d=128 and hidden size 256.
P=4 uses one MPI rank per node and two OpenMP threads per rank (8 cores); P=8
uses two ranks per node and two threads per rank (16 cores). Every measured
condition has three warmups and five trials. Rounds A and B reverse the
condition order. Matched RR and LA cases use identical model, inputs, routes,
node allocation, binding and communication method. The forward timer starts
after routes and the LA owner map are prepared.

## Matched forward times

Each RR/LA time below is the median of four five-trial medians (two rounds ×
Alltoallv/P2P), in milliseconds. The speedup is the median of the four **paired**
RR-time/LA-time ratios, so it need not equal the ratio of the two displayed
time medians. Rank load is the largest actual assignment count divided by the
mean. Remote is the fraction of assignments whose input and result cross ranks.

| MPI ranks | Tokens | Trace | Layout | RR ms | LA ms | Paired speedup | RR max/mean | LA max/mean | RR remote | LA remote |
|---:|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 32,768 | Uniform | A | 169.37 | 169.23 | 1.00× | 1.00 | 1.00 | 75.0% | 75.0% |
| 4 | 32,768 | Moderate | A | 374.99 | 166.44 | 2.21× | 2.40 | 1.00 | 61.7% | 75.0% |
| 4 | 32,768 | Moderate | B | 167.89 | 169.39 | 0.98× | 1.00 | 1.00 | 75.0% | 75.0% |
| 4 | 32,768 | Extreme | A | 326.90 | 254.02 | 1.28× | 2.08 | 1.60 | 72.3% | 71.7% |
| 8 | 32,768 | Uniform | A | 86.78 | 87.23 | 1.00× | 1.00 | 1.00 | 87.5% | 87.5% |
| 8 | 32,768 | Moderate | A | 177.53 | 101.06 | 1.77× | 2.40 | 1.20 | 87.1% | 78.8% |
| 8 | 32,768 | Moderate | B | 118.36 | 104.33 | 1.14× | 1.47 | 1.20 | 81.7% | 78.8% |
| 8 | 32,768 | Extreme | A | 240.24 | 211.33 | 1.14× | 3.52 | 3.20 | 86.2% | 85.9% |
| 8 | 131,072 | Moderate | A | 627.30 | 404.92 | 1.55× | 2.40 | 1.20 | 87.1% | 78.8% |
| 8 | 131,072 | Extreme | A | 888.48 | 786.13 | 1.13× | 3.52 | 3.20 | 86.2% | 85.9% |

Across all 40 pairs, LA is faster in 33, slower in 7. These counts include
near-ties in the uniform and already-balanced moderate-B cases; they are not
evidence of a universal LA advantage. In the moderate-A P=4 case, RR puts all
four hot Experts on rank 0 (actual rank assignments 39,320 versus roughly
8,739 elsewhere). LA spreads them to 16,384 assignments per rank. It does so
despite increasing remote assignment share from 61.7% to 75.0%. Thus reducing
the compute bottleneck outweighs the added remote payload in this controlled
case. Both Alltoallv and P2P show the improvement in both rounds.

Moderate-B is the placement control: its hot Experts already fall on different
RR ranks. At P=4 both mappings have perfectly even rank counts and essentially
the same time. At P=8, RR max/mean is 1.47 and LA improves it to 1.20, yielding
a smaller speedup. With uniform routes both maps are balanced and neither has
a meaningful time advantage.

Under extreme skew, one Expert alone receives most work; moving that Expert
cannot split it across ranks. Consequently LA improves the peak count only
from 3.52 to 3.20 at P=8 and the speedup is about 1.13–1.14×. This is a
structural limit of whole-Expert placement, not a failed mapping.

The earlier communication/skew job `1542686` found a moderate/extreme time
inversion at N=131,072. This job does **not** reproduce it: extreme is slower
than moderate for both placements and communication methods. The earlier
observation should be reported as run dependent; these measurements do not
identify its cause. Phase values are rankwise maxima and include waiting at
MPI calls, so they cannot be added or interpreted as pure network time.

## Calibration and scope

LA uses an independent 8,192-Token calibration trace and LPT placement before
the timed forwards. Its median calibration time is about 5.11 ms; median map
construction is about 1.8 microseconds. These costs are **excluded** from the
table's forward times. For a single forward, include calibration when making an
end-to-end method comparison; for repeated forwards with a frozen map, that
cost can be amortized. The synthetic Trace generator sets exact Expert quotas,
and the observed calibration prediction error is zero in these cases. Real
routing can drift, so these measurements do not demonstrate the same mapping
quality under changing live traffic.

The reported remote payload is a logical estimate based on assignments and
the wire record sizes, not a measurement of physical network bytes. RR and LA
are compared within the same allocation, while this small 8/16-core placement
study is separate from the 256-core strong/weak scaling experiment.

## Reproduce analysis

```bash
python3 scripts/analyze_cpu_placement.py private/placement-1543859 \
  --output results/processed/vanda-cpu-placement-1543859.csv
python3 scripts/plot_cpu_placement.py \
  results/processed/vanda-cpu-placement-1543859.csv
```

The processed CSV stores every condition's time range, phase medians, realized
Expert/rank histograms, owners, predicted loads, remote payload and provenance.
The raw logs remain in ignored `private/placement-1543859/`. The tracked plots are
`figures/cpu-placement-times-1543859.{png,pdf}` and
`figures/cpu-placement-load-1543859.{png,pdf}`.
