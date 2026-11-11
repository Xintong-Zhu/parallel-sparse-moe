#!/usr/bin/env python3
"""Plot validated communication/skew times and observed rank load ranges."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(summary, out_dir):
    with summary.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 24:
        raise ValueError('expected a validated 24-condition summary')
    job = rows[0]['pbs_jobid'].split('.')[0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    skews = ('uniform', 'moderate', 'extreme')
    for i, tokens in enumerate((32768, 131072)):
        for method, color in (('alltoallv', '#1875a1'), ('p2p', '#cb642c')):
            for round_id, marker, linestyle in (('A', 'o', '-'), ('B', 's', '--')):
                selected = [next(r for r in rows if int(r['tokens']) == tokens and r['skew'] == skew
                                 and r['communication'] == method and r['round'] == round_id) for skew in skews]
                times = [float(r['median_seconds']) for r in selected]
                errors = [[t-float(r['min_seconds']) for t,r in zip(times, selected)],
                          [float(r['max_seconds'])-t for t,r in zip(times, selected)]]
                axes[i,0].errorbar(range(3), times, yerr=errors, color=color, linestyle=linestyle,
                                   marker=marker, capsize=3, label=f'{method} {round_id}')
        loads = [next(r for r in rows if int(r['tokens']) == tokens and r['skew'] == skew) for skew in skews]
        axes[i,1].plot(range(3), [int(r['rank_min']) for r in loads], 'o-', color='#1875a1', label='Min rank')
        axes[i,1].plot(range(3), [int(r['rank_max']) for r in loads], 's-', color='#cb642c', label='Max rank')
        axes[i,0].set(title=f'{tokens:,} Tokens: forward time', ylabel='Time (s, log scale)')
        axes[i,0].set_yscale('log')
        axes[i,1].set(title=f'{tokens:,} Tokens: realized assignments', ylabel='Assignments per MPI rank')
        for ax in axes[i]:
            ax.set_xticks(range(3), skews)
            ax.set_xlabel('Routing trace skew')
            ax.grid(alpha=.25, which='both')
            ax.legend(fontsize=8)
    fig.suptitle(f'Vanda EP communication/skew job {job}: 8 MPI ranks × 2 threads, 16 Experts')
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ('png','pdf'):
        fig.savefig(out_dir/f'cpu-comm-skew-{job}.{suffix}',dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('figures'))
    args=parser.parse_args()
    plot(args.summary,args.output_dir)
