#!/usr/bin/env python3
"""Plot validated hybrid rounds separately, including the unstable DP point."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(summary, output_dir):
    with summary.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 20:
        raise ValueError('expected 20 validated runs')
    job = rows[0]['pbs_jobid'].split('.')[0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    labels = ['256x1', '128x2', '64x4', '32x8', '16x16']
    for col, mode in enumerate(('ep', 'dp')):
        for rnd, color, marker in (('A', '#1875a1', 'o'), ('B', '#cb642c', 's')):
            selected = sorted((r for r in rows if r['mode'] == mode and r['round'] == rnd), key=lambda r: int(r['threads']))
            values = [float(r['median_seconds']) for r in selected]
            errors = [[v - float(r['min_seconds']) for r, v in zip(selected, values)],
                      [float(r['max_seconds']) - v for r, v in zip(selected, values)]]
            axes[0, col].errorbar(range(5), values, yerr=errors, color=color, marker=marker,
                                  capsize=3, label=f'Round {rnd}')
            axes[1, col].plot(range(5), [float(r['speedup_vs_same_round_256x1']) for r in selected],
                              color=color, marker=marker, label=f'Round {rnd}')
        axes[0, col].set_yscale('log')
        axes[0, col].set(title=f'{mode.upper()} forward time', ylabel='Time (s, log scale)')
        axes[1, col].set(title=f'{mode.upper()} gain vs same-round 256x1', ylabel='Relative speedup')
        axes[1, col].axhline(1, color='gray', linestyle=':')
        for ax in axes[:, col]:
            ax.set_xticks(range(5), labels)
            ax.set_xlabel('MPI ranks x OpenMP threads (256 active cores)')
            ax.grid(alpha=.25, which='both'); ax.legend()
    axes[0, 1].set_title('DP forward time (128x2: unstable across rounds)')
    fig.suptitle(f'Vanda hybrid job {job}: N=131072, E=256; error bars show trial min/max')
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(output_dir / f'cpu-hybrid-{job}.{extension}', dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('figures'))
    args = parser.parse_args()
    plot(args.summary, args.output_dir)
