#!/usr/bin/env python3
"""Create report figures from the validated single-A40 summary CSV."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIZES = (2048, 8192, 32768)
CASES = (('uniform', 'A'), ('moderate', 'A'), ('moderate', 'B'), ('extreme', 'A'))


def select(rows, round_id, tokens, skew, layout):
    matches = [row for row in rows if row['round'] == round_id and
               int(row['tokens']) == tokens and row['skew'] == skew and
               row['hot_layout'] == layout]
    if len(matches) != 1:
        raise ValueError('validated matrix is missing or duplicated')
    return matches[0]


def save(fig, directory, stem):
    for suffix in ('png', 'pdf'):
        fig.savefig(directory / f'{stem}.{suffix}', dpi=220)
    plt.close(fig)


def plot(summary, directory):
    with summary.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 24 or len({row['pbs_jobid'] for row in rows}) != 1:
        raise ValueError('expected one validated 24-condition GPU job')
    job = rows[0]['pbs_jobid'].split('.')[0]
    directory.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.7), constrained_layout=True)
    for backend, prefix, color in (('GPU end-to-end', 'gpu', '#196c9c'),
                                   ('CPU serial', 'cpu', '#c56a2e')):
        for round_id, marker, style in (('A', 'o', '-'), ('B', 's', '--')):
            selected = [select(rows, round_id, n, 'uniform', 'A') for n in SIZES]
            y = [1000 * float(row[f'{prefix}_median_seconds']) for row in selected]
            low = [v - 1000 * float(row[f'{prefix}_min_seconds']) for v, row in zip(y, selected)]
            high = [1000 * float(row[f'{prefix}_max_seconds']) - v for v, row in zip(y, selected)]
            ax.errorbar(SIZES, y, yerr=[low, high], color=color, marker=marker,
                        linestyle=style, capsize=3, label=f'{backend}, round {round_id}')
    ax.set(xscale='log', yscale='log', xticks=SIZES, xticklabels=[f'{n:,}' for n in SIZES],
           xlabel='Tokens, uniform trace', ylabel='Forward time (ms)',
           title=f'Single A40 vs same-node CPU serial: job {job}')
    ax.grid(alpha=.25, which='both')
    ax.legend(fontsize=8)
    save(fig, directory, f'gpu-formal-size-{job}')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    phase_names = ('host_pack', 'h2d', 'kernel', 'd2h', 'host_combine')
    labels = ('Host pack', 'H2D', 'CUDA kernels', 'D2H', 'Host combine')
    colors = ('#7b6ba2', '#55a1a8', '#196c9c', '#92b86d', '#c56a2e')
    for i, n in enumerate(SIZES):
        row_a = select(rows, 'A', n, 'uniform', 'A')
        row_b = select(rows, 'B', n, 'uniform', 'A')
        for round_id, row in (('A', row_a), ('B', row_b)):
            x = i + (-.18 if round_id == 'A' else .18)
            for j, (phase, color) in enumerate(zip(phase_names, colors)):
                axes[0].bar(x + (j-2)*.06, 1000*float(row[f'median_{phase}_seconds']),
                            width=.055, color=color, label=labels[j] if i == 0 and round_id == 'A' else None)
            axes[0].plot(x, 1000*float(row['weights_h2d_seconds']), marker='x',
                         linestyle='none', color='black', markersize=7,
                         label='One-time weight H2D' if i == 0 and round_id == 'A' else None)
            axes[1].bar(x, 100*float(row['non_kernel_share']), width=.28,
                        color='#196c9c' if round_id == 'A' else '#c56a2e',
                        label=f'Round {round_id}' if i == 0 else None)
    for ax in axes:
        ax.set_xticks(range(3), [f'{n:,}' for n in SIZES])
        ax.set_xlabel('Tokens, uniform trace')
        ax.grid(axis='y', alpha=.25)
    axes[0].set(ylabel='Median component time (ms)', title='Measured GPU forward components')
    axes[1].set(ylabel='Share of end-to-end time (%)', title='Time outside CUDA kernels')
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].legend(fontsize=8)
    fig.suptitle('Component medians are diagnostic; one-time weight upload is excluded')
    save(fig, directory, f'gpu-formal-breakdown-{job}')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    case_labels = [f'{skew}-{layout}' for skew, layout in CASES]
    for round_id, offset, color in (('A', -.17, '#196c9c'), ('B', .17, '#c56a2e')):
        selected = [select(rows, round_id, 32768, skew, layout) for skew, layout in CASES]
        x = [i + offset for i in range(4)]
        axes[0].bar(x, [1000*float(row['gpu_median_seconds']) for row in selected],
                    width=.32, color=color, label=f'End-to-end, round {round_id}')
        axes[0].plot(x, [1000*float(row['median_kernel_seconds']) for row in selected],
                     marker='o', linestyle='none', color='black',
                     label='CUDA kernel' if round_id == 'A' else None)
    selected = [select(rows, 'A', 32768, skew, layout) for skew, layout in CASES]
    axes[1].bar(range(4), [int(row['expert_max']) for row in selected], color='#7b6ba2')
    axes[1].axhline(4096, color='black', linestyle='--', linewidth=1, label='Uniform Expert load')
    for ax in axes:
        ax.set_xticks(range(4), case_labels, rotation=18)
        ax.set_xlabel('Controlled routing trace')
        ax.grid(axis='y', alpha=.25)
    axes[0].set(ylabel='Time (ms)', title='Single-A40 forward at 32,768 Tokens')
    axes[1].set(ylabel='Maximum assignments to one Expert', title='Realized Expert demand')
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    save(fig, directory, f'gpu-formal-skew-{job}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('figures'))
    args = parser.parse_args()
    plot(args.summary, args.output_dir)
