#!/usr/bin/env python3
"""Plot validated RR/LA forward times, realized load and remote assignment share."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(summary, out_dir):
    with summary.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 80:
        raise ValueError('expected a validated 80-condition placement summary')
    job = rows[0]['pbs_jobid'].split('.')[0]
    conditions = ((4,32768), (8,32768), (8,131072))
    colors = {'rr':'#1875a1', 'la':'#cb642c'}
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for i, (ranks, tokens) in enumerate(conditions):
        cases = [('moderate','A'),('extreme','A')] if tokens == 131072 else [
            ('uniform','A'),('moderate','A'),('moderate','B'),('extreme','A')]
        for j, comm in enumerate(('alltoallv','p2p')):
            ax = axes[i,j]
            for placement in ('rr','la'):
                for round_id, marker, linestyle in (('A','o','-'),('B','s','--')):
                    selected = [next(r for r in rows if int(r['ranks']) == ranks and
                                 int(r['tokens']) == tokens and r['skew'] == skew and
                                 r['hot_layout'] == layout and r['communication'] == comm and
                                 r['placement'] == placement and r['round'] == round_id)
                                for skew,layout in cases]
                    times = [1000*float(r['median_seconds']) for r in selected]
                    error = [[t-1000*float(r['min_seconds']) for t,r in zip(times,selected)],
                             [1000*float(r['max_seconds'])-t for t,r in zip(times,selected)]]
                    ax.errorbar(range(len(cases)),times,yerr=error,
                                color=colors[placement],marker=marker,linestyle=linestyle,
                                capsize=3,label=f'{placement.upper()} {round_id}')
            ax.set(title=f'P={ranks}, N={tokens:,}, {comm}',
                   ylabel='Forward time (ms, log scale)',xlabel='Route trace')
            ax.set_xticks(range(len(cases)),[f'{skew} {layout}' for skew,layout in cases])
            ax.set_yscale('log')
            ax.grid(alpha=.25,which='both')
            ax.legend(fontsize=8)
    fig.suptitle(f'Vanda placement job {job}: RR versus calibrated LPT')
    out_dir.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf'):
        fig.savefig(out_dir/f'cpu-placement-times-{job}.{suffix}',dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for i,ranks in enumerate((4,8)):
        cases = [('uniform','A'),('moderate','A'),('moderate','B'),('extreme','A')]
        for placement in ('rr','la'):
            selected = [next(r for r in rows if r['round']=='A' and int(r['ranks'])==ranks
                             and int(r['tokens'])==32768 and r['skew']==skew and
                             r['hot_layout']==layout and r['communication']=='alltoallv' and
                             r['placement']==placement) for skew,layout in cases]
            labels=[f'{skew} {layout}' for skew,layout in cases]
            axes[i,0].plot(range(4),[float(r['rank_load_max_over_mean']) for r in selected],
                           marker='o',color=colors[placement],label=placement.upper())
            axes[i,1].plot(range(4),[100*float(r['remote_assignment_ratio']) for r in selected],
                           marker='o',color=colors[placement],label=placement.upper())
        for j, ylabel in enumerate(('Max / mean rank assignments',
                                     'Remote assignments (%)')):
            ax=axes[i,j]
            ax.set(title=f'P={ranks}, N=32,768',ylabel=ylabel,xlabel='Route trace')
            ax.set_xticks(range(4),labels)
            ax.grid(alpha=.25)
            ax.legend()
    fig.suptitle(f'Vanda placement job {job}: load balance and communication volume')
    for suffix in ('png','pdf'):
        fig.savefig(out_dir/f'cpu-placement-load-{job}.{suffix}',dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('figures'))
    args=parser.parse_args()
    plot(args.summary,args.output_dir)
