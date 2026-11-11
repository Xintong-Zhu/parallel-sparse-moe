#!/usr/bin/env python3
"""Validate both rounds of Vanda EP communication/skew data and summarize trials."""
import argparse
import csv
import math
import re
import statistics
from pathlib import Path

from analyze_cpu_formal import PHASES, pairs, require

SIZES = (32768, 131072)
SKEWS = ('uniform', 'moderate', 'extreme')
COMMS = ('alltoallv', 'p2p')


def single(lines, prefix, name):
    values = [line[len(prefix):] for line in lines if line.startswith(prefix)]
    require(len(values) == 1, f'{name}: expected one {prefix}')
    return values[0]


def analyze(directory, output):
    status = dict(line.split('=', 1) for line in (directory/'job_status.txt').read_text().splitlines())
    require(status == {'exit_status': '0', 'stage': 'complete'}, 'job not complete')
    meta = dict(line.split('=', 1) for line in (directory/'metadata.txt').read_text().splitlines())
    require(re.fullmatch('[0-9a-f]{40}', meta['git_commit']), 'invalid Git commit')
    require(directory.name == 'comm-skew-' + meta['pbs_jobid'].split('.')[0], 'job ID mismatch')
    expected_meta = dict(requested_cores='16', used_cores='16', node_count='4', mpi_slots='8',
                         ranks='8', threads='2', tokens='32768,131072', experts='16',
                         dimension='128', hidden='256', router_mode='trace',
                         skew='uniform,moderate,extreme', hot_layout='A', placement='rr',
                         communication='alltoallv,p2p', omp_schedule='static', rounds='A,B',
                         warmups='3', repeats_per_round='5')
    for key, value in expected_meta.items():
        require(meta.get(key) == value, f'metadata mismatch: {key}')
    models = list(csv.reader((directory/'node_models.csv').read_text().splitlines()))
    require(len(models) == 4 and all(len(row) == 2 and all(row) for row in models), 'bad node model records')
    hosts = {row[0].split('.')[0].lower() for row in models}
    require(len(hosts) == 4 and len({row[1] for row in models}) == 1, 'mixed CPU models')
    recorded = {h.split('.')[0].lower() for h in (directory/'nodes.txt').read_text().splitlines()}
    require(hosts == recorded, 'PBS hosts differ from model probe')
    slots = [h.split('.')[0].lower() for h in (directory/'pbs_nodefile.txt').read_text().splitlines()]
    require(len(slots) == 8 and all(slots.count(h) == 2 for h in hosts), 'MPI slot allocation differs')
    lines = (directory/'launch_probe.txt').read_text().splitlines()
    require('RESULT: PASS' in lines and '  MPI ranks: 8' in lines and '  OMP_NUM_THREADS: 2' in lines, 'launch probe failed')
    for comm in COMMS:
        require('CPU ep correctness: PASS' in (directory/f'correctness-{comm}.txt').read_text(), f'{comm} preflight failed')

    rows = []
    reference_histograms = {}
    for round_id in 'AB':
        for tokens in SIZES:
            for skew in SKEWS:
                for comm in COMMS:
                    name = f'{round_id}-n{tokens}-{skew}-{comm}.txt'
                    lines = (directory/name).read_text().splitlines()
                    config = pairs('benchmark_config,' + single(lines, 'benchmark_config,', name), 'benchmark_config,')
                    expected = dict(tokens=str(tokens), dimension='128', hidden='256', experts='16',
                                    ranks='8', threads='2', parallelism='ep', communication=comm,
                                    router_mode='trace', skew=skew, placement='rr', omp_schedule='static',
                                    model_seed='42', input_seed='314159', routing_seed='271828', calibration_seed='161803')
                    for key, value in expected.items():
                        require(config.get(key) == value, f'{name}: config {key} differs')
                    expert = [int(v) for v in single(lines, 'benchmark_expert_histogram=', name).split(',')]
                    rank = [int(v) for v in single(lines, 'benchmark_rank_assignment_histogram=', name).split(',')]
                    require(len(expert) == 16 and len(rank) == 8 and min(expert) >= 0, f'{name}: bad histogram')
                    require(sum(expert) == 2*tokens and sum(rank) == 2*tokens, f'{name}: assignment count differs')
                    require(rank == [expert[i]+expert[i+8] for i in range(8)], f'{name}: RR load mismatch')
                    key = (tokens, skew)
                    if key in reference_histograms:
                        require((expert, rank) == reference_histograms[key], f'{name}: paired input changed')
                    else:
                        reference_histograms[key] = (expert, rank)
                    if skew == 'uniform':
                        require(expert == [2*tokens//16]*16, f'{name}: realized uniform load differs')
                    trials = [pairs(line, 'benchmark,') for line in lines if line.startswith('benchmark,')]
                    require(len(trials) == 5 and sorted(int(t['trial']) for t in trials) == list(range(5)), f'{name}: trial IDs differ')
                    for trial in trials:
                        for field in ('tokens','experts','ranks','threads','parallelism','communication'):
                            require(trial.get(field) == expected[field], f'{name}: trial {field} differs')
                        require(trial.get('bitwise_ok') == '1', f'{name}: correctness failed')
                        for field in ('end_to_end_seconds', *PHASES):
                            value = float(trial[field])
                            require(math.isfinite(value) and value >= 0, f'{name}: invalid {field}')
                        require(float(trial['end_to_end_seconds']) > 0, f'{name}: zero elapsed time')
                    times = [float(t['end_to_end_seconds']) for t in trials]
                    row = dict(round=round_id, tokens=tokens, skew=skew, communication=comm,
                               ranks=8, threads=2, experts=16, trials=5,
                               min_seconds=min(times), median_seconds=statistics.median(times), max_seconds=max(times),
                               expert_min=min(expert), expert_max=max(expert),
                               rank_min=min(rank), rank_max=max(rank),
                               expert_histogram=';'.join(map(str,expert)), rank_histogram=';'.join(map(str,rank)))
                    row.update({f'median_{phase}':statistics.median(float(t[phase]) for t in trials) for phase in PHASES})
                    rows.append(row)
    for row in rows:
        alltoallv = next(r['median_seconds'] for r in rows if
                         r['round'] == row['round'] and r['tokens'] == row['tokens'] and
                         r['skew'] == row['skew'] and r['communication'] == 'alltoallv')
        uniform = next(r['median_seconds'] for r in rows if
                       r['round'] == row['round'] and r['tokens'] == row['tokens'] and
                       r['communication'] == row['communication'] and r['skew'] == 'uniform')
        row.update(speedup_vs_paired_alltoallv=alltoallv/row['median_seconds'],
                   slowdown_vs_same_method_uniform=row['median_seconds']/uniform,
                   git_commit=meta['git_commit'], pbs_jobid=meta['pbs_jobid'],
                   cpu_model=models[0][1], hosts=';'.join(sorted(hosts)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print('Validated 24 matched conditions, 120 correct trials and realized Expert/rank loads.')
    for row in rows:
        print(f"{row['round']} N={row['tokens']} {row['skew']} {row['communication']}: "
              f"{row['median_seconds']:.6f}s; rank load {row['rank_min']}..{row['rank_max']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        analyze(args.input_dir, args.output)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f'Communication/skew data invalid: {error}\n')
