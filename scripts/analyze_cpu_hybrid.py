#!/usr/bin/env python3
"""Validate physical affinities and all two-round hybrid benchmark records."""
import argparse
import csv
import math
import re
import statistics
from collections import Counter
from pathlib import Path

from analyze_cpu_formal import PHASES, pairs, require

THREADS = (1, 2, 4, 8, 16)


def analyze(directory, output):
    def kv(name):
        return dict(line.split('=', 1) for line in (directory / name).read_text().splitlines())
    require(kv('job_status.txt') == {'exit_status': '0', 'stage': 'complete'}, 'job incomplete')
    meta = kv('metadata.txt')
    require(re.fullmatch('[0-9a-f]{40}', meta['git_commit']), 'invalid commit')
    expected_meta = dict(requested_cores='288', used_cores='256', node_count='4', mpi_slots='256',
                         tokens='131072', experts='256', dimension='128', hidden='256',
                         router_mode='trace', skew='uniform', placement='rr', communication='alltoallv',
                         rounds='A,B', warmups='3', repeats_per_round='5', omp_schedule='static',
                         OMP_PLACES='cores', OMP_PROC_BIND='close', OMP_DYNAMIC='FALSE')
    for key, value in expected_meta.items():
        require(meta.get(key) == value, f'metadata mismatch: {key}')
    require(directory.name == 'hybrid-cpu-' + meta['pbs_jobid'].split('.')[0], 'job ID mismatch')
    models = list(csv.reader((directory / 'node_models.csv').read_text().splitlines()))
    require(len(models) == 4 and all(len(row) == 2 for row in models), 'invalid node records')
    hosts = {row[0] for row in models}
    require(len(hosts) == 4 and len({row[1] for row in models}) == 1, 'mixed CPU models')
    normal = lambda host: host.split('.')[0].lower()
    require({normal(h) for h in hosts} == {normal(h) for h in (directory / 'nodes.txt').read_text().splitlines()}, 'host list mismatch')
    nodefile = Counter(normal(h) for h in (directory / 'pbs_nodefile.txt').read_text().splitlines())
    require(set(nodefile) == {normal(h) for h in hosts} and all(n == 64 for n in nodefile.values()), 'slot allocation mismatch')
    common_cores = None
    for threads in THREADS:
        ranks = 256 // threads
        lines = (directory / f'affinity-p{ranks}-t{threads}.txt').read_text().splitlines()
        require('RESULT: PASS' in lines and 'Affinity check: PASS' in lines, 'probe failed')
        records = [pairs(line, 'affinity,') for line in lines if line.startswith('affinity,')]
        require(len(records) == 256, 'missing affinity records')
        cores, tids, rank_hosts = set(), set(), {}
        host_counts, sockets = Counter(), Counter()
        for record in records:
            rank, tid = int(record['rank']), int(record['thread'])
            host, package, core = record['host'], int(record['package']), int(record['core'])
            require(0 <= rank < ranks and 0 <= tid < threads and host in hosts and record['valid'] == '1', 'invalid affinity record')
            require((rank, tid) not in tids and (host, package, core) not in cores, 'overlapping thread/core')
            require(rank_hosts.setdefault(rank, (host, package)) == (host, package), 'rank spans hosts or sockets')
            tids.add((rank, tid)); cores.add((host, package, core))
            host_counts[host] += 1; sockets[(host, package)] += 1
        require(len(host_counts) == 4 and all(n == 64 for n in host_counts.values()), 'host core counts differ')
        require(len(sockets) == 8 and all(n == 32 for n in sockets.values()), 'socket core counts differ')
        if common_cores is None:
            common_cores = cores
        require(cores == common_cores, 'physical core sets differ across layouts')

    rows = []
    for round_id in 'AB':
        for mode in ('ep', 'dp'):
            for threads in THREADS:
                ranks = 256 // threads
                name = f'{round_id}-{mode}-p{ranks}-t{threads}.txt'
                lines = (directory / name).read_text().splitlines()
                configs = [pairs(line, 'benchmark_config,') for line in lines if line.startswith('benchmark_config,')]
                require(len(configs) == 1, f'{name}: missing config')
                expected = dict(tokens='131072', dimension='128', hidden='256', experts='256',
                                ranks=str(ranks), threads=str(threads), parallelism=mode, communication='alltoallv',
                                router_mode='trace', skew='uniform', placement='rr', omp_schedule='static',
                                model_seed='42', input_seed='314159', routing_seed='271828', calibration_seed='161803')
                for key, value in expected.items():
                    require(configs[0].get(key) == value, f'{name}: config {key} differs')
                for prefix, count, target in (('benchmark_expert_histogram=', 256, 1024),
                                              ('benchmark_rank_assignment_histogram=', ranks, 262144 // ranks)):
                    matches = [line[len(prefix):] for line in lines if line.startswith(prefix)]
                    require(len(matches) == 1, f'{name}: missing histogram')
                    require([int(x) for x in matches[0].split(',')] == [target] * count, f'{name}: actual load differs')
                trials = [pairs(line, 'benchmark,') for line in lines if line.startswith('benchmark,')]
                require(len(trials) == 5 and sorted(int(t['trial']) for t in trials) == list(range(5)), f'{name}: trial IDs differ')
                for trial in trials:
                    require(trial.get('bitwise_ok') == '1', f'{name}: correctness failed')
                    for key in ('tokens', 'experts', 'ranks', 'threads', 'parallelism', 'communication'):
                        require(trial.get(key) == expected[key], f'{name}: trial config differs')
                    for key in ('end_to_end_seconds', *PHASES):
                        value = float(trial[key])
                        require(math.isfinite(value) and value >= 0, f'{name}: invalid timing')
                    require(float(trial['end_to_end_seconds']) > 0, f'{name}: zero elapsed')
                times = [float(t['end_to_end_seconds']) for t in trials]
                row = dict(round=round_id, mode=mode, ranks=ranks, threads=threads, used_cores=256,
                           tokens=131072, experts=256, trial_count=5,
                           min_seconds=min(times), median_seconds=statistics.median(times), max_seconds=max(times))
                row.update({f'median_{key}': statistics.median(float(t[key]) for t in trials) for key in PHASES})
                rows.append(row)
    for row in rows:
        baseline = next(r['median_seconds'] for r in rows if r['round'] == row['round'] and r['mode'] == row['mode'] and r['threads'] == 1)
        a, b = [next(r['median_seconds'] for r in rows if r['round'] == rnd and r['mode'] == row['mode'] and r['threads'] == row['threads']) for rnd in 'AB']
        row.update(speedup_vs_same_round_256x1=baseline / row['median_seconds'],
                   round_drift_fraction=abs(b / a - 1), unstable_over_10pct=int(abs(b / a - 1) > .1),
                   git_commit=meta['git_commit'], pbs_jobid=meta['pbs_jobid'], cpu_model=models[0][1], hosts=';'.join(sorted(hosts)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print('Validated 5 physical-core layouts, 20 runs and 100 correct trials.')
    for row in rows:
        print(f"{row['round']} {row['mode']} {row['ranks']}x{row['threads']}: {row['median_seconds']:.7f}s, gain={row['speedup_vs_same_round_256x1']:.3f}, unstable={row['unstable_over_10pct']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        analyze(args.input_dir, args.output)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f'Hybrid data invalid: {error}\n')
