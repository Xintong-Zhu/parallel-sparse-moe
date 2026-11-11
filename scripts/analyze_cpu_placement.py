#!/usr/bin/env python3
"""Reject invalid Vanda placement runs and summarize all paired RR/LA trials."""
import argparse
import csv
import math
import re
import statistics
from pathlib import Path

from analyze_cpu_formal import PHASES, pairs, require

CASES = (('uniform', 'A'), ('moderate', 'A'), ('moderate', 'B'), ('extreme', 'A'))
COMMS = ('alltoallv', 'p2p')
PLACEMENTS = ('rr', 'la')


def single(lines, prefix, name):
    values = [line[len(prefix):] for line in lines if line.startswith(prefix)]
    require(len(values) == 1, f'{name}: expected exactly one {prefix}')
    return values[0]


def numbers(lines, prefix, name):
    return [int(value) for value in single(lines, prefix, name).split(',')]


def rank_loads(expert, owners, ranks):
    result = [0] * ranks
    for load, owner in zip(expert, owners):
        result[owner] += load
    return result


def lpt(predicted, ranks):
    order = sorted(range(len(predicted)), key=lambda expert: (-predicted[expert], expert))
    owners = [-1] * len(predicted)
    loads = [0] * ranks
    for expert in order:
        owner = min(range(ranks), key=lambda rank: (loads[rank], rank))
        owners[expert] = owner
        loads[owner] += predicted[expert]
    return owners


def analyze(directory, output):
    status = dict(line.split('=', 1) for line in (directory/'job_status.txt').read_text().splitlines())
    require(status == {'exit_status': '0', 'stage': 'complete'}, 'job did not complete')
    meta = dict(line.split('=', 1) for line in (directory/'metadata.txt').read_text().splitlines())
    require(re.fullmatch('[0-9a-f]{40}', meta['git_commit']), 'invalid source commit')
    require(directory.name == 'placement-' + meta['pbs_jobid'].split('.')[0], 'job ID mismatch')
    expected_meta = dict(requested_cores='16', used_cores_p4='8', used_cores_p8='16',
                         node_count='4', mpi_slots='8', threads='2', tokens='32768,131072',
                         experts='16', dimension='128', hidden='256', router_mode='trace',
                         skew='uniform,moderate,extreme', hot_layout='A,B', placement='rr,la',
                         communication='alltoallv,p2p', omp_schedule='static',
                         calibration_tokens='8192', rounds='A,B', warmups='3',
                         repeats_per_round='5')
    for key, value in expected_meta.items():
        require(meta.get(key) == value, f'metadata mismatch: {key}')
    models = list(csv.reader((directory/'node_models.csv').read_text().splitlines()))
    require(len(models) == 4 and all(len(row) == 2 and all(row) for row in models),
            'invalid node model records')
    hosts = {row[0].split('.')[0].lower() for row in models}
    require(len(hosts) == 4 and len({row[1] for row in models}) == 1,
            'mixed or repeated CPU models')
    nodes = {host.split('.')[0].lower() for host in (directory/'nodes.txt').read_text().splitlines()}
    slots = [host.split('.')[0].lower() for host in (directory/'pbs_nodefile.txt').read_text().splitlines()]
    require(nodes == hosts and len(slots) == 8 and all(slots.count(host) == 2 for host in hosts),
            'PBS node allocation differs from model probe')
    for ranks in (4, 8):
        lines = (directory/f'launch-p{ranks}.txt').read_text().splitlines()
        require('RESULT: PASS' in lines and f'  MPI ranks: {ranks}' in lines and
                '  OMP_NUM_THREADS: 2' in lines, f'P={ranks}: launch probe failed')
        for comm in COMMS:
            preflight = (directory/f'correctness-p{ranks}-{comm}-la.txt').read_text()
            require('CPU ep correctness: PASS' in preflight,
                    f'P={ranks} {comm}: LA correctness preflight failed')

    rows = []
    reference_histograms = {}
    for round_id in 'AB':
        for ranks in (4, 8):
            workloads = [(32768, *case) for case in CASES]
            if ranks == 8:
                workloads += [(131072, 'moderate', 'A'), (131072, 'extreme', 'A')]
            for tokens, skew, layout in workloads:
                for comm in COMMS:
                    for placement in PLACEMENTS:
                        name = f'{round_id}-p{ranks}-n{tokens}-{skew}-{layout}-{comm}-{placement}.txt'
                        lines = (directory/name).read_text().splitlines()
                        config = pairs('benchmark_config,' + single(lines, 'benchmark_config,', name),
                                       'benchmark_config,')
                        expected = dict(tokens=str(tokens), dimension='128', hidden='256',
                                        experts='16', ranks=str(ranks), threads='2',
                                        parallelism='ep', communication=comm, router_mode='trace',
                                        skew=skew, hot_layout=layout, placement=placement,
                                        omp_schedule='static', model_seed='42', input_seed='314159',
                                        routing_seed='271828', calibration_seed='161803')
                        for key, value in expected.items():
                            require(config.get(key) == value, f'{name}: config {key} differs')
                        expert = numbers(lines, 'benchmark_expert_histogram=', name)
                        owners = numbers(lines, 'benchmark_expert_owners=', name)
                        actual = numbers(lines, 'benchmark_rank_assignment_histogram=', name)
                        require(len(expert) == len(owners) == 16 and len(actual) == ranks,
                                f'{name}: invalid histogram/map length')
                        require(min(expert) >= 0 and sum(expert) == 2*tokens and
                                all(0 <= owner < ranks for owner in owners),
                                f'{name}: invalid Expert counts or owners')
                        require(actual == rank_loads(expert, owners, ranks),
                                f'{name}: actual rank counts disagree with owner map')
                        key = (tokens, skew, layout)
                        if key in reference_histograms:
                            require(expert == reference_histograms[key],
                                    f'{name}: compared methods used different routes')
                        else:
                            reference_histograms[key] = expert
                        calibration_seconds = float(single(lines, 'benchmark_calibration_seconds=', name))
                        mapping_seconds = float(single(lines, 'benchmark_mapping_seconds=', name))
                        require(all(math.isfinite(x) and x >= 0 for x in
                                    (calibration_seconds, mapping_seconds)),
                                f'{name}: invalid setup timing')
                        predicted = []
                        max_prediction_error_pp = None
                        if placement == 'rr':
                            require(owners == [expert_id % ranks for expert_id in range(16)] and
                                    calibration_seconds == mapping_seconds == 0,
                                    f'{name}: RR mapping or setup cost differs')
                        else:
                            require(int(single(lines, 'benchmark_calibration_tokens=', name)) == 8192,
                                    f'{name}: calibration sample size differs')
                            cal_expert = numbers(lines, 'benchmark_calibration_expert_histogram=', name)
                            require(len(cal_expert) == 16 and sum(cal_expert) == 16384,
                                    f'{name}: invalid calibration histogram')
                            require(owners == lpt(cal_expert, ranks),
                                    f'{name}: mapping differs from deterministic LPT')
                            predicted = numbers(lines, 'benchmark_predicted_rank_assignment_histogram=', name)
                            expected_predicted = [count*tokens//8192 for count in
                                                  rank_loads(cal_expert, owners, ranks)]
                            require(predicted == expected_predicted,
                                    f'{name}: predicted rank counts differ')
                            max_prediction_error_pp = 100 * max(
                                abs(p/(2*tokens) - a/(2*tokens)) for p,a in zip(predicted,actual))
                            printed_error = float(single(lines, 'benchmark_max_prediction_error_pp=', name))
                            require(abs(printed_error-max_prediction_error_pp) < 1e-4,
                                    f'{name}: prediction error differs')
                        max_mean = float(single(lines, 'benchmark_rank_load_max_over_mean=', name))
                        require(abs(max_mean - max(actual)/(2*tokens/ranks)) < 1e-4,
                                f'{name}: reported max/mean load differs')
                        remote_ratio = float(single(lines, 'benchmark_remote_assignment_ratio=', name))
                        remote_bytes = int(single(lines, 'benchmark_remote_payload_bytes=', name))
                        require(remote_bytes % 1056 == 0 and
                                abs(remote_ratio - remote_bytes/1056/(2*tokens)) < 1e-6,
                                f'{name}: remote assignment count/bytes differ')
                        trials = [pairs(line, 'benchmark,') for line in lines
                                  if line.startswith('benchmark,')]
                        require(len(trials) == 5 and sorted(int(t['trial']) for t in trials) ==
                                list(range(5)), f'{name}: trial IDs differ')
                        for trial in trials:
                            for field in ('tokens','experts','ranks','threads','parallelism','communication'):
                                require(trial.get(field) == expected[field],
                                        f'{name}: trial {field} differs')
                            require(trial.get('bitwise_ok') == '1',
                                    f'{name}: correctness failed')
                            for field in ('end_to_end_seconds', *PHASES):
                                value = float(trial[field])
                                require(math.isfinite(value) and value >= 0,
                                        f'{name}: invalid {field}')
                            require(float(trial['end_to_end_seconds']) > 0,
                                    f'{name}: zero elapsed time')
                        times = [float(t['end_to_end_seconds']) for t in trials]
                        row = dict(round=round_id, ranks=ranks, threads=2, tokens=tokens,
                                   skew=skew, hot_layout=layout, communication=comm,
                                   placement=placement, trials=5,
                                   min_seconds=min(times), median_seconds=statistics.median(times),
                                   max_seconds=max(times), calibration_seconds=calibration_seconds,
                                   mapping_seconds=mapping_seconds,
                                   expert_min=min(expert), expert_max=max(expert),
                                   rank_min=min(actual), rank_max=max(actual),
                                   rank_load_max_over_mean=max(actual)/(2*tokens/ranks),
                                   max_prediction_error_pp=max_prediction_error_pp,
                                   remote_assignment_ratio=remote_ratio,
                                   remote_payload_bytes=remote_bytes,
                                   expert_histogram=';'.join(map(str,expert)),
                                   expert_owners=';'.join(map(str,owners)),
                                   predicted_rank_histogram=';'.join(map(str,predicted)),
                                   actual_rank_histogram=';'.join(map(str,actual)))
                        row.update({f'median_{phase}':statistics.median(float(t[phase]) for t in trials)
                                    for phase in PHASES})
                        rows.append(row)
    require(len(rows) == 80, 'expected 80 paired configurations')
    for row in rows:
        rr = next(r for r in rows if all(r[k] == row[k] for k in
                  ('round','ranks','tokens','skew','hot_layout','communication')) and
                  r['placement'] == 'rr')
        row.update(speedup_vs_paired_rr=rr['median_seconds']/row['median_seconds'],
                   git_commit=meta['git_commit'], pbs_jobid=meta['pbs_jobid'],
                   cpu_model=models[0][1], hosts=';'.join(sorted(hosts)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Validated 80 paired conditions and 400 bitwise-correct timed trials.')
    print(f'Wrote {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        analyze(args.input_dir, args.output)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f'Placement data invalid: {error}\n')
