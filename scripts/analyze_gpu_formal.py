#!/usr/bin/env python3
"""Strictly validate and summarize the Vanda single-A40 formal experiment."""
import argparse
import csv
import math
import re
import statistics
from pathlib import Path

SIZES = (2048, 8192, 32768)
CASES = (('uniform', 'A'), ('moderate', 'A'), ('moderate', 'B'), ('extreme', 'A'))
PHASES = ('host_pack_seconds', 'h2d_seconds', 'kernel_seconds',
          'd2h_seconds', 'host_combine_seconds')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def key_values(text, separator='='):
    result = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        key, value = line.split(separator, 1)
        require(key not in result, f'duplicate {key}')
        result[key] = value
    return result


def csv_pairs(line):
    result = {}
    for item in line.split(','):
        key, value = item.split('=', 1)
        require(key not in result, f'duplicate {key}')
        result[key] = value
    return result


def finite_positive(value, label):
    number = float(value)
    require(math.isfinite(number) and number > 0, f'invalid {label}')
    return number


def check_histogram(histogram, tokens, skew, layout, label):
    require(len(histogram) == 16 and all(x >= 0 for x in histogram), f'{label}: bad histogram')
    require(sum(histogram) == 2 * tokens, f'{label}: assignment total differs')
    if skew == 'uniform':
        require(histogram == [tokens // 8] * 16, f'{label}: uniform trace differs')
    elif skew == 'moderate':
        hot = {0, 4, 8, 12} if layout == 'A' else {0, 1, 2, 3}
        for expert, count in enumerate(histogram):
            target = 0.15 if expert in hot else 1 / 30
            require(abs(count / (2 * tokens) - target) <= 0.01,
                    f'{label}: moderate Expert {expert} share differs')
    else:
        for expert, count in enumerate(histogram):
            target = 0.40 if expert == 0 else 0.04
            require(abs(count / (2 * tokens) - target) <= 0.01,
                    f'{label}: extreme Expert {expert} share differs')


def analyze(directory, output):
    status = key_values((directory / 'job_status.txt').read_text(encoding='utf-8'))
    require(status == {'exit_status': '0', 'stage': 'complete'}, 'job not complete')
    meta = key_values((directory / 'metadata.txt').read_text(encoding='utf-8'))
    require(re.fullmatch(r'[0-9a-f]{40}', meta['git_commit']), 'invalid source commit')
    require(directory.name == 'gpu-formal-' + meta['pbs_jobid'].split('.')[0], 'job ID mismatch')
    expected = dict(node_count='1', gpus='1', processes='1', threads='1',
                    tokens='2048,8192,32768', experts='16', dimension='128',
                    hidden='256', topk='2', router_mode='trace',
                    cases='uniform-A,moderate-A,moderate-B,extreme-A',
                    model_seed='42', input_seed='314159', routing_seed='271828',
                    calibration_seed='161803', rounds='A,B', warmups='3', repeats='5')
    for key, value in expected.items():
        require(meta.get(key) == value, f'metadata mismatch: {key}')
    nodes = (directory / 'nodes.txt').read_text(encoding='utf-8').splitlines()
    require(len(nodes) == 1 and nodes[0], 'expected one node')
    gpu_info = (directory / 'gpu_info.csv').read_text(encoding='utf-8').splitlines()
    require(gpu_info and 'A40' in '\n'.join(gpu_info), 'A40 not recorded')

    expected_logs = {f'{round_id}-n{tokens}-{skew}-{layout}.txt'
                     for round_id in 'AB' for tokens in SIZES for skew, layout in CASES}
    actual_logs = {p.name for p in directory.glob('[AB]-n*.txt')}
    require(actual_logs == expected_logs, 'condition log matrix differs')
    rows = []
    reference_histograms = {}
    device_name = None
    runtime = None
    for round_id in 'AB':
        for tokens in SIZES:
            for skew, layout in CASES:
                label = f'{round_id}-n{tokens}-{skew}-{layout}'
                lines = (directory / f'{label}.txt').read_text(encoding='utf-8').splitlines()
                require((directory / f'{label}.stderr').exists(), f'{label}: stderr missing')
                require(lines[-1:] == ['GPU correctness: PASS'], f'{label}: no final PASS')
                header = [line for line in lines if line.startswith('gpu_device=')]
                require(len(header) == 1, f'{label}: GPU header differs')
                fields = csv_pairs(header[0])
                require(fields.get('visible_devices') == '1', f'{label}: GPU visibility differs')
                require('A40' in fields.get('gpu_device', ''), f'{label}: GPU model differs')
                finite_positive(fields['weights_h2d_seconds'], f'{label} weights H2D')
                if device_name is None:
                    device_name, runtime = fields['gpu_device'], fields['cuda_runtime']
                require((fields['gpu_device'], fields['cuda_runtime']) == (device_name, runtime),
                        f'{label}: GPU identity/runtime changed')
                hist_lines = [line for line in lines if line.startswith('expert_histogram=')]
                require(len(hist_lines) == 1, f'{label}: histogram missing')
                histogram = [int(x) for x in hist_lines[0].split('=', 1)[1].split(',')]
                check_histogram(histogram, tokens, skew, layout, label)
                key = (tokens, skew, layout)
                if key in reference_histograms:
                    require(histogram == reference_histograms[key], f'{label}: matched trace differs')
                reference_histograms[key] = histogram

                gpu = [csv_pairs(line) for line in lines if line.startswith('gpu_trial=')]
                cpu = [csv_pairs(line) for line in lines if line.startswith('cpu_trial=')]
                require(len(gpu) == len(cpu) == 5, f'{label}: trial count differs')
                require(sorted(int(x['gpu_trial']) for x in gpu) == list(range(5)),
                        f'{label}: GPU trial IDs differ')
                require(sorted(int(x['cpu_trial']) for x in cpu) == list(range(5)),
                        f'{label}: CPU trial IDs differ')
                for trial in gpu:
                    require(trial.get('tokens') == str(tokens) and trial.get('experts') == '16'
                            and trial.get('router_mode') == 'trace'
                            and trial.get('correctness') == 'PASS', f'{label}: GPU trial config failed')
                    require(float(trial['max_abs_error']) >= 0 and
                            math.isfinite(float(trial['max_abs_error'])) and
                            0 <= float(trial['relative_l2_error']) <= 1e-5,
                            f'{label}: numerical check failed')
                    for phase in (*PHASES, 'end_to_end_seconds'):
                        finite_positive(trial[phase], f'{label} {phase}')
                for trial in cpu:
                    require(trial.get('tokens') == str(tokens) and trial.get('correctness') == 'PASS',
                            f'{label}: CPU trial failed')
                    finite_positive(trial['serial_seconds'], f'{label} serial')
                gpu_times = [float(x['end_to_end_seconds']) for x in gpu]
                cpu_times = [float(x['serial_seconds']) for x in cpu]
                gpu_median = statistics.median(gpu_times)
                cpu_median = statistics.median(cpu_times)
                kernel_median = statistics.median(float(x['kernel_seconds']) for x in gpu)
                row = dict(round=round_id, tokens=tokens, skew=skew, hot_layout=layout,
                           experts=16, gpu_trials=5, cpu_trials=5,
                           gpu_min_seconds=min(gpu_times), gpu_median_seconds=gpu_median,
                           gpu_max_seconds=max(gpu_times), cpu_min_seconds=min(cpu_times),
                           cpu_median_seconds=cpu_median, cpu_max_seconds=max(cpu_times),
                           implementation_ratio=cpu_median / gpu_median,
                           non_kernel_share=1 - kernel_median / gpu_median,
                           weights_h2d_seconds=float(fields['weights_h2d_seconds']),
                           expert_min=min(histogram), expert_max=max(histogram),
                           expert_histogram=';'.join(map(str, histogram)),
                           max_abs_error=max(float(x['max_abs_error']) for x in gpu),
                           max_relative_l2_error=max(float(x['relative_l2_error']) for x in gpu),
                           gpu_device=device_name, cuda_runtime=runtime,
                           host=nodes[0], git_commit=meta['git_commit'], pbs_jobid=meta['pbs_jobid'])
                row.update({f'median_{phase}': statistics.median(float(x[phase]) for x in gpu)
                            for phase in PHASES})
                rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Validated {len(rows)} conditions, {sum(r["gpu_trials"] for r in rows)} GPU '
          f'and {sum(r["cpu_trials"] for r in rows)} CPU trials on {device_name}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        analyze(args.input_dir, args.output)
    except (OSError, KeyError, ValueError, IndexError) as error:
        parser.exit(1, f'GPU formal data invalid: {error}\n')
