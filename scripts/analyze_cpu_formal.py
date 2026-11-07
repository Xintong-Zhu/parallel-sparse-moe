#!/usr/bin/env python3
"""Validate a Vanda formal CPU job and summarize its five-trial scaling points."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path


RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
PHASES = (
    "dispatch_pack_seconds",
    "dispatch_comm_seconds",
    "expert_compute_seconds",
    "return_pack_seconds",
    "return_comm_seconds",
    "validation_unpack_seconds",
    "combine_seconds",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pairs(line: str, prefix: str) -> dict[str, str]:
    require(line.startswith(prefix), f"expected {prefix!r}: {line[:80]}")
    fields = line[len(prefix) :].split(",")
    result: dict[str, str] = {}
    for field in fields:
        require("=" in field, f"invalid field: {field!r}")
        key, value = field.split("=", 1)
        require(key not in result, f"duplicate field: {key}")
        result[key] = value
    return result


def read_metadata(path: Path) -> tuple[str, str, list[str]]:
    metadata = {}
    for line in (path / "metadata.txt").read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        metadata[key] = value
    commit = metadata["git_commit"]
    require(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "invalid Git commit")
    require(metadata["mpi_slots"] == "256", "job did not record 256 MPI slots")
    require(metadata["node_count"] == "4", "job did not record four nodes")
    require(metadata["warmups"] == "3" and metadata["repeats"] == "5", "trial protocol changed")
    require(metadata["strong_tokens"] == "131072", "strong workload changed")
    require(metadata["weak_tokens_per_rank"] == "4096", "weak workload changed")

    models = []
    hosts = []
    with (path / "node_models.csv").open(newline="", encoding="utf-8") as source:
        for row in csv.reader(source):
            require(len(row) == 2, "invalid node model line")
            host, model = row
            require(host and model, "missing host or CPU model")
            hosts.append(host)
            models.append(model)
    require(len(hosts) == 4 and len(set(hosts)) == 4, "expected four distinct hosts")
    require(len(set(models)) == 1, "mixed CPU models")
    for probe in ("four_node_probe.txt", "256_rank_probe.txt"):
        lines = (path / probe).read_text(encoding="utf-8").splitlines()
        require("RESULT: PASS" in lines, f"{probe} did not pass")
    return commit, models[0], sorted(hosts)


def parse_point(path: Path, series: str, ranks: int) -> dict[str, object]:
    tokens = 131072 if series == "strong" else 4096 * ranks
    name = f"{series}-p{ranks}.txt"
    lines = (path / name).read_text(encoding="utf-8").splitlines()
    config_lines = [line for line in lines if line.startswith("benchmark_config,")]
    require(len(config_lines) == 1, f"{name}: missing or duplicate configuration")
    config = pairs(config_lines[0], "benchmark_config,")
    expected_config = {
        "tokens": str(tokens),
        "dimension": "128",
        "hidden": "256",
        "experts": "256",
        "ranks": str(ranks),
        "threads": "1",
        "parallelism": "ep",
        "communication": "alltoallv",
        "router_mode": "trace",
        "skew": "uniform",
        "placement": "rr",
        "omp_schedule": "static",
        "model_seed": "42",
        "input_seed": "314159",
        "routing_seed": "271828",
        "calibration_seed": "161803",
    }
    for key, expected in expected_config.items():
        require(config.get(key) == expected, f"{name}: {key} differs from frozen configuration")

    for prefix, count, expected in (
        ("benchmark_expert_histogram=", 256, 2 * tokens // 256),
        ("benchmark_rank_assignment_histogram=", ranks, 2 * tokens // ranks),
    ):
        matches = [line for line in lines if line.startswith(prefix)]
        require(len(matches) == 1, f"{name}: missing or duplicate {prefix}")
        histogram = [int(value) for value in matches[0][len(prefix) :].split(",")]
        require(len(histogram) == count, f"{name}: histogram length mismatch")
        require(all(value == expected for value in histogram), f"{name}: realized load mismatch")

    trial_lines = [line for line in lines if line.startswith("benchmark,trial=")]
    require(len(trial_lines) == 5, f"{name}: expected five measured trials")
    trials = [pairs(line, "benchmark,") for line in trial_lines]
    require(sorted(int(trial["trial"]) for trial in trials) == list(range(5)), f"{name}: trial IDs differ")
    times: list[float] = []
    phase_times: dict[str, list[float]] = {phase: [] for phase in PHASES}
    for trial in trials:
        for key, expected in (
            ("parallelism", "ep"), ("communication", "alltoallv"),
            ("tokens", str(tokens)), ("experts", "256"),
            ("ranks", str(ranks)), ("threads", "1"), ("bitwise_ok", "1"),
        ):
            require(trial.get(key) == expected, f"{name}: trial {trial.get('trial')} failed {key}")
        elapsed = float(trial["end_to_end_seconds"])
        require(math.isfinite(elapsed) and elapsed > 0, f"{name}: invalid elapsed time")
        times.append(elapsed)
        for phase in PHASES:
            value = float(trial[phase])
            require(math.isfinite(value) and value >= 0, f"{name}: invalid {phase}")
            phase_times[phase].append(value)

    result: dict[str, object] = {
        "series": series,
        "ranks": ranks,
        "threads": 1,
        "tokens": tokens,
        "experts": 256,
        "trial_count": 5,
        "min_seconds": min(times),
        "median_seconds": statistics.median(times),
        "max_seconds": max(times),
    }
    for phase in PHASES:
        result[f"median_{phase}"] = statistics.median(phase_times[phase])
    return result


def analyze(input_dir: Path, output_csv: Path) -> None:
    commit, model, hosts = read_metadata(input_dir)
    rows = [parse_point(input_dir, series, ranks) for series in ("strong", "weak") for ranks in RANKS]
    baselines = {series: next(row["median_seconds"] for row in rows
                              if row["series"] == series and row["ranks"] == 1)
                 for series in ("strong", "weak")}
    for row in rows:
        baseline = float(baselines[str(row["series"])])
        elapsed = float(row["median_seconds"])
        ranks = int(row["ranks"])
        row["speedup"] = baseline / elapsed if row["series"] == "strong" else ""
        row["efficiency"] = baseline / (ranks * elapsed) if row["series"] == "strong" else baseline / elapsed
        row["git_commit"] = commit
        row["cpu_model"] = model
        row["hosts"] = ";".join(hosts)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for series in ("strong", "weak"):
        endpoint = next(row for row in rows if row["series"] == series and row["ranks"] == 256)
        print(f"{series}: P=256, median={endpoint['median_seconds']:.6f} s, "
              f"efficiency={endpoint['efficiency']:.4f}")
    print(f"Validated 18 points and 90 trials from {commit[:7]} on {model}.")
    print(f"Summary: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="private/formal-cpu-<jobid> directory")
    parser.add_argument("--output", type=Path, required=True, help="validated summary CSV")
    arguments = parser.parse_args()
    try:
        analyze(arguments.input_dir, arguments.output)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Formal CPU data invalid: {error}\n")
