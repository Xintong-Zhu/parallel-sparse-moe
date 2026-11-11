#!/usr/bin/env python3
"""Validate one Vanda serial/EP/DP comparison job and summarize its raw trials."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path

from analyze_cpu_formal import PHASES, RANKS, pairs, require

SIZES = (512, 2048, 8192, 32768, 131072)
SEEDS = {"model_seed": "42", "input_seed": "314159", "routing_seed": "271828"}


def read_context(directory: Path) -> tuple[str, str, str]:
    status = dict(line.split("=", 1) for line in (directory / "job_status.txt").read_text().splitlines())
    require(status == {"exit_status": "0", "stage": "complete"}, "job did not complete")
    meta = dict(line.split("=", 1) for line in (directory / "metadata.txt").read_text().splitlines())
    commit = meta["git_commit"]
    require(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "invalid Git commit")
    expected = {
        "mpi_slots": "256", "node_count": "4", "ranks": "1,2,4,8,16,32,64,128,256",
        "threads": "1", "tokens": "131072", "experts": "256",
        "router_mode": "trace", "skew": "uniform", "warmups": "3", "repeats": "5",
        "cpu_binding": "slot:PE=1,bind-to=core",
    }
    for key, value in expected.items():
        require(meta.get(key) == value, f"metadata {key} differs from frozen settings")
    job_id = meta["pbs_jobid"].split(".", 1)[0]
    require(directory.name == f"compare-cpu-{job_id}", "job ID and directory disagree")
    models = list(csv.reader((directory / "node_models.csv").open(newline="", encoding="utf-8")))
    require(len(models) == 4 and all(len(row) == 2 and all(row) for row in models), "invalid node model records")
    hosts = [row[0] for row in models]
    require(len(set(hosts)) == 4 and len({row[1] for row in models}) == 1, "nodes are not four distinct same-model hosts")
    recorded_hosts = (directory / "nodes.txt").read_text().splitlines()
    short_host = lambda value: value.split(".", 1)[0].lower()
    require(sorted(map(short_host, hosts)) == sorted(map(short_host, recorded_hosts)),
            "PBS hosts differ from model probe")
    for probe in ("four_node_probe.txt", "256_rank_probe.txt"):
        require("RESULT: PASS" in (directory / probe).read_text().splitlines(), f"{probe} failed")
    return commit, models[0][1], ";".join(sorted(hosts))


def one_line(lines: list[str], prefix: str, name: str) -> str:
    found = [line for line in lines if line.startswith(prefix)]
    require(len(found) == 1, f"{name}: expected one {prefix}")
    return found[0]


def elapsed(trials: list[dict[str, str]], name: str) -> list[float]:
    require(len(trials) == 5, f"{name}: expected five trials")
    require(sorted(int(t["trial"]) for t in trials) == list(range(5)), f"{name}: trial IDs differ")
    times = []
    for trial in trials:
        require(trial.get("bitwise_ok") == "1", f"{name}: incorrect trial")
        value = float(trial["end_to_end_seconds"])
        require(math.isfinite(value) and value > 0, f"{name}: invalid elapsed time")
        times.append(value)
    return times


def summarize(mode: str, ranks: int, tokens: int, trials: list[dict[str, str]]) -> dict[str, object]:
    times = elapsed(trials, f"{mode} P={ranks} N={tokens}")
    return {"mode": mode, "ranks": ranks, "threads": 1, "tokens": tokens,
            "experts": 256, "trials": 5, "min_seconds": min(times),
            "median_seconds": statistics.median(times), "max_seconds": max(times)}


def serial_point(directory: Path, tokens: int) -> dict[str, object]:
    name = f"serial-n{tokens}.txt"
    lines = (directory / name).read_text(encoding="utf-8").splitlines()
    require(any("Rank 0 bound to" in line for line in lines), f"{name}: no binding report")
    config = pairs(one_line(lines, "serial_config,", name), "serial_config,")
    expected = {"tokens": str(tokens), "experts": "256", "dimension": "128",
                "hidden": "256", "router_mode": "trace", "skew": "uniform", **SEEDS}
    for key, value in expected.items():
        require(config.get(key) == value, f"{name}: {key} differs")
    trials = [pairs(line, "serial_benchmark,") for line in lines if line.startswith("serial_benchmark,")]
    for trial in trials:
        require(trial.get("tokens") == str(tokens) and trial.get("experts") == "256", f"{name}: trial configuration differs")
    return summarize("serial", 1, tokens, trials)


def parallel_point(directory: Path, mode: str, ranks: int) -> dict[str, object]:
    name = f"{mode}-p{ranks}.txt"
    lines = (directory / name).read_text(encoding="utf-8").splitlines()
    config = pairs(one_line(lines, "benchmark_config,", name), "benchmark_config,")
    expected = {"tokens": "131072", "dimension": "128", "hidden": "256", "experts": "256",
                "ranks": str(ranks), "threads": "1", "parallelism": mode,
                "communication": "alltoallv", "router_mode": "trace", "skew": "uniform",
                "placement": "rr", "omp_schedule": "static", "calibration_seed": "161803", **SEEDS}
    for key, value in expected.items():
        require(config.get(key) == value, f"{name}: {key} differs")
    for prefix, count, target in (("benchmark_expert_histogram=", 256, 1024),
                                  ("benchmark_rank_assignment_histogram=", ranks, 262144 // ranks)):
        values = [int(x) for x in one_line(lines, prefix, name)[len(prefix):].split(",")]
        require(len(values) == count and all(x == target for x in values), f"{name}: realized load differs")
    trials = [pairs(line, "benchmark,") for line in lines if line.startswith("benchmark,")]
    for trial in trials:
        for key, value in (("parallelism", mode), ("communication", "alltoallv"),
                           ("tokens", "131072"), ("experts", "256"),
                           ("ranks", str(ranks)), ("threads", "1")):
            require(trial.get(key) == value, f"{name}: trial {key} differs")
        for phase in PHASES:
            value = float(trial[phase])
            require(math.isfinite(value) and value >= 0, f"{name}: invalid {phase}")
    row = summarize(mode, ranks, 131072, trials)
    for phase in PHASES:
        row[f"median_{phase}"] = statistics.median(float(t[phase]) for t in trials)
    return row


def analyze(directory: Path, output: Path) -> None:
    commit, model, hosts = read_context(directory)
    rows = [serial_point(directory, size) for size in SIZES]
    rows += [parallel_point(directory, mode, ranks) for mode in ("ep", "dp") for ranks in RANKS]
    serial = next(float(r["median_seconds"]) for r in rows if r["mode"] == "serial" and r["tokens"] == 131072)
    baselines = {mode: next(float(r["median_seconds"]) for r in rows if r["mode"] == mode and r["ranks"] == 1)
                 for mode in ("ep", "dp")}
    fields = ["mode", "ranks", "threads", "tokens", "experts", "trials", "min_seconds",
              "median_seconds", "max_seconds", "speedup_vs_serial", "efficiency_vs_serial",
              "speedup_vs_mode_p1", "efficiency_vs_mode_p1"]
    fields += [f"median_{phase}" for phase in PHASES]
    fields += ["git_commit", "cpu_model", "hosts"]
    for row in rows:
        mode = str(row["mode"])
        if mode != "serial":
            time = float(row["median_seconds"])
            ranks = int(row["ranks"])
            row["speedup_vs_serial"] = serial / time
            row["efficiency_vs_serial"] = serial / (ranks * time)
            row["speedup_vs_mode_p1"] = baselines[mode] / time
            row["efficiency_vs_mode_p1"] = baselines[mode] / (ranks * time)
        row.update(git_commit=commit, cpu_model=model, hosts=hosts)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Validated 23 points / 115 correct trials from {commit[:7]} on {model}.")
    for mode in ("serial", "ep", "dp"):
        row = next(r for r in rows if r["mode"] == mode and r["tokens"] == 131072 and r["ranks"] == (1 if mode == "serial" else 256))
        print(f"{mode} P={row['ranks']}: median={row['median_seconds']:.6f} s")
    print(f"Summary: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        analyze(args.input_dir, args.output)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"CPU comparison invalid: {error}\n")
