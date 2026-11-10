#!/usr/bin/env python3
"""Plot the validated four-node EP scaling summary (never raw benchmark logs)."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def read_summary(path: Path) -> dict[str, list[dict[str, float]]]:
    with path.open(newline="", encoding="utf-8") as source:
        raw = list(csv.DictReader(source))
    if len(raw) != 18 or {row["series"] for row in raw} != {"strong", "weak"}:
        raise ValueError("expected the 18-point validated EP summary")
    if len({row["git_commit"] for row in raw}) != 1 or len({row["hosts"] for row in raw}) != 1:
        raise ValueError("summary mixes code revisions or node allocations")
    data: dict[str, list[dict[str, float]]] = {}
    for series in ("strong", "weak"):
        subset = sorted((row for row in raw if row["series"] == series),
                        key=lambda row: int(row["ranks"]))
        if tuple(int(row["ranks"]) for row in subset) != RANKS:
            raise ValueError(f"{series}: missing or duplicate rank count")
        for row in subset:
            if (int(row["trial_count"]) != 5 or int(row["threads"]) != 1 or
                int(row["experts"]) != 256 or
                int(row["tokens"]) != (131072 if series == "strong" else 4096 * int(row["ranks"]))):
                raise ValueError(f"{series}: unexpected experiment configuration")
            for field in ("min_seconds", "median_seconds", "max_seconds", "efficiency"):
                row[field] = float(row[field])
            if not (all(math.isfinite(row[field]) for field in
                        ("min_seconds", "median_seconds", "max_seconds", "efficiency")) and
                    0 < row["min_seconds"] <= row["median_seconds"] <= row["max_seconds"] and
                    row["efficiency"] > 0):
                raise ValueError(f"{series}: invalid timing or efficiency")
        data[series] = subset
    return data


def save(fig: plt.Figure, folder: Path, name: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{suffix}", dpi=220)
    plt.close(fig)


def plot(data: dict[str, list[dict[str, float]]], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    strong, weak = data["strong"], data["weak"]
    ranks = list(RANKS)
    elapsed = lambda rows: [row["median_seconds"] for row in rows]
    spread = lambda rows: [
        [row["median_seconds"] - row["min_seconds"] for row in rows],
        [row["max_seconds"] - row["median_seconds"] for row in rows],
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(ranks, elapsed(strong), yerr=spread(strong), fmt="o-",
                capsize=3, label="EP median (5 trials; min–max)")
    ax.plot(ranks, [strong[0]["median_seconds"] / p for p in ranks], "--", label="Ideal from P=1 EP")
    ax.set(xscale="log", yscale="log", xlabel="CPU cores / MPI ranks", ylabel="Forward time (s)",
           title="Strong scaling: 131,072 tokens, 256 experts")
    ax.set_xticks(ranks, [str(p) for p in ranks])
    ax.legend()
    save(fig, folder, "cpu_strong_time")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ranks, [float(row["speedup"]) for row in strong], "o-", label="EP measured")
    ax.plot(ranks, ranks, "--", label="Ideal P")
    ax.set(xscale="log", yscale="log", xlabel="CPU cores / MPI ranks", ylabel="Speedup vs P=1 EP",
           title="Strong scaling speedup")
    ax.set_xticks(ranks, [str(p) for p in ranks])
    ax.legend()
    save(fig, folder, "cpu_strong_speedup")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(ranks, elapsed(weak), yerr=spread(weak), fmt="o-",
                capsize=3, label="EP median (5 trials; min–max)")
    ax.axhline(weak[0]["median_seconds"], linestyle="--", color="gray", label="Ideal constant time")
    ax.set(xscale="log", yscale="log", xlabel="CPU cores / MPI ranks (4,096 tokens per rank)",
           ylabel="Forward time (s)", title="Weak scaling: 256 experts")
    ax.set_xticks(ranks, [str(p) for p in ranks])
    ax.legend()
    save(fig, folder, "cpu_weak_time")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ranks, [row["efficiency"] for row in strong], "o-", label="Strong: T1 / (P·TP)")
    ax.plot(ranks, [row["efficiency"] for row in weak], "s-", label="Weak: T1 / TP")
    ax.axhline(1, linestyle="--", color="gray")
    ax.set(xscale="log", xlabel="CPU cores / MPI ranks", ylabel="Parallel efficiency",
           title="EP scaling efficiency")
    ax.set_xticks(ranks, [str(p) for p in ranks])
    ax.legend()
    save(fig, folder, "cpu_efficiency")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validated_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    try:
        plot(read_summary(args.validated_csv), args.output_dir)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Cannot plot CPU results: {error}\n")
