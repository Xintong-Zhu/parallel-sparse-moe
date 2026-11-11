#!/usr/bin/env python3
"""Plot validated same-job serial, EP, and DP CPU comparison data."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(summary: Path, output_dir: Path) -> None:
    with summary.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 23 or {r["mode"] for r in rows} != {"serial", "ep", "dp"}:
        raise ValueError("expected a validated 23-point serial/EP/DP summary")
    match = re.fullmatch(r"vanda-cpu-compare-(\d+)\.csv", summary.name)
    if not match:
        raise ValueError("summary filename must contain the PBS job number")
    job_id = match.group(1)
    colors = {"serial": "#373737", "ep": "#1875a1", "dp": "#cb642c"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    serial = [r for r in rows if r["mode"] == "serial"]
    axes[0, 0].loglog([int(r["tokens"]) for r in serial],
                      [float(r["median_seconds"]) for r in serial],
                      "o-", color=colors["serial"], label="Serial")
    axes[0, 0].set(xlabel="Tokens", ylabel="Forward time (s)", title="Serial problem size")
    for mode in ("ep", "dp"):
        series = sorted((r for r in rows if r["mode"] == mode), key=lambda r: int(r["ranks"]))
        x = [int(r["ranks"]) for r in series]
        axes[0, 1].loglog(x, [float(r["median_seconds"]) for r in series],
                          "o-", color=colors[mode], label=mode.upper())
        axes[1, 0].loglog(x, [float(r["speedup_vs_serial"]) for r in series],
                          "o-", color=colors[mode], label=mode.upper())
        axes[1, 1].semilogx(x, [100 * float(r["efficiency_vs_serial"]) for r in series],
                            "o-", color=colors[mode], label=mode.upper())
    axes[0, 1].set(xlabel="Physical CPU cores / MPI ranks", ylabel="Forward time (s)",
                   title="Strong scaling, 131072 Tokens")
    axes[1, 0].loglog([1, 256], [1, 256], ":", color="#777777", label="Ideal")
    axes[1, 0].set(xlabel="Physical CPU cores / MPI ranks", ylabel="Serial / parallel speedup",
                   title="Speedup vs 1-core serial")
    axes[1, 1].axhline(100, linestyle=":", color="#777777")
    axes[1, 1].set(xlabel="Physical CPU cores / MPI ranks", ylabel="Efficiency vs serial (%)",
                   title="Serial-relative efficiency")
    for ax in axes.flat:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    fig.suptitle(f"Vanda CPU comparison: job {job_id}, 256 Experts, uniform Top-2 trace", fontsize=13)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"cpu-compare-{job_id}.{suffix}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    main(args.summary, args.output_dir)
