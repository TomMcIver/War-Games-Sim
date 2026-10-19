"""Plot a timed experiment's training and fixed-validation learning curves."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _number(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def _interval(rate: float, count: float) -> tuple[float, float]:
    z = 1.96
    den = 1 + z * z / count
    mid = (rate + z * z / (2 * count)) / den
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / den
    return max(0, mid - radius), min(1, mid + radius)


def plot(run_dir: Path) -> tuple[Path, Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install the optional plot dependency: pip install -e 'python[plot]'") from exc

    train = _rows(run_dir / "metrics.csv")
    val = _rows(run_dir / "learning_curve.csv")
    if not train or not val:
        raise ValueError("run needs both training metrics and validation samples")
    updates = [_number(r, "update") for r in train]
    checkpoints = [_number(r, "update") for r in val]
    stages = [r["stage_scenario"] for r in val]
    elapsed = _number(val[-1], "elapsed_minutes")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "savefig.facecolor": "white"})
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=True)
    fig.patch.set_facecolor("#f8fafc")
    colors = {"train": "#678399", "anchor": "#2065a5", "stage": "#117f72",
              "draw": "#bc7c28", "hit": "#a94e67", "length": "#614c9e"}
    for ax in axs.flat:
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#e6ebef", linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, max(updates))

    ax = axs[0, 0]
    ax.plot(updates, [_number(r, "win_rate") for r in train],
            color=colors["train"], lw=1.5, alpha=0.65,
            label="Training win rate vs league (rolling)")
    anchor_rate = [_number(r, "anchor_win_rate") for r in val]
    ax.plot(checkpoints, anchor_rate, "o-", color=colors["anchor"],
            lw=2.2, ms=4, label="Fixed duel validation vs random")
    bands = [_interval(p, _number(r, "anchor_episodes"))
             for p, r in zip(anchor_rate, val)]
    ax.fill_between(checkpoints, [b[0] for b in bands], [b[1] for b in bands],
                    color=colors["anchor"], alpha=0.12, label="Approx. 95% win interval")
    ax.set(title="Win rate", ylabel="Fraction of matches", ylim=(0, 1))
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    ax = axs[0, 1]
    ax.plot(checkpoints, [_number(r, "validation_win_rate") for r in val],
            "o-", color=colors["stage"], lw=2, ms=4, label="Fixed validation wins")
    # A promotion is applied before the trainer writes its row; at that
    # update, its evaluation was played on the previous stage. Omit that
    # boundary point from the current-stage comparison.
    selection = [
        (u, _number(r, "eval_win_rate_random"))
        for i, (u, r) in enumerate(zip(updates, train))
        if math.isfinite(_number(r, "eval_win_rate_random"))
        and (i == 0 or r["stage"] == train[i - 1]["stage"])
    ]
    if selection:
        ax.plot([p[0] for p in selection], [p[1] for p in selection],
                "s--", color=colors["train"], lw=1.5, ms=3,
                label="Promotion evaluation wins")
    ax.plot(checkpoints, [_number(r, "validation_draw_rate") for r in val],
            "o-", color=colors["draw"], lw=1.8, ms=3, label="Fixed validation draws")
    ax.set(title="Validation on the current curriculum stage",
           ylabel="Fraction of matches", ylim=(0, 1))
    ax.legend(frameon=False, fontsize=8)

    ax = axs[1, 0]
    ax.plot(checkpoints, [_number(r, "anchor_timeout_rate") for r in val],
            "o-", color=colors["draw"], lw=2, ms=4, label="Duel timeout rate")
    ax.plot(checkpoints, [_number(r, "anchor_hit_rate") for r in val],
            "o-", color=colors["hit"], lw=2, ms=4, label="Learner shot hit rate")
    ax.set(title="Combat quality on fixed duels", xlabel="Training update",
           ylabel="Fraction", ylim=(0, 1))
    ax.legend(frameon=False, fontsize=8)

    ax = axs[1, 1]
    ax.plot(updates, [_number(r, "ep_len") for r in train],
            color=colors["train"], lw=1.5, alpha=0.65,
            label="Training episode length (rolling)")
    ax.plot(checkpoints, [_number(r, "anchor_mean_len") for r in val],
            "o-", color=colors["length"], lw=2.2, ms=4,
            label="Fixed duel episode length")
    ax.set(title="How quickly matches finish", xlabel="Training update",
           ylabel="Ticks")
    ax.legend(frameon=False, fontsize=8)

    boundaries = [(checkpoints[i], stages[i]) for i in range(1, len(stages))
                  if stages[i] != stages[i - 1]]
    for axis in axs.flat:
        for update, scenario in boundaries:
            axis.axvline(update, color="#8293a0", ls="--", lw=1, alpha=0.7)
    for update, scenario in boundaries:
        axs[0, 1].annotate(scenario, (update, 0.99), xytext=(3, -3),
                           textcoords="offset points", ha="left", va="top",
                           fontsize=8, color="#526270", rotation=90)

    fig.suptitle("Real-simulator learning curve", x=0.07, y=0.985,
                 ha="left", fontsize=19, fontweight="bold", color="#183044")
    fig.text(0.07, 0.925,
             f"{run_dir.name}  •  {int(max(updates))} updates  •  {elapsed:.1f} min  "
             "•  fixed validation seeds 31001 and 42002",
             color="#526270", fontsize=10)
    fig.text(0.07, 0.02,
             "Training and validation face different opponents; compare their trends, not their absolute levels. "
             "Shaded intervals reflect match count only.",
             color="#607180", fontsize=8)
    fig.tight_layout(rect=(0.055, 0.045, 0.985, 0.865), w_pad=2.6, h_pad=2.5)
    png = run_dir / "learning_curve.png"
    svg = run_dir / "learning_curve.svg"
    fig.savefig(png, dpi=180)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot a timed training experiment")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    for path in plot(args.run_dir):
        print(path)


if __name__ == "__main__":
    main()
