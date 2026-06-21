from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BACKGROUND = "#0d1829"
PANEL = "#101c2f"
TEXT = "#e8eef7"
MUTED = "#9fb0c6"
GRID = "#334762"
OLD_COLOR = "#60a5fa"
NEW_COLOR = "#22d3ee"
FORGETTING_COLOR = "#f87171"
TRANSFER_COLOR = "#34d399"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NestDetect forgetting results")
    parser.add_argument("--results-dir", default="results/baseline")
    parser.add_argument("--title", default="NestDetect Research Results")
    args = parser.parse_args()
    results = Path(args.results_dir)
    if not results.is_absolute():
        results = ROOT / results

    comparison = pd.read_csv(results / "comparison.csv")
    incremental = comparison[comparison["strategy"] != "base"].copy()
    labels = {
        "with_replay": "With replay",
        "hope_no_replay": "HoPe without replay",
        "hope_with_replay": "HoPe with replay",
        "no_replay": "Without replay",
        "cms_v1": "CMS V1",
        "cms_v2": "CMS V2",
        "cms_v3": "CMS V3",
        "cms_v4": "CMS V4",
        "cms_v5": "CMS V5",
        "replay_fusion": "Replay-Fusion",
        "replay": "With replay",
    }

    plt.rcParams.update(
        {
            "text.color": TEXT,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 5),
        facecolor=BACKGROUND,
    )
    for axis in axes:
        axis.set_facecolor(PANEL)
        for spine in axis.spines.values():
            spine.set_color(GRID)

    x = range(len(incremental))
    width = 0.34
    axes[0].bar(
        [index - width / 2 for index in x],
        incremental["old_map50_95"],
        width,
        label="Old classes",
        color=OLD_COLOR,
    )
    axes[0].bar(
        [index + width / 2 for index in x],
        incremental["new_map50_95"],
        width,
        label="New classes",
        color=NEW_COLOR,
    )
    display_labels = [labels.get(item, item) for item in incremental["strategy"]]
    axes[0].set_xticks(list(x), display_labels)
    axes[0].tick_params(axis="x", labelrotation=20, labelsize=9)
    for label in axes[0].get_xticklabels():
        label.set_horizontalalignment("right")
    axes[0].set_ylim(0, 0.6)
    axes[0].set_ylabel("mAP50-95")
    axes[0].set_title("Knowledge Retention and Acquisition")
    legend = axes[0].legend(
        facecolor=BACKGROUND,
        edgecolor=GRID,
        labelcolor=TEXT,
    )
    legend.get_frame().set_alpha(0.95)
    axes[0].grid(axis="y", color=GRID, alpha=0.65)

    forgetting = incremental["forgetting"] * 100
    bars = axes[1].bar(
        display_labels,
        forgetting,
        color=[
            FORGETTING_COLOR if value > 0 else TRANSFER_COLOR
            for value in forgetting
        ],
    )
    axes[1].set_ylabel("mAP50-95 decrease (percentage points)")
    axes[1].set_title("Catastrophic Forgetting")
    axes[1].grid(axis="y", color=GRID, alpha=0.65)
    axes[1].tick_params(axis="x", labelrotation=20, labelsize=9)
    for label in axes[1].get_xticklabels():
        label.set_horizontalalignment("right")
    for bar, value in zip(bars, forgetting):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.5,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            color=TEXT,
        )

    figure.suptitle(args.title, fontsize=14, fontweight="bold", color=TEXT)
    figure.tight_layout()
    figure.savefig(results / "forgetting-comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
