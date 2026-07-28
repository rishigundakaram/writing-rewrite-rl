"""Render the three training-reward curves and the validation win-rate curve
as standalone PNG + SVG images (for pasting into external docs like Notion).

Reads the same cached data the blog uses:
  outputs/blog_curves.json          - per-step training reward, 3 runs
  outputs/winrate_train_curves.json - win rate vs gpt-5.6 every 40 steps
Writes outputs/img/{training_curves,validation_curve}.{png,svg}.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {"ranked": "#3567b3", "absolute_attribute": "#b3403a", "ranked_attribute": "#2e7d4f"}
# blog_curves keys -> reward-design names
BLOG_KEY = {"group": "ranked", "parallel": "absolute_attribute", "group_parallel": "ranked_attribute"}
PANEL_ORDER = ["parallel", "group", "group_parallel"]
PANEL_TITLE = {"parallel": "absolute_attribute",
               "group": "ranked",
               "group_parallel": "ranked_attribute"}
PANEL_YMAX = {"parallel": 1.0, "group": 0.5, "group_parallel": 0.5}


def ema(ys: list[float], alpha: float) -> list[float]:
    acc = ys[0]
    out = []
    for y in ys:
        acc = acc * alpha + y * (1 - alpha)
        out.append(acc)
    return out


def training_curves() -> None:
    curves = json.loads((ROOT / "outputs" / "blog_curves.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), dpi=200)
    for ax, key in zip(axes, PANEL_ORDER):
        name = BLOG_KEY[key]
        color = COLORS[name]
        pts = curves[key]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=color, alpha=0.22, linewidth=0.8)
        ax.plot(xs, ema(ys, 0.9), color=color, linewidth=2)
        ax.set_ylim(0, PANEL_YMAX[key])
        ax.set_xlim(0, max(xs))
        ax.set_title(PANEL_TITLE[key], fontsize=9, color=color, loc="left",
                     fontweight="bold", fontfamily="monospace")
        ax.set_xlabel("step", fontsize=9)
        ax.grid(True, alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("reward", fontsize=9)
    fig.suptitle("Training reward per step", fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"training_curves.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def validation_curve() -> None:
    data = json.loads((ROOT / "outputs" / "winrate_train_curves.json").read_text())
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
    ax.axhline(0.5, color="#888", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(2, 0.51, "parity with gpt-5.6", fontsize=8, color="#888")
    for name in ("ranked", "absolute_attribute", "ranked_attribute"):
        pts = data[name]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=COLORS[name], linewidth=2, marker="o", markersize=3.5, label=name)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(p[0] for p in data["ranked"]))
    ax.set_xlabel("training step", fontsize=10)
    ax.set_ylabel("win rate vs gpt-5.6", fontsize=10)
    ax.set_title("Validation: win rate vs gpt-5.6 during training",
                 fontsize=12, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"validation_curve.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    training_curves()
    validation_curve()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*"))))
