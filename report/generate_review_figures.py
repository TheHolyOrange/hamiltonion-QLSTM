"""
Generates the architecture diagram and project-timeline Gantt chart used in
both the Review Report PDF and the Review Presentation PPTX, so the two
documents show identical figures.

Outputs (PNG, high-DPI, transparent-safe white background):
  figs/architecture.png
  figs/timeline_gantt.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = os.path.dirname(os.path.abspath(__file__)) + "/figures"
os.makedirs(OUT, exist_ok=True)

NAVY = "#1a3a6b"
NAVY_LIGHT = "#e8eef7"
BLUE_INPUT = "#c9d9f2"
GREEN_PROC = "#d3ecd9"
ORANGE_OUT = "#fbe3c7"
GREY_TXT = "#333333"


def box(ax, x, y, w, h, text, fc, ec=NAVY, fontsize=10, weight="bold", textcolor=GREY_TXT, lw=1.4, zorder=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, weight=weight, color=textcolor, wrap=True, zorder=zorder + 1)
    return p


def arrow(ax, x1, y1, x2, y2, color=NAVY):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                         linewidth=1.6, color=color, zorder=3)
    ax.add_patch(a)


def architecture_diagram():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    # Input layer
    box(ax, 0.3, 3.3, 2.5, 1.5,
        "Input Layer\n\nRaw multivariate series\n(ETTh1 / Electricity Load /\nJena Climate)",
        BLUE_INPUT, fontsize=9.5)

    # Preprocessing
    box(ax, 0.3, 0.7, 2.5, 2.0,
        "Preprocessing\n\nCyclical time features\nChronological split\nScaling\n24-hour windowing",
        BLUE_INPUT, fontsize=9.5)

    arrow(ax, 2.8, 4.0, 3.3, 3.0)
    arrow(ax, 2.8, 1.7, 3.3, 2.4)

    # Processing layer big box
    box(ax, 3.3, 0.4, 4.6, 4.4, "", NAVY_LIGHT, lw=1.8)
    ax.text(3.3 + 4.6 / 2, 4.55, "Processing Layer — Recurrent Model", ha="center", va="center",
            fontsize=10.5, weight="bold", color=NAVY)

    box(ax, 3.55, 3.35, 4.1, 0.9, "Classical LSTM (baseline)", GREEN_PROC, fontsize=9.5)
    box(ax, 3.55, 2.25, 4.1, 0.9, "Existing QLSTM\n(angle-embedding VQC gates)", GREEN_PROC, fontsize=9.5)
    box(ax, 3.55, 0.65, 4.1, 1.35,
        "Proposed H-QLSTM\nHamiltonian encoding  H(x,θ)\n+ Trotterized evolution  U = exp(−iHΔt)",
        "#bfe3c9", ec="#1f7a3f", fontsize=9.5)

    # Output layer
    arrow(ax, 7.9, 2.6, 8.4, 2.6)
    box(ax, 8.4, 1.7, 2.3, 1.8,
        "Output Layer\n\nRegression head\nForecast\nRMSE / MAE / MAPE",
        ORANGE_OUT, fontsize=9.5)

    plt.tight_layout()
    fig.savefig(f"{OUT}/architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def timeline_gantt():
    months = ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    years = ["2026"] * 5 + ["2027"] * 6

    tasks = [
        ("Data preprocessing (cleaning, normalization,\nsequence generation, train-test split)", 0, 1),
        ("Implement & train Classical LSTM", 0, 2),
        ("Implement & train existing QLSTM", 1, 2),
        ("Design proposed H-QLSTM architecture\n& implementation methodology", 2, 2),
        ("Finalize architecture; prepare implementation", 4, 1),
        ("Develop Hamiltonian-based data\nencoding strategy", 5, 1),
        ("Implement Hamiltonian-based QLSTM model", 5, 2),
        ("Integrate, train & refine complete H-QLSTM", 6, 2),
        ("Optimize parameters; evaluate performance", 8, 1),
        ("Compare with baselines; analyze results;\nprepare dissertation, presentation, demo", 9, 2),
    ]

    n = len(tasks)
    fig, ax = plt.subplots(figsize=(12, 5.6))

    for i, (label, start, dur) in enumerate(tasks):
        y = n - i
        ax.barh(y, dur, left=start, height=0.55, color="#5b7fd6", edgecolor="#2a4a9c", zorder=3)

    ax.scatter([11], [0.5], marker="D", s=110, color="#c0392b", zorder=4)
    ax.text(11.25, 0.5, "Final submission (Review II)", va="center", fontsize=9, color="#c0392b", weight="bold")

    ax.set_yticks(range(1, n + 1))
    ax.set_yticklabels([t[0] for t in tasks][::-1], fontsize=8.8)
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, n + 1)
    ax.set_xticks(range(11))
    ax.set_xticklabels(months, fontsize=9)
    ax.xaxis.tick_top()
    ax.grid(axis="x", linestyle=":", color="#999999", alpha=0.7, zorder=0)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    # Year band on top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([2, 7.5])
    ax2.set_xticklabels(["2026", "2027"], fontsize=11, weight="bold")
    ax2.xaxis.set_ticks_position("top")
    ax2.xaxis.set_label_position("top")
    ax2.spines["top"].set_position(("outward", 22))
    for spine in ["top", "right", "left", "bottom"]:
        ax2.spines[spine].set_visible(False)
    ax2.tick_params(length=0)

    # Phase divider
    ax.axvline(5, color="#888888", linestyle="--", linewidth=1)
    ax.text(2.5, n + 0.6, "Phase I", ha="center", fontsize=10, weight="bold", color=NAVY)
    ax.text(8, n + 0.6, "Phase II", ha="center", fontsize=10, weight="bold", color=NAVY)

    plt.tight_layout()
    fig.savefig(f"{OUT}/timeline_gantt.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    architecture_diagram()
    timeline_gantt()
    print("done")
