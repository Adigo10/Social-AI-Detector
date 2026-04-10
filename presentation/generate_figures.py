"""
Generate figures for the Data Preparation presentation section.
Run: python presentation/generate_figures.py
Outputs: presentation/figures/pipeline_diagram.png
         presentation/figures/leakage_prevention.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ── shared style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE   = "#3B82F6"
INDIGO = "#6366F1"
GRAY   = "#6B7280"
GREEN  = "#10B981"
ORANGE = "#F59E0B"
RED    = "#EF4444"
BG     = "#F9FAFB"


# ── Figure 1: Pipeline Diagram ───────────────────────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    steps = [
        ("Download", "3 datasets\n(MultiSocial,\nHC3, RAID)", "#DBEAFE", BLUE),
        ("Preprocess", "Unified schema\n495K records\ncorpus.jsonl", "#EDE9FE", INDIGO),
        ("Embed", "Gemini Emb-2\n768-dim\n1.4 GB .npy", "#D1FAE5", GREEN),
        ("FAISS Index", "Train split only\n(346K vectors)\nno leakage", "#FEF3C7", ORANGE),
        ("Training Data", "RAG / plain\nbalanced 50/50\n12 JSONL files", "#FEE2E2", RED),
    ]

    box_w, box_h = 0.16, 0.54
    gap = 0.045
    total = len(steps) * box_w + (len(steps) - 1) * gap
    x0 = (1 - total) / 2

    for i, (title, subtitle, bg, border) in enumerate(steps):
        x = x0 + i * (box_w + gap)
        y = 0.22
        fancy = FancyBboxPatch((x, y), box_w, box_h,
                               boxstyle="round,pad=0.01",
                               facecolor=bg, edgecolor=border, linewidth=2,
                               transform=ax.transAxes, clip_on=False)
        ax.add_patch(fancy)
        ax.text(x + box_w / 2, y + box_h - 0.065, title,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=border, transform=ax.transAxes)
        ax.text(x + box_w / 2, y + box_h / 2 - 0.04, subtitle,
                ha="center", va="center", fontsize=8.5, color="#374151",
                transform=ax.transAxes, linespacing=1.4)

        # step number bubble
        ax.text(x + box_w / 2, y - 0.08, f"Step {i+1}" if i < 4 else "Step 5",
                ha="center", va="center", fontsize=8, color=GRAY,
                transform=ax.transAxes)

        if i < len(steps) - 1:
            ax_x = x + box_w + 0.005
            ax.annotate("", xy=(ax_x + gap - 0.01, y + box_h / 2),
                        xytext=(ax_x, y + box_h / 2),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=GRAY,
                                        lw=1.8, mutation_scale=16))

    ax.set_title("Data Preparation Pipeline  ·  NTU AI6130 Group G30",
                 fontsize=13, fontweight="bold", color="#111827", pad=12)
    plt.tight_layout()
    path = os.path.join(OUT, "pipeline_diagram.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved {path}")


# ── Figure 2: Leakage Prevention ────────────────────────────────────────────
def fig_leakage_prevention():
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)

    def box(cx, cy, w, h, fc, ec, label, sublabel=""):
        r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0.15",
                           facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(r)
        ax.text(cx, cy + (0.18 if sublabel else 0), label,
                ha="center", va="center", fontsize=10, fontweight="bold", color=ec)
        if sublabel:
            ax.text(cx, cy - 0.3, sublabel,
                    ha="center", va="center", fontsize=8, color=GRAY)

    def arrow(x1, y1, x2, y2, color=GRAY, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=1.8, mutation_scale=14))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.1, my + 0.15, label, fontsize=8, color=color, style="italic")

    # Split boxes
    box(1.4, 3.4, 2.0, 0.8, "#DBEAFE", BLUE, "Train Split", "346,853 records")
    box(1.4, 2.1, 2.0, 0.8, "#D1FAE5", GREEN, "Val Split",  "74,326 records")
    box(1.4, 0.8, 2.0, 0.8, "#FEF3C7", ORANGE, "Test Split", "74,326 records")

    # FAISS index box
    box(5.5, 3.4, 2.2, 0.8, "#EDE9FE", INDIGO, "FAISS Index", "train-only (346K vectors)")

    # Training data box
    box(9.0, 3.4, 1.6, 0.8, "#FEE2E2", RED, "Training\nData", "")

    # arrows
    arrow(2.4, 3.4, 4.4, 3.4, INDIGO, "indexes")        # train → index
    arrow(6.6, 3.4, 8.2, 3.4, RED, "builds")            # index → training data

    # val and test query the index
    arrow(2.4, 2.1, 4.4, 3.1, GREEN, "queries only")
    arrow(2.4, 0.8, 4.4, 2.9, ORANGE, "queries only")

    # X marks to show val/test NOT in index
    ax.text(4.9, 2.6, "val/test\nnot indexed", ha="center", va="center",
            fontsize=8, color=RED, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEE2E2", edgecolor=RED, lw=1.2))

    ax.set_title("Data Leakage Prevention: Train-Only FAISS Index",
                 fontsize=12, fontweight="bold", color="#111827", pad=10)
    plt.tight_layout()
    path = os.path.join(OUT, "leakage_prevention.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved {path}")


# ── Figure 3: Dataset Composition ───────────────────────────────────────────
def fig_dataset_composition():
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Approximate counts derived from processed corpus (410,087 + 85,418 = 495,505)
    datasets = ["MultiSocial", "HC3", "Total"]
    ai_counts    = [352_675,  48_688, 401_363]
    human_counts = [ 57_412,  36_730,  94_142]
    totals       = [a + h for a, h in zip(ai_counts, human_counts)]

    x = range(len(datasets))
    bars_ai    = ax.bar(x, ai_counts,    color="#EF4444", label="AI-generated")
    bars_human = ax.bar(x, human_counts, bottom=ai_counts, color="#60A5FA", label="Human")

    # Percentage labels inside bars
    for i, (a, h, t) in enumerate(zip(ai_counts, human_counts, totals)):
        ax.text(i, a / 2,         f"{round(a/t*100)}%", ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        ax.text(i, a + h / 2,     f"{round(h/t*100)}%", ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")

    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylabel("Records", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v/1000)}K"))
    ax.set_ylim(0, 550_000)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Dataset Composition — Processed Records",
                 fontsize=13, fontweight="bold", color="#111827", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    path = os.path.join(OUT, "dataset_composition.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    print("Generating figures...")
    fig_dataset_composition()
    fig_pipeline()
    fig_leakage_prevention()
    print("Done. Figures saved to presentation/figures/")
