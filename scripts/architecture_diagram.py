"""
architecture_diagram.py
=======================
Generates Figure 4.2 — Proposed system architecture.

A clean block-and-arrow diagram (matplotlib) of the five-component modular
system: PACS frontend workstation, preprocessing module, BiomedCLIP VLM,
Qwen2.5-1.5B-Instruct report generator, and MongoDB database — showing the end-to-end flow.

Run from project root:
    python scripts/architecture_diagram.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).parent.parent

# ── palette (muted, technical-document look) ─────────────────────
C_FRONT = "#dce8fb"; E_FRONT = "#2f6fd0"
C_PROC  = "#e6f0e0"; E_PROC  = "#4f8a3d"
C_VLM   = "#fdeede"; E_VLM   = "#c97a1e"
C_LLM   = "#f3e3f5"; E_LLM   = "#8b3fa0"
C_DB    = "#e3e8ee"; E_DB    = "#566273"


def box(ax, x, y, w, h, title, lines, fc, ec):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h - 0.26, title,
            ha="center", va="top", fontsize=11, fontweight="bold",
            color="#1c2530", zorder=3)
    ax.text(x + w / 2, y + h - 0.62, "\n".join(lines),
            ha="center", va="top", fontsize=8.4, color="#39434f", zorder=3)


def arrow(ax, p0, p1, label="", rad=0.0, off=(0, 0)):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=16,
        linewidth=1.5, color="#3a4350",
        connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        mx = (p0[0] + p1[0]) / 2 + off[0]
        my = (p0[1] + p1[1]) / 2 + off[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=7.8,
                style="italic", color="#283040",
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec="none", alpha=0.9), zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(13.6, 7.2))
    ax.set_xlim(0, 13.6); ax.set_ylim(0, 7.2); ax.axis("off")

    ax.text(6.8, 6.95, "Figure 4.2  Proposed System Architecture",
            ha="center", fontsize=14, fontweight="bold", color="#16202b")

    BW, BH = 2.9, 1.5

    # ── Frontend (top, spanning) ─────────────────────────────────
    box(ax, 4.85, 5.0, 3.9, 1.3,
        "PACS-style Frontend Workstation",
        ["Web UI · upload X-ray, view image,",
         "review findings & confidence,",
         "explainability, export PDF"],
        C_FRONT, E_FRONT)

    # ── Pipeline row (left -> right) ─────────────────────────────
    y = 2.7
    box(ax, 0.4,  y, BW, BH, "Image Preprocessing",
        ["Grayscale · denoise · CLAHE", "resize 224x224 · normalize"], C_PROC, E_PROC)
    box(ax, 3.7,  y, BW, BH, "BiomedCLIP (VLM)",
        ["Image-text matching vs.", "thoracic disease descriptions",
         "-> disease + confidence"], C_VLM, E_VLM)
    box(ax, 7.0,  y, BW, BH, "Qwen2.5 Report Generator",
        ["Structured prompt template", "-> radiology-style", "observational report"], C_LLM, E_LLM)
    box(ax, 10.3, y, BW, BH, "MongoDB Database",
        ["Stores report, predictions,", "metrics & metadata", "for retrieval/analysis"], C_DB, E_DB)

    # ── Flow arrows ──────────────────────────────────────────────
    # Frontend -> Preprocessing (upload)
    arrow(ax, (5.2, 5.0), (1.85, y + BH), "1. upload CXR", rad=-0.25, off=(-0.2, 0.35))
    # Preprocessing -> VLM
    arrow(ax, (0.4 + BW, y + BH / 2), (3.7, y + BH / 2), "2. preprocessed", off=(0, 0.32))
    # VLM -> LLM
    arrow(ax, (3.7 + BW, y + BH / 2), (7.0, y + BH / 2), "3. predictions", off=(0, 0.32))
    # LLM -> DB
    arrow(ax, (7.0 + BW, y + BH / 2), (10.3, y + BH / 2), "4. store", off=(0, 0.32))
    # DB / results -> Frontend (display)
    arrow(ax, (11.75, y + BH), (8.4, 5.0), "5. results & report", rad=-0.25, off=(0.5, 0.35))

    ax.text(6.8, 0.55,
            "End-to-end AI-assisted radiology reporting workflow",
            ha="center", fontsize=9.5, style="italic", color="#566273")

    fig.tight_layout()
    out = ROOT / "data" / "processed" / "system_architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


if __name__ == "__main__":
    main()
