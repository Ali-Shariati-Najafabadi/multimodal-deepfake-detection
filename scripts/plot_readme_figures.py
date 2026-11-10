"""Generate the summary figures shown in the README.

All numbers are copied from the bachelor thesis (Tables 5.2, 5.4, 5.8 and A.1).
Usage: python scripts/plot_readme_figures.py [--out assets]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"Phase 1": "#1f77b4", "Phase 2": "#ff7f0e", "Phase 3": "#2ca02c"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def fig_fusion_auc(out):
    """Fusion AUC: in-domain vs. cross-dataset (binary, AV-Deepfake1M++ training)."""
    groups = ["AVD → AVD\n(in-domain)", "AVD → FakeAVCeleb", "AVD → DeepSpeak"]
    data = {
        "Phase 1": [0.968, 0.823, 0.453],
        "Phase 2": [0.948, 0.877, 0.460],
        "Phase 3": [0.769, 0.657, 0.536],
    }
    x = np.arange(len(groups))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    for i, (name, vals) in enumerate(data.items()):
        bars = ax.bar(x + (i - 1) * w, vals, w, label=name, color=COLORS[name])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.text(1.5, 0.505, "chance", color="grey", fontsize=8, ha="center", va="bottom")
    ax.set_xticks(x, groups)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Fusion AUC")
    ax.set_title("Binary detection: great in-domain, fragile under dataset shift")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "fusion_auc.png")
    plt.close(fig)


def fig_modalities(out):
    """Video vs. audio branch AUC for AVD (4-class) -> FakeAVCeleb."""
    phases = ["Phase 1", "Phase 2", "Phase 3"]
    video = [0.511, 0.833, 0.492]
    audio = [0.872, 0.836, 0.847]
    fusion = [0.727, 0.699, 0.743]
    x = np.arange(len(phases))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    for off, vals, lab, col in [(-w, video, "Video branch", "#d62728"),
                                (0, audio, "Audio branch (HuBERT)", "#17becf"),
                                (w, fusion, "Fusion", "#7f7f7f")]:
        bars = ax.bar(x + off, vals, w, label=lab, color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xticks(x, phases)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC on FakeAVCeleb (unseen)")
    ax.set_title("Audio generalizes, video collapses (AV-Deepfake1M++ 4-class → FakeAVCeleb)")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "modality_asymmetry.png")
    plt.close(fig)


def fig_sampling(out):
    """Uniform vs. segment-based frame sampling (Phase 1, 4-class), video branch."""
    labels = ["In-domain\n(AV-Deepfake1M++)", "Cross-dataset\n(FakeAVCeleb)"]
    uniform = [0.725, 0.511]
    segment = [0.815, 0.869]
    x = np.arange(2)
    w = 0.3
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=160)
    for off, vals, lab, col in [(-w / 2, uniform, "Uniform sampling", "#9aa5b1"),
                                (w / 2, segment, "Segment-based training", "#1f77b4")]:
        bars = ax.bar(x + off, vals, w, label=lab, color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Video-branch AUC")
    ax.set_title("Sampling strategy decides video generalization")
    ax.legend(frameon=False, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "sampling_strategy.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    fig_fusion_auc(out)
    fig_modalities(out)
    fig_sampling(out)
    print(f"Figures written to {out}/")
