import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# AUC-Werte (letzter Checkpoint / hoechste Epoch je Lauf).
# Struktur: data[modalitaet][zeile] = {ziel: auc}
# Zeilen: AVD-trainierte Modelle (Phase x Aufgabe). Ziel-Spalten: Trainings-
# datensatz (in-domain) sowie die beiden Cross-Datensaetze.
ROWS = ["P1 4-Kl.", "P2 4-Kl.", "P3 4-Kl.", "P1 binaer", "P2 binaer", "P3 binaer"]
COLS = ["AVD (in-domain)", "FakeAVCeleb", "DeepSpeak"]

VIDEO = [
    [0.7248, 0.5106, np.nan],
    [0.7088, 0.8325, np.nan],
    [0.5359, 0.4921, np.nan],
    [0.9560, 0.7401, 0.4553],
    [0.9242, 0.8286, 0.4664],
    [0.7022, 0.5254, 0.5862],
]
AUDIO = [
    [0.8018, 0.8716, np.nan],
    [0.7955, 0.8363, np.nan],
    [0.7989, 0.8465, np.nan],
    [0.7118, 0.7187, 0.4729],
    [0.7155, 0.6971, 0.5038],
    [0.6895, 0.6771, 0.4846],
]
FUSION = [
    [0.8236, 0.7267, np.nan],
    [0.8097, 0.6990, np.nan],
    [0.7086, 0.7430, np.nan],
    [0.9682, 0.8233, 0.4528],
    [0.9485, 0.8770, 0.4595],
    [0.7692, 0.6572, 0.5363],
]

# FakeAVCeleb-trainierte Modelle -> AVD (eigene kleine Matrix)
FAV_ROWS = ["P1", "P2", "P3"]
FAV_TO_AVD = {  # (video, audio, fusion)
    "P1": [0.5355, 0.5115, 0.5131],
    "P2": [0.5104, 0.4866, 0.5177],
    "P3": [0.5058, 0.5307, 0.5014],
}

TUDA_BLUE = "#004E8A"


def annotated_heatmap(ax, mat, rows, cols, title, vmin=0.4, vmax=1.0):
    m = np.array(mat, dtype=float)
    im = ax.imshow(m, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_title(title, fontsize=10, color=TUDA_BLUE)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=8, color="grey")
            else:
                col = "white" if v < 0.72 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=col)
    return im


def main():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, mat, name in zip(axes, [VIDEO, AUDIO, FUSION], ["Video", "Audio", "Fusion"]):
        im = annotated_heatmap(ax, mat, ROWS, COLS, name)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="AUC")
    fig.suptitle("Cross-Dataset-Generalisierung (AVD-trainiert)", fontsize=12)
    fig.savefig("cross_heatmap_avd.pdf", bbox_inches="tight")
    fig.savefig("cross_heatmap_avd.png", dpi=200, bbox_inches="tight")

    fav = np.array([FAV_TO_AVD[r] for r in FAV_ROWS])
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    annotated_heatmap(ax2, fav, FAV_ROWS, ["Video", "Audio", "Fusion"],
                      "FakeAVCeleb-trainiert \u2192 AVD")
    fig2.savefig("cross_heatmap_fav2avd.pdf", bbox_inches="tight")
    fig2.savefig("cross_heatmap_fav2avd.png", dpi=200, bbox_inches="tight")
    print("gespeichert: cross_heatmap_avd.pdf/.png, cross_heatmap_fav2avd.pdf/.png")


if __name__ == "__main__":
    main()