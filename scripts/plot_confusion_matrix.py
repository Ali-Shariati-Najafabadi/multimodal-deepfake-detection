import numpy as np
import matplotlib.pyplot as plt

# Konfusionsmatrizen aus den Logs (bestes Modell je Lauf).
# Klassenreihenfolge 4-Klassen: Real, FARV, RAFV, FAFV.
LABELS_4 = ["Real", "FARV", "RAFV", "FAFV"]
LABELS_2 = ["Real", "Fake"]

# AVD 4-Klassen, Fusion, Phase 1 (uniform, In-Domain)
CM_AVD_P1_UNIFORM = np.array([
    [9681, 196, 112, 34],
    [4651, 5200, 81, 112],
    [6486, 132, 3254, 139],
    [3651, 2344, 1039, 2997],
])

# AVD 4-Klassen, Fusion, Phase 1 (seg-trainiert, uniform evaluiert, Cross FakeAVCeleb test)
CM_FAV_P1_SEG = np.array([
    [75, 0, 0, 0],
    [28, 47, 0, 0],
    [1265, 1, 220, 1],
    [421, 605, 312, 334],
])


def plot_cm(cm, labels, title, normalize=True, outfile="confusion_matrix"):
    cm = np.array(cm, dtype=float)
    disp = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None) if normalize else cm
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=1 if normalize else disp.max())
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Vorhergesagt")
    ax.set_ylabel("Tatsaechlich")
    ax.set_title(title, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = disp[i, j]
            txt = f"{v:.2f}" if normalize else f"{int(v)}"
            col = "white" if v > (0.5 if normalize else disp.max() * 0.5) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=col)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(f"{outfile}.pdf", bbox_inches="tight")
    fig.savefig(f"{outfile}.png", dpi=200, bbox_inches="tight")
    print(f"gespeichert: {outfile}.pdf/.png")


if __name__ == "__main__":
    plot_cm(CM_AVD_P1_UNIFORM, LABELS_4, "AVD 4-Klassen, Fusion (Phase 1, In-Domain)",
            normalize=True, outfile="cm_avd_4class_p1_indomain")
    plot_cm(CM_FAV_P1_SEG, LABELS_4, "FakeAVCeleb, Fusion (Phase 1, seg-trainiert)",
            normalize=True, outfile="cm_fav_4class_p1_seg")