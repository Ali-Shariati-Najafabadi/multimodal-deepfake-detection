import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ROC-Kurven mehrerer Phasen in einer Abbildung.
# Erwartet je Phase eine .npz mit Arrays y_true (0/1) und y_score (Fake-Wahrscheinlichkeit).
# Diese .npz kann die Eval-Routine speichern:  np.savez('phase1.npz', y_true=..., y_score=...)
# Aufruf: python plot_roc_curves.py --inputs phase1.npz phase2.npz phase3.npz --labels "Phase 1" "Phase 2" "Phase 3"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", default="roc_indomain")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(5, 4.2))
    for path, label in zip(args.inputs, args.labels):
        d = np.load(path)
        fpr, tpr, _ = roc_curve(d["y_true"], d["y_score"])
        ax.plot(fpr, tpr, linewidth=1.8, label=f"{label} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("Falsch-Positiv-Rate")
    ax.set_ylabel("Richtig-Positiv-Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}.png", dpi=200, bbox_inches="tight")
    print(f"gespeichert: {args.out}.pdf/.png")

if __name__ == "__main__":
    main()