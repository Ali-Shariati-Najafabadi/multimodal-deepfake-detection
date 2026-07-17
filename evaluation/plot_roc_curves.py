import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


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