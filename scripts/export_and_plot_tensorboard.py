import os
import glob
import argparse
import csv

import matplotlib.pyplot as plt


try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    raise SystemExit("Bitte 'pip install tensorboard' ausfuehren.")


from pathlib import Path

def find_event_dir(logdir):
    p = Path(logdir)
    if list(p.glob('events.out.tfevents*')):
        return str(p)
    hits = sorted(p.rglob('events.out.tfevents*'))
    if not hits:
        raise SystemExit(f'Keine TensorBoard-Event-Datei unter {logdir} gefunden.')
    return str(hits[0].parent)

def load_scalars(logdir):
    acc = EventAccumulator(find_event_dir(logdir), size_guidance={"scalars": 0})
    acc.Reload()
    data = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]
    return data


def export_csv(data, outfile):
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "step", "value"])
        for tag, series in data.items():
            for step, value in series:
                w.writerow([tag, step, value])
    print(f"CSV: {outfile}")


def plot_pair(data, tags, ylabel, title, outfile):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for tag in tags:
        if tag not in data:
            continue
        steps = [s for s, _ in data[tag]]
        vals = [v for _, v in data[tag]]
        ax.plot(steps, vals, label=tag, linewidth=1.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outfile}.pdf", bbox_inches="tight")
    fig.savefig(f"{outfile}.png", dpi=200, bbox_inches="tight")
    print(f"Plot: {outfile}.pdf/.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--out", default="tb_export")
    args = ap.parse_args()

    data = load_scalars(args.logdir)
    if not data:
        raise SystemExit("Keine Scalars gefunden. Pfad pruefen.")
    print("Gefundene Tags:", ", ".join(sorted(data)))
    export_csv(data, f"{args.out}.csv")

    # Haeufige Tag-Namen; bei Bedarf an das eigene Logging anpassen.
    loss_tags = [t for t in data if "loss" in t.lower()]
    acc_tags = [t for t in data if any(k in t.lower() for k in ("acc", "auc", "uar"))]
    if loss_tags:
        plot_pair(data, loss_tags, "Loss", "Trainings-/Validierungsverlust", f"{args.out}_loss")
    if acc_tags:
        plot_pair(data, acc_tags, "Metrik", "Validierungsmetriken", f"{args.out}_metrics")


if __name__ == "__main__":
    main()