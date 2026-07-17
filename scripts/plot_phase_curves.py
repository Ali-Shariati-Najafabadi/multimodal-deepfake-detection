import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Eine Metrik (z.B. Val-UAR) mehrerer Laeufe (Phasen) in EINEM Diagramm.
# Aufruf: pro Phase --run "Phase 1" <logdir>  (mehrfach).

def find_event_dir(logdir):
    p = Path(logdir)
    if list(p.glob("events.out.tfevents*")):
        return str(p)
    hits = sorted(p.rglob("events.out.tfevents*"))
    if not hits:
        raise SystemExit(f"Keine Event-Datei unter {logdir}")
    return str(hits[0].parent)

def series(logdir, metric, prefer="val"):
    acc = EventAccumulator(find_event_dir(logdir), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    cands = [t for t in tags if metric.lower() in t.lower()]
    val = [t for t in cands if prefer.lower() in t.lower()]
    tag = (val or cands)
    if not tag:
        raise SystemExit(f"Metrik '{metric}' nicht gefunden. Verfuegbar: {tags}")
    ev = acc.Scalars(tag[0])
    return [e.step for e in ev], [e.value for e in ev], tag[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "LOGDIR"), required=True)
    ap.add_argument("--metric", default="uar", help="Substring, z.B. uar / auc / loss")
    ap.add_argument("--ylabel", default="Val-UAR")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default="phase_curves")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for label, logdir in args.run:
        steps, vals, tag = series(logdir, args.metric)
        ax.plot(steps, vals, linewidth=1.8, label=label)
        print(f"{label}: Tag '{tag}', {len(steps)} Punkte")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(args.ylabel)
    if args.title:
        ax.set_title(args.title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}.png", dpi=200, bbox_inches="tight")
    print(f"gespeichert: {args.out}.pdf/.png")

if __name__ == "__main__":
    main()