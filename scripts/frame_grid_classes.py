import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Ein repraesentatives Frame je Klasse aus dem HDF5-Cache (kein Video-Decoding noetig).
# Klassen: 0 Real, 1 FARV, 2 RAFV, 3 FAFV.
LABELS = ["Real", "FARV", "RAFV", "FAFV"]

def frame_for_label(cache_dir, dataset, split, label):
    cache_dir = Path(cache_dir)
    meta = json.load(open(cache_dir / f"{dataset}_{split}_metadata.json"))
    index = json.load(open(cache_dir / f"{dataset}_{split}_index.json"))
    vid = next((v for v, m in meta.items() if int(m["label"]) == label), None)
    if vid is None:
        return None
    for shard_path in index["shards"]:
        p = Path(shard_path)
        if not p.exists():
            p = cache_dir / p.name
        if not p.exists():
            continue
        with h5py.File(p, "r") as f:
            key = f"{vid}/faces"
            if key in f:
                faces = np.asarray(f[key][:], dtype=np.float32)   # [T,3,H,W]
                img = faces[len(faces) // 2].transpose(1, 2, 0)   # [H,W,3]
                return (img - img.min()) / (np.ptp(img) + 1e-6)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True,
                    help="4-Klassen-Cache, z.B. .../cache_avdeepfake_uniform_4class")
    ap.add_argument("--dataset", default="av_deepfake")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="klassen_beispiele")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, label in zip(axes, range(4)):
        img = frame_for_label(args.cache_dir, args.dataset, args.split, label)
        if img is not None:
            ax.imshow(np.clip(img, 0, 1))
        ax.set_title(LABELS[label], fontsize=11)
        ax.axis("off")
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}.png", dpi=200, bbox_inches="tight")
    print(f"gespeichert: {args.out}.pdf/.png")

if __name__ == "__main__":
    main()