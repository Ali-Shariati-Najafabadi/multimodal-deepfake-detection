import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Mel-Spektrogramme aus dem HDF5-Cache (kein ffmpeg noetig).
# Liest je ein Real- und ein Fake-Audio direkt aus dem bereits dekodierten Cache.
# Standard: Real (Label 0) vs. FARV (Label 1, manipuliertes Audio).

def read_audio_from_cache(cache_dir, dataset, split, label, sr=16000):
    cache_dir = Path(cache_dir)
    meta = json.load(open(cache_dir / f"{dataset}_{split}_metadata.json"))
    index = json.load(open(cache_dir / f"{dataset}_{split}_index.json"))

    vid = next((v for v, m in meta.items() if int(m["label"]) == label), None)
    if vid is None:
        raise SystemExit(f"Kein Sample mit Label {label} in {dataset}/{split} gefunden.")

    for shard_path in index["shards"]:
        p = Path(shard_path)
        if not p.exists():
            p = cache_dir / p.name
        if not p.exists():
            continue
        with h5py.File(p, "r") as f:
            key = f"{vid}/audio"
            if key in f:
                return np.asarray(f[key][:], dtype=np.float32), vid
    raise SystemExit(f"Audio fuer {vid} in keinem Shard gefunden.")

def mel_db(y, sr=16000, n_mels=80):
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160, n_mels=n_mels)
    return librosa.power_to_db(m, ref=np.max)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True,
                    help="z.B. /data/shared/ali/deepfake-detection/cache_avdeepfake_uniform_4class")
    ap.add_argument("--dataset", default="av_deepfake")
    ap.add_argument("--split", default="val")
    ap.add_argument("--fake-label", type=int, default=1, help="1=FARV, 3=FAFV")
    ap.add_argument("--out", default="spektrogramme")
    args = ap.parse_args()

    real_y, real_id = read_audio_from_cache(args.cache_dir, args.dataset, args.split, 0)
    fake_y, fake_id = read_audio_from_cache(args.cache_dir, args.dataset, args.split, args.fake_label)
    print(f"Real: {real_id} | Fake: {fake_id}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, y, title in zip(axes, [real_y, fake_y], ["Authentisch", "Manipuliert"]):
        im = librosa.display.specshow(mel_db(y), sr=16000, hop_length=160,
                                      x_axis="time", y_axis="mel", ax=ax, cmap="magma")
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=axes, format="%+2.0f dB", fraction=0.02, pad=0.02)
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}.png", dpi=200, bbox_inches="tight")
    print(f"gespeichert: {args.out}.pdf/.png")

if __name__ == "__main__":
    main()