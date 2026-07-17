import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Filmstrip eines Videos: uniform vs. segmentbasiert entnommene Frames.
# Fake-Segmente als Sekundenpaare uebergeben, z.B. --segments 1.2 2.4 3.0 3.8
# pip install opencv-python

def read_frames(path, indices):
    cap = cv2.VideoCapture(path)
    out = {}
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if ok:
            out[idx] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    cap.release()
    return out

def seg_indices(total, fps, segments, num=16, fake_ratio=0.75):
    fake = set()
    for s, e in zip(segments[::2], segments[1::2]):
        fake.update(range(int(s * fps), min(total, int(e * fps))))
    fake = sorted(fake)
    if not fake:
        return np.linspace(0, total - 1, num, dtype=int)
    n_fake = max(1, int(num * fake_ratio))
    step = max(1, len(fake) // n_fake)
    picks = fake[::step][:n_fake]
    rest = [i for i in range(total) if i not in set(picks)]
    picks += list(np.linspace(0, len(rest) - 1, num - len(picks)).astype(int))
    return sorted(set(picks))

def strip(ax, frames, indices, title, fake_frames=None):
    for k, idx in enumerate(indices):
        sub = ax.inset_axes([k / len(indices), 0, 1 / len(indices), 1])
        if idx in frames:
            sub.imshow(frames[idx])
        sub.axis("off")
        if fake_frames and idx in fake_frames:
            sub.add_patch(Rectangle((0, 0), 1, 1, transform=sub.transAxes,
                                    fill=False, edgecolor="red", linewidth=2))
    ax.set_title(title, fontsize=10, loc="left")
    ax.axis("off")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--segments", nargs="+", type=float, default=[])
    ap.add_argument("--num", type=int, default=16)
    ap.add_argument("--out", default="filmstrip_sampling")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()

    uni = np.linspace(0, total - 1, args.num, dtype=int)
    seg = seg_indices(total, fps, args.segments, args.num)
    fake_frames = set()
    for s, e in zip(args.segments[::2], args.segments[1::2]):
        fake_frames.update(range(int(s * fps), min(total, int(e * fps))))

    frames = read_frames(args.video, sorted(set(list(uni) + list(seg))))
    fig, axes = plt.subplots(2, 1, figsize=(12, 3.4))
    strip(axes[0], frames, uni, "uniform", fake_frames)
    strip(axes[1], frames, seg, "segmentbasiert", fake_frames)
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}.png", dpi=200, bbox_inches="tight")
    print(f"gespeichert: {args.out}.pdf/.png")

if __name__ == "__main__":
    main()