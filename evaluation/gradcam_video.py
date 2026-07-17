import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Grad-CAM fuer den ResNet-50-Video-Zweig (Phase 1).
# In den Ordner evaluation/ legen und dort ausfuehren.

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'evaluation'))
from evaluate_modality import ModalityEvaluator, CachedEvalDataset, collate_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--index', type=int, default=0, help="fester Index (falls --label nicht gesetzt)")
    ap.add_argument('--label', type=int, default=None, help="0=Real,1=FARV,2=RAFV,3=FAFV; waehlt erstes Sample dieser Klasse")
    ap.add_argument('--out', default='gradcam')
    args = ap.parse_args()

    evaluator = ModalityEvaluator(args.config, args.checkpoint)
    model, dev = evaluator.model, evaluator.device

    ds = CachedEvalDataset(args.cache_dir, args.dataset, args.split)
    if args.label is not None:
        idx = next((i for i, v in enumerate(ds.video_ids)
                    if int(ds.metadata[v]['label']) == args.label), None)
        if idx is None:
            raise SystemExit(f'Kein Sample mit Label {args.label} gefunden.')
        print(f'Label {args.label} -> Index {idx}')
    else:
        idx = args.index
    batch = collate_fn([ds[idx]])
    v = batch['video_features'].to(dev)
    vm = batch['video_masks'].to(dev)

    target_layer = model.video_model.backbone[-2]   # ResNet layer4 (letzte Conv-Stufe)
    acts, grads = {}, {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.__setitem__('v', o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__('v', go[0].detach()))

    model.zero_grad()
    # cuDNN erlaubt RNN-Backward nicht im Eval-Modus -> fuer diesen Schritt abschalten.
    with torch.backends.cudnn.flags(enabled=False):
        vfeat = model.extract_video_features(v, vm)
        logits = model.video_aux_head(vfeat) if hasattr(model, 'video_aux_head') \
            else model.video_model.classifier(vfeat.mean(dim=1) if vfeat.dim() == 3 else vfeat)
        score = logits[0, logits.argmax(dim=1)].sum()
        score.backward()
    h1.remove(); h2.remove()

    a = acts['v'].detach()          # [T, C, h, w]  (Frames im Batch gefaltet)
    g = grads['v']                  # [T, C, h, w]
    weights = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * a).sum(dim=1))
    cam = (cam - cam.amin(dim=(1, 2), keepdim=True)) / (cam.amax(dim=(1, 2), keepdim=True) + 1e-6)
    cam = cam.cpu().numpy()

    frames = v[0].permute(1, 2, 3, 0).cpu().numpy()   # [T, H, W, 3]
    frames = (frames - frames.min()) / (np.ptp(frames) + 1e-6)
    idxs = np.linspace(0, len(frames) - 1, 6, dtype=int)

    fig, axes = plt.subplots(1, len(idxs), figsize=(12, 2.4))
    for ax, i in zip(axes, idxs):
        ax.imshow(frames[i])
        heat = plt.cm.jet(cam[min(i, len(cam) - 1)])[..., :3]
        ax.imshow(heat, alpha=0.45, extent=(0, frames.shape[2], frames.shape[1], 0))
        ax.axis('off')
    fig.savefig(f"{args.out}.pdf", bbox_inches='tight')
    fig.savefig(f"{args.out}.png", dpi=200, bbox_inches='tight')
    print(f"gespeichert: {args.out}.pdf/.png")


if __name__ == '__main__':
    main()