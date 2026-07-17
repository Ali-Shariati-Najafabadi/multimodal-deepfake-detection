import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as _mp
_mp.set_sharing_strategy('file_system')

# Speichert y_true und y_score (Fake-Wahrscheinlichkeit) der Fusion als .npz,
# damit plot_roc_curves.py die ROC-Kurven zeichnen kann. Nur binaer sinnvoll.
# In den Ordner evaluation/ legen und dort ausfuehren.

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'evaluation'))
from evaluate_modality import ModalityEvaluator, CachedEvalDataset, collate_fn


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--out', required=True, help="z.B. p1.npz")
    args = ap.parse_args()

    evaluator = ModalityEvaluator(args.config, args.checkpoint)
    model, dev = evaluator.model, evaluator.device
    ds = CachedEvalDataset(args.cache_dir, args.dataset, args.split)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False,
                                         num_workers=8, collate_fn=collate_fn)
    ys, ss = [], []
    for batch in loader:
        v = batch['video_features'].to(dev)
        a = batch['audio_features'].to(dev)
        vm = batch['video_masks'].to(dev)
        am = batch['audio_masks'].to(dev)
        logits = model(v, a, vm, am)
        prob = torch.softmax(logits, dim=1)
        fake = 1.0 - prob[:, 0] if prob.shape[1] > 1 else prob[:, 0]
        ys.append(batch['labels'].numpy())
        ss.append(fake.cpu().numpy())
    y = np.concatenate(ys)
    y_bin = (y > 0).astype(int)   # alles ausser Real (0) gilt als Fake
    np.savez(args.out, y_true=y_bin, y_score=np.concatenate(ss))
    print(f"gespeichert: {args.out}  ({len(y_bin)} Samples)")


if __name__ == '__main__':
    main()