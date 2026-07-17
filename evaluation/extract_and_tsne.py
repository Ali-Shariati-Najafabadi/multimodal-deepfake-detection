import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'evaluation'))
from evaluate_modality import ModalityEvaluator, CachedEvalDataset, collate_fn


@torch.no_grad()
def features_for(evaluator, cache_dir, dataset_name, split, max_batches, batch_size):
    ds = CachedEvalDataset(cache_dir, dataset_name, split)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         num_workers=8, collate_fn=collate_fn)
    model, dev = evaluator.model, evaluator.device
    vids, auds = [], []
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        v = batch['video_features'].to(dev)
        a = batch['audio_features'].to(dev)
        vm = batch['video_masks'].to(dev)
        vfeat = model.extract_video_features(v, vm)
        vfeat = vfeat.mean(dim=1) if vfeat.dim() == 3 else vfeat
        B, S, L = a.shape
        enc = model.audio_model.encoder(a.view(B * S, L))
        afeat = enc.last_hidden_state.mean(dim=1).view(B, S, -1).mean(dim=1)
        vids.append(vfeat.cpu().numpy())
        auds.append(afeat.cpu().numpy())
    return np.concatenate(vids), np.concatenate(auds)


def tsne_panel(ax, feats, labels, names, title):
    emb = TSNE(n_components=2, init='pca', perplexity=30, random_state=42).fit_transform(feats)
    for k, name in enumerate(names):
        m = labels == k
        ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.6, label=name)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    # je Datensatz: --dataset NAME CACHE_DIR  (mehrfach angeben)
    ap.add_argument('--dataset', nargs=2, action='append', metavar=('NAME', 'CACHE'), required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--max-batches', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--out', default='tsne_embeddings')
    args = ap.parse_args()

    evaluator = ModalityEvaluator(args.config, args.checkpoint)
    names = [d[0] for d in args.dataset]
    V, A, L = [], [], []
    for k, (name, cache) in enumerate(args.dataset):
        v, a = features_for(evaluator, cache, name, args.split, args.max_batches, args.batch_size)
        V.append(v); A.append(a); L.append(np.full(len(v), k))
    V, A, L = np.concatenate(V), np.concatenate(A), np.concatenate(L)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    tsne_panel(axes[0], V, L, names, "Video-Embeddings")
    tsne_panel(axes[1], A, L, names, "Audio-Embeddings")
    fig.savefig(f"{args.out}.pdf", bbox_inches='tight')
    fig.savefig(f"{args.out}.png", dpi=200, bbox_inches='tight')
    print(f"gespeichert: {args.out}.pdf/.png")


if __name__ == '__main__':
    main()