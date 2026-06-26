import torch
import yaml
import argparse
import json
import h5py
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import get_model
from training.training_utils import MetricsCalculator


class CachedEvalDataset(torch.utils.data.Dataset):
    """Liest faces und audio direkt aus dem Sharded-HDF5-Cache. Kein Live-Preprocessing."""

    def __init__(self, cache_dir, dataset_name, split):
        self.cache_dir = Path(cache_dir)
        index_file = self.cache_dir / f'{dataset_name}_{split}_index.json'
        meta_file = self.cache_dir / f'{dataset_name}_{split}_metadata.json'

        if not index_file.exists():
            raise FileNotFoundError(f"Index nicht gefunden: {index_file}")

        with open(index_file) as f:
            self.index = json.load(f)
        with open(meta_file) as f:
            self.metadata = json.load(f)

        self.video_ids = list(self.metadata.keys())

        # benoetigte Shard-Indizes laut Metadata
        needed = sorted({self.metadata[v]['shard'] for v in self.video_ids})

        self.shards = {}
        for shard_path in self.index['shards']:
            p = Path(shard_path)
            if not p.exists():
                p = self.cache_dir / Path(shard_path).name  # Fallback: Datei lokal im cache_dir
            if not p.exists():
                continue
            digits = ''.join(ch for ch in p.stem if ch.isdigit())
            shard_idx = int(digits[-4:]) if digits else len(self.shards)
            self.shards[shard_idx] = h5py.File(p, 'r')

        missing = [s for s in needed if s not in self.shards]
        if missing:
            raise FileNotFoundError(
                f"Shards {missing} fehlen fuer {dataset_name}/{split}. "
                f"Geladen: {sorted(self.shards.keys())}. Pruefe {self.cache_dir}"
            )

        labels = [self.metadata[v]['label'] for v in self.video_ids]
        dist = {k: labels.count(k) for k in sorted(set(labels))}
        print(f"  {dataset_name}/{split}: {len(self.video_ids)} Samples | Verteilung: {dist}")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        meta = self.metadata[vid]
        f = self.shards[meta['shard']]

        faces = torch.from_numpy(f[f'{vid}/faces'][:]).float()    # [T, 3, H, W]

        if f'{vid}/audio' not in f:
            raise KeyError(f"Kein Audio im Cache fuer {vid} (mit --no-audio gecached?)")
        audio = torch.from_numpy(f[f'{vid}/audio'][:]).float()    # [samples]

        return {'video': faces, 'audio': audio, 'label': int(meta['label'])}

    def __del__(self):
        for f in getattr(self, 'shards', {}).values():
            try:
                f.close()
            except Exception:
                pass


def collate_fn(batch):
    video = torch.stack([b['video'] for b in batch])               # [B, T, 3, H, W]
    video = video.permute(0, 2, 1, 3, 4).contiguous()             # [B, 3, T, H, W]
    audio = torch.stack([b['audio'] for b in batch]).unsqueeze(1)  # [B, 1, samples]
    T = video.shape[2]
    video_masks = torch.ones(len(batch), T)
    audio_masks = torch.ones(len(batch), 1)
    labels = torch.tensor([b['label'] for b in batch])
    return {
        'video_features': video,
        'audio_features': audio,
        'video_masks': video_masks,
        'audio_masks': audio_masks,
        'labels': labels,
    }


class ModalityEvaluator:
    def __init__(self, config_path, checkpoint_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.phase = self.config['experiment']['phase']
        self.num_classes = self.config['models'].get('num_classes', 2)

        self.model = get_model(self.phase, self.config).to(self.device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

        info = self.model.get_model_info()
        print(f"\n  Phase: {info['phase']} | Fusion: {info.get('fusion_type', '-')} "
              f"| Video: {info.get('video_architecture', '-')} | Klassen: {self.num_classes}")

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
        self.model.load_state_dict(state)
        if isinstance(ckpt, dict) and 'epoch' in ckpt:
            print(f"  Checkpoint Epoch: {ckpt['epoch']}")

    @staticmethod
    def _pool(feat):
        return feat.mean(dim=1) if feat.dim() == 3 else feat

    def _modality_labels(self, labels):
        # binaere Modality-Labels analog zum Training, nur relevant bei 4 Klassen
        if self.num_classes == 4:
            video_labels = (labels >= 2).long()
            audio_labels = ((labels == 1) | (labels == 3)).long()
            return video_labels, audio_labels
        return labels, labels

    @torch.no_grad()
    def evaluate(self, cache_dir, dataset_name, split='val', batch_size=None, max_samples=None):
        print(f"\n{'='*80}")
        print(f"MODALITY-EVALUATION  |  {dataset_name}/{split}  |  Cache: {cache_dir}")
        print(f"{'='*80}")

        dataset = CachedEvalDataset(cache_dir, dataset_name, split)
        if max_samples and max_samples < len(dataset):
            import random
            random.seed(42)
            idxs = random.sample(range(len(dataset)), max_samples)
            dataset = torch.utils.data.Subset(dataset, idxs)
            print(f"  Limitiert auf {max_samples} Samples")

        bs = batch_size or self.config['training']['batch_size']
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=bs, shuffle=False,
            num_workers=8, collate_fn=collate_fn, pin_memory=True
        )

        video_m = MetricsCalculator()
        audio_m = MetricsCalculator()
        fusion_m = MetricsCalculator()

        has_video_aux = hasattr(self.model, 'video_aux_head')
        has_audio_aux = hasattr(self.model, 'audio_aux_head')

        for bi, batch in enumerate(loader):
            v = batch['video_features'].to(self.device)
            a = batch['audio_features'].to(self.device)
            vm = batch['video_masks'].to(self.device)
            am = batch['audio_masks'].to(self.device)
            y = batch['labels'].to(self.device)
            vy, ay = self._modality_labels(y)

            # VIDEO ONLY
            vfeat = self._pool(self.model.extract_video_features(v, vm))
            if has_video_aux:
                vlogits = self.model.video_aux_head(vfeat)
            else:
                vlogits = self.model.video_model.classifier(vfeat)
            video_m.update(vy, vlogits.argmax(1), torch.softmax(vlogits, 1))

            # AUDIO ONLY
            B, S, L = a.shape
            enc = self.model.audio_model.encoder(a.view(B * S, L))
            afeat = enc.last_hidden_state.mean(dim=1).view(B, S, -1).mean(dim=1)
            if has_audio_aux:
                alogits = self.model.audio_aux_head(afeat)
            else:
                alogits = self.model.audio_model.classifier(afeat)
            audio_m.update(ay, alogits.argmax(1), torch.softmax(alogits, 1))

            # FUSION
            flogits = self.model(v, a, vm, am)
            fusion_m.update(y, flogits.argmax(1), torch.softmax(flogits, 1))

            if bi == 0:
                src = "aux_head" if has_video_aux else "video_model.classifier"
                print(f"\n  Erste Batch: Video {tuple(v.shape)}, Audio {tuple(a.shape)}")
                print(f"  Logits: Video {tuple(vlogits.shape)}, Audio {tuple(alogits.shape)}, Fusion {tuple(flogits.shape)}")
                print(f"  Per-Modality-Kopf: {src}\n")

            if (bi + 1) % 50 == 0:
                print(f"  {bi+1}/{len(loader)} Batches")

        vres = video_m.compute()
        ares = audio_m.compute()
        fres = fusion_m.compute()

        print(f"\n{'-'*80}\nVIDEO ONLY")
        video_m.print_metrics(vres, prefix="  ")
        print(f"\nAUDIO ONLY")
        audio_m.print_metrics(ares, prefix="  ")
        print(f"\nFUSION")
        fusion_m.print_metrics(fres, prefix="  ")

        print(f"\n{'='*80}")
        print(f"SUMMARY  ({dataset_name}/{split})")
        print(f"  Video   ACC {vres['accuracy']:.1%}  UAR {vres.get('uar', 0):.1%}  AUC {vres.get('auc', 0):.4f}")
        print(f"  Audio   ACC {ares['accuracy']:.1%}  UAR {ares.get('uar', 0):.1%}  AUC {ares.get('auc', 0):.4f}")
        print(f"  Fusion  ACC {fres['accuracy']:.1%}  UAR {fres.get('uar', 0):.1%}  AUC {fres.get('auc', 0):.4f}")
        print(f"{'='*80}\n")

        return {'video': vres, 'audio': ares, 'fusion': fres}


def main():
    parser = argparse.ArgumentParser(description="Cross-Dataset Modality Evaluation (aus Cache)")
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache-dir', required=True, help="Cache-Ordner des Eval-Datensatzes")
    parser.add_argument('--dataset', required=True, help="av_deepfake | fakeavceleb | deepspeak")
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--save-results', default=None)
    args = parser.parse_args()

    evaluator = ModalityEvaluator(args.config, args.checkpoint)
    results = evaluator.evaluate(
        cache_dir=args.cache_dir,
        dataset_name=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    if args.save_results:
        out = {}
        for mod, m in results.items():
            out[mod] = {
                k: (v.tolist() if isinstance(v, np.ndarray)
                    else float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in m.items()
            }
        with open(args.save_results, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"Ergebnisse gespeichert: {args.save_results}")


if __name__ == "__main__":
    main()