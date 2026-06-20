import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data import WeightedRandomSampler
from pathlib import Path
from functools import partial
import json
import random
import h5py
from typing import Dict, List, Optional


class ShardedCachedDataset(Dataset):
    def __init__(
        self,
        cache_dir:       str,
        dataset_name:    str,
        split:           str            = 'train',
        data_config:     Optional[Dict] = None,
        max_samples:     Optional[int]  = None,
        balance_classes: bool           = False,
    ):
        self.cache_dir     = Path(cache_dir)
        self.dataset_name  = dataset_name
        self.split         = split
        self.data_config   = data_config or {}

        self.max_frames             = self.data_config.get('video', {}).get('max_frames', 16)
        self.min_frames             = self.data_config.get('video', {}).get('min_frames', 16)
        self.use_random_crop        = self.data_config.get('video', {}).get('use_random_crop', False)
        self.random_crop_train_only = self.data_config.get('video', {}).get('random_crop_training_only', True)

        index_file = self.cache_dir / f'{dataset_name}_{split}_index.json'
        meta_file  = self.cache_dir / f'{dataset_name}_{split}_metadata.json'

        if not index_file.exists():
            raise FileNotFoundError(f"Index not found: {index_file}")

        with open(index_file) as f:
            self.index = json.load(f)
        with open(meta_file) as f:
            self.metadata = json.load(f)

        self.video_ids = list(self.metadata.keys())

        if balance_classes:
            self.video_ids = self._balance_classes(self.video_ids, max_samples)
        elif max_samples and len(self.video_ids) > max_samples:
            random.seed(42)
            self.video_ids = random.sample(self.video_ids, max_samples)

        # Open all shard H5 files
        self.shard_files = {}
        for shard_path in self.index['shards']:
            p = Path(shard_path)
            if p.exists():
                shard_idx = int(p.stem.split('shard')[-1])
                self.shard_files[shard_idx] = h5py.File(p, 'r')

        # Print summary
        label_counts = {}
        for vid_id in self.video_ids:
            lbl = self.metadata[vid_id]['label']
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        dist = ", ".join(f"Class {k}: {v}" for k, v in sorted(label_counts.items()))
        print(f"  [{dataset_name}/{split}] {len(self.video_ids):,} samples | {dist}")

    def _balance_classes(self, video_ids: List[str], max_samples: Optional[int]) -> List[str]:
        random.seed(42)
        class_samples: Dict[int, List] = {}
        for vid_id in video_ids:
            lbl = self.metadata[vid_id]['label']
            class_samples.setdefault(lbl, []).append(vid_id)

        n = max_samples // len(class_samples) if max_samples else min(len(v) for v in class_samples.values())
        balanced = []
        for lbl, samples in sorted(class_samples.items()):
            selected = random.sample(samples, min(n, len(samples)))
            balanced.extend(selected)

        random.shuffle(balanced)
        print(f"    Balanced: {n:,} per class ({len(class_samples)} classes)")
        return balanced

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id  = self.video_ids[idx]
        meta      = self.metadata[video_id]
        label     = meta['label']
        shard_idx = meta['shard']

        # ── Video ─────────────────────────────────────────────────────────
        try:
            f = self.shard_files[shard_idx]
            video_features = torch.from_numpy(f[f'{video_id}/faces'][:])
        except Exception:
            video_features = torch.zeros(self.min_frames, 3, 224, 224)

        n = video_features.shape[0]
        should_crop = self.use_random_crop and not (
            self.random_crop_train_only and self.split != 'train'
        )

        if should_crop and n > self.max_frames:
            start = torch.randint(0, n - self.max_frames + 1, (1,)).item()
            video_features = video_features[start: start + self.max_frames]
        elif n > self.max_frames:
            video_features = video_features[:self.max_frames]

        n = video_features.shape[0]
        if n == 0:
            video_features = torch.zeros(self.min_frames, 3, 224, 224)
        elif n < self.min_frames:
            pad = video_features[-1:].repeat(self.min_frames - n, 1, 1, 1)
            video_features = torch.cat([video_features, pad], dim=0)

        # ── Audio ─────────────────────────────────────────────────────────
        try:
            f = self.shard_files[shard_idx]
            if f'{video_id}/audio' in f:
                audio          = torch.from_numpy(f[f'{video_id}/audio'][:])  # (64000,)
                audio_features = audio.unsqueeze(0)                            # (1, 64000)
            else:
                audio_features = torch.zeros(1, 64000)
        except Exception:
            audio_features = torch.zeros(1, 64000)

        return {
            'video_features': video_features,    # (T, 3, 224, 224)
            'audio_features': audio_features,    # (1, 64000)
            'label':          label,
            'dataset':        self.dataset_name,
        }

    def __del__(self):
        for f in getattr(self, 'shard_files', {}).values():
            try:
                f.close()
            except Exception:
                pass


# =====================================================================
# COLLATE FN
# =====================================================================
def collate_fn(batch, min_frames: int = 16, max_frames: int = 16):
    max_t = min(max(b['video_features'].shape[0] for b in batch), max_frames)
    max_t = max(max_t, min_frames)

    video_batch, video_masks = [], []
    for b in batch:
        v = b['video_features']
        t = v.shape[0]
        if t > max_frames:
            v = v[:max_frames]; t = max_frames
        if t < min_frames:
            v = torch.cat([v, v[-1:].repeat(min_frames - t, 1, 1, 1)], dim=0); t = min_frames
        if t < max_t:
            v = torch.cat([v, torch.zeros(max_t - t, 3, 224, 224)], dim=0)
        mask = torch.zeros(max_t); mask[:t] = 1.0
        video_batch.append(v)
        video_masks.append(mask)

    audio_batch, audio_masks = [], []
    for b in batch:
        a = b['audio_features']   # (1, 64000)
        audio_batch.append(a)
        audio_masks.append(torch.ones(a.shape[0]))

    video_out = torch.stack(video_batch)                        # (B, T, 3, H, W)
    video_out = video_out.permute(0, 2, 1, 3, 4).contiguous()  # (B, 3, T, H, W)

    max_seg = max(a.shape[0] for a in audio_batch)
    seg_len = audio_batch[0].shape[1]
    audio_padded, audio_mask_padded = [], []
    for a, m in zip(audio_batch, audio_masks):
        if a.shape[0] < max_seg:
            pad = torch.zeros(max_seg - a.shape[0], seg_len)
            a   = torch.cat([a, pad], dim=0)
            m   = torch.cat([m, torch.zeros(max_seg - m.shape[0])], dim=0)
        audio_padded.append(a)
        audio_mask_padded.append(m)

    return {
        'video_features': video_out,
        'audio_features': torch.stack(audio_padded),
        'video_masks':    torch.stack(video_masks),
        'audio_masks':    torch.stack(audio_mask_padded),
        'labels':         torch.tensor([b['label']   for b in batch], dtype=torch.long),
        'datasets':       [b['dataset'] for b in batch],
    }


# =====================================================================
# HELPER: alle Labels aus einem Dataset/ConcatDataset holen
# =====================================================================
def _collect_labels(dataset) -> List[int]:
    """Funktioniert für ShardedCachedDataset UND ConcatDataset."""
    if isinstance(dataset, ConcatDataset):
        labels = []
        for ds in dataset.datasets:
            labels.extend(_collect_labels(ds))
        return labels
    # ShardedCachedDataset
    return [dataset.metadata[vid]['label'] for vid in dataset.video_ids]


# =====================================================================
# CREATE DATALOADERS
# =====================================================================
def create_dataloaders(
    datasets_config:  Dict,
    training_config:  Optional[Dict] = None,
    batch_size:       int            = 16,
    num_workers:      int            = 8,
    max_samples:      Optional[int]  = None,
    data_config:      Optional[Dict] = None,
):
    """
    Unterstützt alle 3 Konfigurationen über YAML:

    Config A — Nur AV-Deepfake 50k (balanced, kein WeightedSampler):
        datasets:
          av_deepfake:  enabled: true,  cache_dir: cache_avdeepfake_50k
          fakeavceleb:  enabled: false
          deepspeak:    enabled: false
        training:
          use_weighted_sampler: false

    Config B — FakeAVCeleb + AV-Deepfake Real (ConcatDataset + WeightedSampler):
        datasets:
          fakeavceleb:  enabled: true,  cache_dir: cache_fakeavceleb_phase1
          av_deepfake:  enabled: true,  cache_dir: cache_avdeepfake_real_only
          deepspeak:    enabled: false
        training:
          use_weighted_sampler: true

    Config C — Nur FakeAVCeleb (Original):
        datasets:
          fakeavceleb:  enabled: true,  cache_dir: cache_fakeavceleb_phase1
          av_deepfake:  enabled: false
          deepspeak:    enabled: false
        training:
          use_weighted_sampler: true
    """
    data_cfg     = data_config    or {}
    training_cfg = training_config or {}

    max_frames = data_cfg.get('video', {}).get('max_frames', 16)
    min_frames = data_cfg.get('video', {}).get('min_frames', 16)

    # use_weighted_sampler: aus training_config lesen (default: True)
    use_weighted_sampler = training_cfg.get('use_weighted_sampler', True)

    DATASETS = ['fakeavceleb', 'av_deepfake', 'deepspeak']
    train_datasets, val_datasets = [], []

    print("\n" + "="*60)
    print("DATASET LOADING")
    print("="*60)

    for name in DATASETS:
        cfg = datasets_config.get(name, {})
        if not cfg.get('enabled', False):
            print(f"  [{name}] DISABLED — skipped")
            continue

        cache_dir       = cfg['cache_dir']
        balance_classes = cfg.get('balance_classes', False)

        print(f"\n  Loading {name.upper()} ...")

        # Train
        train_ds = ShardedCachedDataset(
            cache_dir=cache_dir,
            dataset_name=name,
            split='train',
            data_config=data_cfg,
            max_samples=max_samples,
            balance_classes=balance_classes,
        )
        train_datasets.append(train_ds)

        # Val (suche zuerst 'val', dann 'test')
        val_split = None
        for s in ('val', 'test'):
            if (Path(cache_dir) / f'{name}_{s}_index.json').exists():
                val_split = s
                break

        if val_split:
            val_ds = ShardedCachedDataset(
                cache_dir=cache_dir,
                dataset_name=name,
                split=val_split,
                data_config=data_cfg,
                balance_classes=False,
            )
            val_datasets.append(val_ds)
        else:
            print(f"    WARNING: No val/test split found for {name}, skipping.")

    if not train_datasets:
        raise ValueError("Kein Dataset aktiviert! Bitte YAML Config prüfen.")

    # ConcatDataset wenn mehrere Datasets
    train_dataset = (
        ConcatDataset(train_datasets) if len(train_datasets) > 1
        else train_datasets[0]
    )
    val_dataset = (
        ConcatDataset(val_datasets) if len(val_datasets) > 1
        else val_datasets[0]
    )

    print(f"\n{'='*60}")
    print(f"Total: {len(train_dataset):,} train | {len(val_dataset):,} val")

    _collate = partial(collate_fn, min_frames=min_frames, max_frames=max_frames)

    # ── WeightedRandomSampler ─────────────────────────────────────────
    if use_weighted_sampler:
        all_labels   = _collect_labels(train_dataset)
        class_counts = {}
        for lbl in all_labels:
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

        sample_weights = [1.0 / class_counts[lbl] for lbl in all_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        n_classes = len(class_counts)
        dist = ', '.join(f'Class {k}: {v:,}' for k, v in sorted(class_counts.items()))
        print(f"  WeightedSampler ON  → {dist}")
        print(f"  → ~{batch_size // n_classes} samples/class per batch")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,          # shuffle=True ist nicht kompatibel mit sampler
            num_workers=num_workers,
            collate_fn=_collate,
            pin_memory=True,
            drop_last=True,
        )

    else:
        # Bereits balanced (z.B. Config A: 25k Real + 25k Fake)
        all_labels   = _collect_labels(train_dataset)
        class_counts = {}
        for lbl in all_labels:
            class_counts[lbl] = class_counts.get(lbl, 0) + 1
        dist = ', '.join(f'Class {k}: {v:,}' for k, v in sorted(class_counts.items()))
        print(f"  WeightedSampler OFF → {dist}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=_collate,
            pin_memory=True,
            drop_last=True,
        )

    print("="*60 + "\n")

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
    )

    return train_loader, val_loader
