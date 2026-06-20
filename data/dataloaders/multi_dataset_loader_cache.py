import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import h5py


class ShardedCachedDataset(Dataset):
    def __init__(
        self,
        cache_dir: str,
        dataset_name: str,
        split: str = 'train',
        audio_preprocessor = None,
        data_config: Optional[Dict] = None,
        max_samples: Optional[int] = None,
        balance_classes: bool = False
    ):
        self.cache_dir = Path(cache_dir)
        self.dataset_name = dataset_name
        self.split = split
        self.audio_preprocessor = audio_preprocessor
        
        self.data_config = data_config or {}
        self.max_frames = self.data_config.get('video', {}).get('max_frames', 32)
        self.min_frames = self.data_config.get('video', {}).get('min_frames', 5)
        
        self.use_random_crop = self.data_config.get('video', {}).get('use_random_crop', True)
        self.random_crop_training_only = self.data_config.get('video', {}).get('random_crop_training_only', True)
        
        self.use_temporal_alignment = self.data_config.get('audio', {}).get('use_temporal_alignment', False)
        self.alignment_window = self.data_config.get('audio', {}).get('alignment_window', 8.0)
        
        index_file = self.cache_dir / f'{dataset_name}_{split}_index.json'
        meta_file = self.cache_dir / f'{dataset_name}_{split}_metadata.json'
        
        if not index_file.exists():
            raise FileNotFoundError(f"Index not found: {index_file}")
        
        with open(index_file, 'r') as f:
            self.index = json.load(f)
        
        with open(meta_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.video_ids = list(self.metadata.keys())
        
        if balance_classes:
            self.video_ids = self._balance_classes(self.video_ids, max_samples)
        elif max_samples and len(self.video_ids) > max_samples:
            import random
            random.seed(42)
            self.video_ids = random.sample(self.video_ids, max_samples)
            print(f"Limited to {max_samples} samples")
        
        self.shard_files = {}
        for shard_path in self.index['shards']:
            shard_path = Path(shard_path)
            if shard_path.exists():
                shard_idx = int(shard_path.stem.split('shard')[-1])
                self.shard_files[shard_idx] = h5py.File(shard_path, 'r')
        
        print(f"Loaded {len(self.video_ids)} cached {dataset_name} samples ({split}) from {len(self.shard_files)} shards")
        
        crop_status = "ON" if self.use_random_crop else "OFF"
        if self.use_random_crop and self.random_crop_training_only:
            crop_status += " (training only)"
        print(f"  Video random crop: {crop_status}")
        
        audio_status = "SYNC" if self.use_temporal_alignment else "FULL"
        if self.use_temporal_alignment:
            audio_status += f" (window={self.alignment_window}s)"
        print(f"  Audio alignment: {audio_status}")
        
        label_counts = {}
        for vid_id in self.video_ids:
            label = self.metadata[vid_id]['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        dist_str = ", ".join([f"Class {k}: {v}" for k, v in sorted(label_counts.items())])
        print(f"{split.capitalize()} distribution - {dist_str}")

    def _balance_classes(self, video_ids: List[str], max_samples: Optional[int]) -> List[str]:
        import random
        random.seed(42)
        
        class_samples = {}
        for vid_id in video_ids:
            label = self.metadata[vid_id]['label']
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(vid_id)
        
        num_classes = len(class_samples)
        
        if max_samples:
            samples_per_class = max_samples // num_classes
        else:
            samples_per_class = min(len(samples) for samples in class_samples.values())
        
        balanced_ids = []
        for label, samples in sorted(class_samples.items()):
            if len(samples) >= samples_per_class:
                selected = random.sample(samples, samples_per_class)
            else:
                selected = samples
            balanced_ids.extend(selected)
        
        random.shuffle(balanced_ids)
        
        print(f"Balanced sampling: {samples_per_class} samples per class ({num_classes} classes)")
        
        return balanced_ids

    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        meta = self.metadata[video_id]
        video_path = meta['path']
        label = meta['label']
        shard_idx = meta['shard']
        
        try:
            if shard_idx not in self.shard_files:
                raise KeyError(f"Shard {shard_idx} not loaded")
            
            f = self.shard_files[shard_idx]
            faces_data = f[f'{video_id}/faces'][:]
            video_features = torch.from_numpy(faces_data)
            
            num_frames = video_features.shape[0]
            start_frame_idx = 0
            
            # ✅ NEU: Conditional random crop
            should_random_crop = self.use_random_crop
            if self.random_crop_training_only and self.split != 'train':
                should_random_crop = False
            
            if should_random_crop and num_frames > self.max_frames:
                # Random crop
                max_start = num_frames - self.max_frames
                start_frame_idx = torch.randint(0, max_start + 1, (1,)).item()
                video_features = video_features[start_frame_idx:start_frame_idx + self.max_frames]
            elif num_frames > self.max_frames:
                # Sequential crop (from start)
                video_features = video_features[:self.max_frames]
            
        except Exception:
            video_features = torch.zeros(self.min_frames, 3, 224, 224)
            start_frame_idx = 0
        
        # Padding if needed
        if video_features.shape[0] == 0:
            video_features = torch.zeros(self.min_frames, 3, 224, 224)
        elif video_features.shape[0] < self.min_frames:
            num_needed = self.min_frames - video_features.shape[0]
            last_frame = video_features[-1:].repeat(num_needed, 1, 1, 1)
            video_features = torch.cat([video_features, last_frame], dim=0)
        
        # ✅ NEU: Conditional audio processing
        if self.audio_preprocessor:
            try:
                sample_rate = 3
                fps = 25
                
                if self.use_temporal_alignment:
                    video_duration = self.max_frames * sample_rate / fps
                    video_start_time = start_frame_idx * sample_rate / fps
                    
                    # Center audio window around video
                    window_half = self.alignment_window / 2
                    audio_start = max(0, video_start_time + video_duration/2 - window_half)
                    audio_duration = self.alignment_window
                else:
                    audio_start = 0.0
                    audio_duration = None  # Will use full audio
                
                audio_features, timestamps, duration = \
                    self.audio_preprocessor.process_audio(
                        video_path,
                        start_time=audio_start,
                        duration=audio_duration
                    )
            except Exception:
                audio_features = torch.zeros(1, 64000)
        else:
            audio_features = torch.zeros(1, 64000)
        
        return {
            'video_features': video_features,
            'audio_features': audio_features,
            'label': label,
            'dataset': self.dataset_name,
            'metadata': meta,
            'video_path': video_path
        }
    
    def __del__(self):
        if hasattr(self, 'shard_files'):
            for f in self.shard_files.values():
                try:
                    f.close()
                except:
                    pass



class CachedDataset(Dataset):
    def __init__(
        self,
        cache_dir: str,
        dataset_name: str,
        split: str = 'train',
        audio_preprocessor = None,
        data_config: Optional[Dict] = None,
        max_samples: Optional[int] = None,
        balance_classes: bool = False
    ):
        self.cache_dir = Path(cache_dir)
        self.dataset_name = dataset_name
        self.split = split
        self.audio_preprocessor = audio_preprocessor
        
        self.data_config = data_config or {}
        self.max_frames = self.data_config.get('video', {}).get('max_frames', 32)
        self.min_frames = self.data_config.get('video', {}).get('min_frames', 5)
        
        cache_file = self.cache_dir / f'{dataset_name}_{split}_faces.h5'
        meta_file = self.cache_dir / f'{dataset_name}_{split}_metadata.json'
        
        if not cache_file.exists():
            raise FileNotFoundError(f"Cache not found: {cache_file}")
        
        self.cache_file = h5py.File(cache_file, 'r')
        
        with open(meta_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.video_ids = list(self.metadata.keys())
        
        if balance_classes:
            self.video_ids = self._balance_classes(self.video_ids, max_samples)
        elif max_samples and len(self.video_ids) > max_samples:
            import random
            random.seed(42)
            self.video_ids = random.sample(self.video_ids, max_samples)
            print(f"Limited to {max_samples} samples")
        
        print(f"Loaded {len(self.video_ids)} cached {dataset_name} samples ({split})")
        
        label_counts = {}
        for vid_id in self.video_ids:
            label = self.metadata[vid_id]['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        dist_str = ", ".join([f"Class {k}: {v}" for k, v in sorted(label_counts.items())])
        print(f"{split.capitalize()} distribution - {dist_str}")

    def _balance_classes(self, video_ids: List[str], max_samples: Optional[int]) -> List[str]:
        import random
        random.seed(42)
        
        class_samples = {}
        for vid_id in video_ids:
            label = self.metadata[vid_id]['label']
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(vid_id)
        
        num_classes = len(class_samples)
        
        if max_samples:
            samples_per_class = max_samples // num_classes
        else:
            samples_per_class = min(len(samples) for samples in class_samples.values())
        
        balanced_ids = []
        for label, samples in sorted(class_samples.items()):
            if len(samples) >= samples_per_class:
                selected = random.sample(samples, samples_per_class)
            else:
                selected = samples
            balanced_ids.extend(selected)
        
        random.shuffle(balanced_ids)
        
        print(f"Balanced sampling: {samples_per_class} samples per class ({num_classes} classes)")
        
        return balanced_ids

    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        meta = self.metadata[video_id]
        video_path = meta['path']
        label = meta['label']
        
        # ✅ Get settings
        use_random_crop = self.data_config.get('video', {}).get('use_random_crop', True)
        random_crop_training_only = self.data_config.get('video', {}).get('random_crop_training_only', True)
        use_temporal_alignment = self.data_config.get('audio', {}).get('use_temporal_alignment', False)
        alignment_window = self.data_config.get('audio', {}).get('alignment_window', 8.0)
        
        try:
            faces_data = self.cache_file[f'{video_id}/faces'][:]
            video_features = torch.from_numpy(faces_data)
            
            num_frames = video_features.shape[0]
            start_frame_idx = 0
            
            # ✅ Conditional random crop
            should_random_crop = use_random_crop
            if random_crop_training_only and self.split != 'train':
                should_random_crop = False
            
            if should_random_crop and num_frames > self.max_frames:
                max_start = num_frames - self.max_frames
                start_frame_idx = torch.randint(0, max_start + 1, (1,)).item()
                video_features = video_features[start_frame_idx:start_frame_idx + self.max_frames]
            elif num_frames > self.max_frames:
                video_features = video_features[:self.max_frames]
            
        except Exception:
            video_features = torch.zeros(self.min_frames, 3, 224, 224)
            start_frame_idx = 0
        
        # Padding
        if video_features.shape[0] == 0:
            video_features = torch.zeros(self.min_frames, 3, 224, 224)
        elif video_features.shape[0] < self.min_frames:
            num_needed = self.min_frames - video_features.shape[0]
            last_frame = video_features[-1:].repeat(num_needed, 1, 1, 1)
            video_features = torch.cat([video_features, last_frame], dim=0)
        
        # ✅ Conditional audio
        if self.audio_preprocessor:
            try:
                sample_rate = 3
                fps = 25
                
                if use_temporal_alignment:
                    video_duration = self.max_frames * sample_rate / fps
                    video_start_time = start_frame_idx * sample_rate / fps
                    
                    window_half = alignment_window / 2
                    audio_start = max(0, video_start_time + video_duration/2 - window_half)
                    audio_duration = alignment_window
                else:
                    audio_start = 0.0
                    audio_duration = None
                
                audio_features, timestamps, duration = \
                    self.audio_preprocessor.process_audio(
                        video_path,
                        start_time=audio_start,
                        duration=audio_duration
                    )
            except Exception:
                audio_features = torch.zeros(1, 64000)
        else:
            audio_features = torch.zeros(1, 64000)
        
        return {
            'video_features': video_features,
            'audio_features': audio_features,
            'label': label,
            'dataset': self.dataset_name,
            'metadata': meta,
            'video_path': video_path
        }



    
    def __del__(self):
        if hasattr(self, 'cache_file'):
            self.cache_file.close()


class MultiDatasetLoader(Dataset):
    def __init__(
        self,
        datasets_config: Dict,
        split: str = 'train',
        video_preprocessor = None,
        audio_preprocessor = None,
        max_samples_per_dataset: Optional[int] = None, 
        augment_config: Optional[Dict] = None,
        data_config: Optional[Dict] = None,
        balance_classes: bool = False
    ):
        self.datasets_config = datasets_config
        self.split = split
        self.video_preprocessor = video_preprocessor
        self.audio_preprocessor = audio_preprocessor
        self.samples = []
        self.dataset_labels = []

        self.data_config = data_config or {}
        self.max_frames = self.data_config.get('video', {}).get('max_frames', 32)
        self.min_frames = self.data_config.get('video', {}).get('min_frames', 5)

        self.sample_rate = datasets_config.get('preprocessing', {}).get('video', {}).get('extraction', {}).get('sample_rate', 5)

        if datasets_config.get('av_deepfake', {}).get('enabled', False):
            self._load_av_deepfake(
                datasets_config['av_deepfake']['path'],
                None
            )
        if datasets_config.get('deepspeak', {}).get('enabled', False):
            self._load_deepspeak(
                datasets_config['deepspeak']['path'],
                None
            )

        if balance_classes:
            self.samples = self._balance_classes(self.samples, max_samples_per_dataset)
        elif max_samples_per_dataset and len(self.samples) > max_samples_per_dataset:
            import random
            random.seed(42)
            self.samples = random.sample(self.samples, max_samples_per_dataset)
            print(f"Limited to {max_samples_per_dataset} samples")

        print(f"Loaded {len(self.samples)} samples ({self.split})")

        label_counts = {}
        for s in self.samples:
            label = s['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        dist_str = ", ".join([f"Class {k}: {v}" for k, v in sorted(label_counts.items())])
        print(f"{self.split.capitalize()} distribution - {dist_str}")

        if video_preprocessor and split == 'train' and augment_config:
            video_preprocessor.augment = True
            video_preprocessor.augment_config = augment_config.get('video', {})

    def _balance_classes(self, samples: List[Dict], max_samples: Optional[int]) -> List[Dict]:
        import random
        random.seed(42)
        
        class_samples = {}
        for sample in samples:
            label = sample['label']
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(sample)
        
        num_classes = len(class_samples)
        
        if max_samples:
            samples_per_class = max_samples // num_classes
        else:
            samples_per_class = min(len(samps) for samps in class_samples.values())
        
        balanced_samples = []
        for label, samps in sorted(class_samples.items()):
            if len(samps) >= samples_per_class:
                selected = random.sample(samps, samples_per_class)
            else:
                selected = samps
            balanced_samples.extend(selected)
        
        random.shuffle(balanced_samples)
        
        print(f"Balanced sampling: {samples_per_class} samples per class ({num_classes} classes)")
        
        return balanced_samples

    def _load_av_deepfake(self, dataset_path: str, max_samples: Optional[int]):
        dataset_path = Path(dataset_path)

        if self.split == 'train':
            metadata_file = dataset_path / "train_metadata.json"
            video_dir = dataset_path / "train" / "train"
        elif self.split == 'val':
            metadata_file = dataset_path / "val_metadata.json"
            video_dir = dataset_path / "val" / "val"
        else:
            return

        if not metadata_file.exists():
            return

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        for sample in metadata:
            video_path = video_dir / sample['file']
            
            modify_type = sample.get('modify_type', 'real')
            audio_fake = sample.get('audio', 'real') == 'fake'
            video_fake = modify_type != 'real'
            
            if modify_type == 'real' and not audio_fake:
                label = 0
            elif not audio_fake and video_fake:
                label = 1
            elif audio_fake and not video_fake:
                label = 2
            else:
                label = 3
            
            self.samples.append({
                'dataset': 'av_deepfake',
                'video_path': str(video_path),
                'label': label
            })
            self.dataset_labels.append('av_deepfake')

    def _load_deepspeak(self, dataset_path: str, max_samples: Optional[int]):
        dataset_path = Path(dataset_path)

        split_def = pd.read_csv(dataset_path / "annotations-split-def.csv")
        if self.split == 'train':
            identities = split_def[split_def["split"] == "train"]["identity"].to_list()
        else:
            identities = split_def[split_def["split"] == "test"]["identity"].to_list()
        df_fake = pd.read_csv(dataset_path / "annotations-fake.csv")
        df_real = pd.read_csv(dataset_path / "annotations-real.csv")
        df_fake = df_fake[
            (df_fake["identity-source"].isin(identities)) & 
            (df_fake["identity-target"].isin(identities))
        ]
        df_real = df_real[df_real["identity"].isin(identities)]

        if max_samples:
            n_fake = min(len(df_fake), max_samples // 2)
            n_real = min(len(df_real), max_samples // 2)
            df_fake = df_fake.iloc[:n_fake]
            df_real = df_real.iloc[:n_real]

        for _, row in df_fake.iterrows():
            video_path = dataset_path / row["engine"] / row["video-file"]
            self.samples.append({
                'dataset': 'deepspeak',
                'video_path': str(video_path),
                'label': 1
            })
            self.dataset_labels.append('deepspeak')

        for _, row in df_real.iterrows():
            video_path = dataset_path / "real" / row["video-file"]
            self.samples.append({
                'dataset': 'deepspeak',
                'video_path': str(video_path),
                'label': 0
            })
            self.dataset_labels.append('deepspeak')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['video_path']
        
        if self.video_preprocessor:
            try:
                video_features, valid_indices = self.video_preprocessor.process_video(
                    video_path,
                    sample_rate=self.sample_rate,
                    max_frames=999
                )
                
                num_frames = video_features.shape[0]
                start_frame_idx = 0
                
                if self.split == 'train' and num_frames > self.max_frames:
                    max_start = num_frames - self.max_frames
                    start_frame_idx = torch.randint(0, max_start + 1, (1,)).item()
                    video_features = video_features[start_frame_idx:start_frame_idx + self.max_frames]
                elif num_frames > self.max_frames:
                    video_features = video_features[:self.max_frames]
                
            except Exception:
                video_features = torch.zeros(self.min_frames, 3, 224, 224)
                start_frame_idx = 0
        else:
            video_features = torch.zeros(self.min_frames, 3, 224, 224)
            start_frame_idx = 0
        
        if video_features.shape[0] == 0:
            video_features = torch.zeros(self.min_frames, 3, 224, 224)
        elif video_features.shape[0] < self.min_frames:
            num_needed = self.min_frames - video_features.shape[0]
            last_frame = video_features[-1:].repeat(num_needed, 1, 1, 1)
            video_features = torch.cat([video_features, last_frame], dim=0)
        
        if self.audio_preprocessor:
            try:
                sample_rate = 3
                fps = 25
                target_duration = self.max_frames * sample_rate / fps
                start_time = start_frame_idx * sample_rate / fps
                
                audio_features, timestamps, duration = \
                    self.audio_preprocessor.process_audio(
                        video_path,
                        start_time=start_time,
                        duration=target_duration
                    )
            except Exception:
                audio_features = torch.zeros(1, 64000)
        else:
            audio_features = torch.zeros(1, 64000)
        
        return {
            'video_features': video_features,
            'audio_features': audio_features,
            'label': sample['label'],
            'dataset': sample['dataset'],
            'metadata': sample,
            'video_path': video_path
        }




def collate_fn(batch, min_frames=16, max_frames=32, max_audio_duration=16.0, segment_duration=4.0, overlap=0.5):
    hop_duration = segment_duration * (1.0 - overlap)
    max_audio_segments = int((max_audio_duration - segment_duration) / hop_duration) + 1
    
    # Determine max video frames in this batch
    max_video_frames = min(
        max(max(b['video_features'].shape[0], min_frames) for b in batch),
        max_frames 
    )
    
    # Process video features
    video_batch = []
    video_masks = []
    for b in batch:
        video = b['video_features']
        n_frames = video.shape[0]
        
        # Crop if too long
        if n_frames > max_frames:
            video = video[:max_frames]
            n_frames = max_frames
        
        # Pad if too short (below min_frames)
        if n_frames < min_frames:
            last_frame = video[-1:].repeat(min_frames - n_frames, 1, 1, 1)
            video = torch.cat([video, last_frame], dim=0)
            n_frames = min_frames
        
        # Pad to max_video_frames in batch
        if n_frames < max_video_frames:
            padding = torch.zeros(max_video_frames - n_frames, 3, 224, 224)
            video = torch.cat([video, padding], dim=0)
        
        video_batch.append(video)
        
        # Create mask
        mask = torch.ones(max_video_frames)
        mask[n_frames:] = 0
        video_masks.append(mask)
    
    # Process audio features
    audio_batch = []
    audio_masks = []
    for b in batch:
        audio = b['audio_features']
        n_segments = audio.shape[0]
        
        # Crop if too many segments
        if n_segments > max_audio_segments:
            audio = audio[:max_audio_segments]
            n_segments = max_audio_segments
        
        # Pad if too few segments
        if n_segments < max_audio_segments:
            padding = torch.zeros(max_audio_segments - n_segments, audio.shape[1])
            audio = torch.cat([audio, padding], dim=0)
        
        audio_batch.append(audio)
        
        # Create mask
        mask = torch.ones(max_audio_segments)
        mask[n_segments:] = 0
        audio_masks.append(mask)
    
    # Stack and format batches
    video_batch = torch.stack(video_batch)
    video_batch = video_batch.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
    video_batch = video_batch.contiguous()
    
    audio_batch = torch.stack(audio_batch)
    video_masks = torch.stack(video_masks)
    audio_masks = torch.stack(audio_masks)
    
    # Collect labels and metadata
    labels = torch.tensor([b['label'] for b in batch])
    datasets = [b['dataset'] for b in batch]
    metadata = [b['metadata'] for b in batch]
    
    return {
        'video_features': video_batch,
        'audio_features': audio_batch,
        'video_masks': video_masks,
        'audio_masks': audio_masks,
        'labels': labels,
        'datasets': datasets,
        'metadata': metadata
    }




def create_dataloaders(
    datasets_config: Dict,
    video_preprocessor = None,
    audio_preprocessor = None,
    batch_size: int = 8,
    num_workers: int = 4,
    max_samples_per_dataset: Optional[int] = None,
    augment_config: Optional[Dict] = None,
    data_config: Optional[Dict] = None,
    preprocessing_config: Optional[Dict] = None
):
    if data_config:
        min_frames = data_config.get('video', {}).get('min_frames', 16)
        max_frames = data_config.get('video', {}).get('max_frames', 32)
        max_audio_duration = data_config.get('audio', {}).get('max_duration', 16.0)
    else:
        min_frames = 16
        max_frames = 32
        max_audio_duration = 16.0
    
    if preprocessing_config:
        segment_duration = preprocessing_config.get('audio', {}).get('segmentation', {}).get('segment_duration', 4.0)
        overlap = preprocessing_config.get('audio', {}).get('segmentation', {}).get('overlap', 0.5)
    elif datasets_config and 'preprocessing' in datasets_config:
        segment_duration = datasets_config['preprocessing'].get('audio', {}).get('segmentation', {}).get('segment_duration', 4.0)
        overlap = datasets_config['preprocessing'].get('audio', {}).get('segmentation', {}).get('overlap', 0.5)
    else:
        segment_duration = 4.0
        overlap = 0.5
    
    hop_duration = segment_duration * (1.0 - overlap)
    expected_max_segments = int((max_audio_duration - segment_duration) / hop_duration) + 1
    
    print(f"\n{'='*60}")
    print(f"DATALOADER CONFIGURATION")
    print(f"{'='*60}")
    print(f"Video:")
    print(f"  Min frames: {min_frames}")
    print(f"  Max frames: {max_frames}")
    print(f"Audio:")
    print(f"  Max duration: {max_audio_duration}s")
    print(f"  Segment duration: {segment_duration}s")
    print(f"  Overlap: {overlap}")
    print(f"  Hop duration: {hop_duration}s")
    print(f"  Expected max segments: {expected_max_segments}")
    print(f"{'='*60}\n")
    
    cache_dir_old = Path('/data/shared/ali/deepfake-detection/cache')
    cache_dir_sharded = Path('/data/shared/ali/deepfake-detection/cache_sharded45')
    
    av_enabled = datasets_config.get('av_deepfake', {}).get('enabled', False)
    deepspeak_enabled = datasets_config.get('deepspeak', {}).get('enabled', False)
    
    balance_classes = datasets_config.get('av_deepfake', {}).get('balance_classes', False) or datasets_config.get('deepspeak', {}).get('balance_classes', False)
    
    av_sharded_exists = (cache_dir_sharded / 'av_deepfake_train_index.json').exists()
    deepspeak_sharded_exists = (cache_dir_sharded / 'deepspeak_train_index.json').exists()
    
    av_cache_exists = (cache_dir_old / 'av_deepfake_train_faces.h5').exists()
    deepspeak_cache_exists = (cache_dir_old / 'deepspeak_train_faces.h5').exists()
    
    train_datasets = []
    val_datasets = []
    
    if av_enabled:
        if av_sharded_exists:
            print(f"Loading AV-Deepfake from SHARDED CACHE: {cache_dir_sharded}")
            train_datasets.append(ShardedCachedDataset(
                cache_dir=str(cache_dir_sharded),
                dataset_name='av_deepfake',
                split='train',
                audio_preprocessor=audio_preprocessor,
                data_config=data_config,
                max_samples=max_samples_per_dataset,
                balance_classes=balance_classes
            ))
            if (cache_dir_sharded / 'av_deepfake_val_index.json').exists():
                val_datasets.append(ShardedCachedDataset(
                    cache_dir=str(cache_dir_sharded),
                    dataset_name='av_deepfake',
                    split='val',
                    audio_preprocessor=audio_preprocessor,
                    data_config=data_config,
                    balance_classes=False
                ))
        elif av_cache_exists:
            print(f"Loading AV-Deepfake from CACHE: {cache_dir_old}")
            train_datasets.append(CachedDataset(
                cache_dir=str(cache_dir_old),
                dataset_name='av_deepfake',
                split='train',
                audio_preprocessor=audio_preprocessor,
                data_config=data_config,
                max_samples=max_samples_per_dataset,
                balance_classes=balance_classes
            ))
            val_datasets.append(CachedDataset(
                cache_dir=str(cache_dir_old),
                dataset_name='av_deepfake',
                split='val',
                audio_preprocessor=audio_preprocessor,
                data_config=data_config,
                balance_classes=False
            ))
        else:
            print(f"Loading AV-Deepfake LIVE (no cache found)")
            av_only_config = {
                'av_deepfake': datasets_config['av_deepfake'],
                'deepspeak': {'enabled': False},
                'preprocessing': datasets_config.get('preprocessing', {})
            }
            train_datasets.append(MultiDatasetLoader(
                datasets_config=av_only_config,
                split='train',
                video_preprocessor=video_preprocessor,
                audio_preprocessor=audio_preprocessor,
                max_samples_per_dataset=max_samples_per_dataset,
                augment_config=augment_config,
                data_config=data_config,
                balance_classes=balance_classes
            ))
            val_datasets.append(MultiDatasetLoader(
                datasets_config=av_only_config,
                split='val',
                video_preprocessor=video_preprocessor,
                audio_preprocessor=audio_preprocessor,
                max_samples_per_dataset=None,
                augment_config=None,
                data_config=data_config,
                balance_classes=balance_classes
            ))
    
    if deepspeak_enabled:
        if deepspeak_sharded_exists:
            print(f"Loading DeepSpeak from SHARDED CACHE: {cache_dir_sharded}")
            train_datasets.append(ShardedCachedDataset(
                cache_dir=str(cache_dir_sharded),
                dataset_name='deepspeak',
                split='train',
                audio_preprocessor=audio_preprocessor,
                data_config=data_config,
                max_samples=max_samples_per_dataset,
                balance_classes=balance_classes
            ))
            if (cache_dir_sharded / 'deepspeak_val_index.json').exists():
                val_datasets.append(ShardedCachedDataset(
                    cache_dir=str(cache_dir_sharded),
                    dataset_name='deepspeak',
                    split='val',
                    audio_preprocessor=audio_preprocessor,
                    data_config=data_config,
                    balance_classes=False
                ))
        elif deepspeak_cache_exists:
            print(f"Loading DeepSpeak from CACHE: {cache_dir_old}")
            train_datasets.append(CachedDataset(
                cache_dir=str(cache_dir_old),
                dataset_name='deepspeak',
                split='train',
                audio_preprocessor=audio_preprocessor,
                data_config=data_config,
                max_samples=max_samples_per_dataset,
                balance_classes=balance_classes
            ))
            val_datasets.append(CachedDataset(
                cache_dir=str(cache_dir_old),
                dataset_name='deepspeak',
                split='val',
                audio_preprocessor=audio_preprocessor,
                data_config=data_config,
                balance_classes=False
            ))
        else:
            print(f"Loading DeepSpeak LIVE (no cache found)")
            ds_only_config = {
                'av_deepfake': {'enabled': False},
                'deepspeak': datasets_config['deepspeak'],
                'preprocessing': datasets_config.get('preprocessing', {})
            }
            train_datasets.append(MultiDatasetLoader(
                datasets_config=ds_only_config,
                split='train',
                video_preprocessor=video_preprocessor,
                audio_preprocessor=audio_preprocessor,
                max_samples_per_dataset=max_samples_per_dataset,
                augment_config=augment_config,
                data_config=data_config
            ))
            val_datasets.append(MultiDatasetLoader(
                datasets_config=ds_only_config,
                split='val',
                video_preprocessor=video_preprocessor,
                audio_preprocessor=audio_preprocessor,
                max_samples_per_dataset=None,
                augment_config=None,
                data_config=data_config
            ))
    
    if len(train_datasets) == 0:
        raise ValueError("No datasets enabled! Check your config.")
    
    if len(train_datasets) > 1:
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
        print(f"Combined {len(train_datasets)} datasets")
    else:
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]
    
    print(f"Total: {len(train_dataset)} train samples, {len(val_dataset)} val samples\n")
    
    from functools import partial
    collate_fn_with_config = partial(
        collate_fn, 
        min_frames=min_frames, 
        max_frames=max_frames, 
        max_audio_duration=max_audio_duration,
        segment_duration=segment_duration,
        overlap=overlap
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_with_config,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_with_config,
        pin_memory=True
    )
    
    return train_loader, val_loader
