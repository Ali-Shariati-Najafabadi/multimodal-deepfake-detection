import torch
import yaml
import argparse
from pathlib import Path
import numpy as np
import sys
import json


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from models import get_model
from data.dataloaders.multi_dataset_loader import create_dataloaders, CachedDataset, ShardedCachedDataset, collate_fn
from data.preprocessing.audio_preprocessing import AudioPreprocessor
from training.training_utils import MetricsCalculator



class Evaluator:
    def __init__(self, config_path: str, checkpoint_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.phase = self.config['experiment']['phase']
        
        self.setup_preprocessors()
        self.setup_model()
        self.load_checkpoint(checkpoint_path)
    
    
    def setup_preprocessors(self):
        audio_config = self.config['preprocessing']['audio']
        self.audio_preprocessor = AudioPreprocessor(
            target_sr=audio_config['extraction']['target_sr'],
            segment_duration=audio_config['segmentation']['segment_duration'],
            overlap=audio_config['segmentation']['overlap'],
            normalize=audio_config['normalize']
        )
    
    
    def setup_model(self):
        print(f"\n{'='*60}")
        print(f"Setting up model for phase: {self.phase}")
        print(f"{'='*60}")
        
        self.model = get_model(self.phase, self.config)
        
        print(f"Model type: {type(self.model).__name__}")
        self.model = self.model.to(self.device)
        print(f"Model moved to device: {self.device}")
        
        model_info = self.model.get_model_info()
        print(f"\n MODEL INFO:")
        print(f"   Phase: {model_info['phase']}")
        
        if 'fusion_type' in model_info:
            print(f"   Fusion: {model_info['fusion_type']}")
        
        if 'video_architecture' in model_info:
            print(f"   Video: {model_info['video_architecture']}")
            
        if 'audio_architecture' in model_info:
            print(f"   Audio: {model_info['audio_architecture']}")
        
        print(f"   Total params: {model_info['total_params']:,}")
        print(f"   Trainable: {model_info['total_trainable']:,}")
        
        self.model.eval()
        print(f"Model set to eval mode")
        print(f"{'='*60}\n")
    
    
    def load_checkpoint(self, checkpoint_path: str):
        print(f"\n{'='*60}")
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            if 'epoch' in checkpoint:
                print(f"Checkpoint epoch: {checkpoint['epoch']}")
        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"Checkpoint epoch: {checkpoint['epoch']}")
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Checkpoint loaded successfully")
        print(f"{'='*60}\n")
    
    
    def _get_cached_dataloader(self, dataset_name: str, split: str, max_samples: int = None):
        cache_dir_old = Path('/data/shared/ali/deepfake-detection/cache')
        cache_dir_sharded = Path('/data/shared/ali/deepfake-detection/cache_sharded')
        
        data_config = self.config.get('data', {})
        batch_size = self.config['training']['batch_size']
        
        sharded_exists = (cache_dir_sharded / f'{dataset_name}_{split}_index.json').exists()
        single_exists = (cache_dir_old / f'{dataset_name}_{split}_faces.h5').exists()
        
        if sharded_exists:
            print(f"Loading {dataset_name} {split} from SHARDED CACHE: {cache_dir_sharded}")
            dataset = ShardedCachedDataset(
                cache_dir=str(cache_dir_sharded),
                dataset_name=dataset_name,
                split=split,
                audio_preprocessor=self.audio_preprocessor,
                data_config=data_config
            )
        elif single_exists:
            print(f"Loading {dataset_name} {split} from CACHE: {cache_dir_old}")
            dataset = CachedDataset(
                cache_dir=str(cache_dir_old),
                dataset_name=dataset_name,
                split=split,
                audio_preprocessor=self.audio_preprocessor,
                data_config=data_config
            )
        else:
            print(f"No cache found for {dataset_name} {split}")
            return None
        
        if max_samples and max_samples < len(dataset):
            import random
            random.seed(42)
            indices = random.sample(range(len(dataset)), max_samples)
            dataset = torch.utils.data.Subset(dataset, indices)
            print(f"Limited to {max_samples} samples (random subset)")
        
        from functools import partial
        min_frames = data_config.get('video', {}).get('min_frames', 16)
        max_frames = data_config.get('video', {}).get('max_frames', 32)
        collate_fn_cfg = partial(collate_fn, min_frames=min_frames, max_frames=max_frames)
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=16,
            collate_fn=collate_fn_cfg,
            pin_memory=True
        )
        
        return dataloader
    
    
    def _get_live_dataloader(self, dataset_name: str, split: str, max_samples: int = None):
        from data.preprocessing.video_preprocessing import VideoPreprocessor
        from data.dataloaders.multi_dataset_loader import MultiDatasetLoader
        
        video_config = self.config['preprocessing']['video']
        video_preprocessor = VideoPreprocessor(
            face_size=video_config['face_detection']['face_size'],
            device=video_config['face_detection']['device'],
            min_face_size=video_config['face_detection']['min_face_size'],
            confidence_threshold=video_config['face_detection']['confidence_threshold']
        )
        
        datasets_config = self.config['datasets'].copy()
        for ds_name in datasets_config.keys():
            if ds_name not in ['sampling']:
                datasets_config[ds_name]['enabled'] = (ds_name == dataset_name)
        
        data_config = self.config.get('data', {})
        
        dataset = MultiDatasetLoader(
            datasets_config=datasets_config,
            split=split,
            video_preprocessor=video_preprocessor,
            audio_preprocessor=self.audio_preprocessor,
            max_samples_per_dataset=max_samples,
            augment_config=None,
            data_config=data_config
        )
        
        from functools import partial
        min_frames = data_config.get('video', {}).get('min_frames', 16)
        max_frames = data_config.get('video', {}).get('max_frames', 32)
        collate_fn_cfg = partial(collate_fn, min_frames=min_frames, max_frames=max_frames)
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=16,
            collate_fn=collate_fn_cfg,
            pin_memory=True
        )
        
        print(f"Loading {dataset_name} {split} LIVE (no cache)")
        return dataloader
    
    
    @torch.no_grad()
    def evaluate_dataset(self, split: str = 'val', max_samples: int = None, dataset_name: str = None):
        print(f"\n{'='*80}")
        print(f"EVALUATING ON {split.upper()} SET")
        if dataset_name:
            print(f"Dataset filter: {dataset_name}")
        print(f"{'='*80}\n")
        
        if dataset_name:
            dataloader = self._get_cached_dataloader(dataset_name, split, max_samples)
            if dataloader is None:
                dataloader = self._get_live_dataloader(dataset_name, split, max_samples)
        else:
            datasets_config = self.config['datasets'].copy()
            _, dataloader = create_dataloaders(
                datasets_config=datasets_config,
                audio_preprocessor=self.audio_preprocessor,
                batch_size=self.config['training']['batch_size'],
                num_workers=16,
                max_samples_per_dataset=max_samples,
                data_config=self.config.get('data', {})
            )
        
        metrics_calc = MetricsCalculator()
        first_batch_done = False
        
        for batch_idx, batch in enumerate(dataloader):
            video_features = batch['video_features'].to(self.device)
            audio_features = batch['audio_features'].to(self.device)
            video_masks = batch['video_masks'].to(self.device)
            audio_masks = batch['audio_masks'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if not first_batch_done:
                print(f"\nFirst batch info:")
                print(f"  Video features: {video_features.shape}")
                print(f"  Audio features: {audio_features.shape}")
                print(f"  Labels distribution: Real={sum(labels==0).item()}, Fake={sum(labels==1).item()}")
                first_batch_done = True
            
            logits = self.model(video_features, audio_features, video_masks, audio_masks)
            
            if batch_idx == 0:
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
                print(f"  Logits std: {logits.std():.4f}")
                
                preds_sample = torch.argmax(logits[:5], dim=1)
                print(f"  First 5 predictions: {preds_sample.cpu().tolist()}")
                print(f"  First 5 labels: {labels[:5].cpu().tolist()}")
                print()
            
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            metrics_calc.update(labels, preds, probs)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        metrics = metrics_calc.compute()
        
        print(f"\n{'='*80}")
        print(f"RESULTS ON {split.upper()} SET" + (f" ({dataset_name})" if dataset_name else ""))
        print(f"{'='*80}")
        metrics_calc.print_metrics(metrics, prefix="  ")
        
        return metrics
    
    
    @torch.no_grad()
    def evaluate_modalities_separate(self, split: str = 'val', max_samples: int = None, 
                                dataset_name: str = None, audio_scale: float = 1.0):
        print(f"\n{'='*80}")
        print(f"MODALITY SEPARATE EVALUATION ({split} set)")
        if dataset_name:
            print(f"Dataset filter: {dataset_name}")
        if audio_scale != 1.0:
            print(f"Audio scale: {audio_scale}")
        print(f"{'='*80}\n")
        
        if dataset_name:
            dataloader = self._get_cached_dataloader(dataset_name, split, max_samples)
            if dataloader is None:
                dataloader = self._get_live_dataloader(dataset_name, split, max_samples)
        else:
            datasets_config = self.config['datasets'].copy()
            _, dataloader = create_dataloaders(
                datasets_config=datasets_config,
                audio_preprocessor=self.audio_preprocessor,
                batch_size=self.config['training']['batch_size'],
                num_workers=16,
                max_samples_per_dataset=max_samples,
                data_config=self.config.get('data', {})
            )
        
        video_metrics = MetricsCalculator()
        audio_metrics = MetricsCalculator()
        fusion_metrics = MetricsCalculator()
        
        # Read audio pooling type from config
        audio_pooling = self.config['models']['audio'].get('pooling', 'mean')
        
        for batch_idx, batch in enumerate(dataloader):
            video_features = batch['video_features'].to(self.device)
            audio_features = batch['audio_features'].to(self.device)
            video_masks = batch['video_masks'].to(self.device)
            audio_masks = batch['audio_masks'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if batch_idx == 0:
                print(f"INPUT SHAPES:")
                print(f"  Video features: {video_features.shape}")
                print(f"  Audio features: {audio_features.shape}")
                print(f"  Video masks: {video_masks.shape}")
                print(f"  Audio masks: {audio_masks.shape}")
                print(f"  Audio pooling type: {audio_pooling}\n")
            
            # ==================== VIDEO EVALUATION ====================
            video_model_type = type(self.model.video_model).__name__
            
            if video_model_type == 'TimeSformer':
                video_feat = self.model.extract_video_features(video_features, video_masks)
                video_logits = self.model.video_model.classifier(video_feat)
                
                if batch_idx == 0:
                    print(f"VIDEO EVALUATION DEBUG (TimeSformer):")
                    print(f"  Features shape: {video_feat.shape}")
                    print(f"  Logits shape: {video_logits.shape}")
                    print(f"  Logits range: [{video_logits.min():.2f}, {video_logits.max():.2f}]")
                    print(f"  Logits std: {video_logits.std():.4f}\n")
            
            elif video_model_type in ['VideoSwinTransformer', 'X3D']:
                video_feat = self.model.extract_video_features(video_features, video_masks)
                video_logits = self.model.video_model.classifier(video_feat)
                
                if batch_idx == 0:
                    print(f"VIDEO EVALUATION DEBUG ({video_model_type}):")
                    print(f"  Features shape: {video_feat.shape}")
                    print(f"  Logits shape: {video_logits.shape}")
                    print(f"  Logits range: [{video_logits.min():.2f}, {video_logits.max():.2f}]")
                    print(f"  Logits std: {video_logits.std():.4f}\n")
            
            elif video_model_type == 'ResNetLSTM':
                video_feat = self.model.extract_video_features(video_features, video_masks)
                video_logits = self.model.video_model.classifier(video_feat)
                
                if batch_idx == 0:
                    print(f"VIDEO EVALUATION DEBUG (ResNetLSTM):")
                    print(f"  Aggregated features shape: {video_feat.shape}")
                    print(f"  Logits shape: {video_logits.shape}")
                    print(f"  Logits range: [{video_logits.min():.2f}, {video_logits.max():.2f}]")
                    print(f"  Logits std: {video_logits.std():.4f}\n")
            
            elif self.phase == 'phase2':
                video_feat = self.model.extract_video_features(video_features, video_masks)
                video_logits = self.model.video_model.classifier(video_feat)
                
                if batch_idx == 0:
                    print(f"VIDEO EVALUATION DEBUG (Phase 2 - FTCN):")
                    print(f"  Aggregated features shape: {video_feat.shape}")
                    print(f"  Logits shape: {video_logits.shape}")
                    print(f"  Logits range: [{video_logits.min():.2f}, {video_logits.max():.2f}]")
                    print(f"  Logits std: {video_logits.std():.4f}\n")
            
            else:
                B, C, T, H, W = video_features.shape
                video_frames = video_features.permute(0, 2, 1, 3, 4).contiguous()
                video_frames = video_frames.view(B * T, C, H, W)
                
                frame_features = self.model.video_model.backbone(video_frames)
                
                pooling_method = getattr(self.model.video_model, 'pooling', 'avg')
                
                if pooling_method == "avg":
                    frame_features = torch.nn.functional.adaptive_avg_pool2d(frame_features, (1, 1))
                elif pooling_method == "max":
                    frame_features = torch.nn.functional.adaptive_max_pool2d(frame_features, (1, 1))
                
                frame_features = frame_features.view(B, T, -1)
                
                valid_frames = video_masks.sum(dim=1, keepdim=True).clamp(min=1)
                masked_features = frame_features * video_masks.unsqueeze(-1).float()
                video_feat = masked_features.sum(dim=1) / valid_frames.float()
                
                video_logits = self.model.video_model.classifier(video_feat)
                
                if batch_idx == 0:
                    print(f"VIDEO EVALUATION DEBUG (Standard VideoModel):")
                    print(f"  Frame features shape: {frame_features.shape}")
                    print(f"  Aggregated features shape: {video_feat.shape}")
                    print(f"  Logits shape: {video_logits.shape}")
                    print(f"  Logits range: [{video_logits.min():.2f}, {video_logits.max():.2f}]")
                    print(f"  Logits std: {video_logits.std():.4f}\n")
            
            video_preds = torch.argmax(video_logits, dim=1)
            video_probs = torch.softmax(video_logits, dim=1)
            video_metrics.update(labels, video_preds, video_probs)
            
            # ==================== AUDIO EVALUATION ====================
            B_audio, num_segments, segment_length = audio_features.shape
            audio_flat = audio_features.view(B_audio * num_segments, segment_length)
            
            # Extract segment features from HuBERT encoder
            outputs = self.model.audio_model.encoder(audio_flat)
            segment_features = outputs.last_hidden_state.mean(dim=1)  # [B*segs, 768]
            segment_features = segment_features.view(B_audio, num_segments, -1)
            
            # Apply mask
            valid_segments = audio_masks.sum(dim=1, keepdim=True).clamp(min=1)
            masked_audio = segment_features * audio_masks.unsqueeze(-1).float()
            
            # Apply pooling based on config
            if audio_pooling == 'mean':
                audio_feat = masked_audio.sum(dim=1) / valid_segments.float()  # [B, 768]
            
            elif audio_pooling == 'max':
                audio_feat = (segment_features * audio_masks.unsqueeze(-1).float()).max(dim=1)[0]  # [B, 768]
            
            elif audio_pooling == 'max_mean':
                mean_feat = masked_audio.sum(dim=1) / valid_segments.float()  # [B, 768]
                max_feat = (segment_features * audio_masks.unsqueeze(-1).float()).max(dim=1)[0]  # [B, 768]
                audio_feat = torch.cat([max_feat, mean_feat], dim=1)  # [B, 1536]
            
            elif audio_pooling == 'avg':  # alias for mean
                audio_feat = masked_audio.sum(dim=1) / valid_segments.float()  # [B, 768]
            
            else:
                raise ValueError(f"Unknown audio pooling type: {audio_pooling}")
            
            audio_logits = self.model.audio_model.classifier(audio_feat)
            audio_logits = audio_logits * audio_scale
            
            if batch_idx == 0:
                print(f"AUDIO EVALUATION DEBUG:")
                print(f"  Segment features shape: {segment_features.shape}")
                print(f"  Pooling type: {audio_pooling}")
                print(f"  Aggregated features shape: {audio_feat.shape}")
                print(f"  Logits shape: {audio_logits.shape}")
                print(f"  Logits range: [{audio_logits.min():.2f}, {audio_logits.max():.2f}]")
                print(f"  Logits std: {audio_logits.std():.4f}\n")
            
            audio_preds = torch.argmax(audio_logits, dim=1)
            audio_probs = torch.softmax(audio_logits, dim=1)
            audio_metrics.update(labels, audio_preds, audio_probs)
            
            # ==================== FUSION EVALUATION ====================
            fusion_logits = self.model(video_features, audio_features, video_masks, audio_masks)
            fusion_preds = torch.argmax(fusion_logits, dim=1)
            fusion_probs = torch.softmax(fusion_logits, dim=1)
            fusion_metrics.update(labels, fusion_preds, fusion_probs)
            
            if batch_idx == 0:
                print(f"FUSION EVALUATION DEBUG:")
                print(f"  Logits shape: {fusion_logits.shape}")
                print(f"  Logits range: [{fusion_logits.min():.2f}, {fusion_logits.max():.2f}]")
                print(f"  Logits std: {fusion_logits.std():.4f}\n")
                
                print(f"FIRST BATCH PREDICTIONS (first 5 samples):")
                print(f"  Labels:  {labels[:5].cpu().tolist()}")
                print(f"  Video:   {video_preds[:5].cpu().tolist()}")
                print(f"  Audio:   {audio_preds[:5].cpu().tolist()}")
                print(f"  Fusion:  {fusion_preds[:5].cpu().tolist()}\n")
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # ==================== COMPUTE RESULTS ====================
        video_res = video_metrics.compute()
        audio_res = audio_metrics.compute()
        fusion_res = fusion_metrics.compute()
        
        print(f"\n{'='*80}")
        print(f"RESULTS COMPARISON")
        print(f"{'='*80}\n")
        
        print(f"VIDEO ONLY:")
        video_metrics.print_metrics(video_res, prefix="  ")
        
        print(f"\nAUDIO ONLY (pooling={audio_pooling}, scale={audio_scale}):")
        audio_metrics.print_metrics(audio_res, prefix="  ")
        
        print(f"\nFUSION:")
        fusion_metrics.print_metrics(fusion_res, prefix="  ")
        
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"  Video ACC:  {video_res['accuracy']:.1%} | AUC: {video_res['auc']:.4f}")
        print(f"  Audio ACC:  {audio_res['accuracy']:.1%} | AUC: {audio_res['auc']:.4f}")
        print(f"  Fusion ACC: {fusion_res['accuracy']:.1%} | AUC: {fusion_res['auc']:.4f}")
        print(f"  Best modality: {'VIDEO' if video_res['accuracy'] > audio_res['accuracy'] else 'AUDIO'}")
        print(f"  Fusion improvement: {(fusion_res['accuracy'] - max(video_res['accuracy'], audio_res['accuracy']))*100:.2f}%")
        print(f"{'='*80}\n")
        
        return {
            'video': video_res,
            'audio': audio_res,
            'fusion': fusion_res
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['dataset', 'modality'], default='dataset')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--dataset', type=str, help='Specific dataset to evaluate on')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--audio-scale', type=float, default=1.0)
    parser.add_argument('--save-results', type=str, default=None)
    
    args = parser.parse_args()
    
    evaluator = Evaluator(config_path=args.config, checkpoint_path=args.checkpoint)
    
    if args.mode == 'dataset':
        metrics = evaluator.evaluate_dataset(
            split=args.split, 
            max_samples=args.max_samples,
            dataset_name=args.dataset
        )
        
        if args.save_results:
            serializable_metrics = {}
            for key, val in metrics.items():
                if isinstance(val, np.ndarray):
                    serializable_metrics[key] = val.tolist()
                elif isinstance(val, (np.float32, np.float64)):
                    serializable_metrics[key] = float(val)
                else:
                    serializable_metrics[key] = val
            
            with open(args.save_results, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
            print(f"\nResults saved to: {args.save_results}")
    
    elif args.mode == 'modality':
        results = evaluator.evaluate_modalities_separate(
            split=args.split,
            max_samples=args.max_samples,
            dataset_name=args.dataset,
            audio_scale=args.audio_scale
        )
        
        if args.save_results:
            import pickle
            with open(args.save_results, 'wb') as f:
                pickle.dump(results, f)
            print(f"Results saved to: {args.save_results}")


if __name__ == "__main__":
    main()