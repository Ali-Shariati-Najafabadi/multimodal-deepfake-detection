import torch
import yaml
import argparse
from pathlib import Path
import sys
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import get_model
from training.training_utils import MetricsCalculator
from data.dataloaders.fakeavceleb_dataset import FakeAVCelebDataset


class FakeAVCelebEvaluator:
    def __init__(self, config_path: str, checkpoint_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.phase = self.config['experiment']['phase']
        
        self.setup_preprocessors()
        self.setup_model()
        self.load_checkpoint(checkpoint_path)
    
    
    @staticmethod
    def fakeavceleb_collate_fn(batch):
        """Custom collate function for FakeAVCeleb"""
        video_features = torch.stack([b['video_features'] for b in batch])
        audio_features = torch.nn.utils.rnn.pad_sequence(
            [b['audio_features'] for b in batch],
            batch_first=True
        )
        video_masks = torch.stack([b['video_masks'] for b in batch])
        audio_masks = torch.nn.utils.rnn.pad_sequence(
            [b['audio_masks'] for b in batch],
            batch_first=True
        )
        labels = torch.stack([b['labels'] for b in batch])
        
        return {
            'video_features': video_features,
            'audio_features': audio_features,
            'video_masks': video_masks,
            'audio_masks': audio_masks,
            'labels': labels
        }
    
    
    def setup_preprocessors(self):
        # FakeAVCeleb uses direct video/audio loading, no preprocessors needed
        self.video_preprocessor = None
        self.audio_preprocessor = None
    
    
    def setup_model(self):
        print(f"\n{'='*60}")
        print(f"Setting up model for phase: {self.phase}")
        print(f"{'='*60}")
        
        self.model = get_model(self.phase, self.config).to(self.device)
        self.model.eval()
        
        model_info = self.model.get_model_info()
        print(f"   Phase: {model_info['phase']}")
        print(f"   Total params: {model_info['total_params']:,}")
        print(f"   Trainable: {model_info['total_trainable']:,}")
        print(f"{'='*60}\n")
    
    
    def load_checkpoint(self, checkpoint_path: str):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"Checkpoint epoch: {checkpoint['epoch']}")
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Checkpoint loaded successfully\n")
    
    
    @torch.no_grad()
    def evaluate_by_modality(
        self,
        root_dir: str,
        metadata_csv: str,
        split: str = 'test',
        max_samples: int = None,
        batch_size: int = 4
    ):
        """Evaluate performance by modality (Video-only, Audio-only, Fusion)"""
        
        print(f"\n{'='*80}")
        print(f"MODALITY EVALUATION ON FAKEAVCELEB ({split} set)")
        print(f"{'='*80}\n")
        
        # Create dataset
        dataset = FakeAVCelebDataset(
            root_dir=root_dir,
            metadata_csv=metadata_csv,
            split=split,
            video_preprocessor=self.video_preprocessor,
            audio_preprocessor=self.audio_preprocessor,
            data_config=self.config.get('data', {}),
            max_samples=max_samples
        )
        
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=self.fakeavceleb_collate_fn
        )
        
        # Three separate metrics calculators
        video_metrics_calc = MetricsCalculator()
        audio_metrics_calc = MetricsCalculator()
        fusion_metrics_calc = MetricsCalculator()
        
        print("Evaluating all modalities...")
        
        for batch_idx, batch in enumerate(dataloader):
            video_features = batch['video_features'].to(self.device)
            audio_features = batch['audio_features'].to(self.device)
            video_masks = batch['video_masks'].to(self.device)
            audio_masks = batch['audio_masks'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Get model outputs for each modality
            # Simple approach: call models directly and check output shape
            
            # 1. VIDEO-ONLY
            video_output = self.model.video_model(video_features, video_masks)
            # Check if output is logits (shape: [batch, 2]) or features (shape: [batch, feature_dim])
            if video_output.shape[-1] == 2:
                video_logits = video_output
            else:
                # It's features - need to pass through head
                # Create dummy audio features (zeros)
                batch_size = video_output.shape[0]
                audio_dim = 768  # HuBERT feature dim
                dummy_audio_feat = torch.zeros(batch_size, audio_dim, device=self.device)
                
                # Concatenate and pass through fusion head
                fused_feat = torch.cat([video_output, dummy_audio_feat], dim=1)
                video_logits = self.model.head(fused_feat)
            
            # 2. AUDIO-ONLY  
            audio_output = self.model.audio_model(audio_features, audio_masks)
            # Check if output is logits or features
            if audio_output.shape[-1] == 2:
                audio_logits = audio_output
            else:
                # It's features - need to pass through head
                # Create dummy video features (zeros)
                batch_size = audio_output.shape[0]
                video_dim = video_output.shape[1] if video_output.shape[-1] != 2 else 512
                dummy_video_feat = torch.zeros(batch_size, video_dim, device=self.device)
                
                # Concatenate and pass through fusion head
                fused_feat = torch.cat([dummy_video_feat, audio_output], dim=1)
                audio_logits = self.model.head(fused_feat)
            
            # 3. FUSION: Use normal forward pass
            fusion_logits = self.model(video_features, audio_features, video_masks, audio_masks)
            video_preds = torch.argmax(video_logits, dim=1)
            video_probs = torch.softmax(video_logits, dim=1)
            video_metrics_calc.update(labels, video_preds, video_probs)
            
            # Audio-only predictions
            audio_preds = torch.argmax(audio_logits, dim=1)
            audio_probs = torch.softmax(audio_logits, dim=1)
            audio_metrics_calc.update(labels, audio_preds, audio_probs)
            
            # Fusion predictions
            fusion_preds = torch.argmax(fusion_logits, dim=1)
            fusion_probs = torch.softmax(fusion_logits, dim=1)
            fusion_metrics_calc.update(labels, fusion_preds, fusion_probs)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # Compute metrics
        video_metrics = video_metrics_calc.compute()
        audio_metrics = audio_metrics_calc.compute()
        fusion_metrics = fusion_metrics_calc.compute()
        
        # Print results
        print(f"\n{'='*80}")
        print(f"VIDEO-ONLY RESULTS")
        print(f"{'='*80}")
        video_metrics_calc.print_metrics(video_metrics, prefix="  ")
        
        print(f"\n{'='*80}")
        print(f"AUDIO-ONLY RESULTS")
        print(f"{'='*80}")
        audio_metrics_calc.print_metrics(audio_metrics, prefix="  ")
        
        print(f"\n{'='*80}")
        print(f"FUSION RESULTS")
        print(f"{'='*80}")
        fusion_metrics_calc.print_metrics(fusion_metrics, prefix="  ")
        
        # Summary comparison
        print(f"\n{'='*80}")
        print(f"MODALITY COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"{'Modality':<15} {'Accuracy':<12} {'UAR':<12} {'AUC':<12} {'F1-Score':<12}")
        print(f"{'-'*80}")
        
        modalities = {
            'Video-only': video_metrics,
            'Audio-only': audio_metrics,
            'Fusion': fusion_metrics
        }
        
        for name, metrics in modalities.items():
            acc = metrics['accuracy'] * 100
            uar = metrics.get('uar', 0) * 100
            auc = metrics.get('auc', 0)
            f1 = metrics.get('f1', 0)
            print(f"{name:<15} {acc:>10.2f}%  {uar:>10.2f}%  {auc:>10.4f}    {f1:>10.4f}")
        
        # Best modality
        best_modality = max(modalities.items(), key=lambda x: x[1].get('uar', 0))
        print(f"\n{'='*80}")
        print(f"Best Modality: {best_modality[0]} (UAR: {best_modality[1]['uar']*100:.2f}%)")
        print(f"{'='*80}\n")
        
        return {
            'video': video_metrics,
            'audio': audio_metrics,
            'fusion': fusion_metrics
        }
    
    
    @torch.no_grad()
    def evaluate_overall(
        self,
        root_dir: str,
        metadata_csv: str,
        split: str = 'test',
        max_samples: int = None,
        batch_size: int = 4
    ):
        """Evaluate overall performance on FakeAVCeleb"""
        
        print(f"\n{'='*80}")
        print(f"OVERALL EVALUATION ON FAKEAVCELEB ({split} set)")
        print(f"{'='*80}\n")
        
        # Create dataset
        dataset = FakeAVCelebDataset(
            root_dir=root_dir,
            metadata_csv=metadata_csv,
            split=split,
            video_preprocessor=self.video_preprocessor,
            audio_preprocessor=self.audio_preprocessor,
            data_config=self.config.get('data', {}),
            max_samples=max_samples
        )
        
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=self.fakeavceleb_collate_fn
        )
        
        metrics_calc = MetricsCalculator()
        
        for batch_idx, batch in enumerate(dataloader):
            video_features = batch['video_features'].to(self.device)
            audio_features = batch['audio_features'].to(self.device)
            video_masks = batch['video_masks'].to(self.device)
            audio_masks = batch['audio_masks'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            logits = self.model(video_features, audio_features, video_masks, audio_masks)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            
            metrics_calc.update(labels, preds, probs)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        metrics = metrics_calc.compute()
        
        print(f"\n{'='*80}")
        print(f"OVERALL RESULTS")
        print(f"{'='*80}")
        metrics_calc.print_metrics(metrics, prefix="  ")
        
        return metrics
    
    
    @torch.no_grad()
    def evaluate_by_method(
        self,
        root_dir: str,
        metadata_csv: str,
        split: str = 'test',
        max_samples_per_method: int = None,
        batch_size: int = 4
    ):
        """Evaluate performance per deepfake method"""
        
        print(f"\n{'='*80}")
        print(f"METHOD-SPECIFIC EVALUATION ON FAKEAVCELEB ({split} set)")
        print(f"{'='*80}\n")
        
        methods = ['real', 'faceswap', 'fsgan', 'wav2lip', 'sv2tts']
        
        results = {}
        
        for method in methods:
            print(f"\n{'='*60}")
            print(f"Evaluating: {method.upper()}")
            print(f"{'='*60}\n")
            
            # Create dataset filtered by method
            if method == 'real':
                categories = ['A']
                method_filter = None
            else:
                categories = ['B', 'C', 'D']
                method_filter = [method]
            
            dataset = FakeAVCelebDataset(
                root_dir=root_dir,
                metadata_csv=metadata_csv,
                split=split,
                video_preprocessor=self.video_preprocessor,
                audio_preprocessor=self.audio_preprocessor,
                data_config=self.config.get('data', {}),
                max_samples=max_samples_per_method,
                categories=categories,
                methods=method_filter
            )
            
            if len(dataset) == 0:
                print(f"No samples for method: {method}")
                continue
            
            from torch.utils.data import DataLoader
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                collate_fn=self.fakeavceleb_collate_fn
            )
            
            metrics_calc = MetricsCalculator()
            
            for batch in dataloader:
                video_features = batch['video_features'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                video_masks = batch['video_masks'].to(self.device)
                audio_masks = batch['audio_masks'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(video_features, audio_features, video_masks, audio_masks)
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)
                
                metrics_calc.update(labels, preds, probs)
            
            metrics = metrics_calc.compute()
            results[method] = metrics
            
            print(f"\nResults for {method.upper()}:")
            metrics_calc.print_metrics(metrics, prefix="  ")
        
        # Summary
        print(f"\n{'='*80}")
        print(f"SUMMARY BY METHOD")
        print(f"{'='*80}")
        print(f"{'Method':<15} {'Accuracy':<12} {'UAR':<12} {'AUC':<12}")
        print(f"{'-'*80}")
        
        for method, metrics in results.items():
            acc = metrics['accuracy'] * 100
            uar = metrics.get('uar', 0) * 100
            auc = metrics.get('auc', 0)
            print(f"{method:<15} {acc:>10.2f}%  {uar:>10.2f}%  {auc:>10.4f}")
        
        print(f"{'='*80}\n")
        
        return results
    
    
    @torch.no_grad()
    def evaluate_by_category(
        self,
        root_dir: str,
        metadata_csv: str,
        split: str = 'test',
        max_samples_per_category: int = None,
        batch_size: int = 4
    ):
        """Evaluate performance per category (A/B/C/D)"""
        
        print(f"\n{'='*80}")
        print(f"CATEGORY-SPECIFIC EVALUATION ON FAKEAVCELEB ({split} set)")
        print(f"{'='*80}\n")
        
        category_names = {
            'A': 'RealVideo-RealAudio',
            'B': 'RealVideo-FakeAudio',
            'C': 'FakeVideo-RealAudio',
            'D': 'FakeVideo-FakeAudio'
        }
        
        results = {}
        
        for category, name in category_names.items():
            print(f"\n{'='*60}")
            print(f"Evaluating Category {category}: {name}")
            print(f"{'='*60}\n")
            
            dataset = FakeAVCelebDataset(
                root_dir=root_dir,
                metadata_csv=metadata_csv,
                split=split,
                video_preprocessor=self.video_preprocessor,
                audio_preprocessor=self.audio_preprocessor,
                data_config=self.config.get('data', {}),
                max_samples=max_samples_per_category,
                categories=[category]
            )
            
            from torch.utils.data import DataLoader
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                collate_fn=self.fakeavceleb_collate_fn
            )
            metrics_calc = MetricsCalculator()
            
            for batch in dataloader:
                video_features = batch['video_features'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                video_masks = batch['video_masks'].to(self.device)
                audio_masks = batch['audio_masks'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(video_features, audio_features, video_masks, audio_masks)
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)
                
                metrics_calc.update(labels, preds, probs)
            
            metrics = metrics_calc.compute()
            results[category] = metrics
            
            print(f"\nResults for Category {category} ({name}):")
            metrics_calc.print_metrics(metrics, prefix="  ")
        
        # Summary
        print(f"\n{'='*80}")
        print(f"SUMMARY BY CATEGORY")
        print(f"{'='*80}")
        print(f"{'Category':<30} {'Accuracy':<12} {'UAR':<12} {'AUC':<12}")
        print(f"{'-'*80}")
        
        for category, metrics in results.items():
            name = category_names[category]
            acc = metrics['accuracy'] * 100
            uar = metrics.get('uar', 0) * 100
            auc = metrics.get('auc', 0)
            print(f"{category} ({name}):  {acc:>8.2f}%  {uar:>10.2f}%  {auc:>10.4f}")
        
        print(f"{'='*80}\n")
        
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--root-dir', type=str, default='/data/shared/Datasets/FakeAVCeleb_v1.2')
    parser.add_argument('--metadata-csv', type=str, default='/data/shared/Datasets/FakeAVCeleb_v1.2/meta_data.csv')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--mode', type=str, choices=['overall', 'by_method', 'by_category', 'by_modality', 'all'], default='overall')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    
    args = parser.parse_args()
    
    evaluator = FakeAVCelebEvaluator(
        config_path=args.config,
        checkpoint_path=args.checkpoint
    )
    
    if args.mode == 'overall' or args.mode == 'all':
        evaluator.evaluate_overall(
            root_dir=args.root_dir,
            metadata_csv=args.metadata_csv,
            split=args.split,
            max_samples=args.max_samples,
            batch_size=args.batch_size
        )
    
    if args.mode == 'by_method' or args.mode == 'all':
        evaluator.evaluate_by_method(
            root_dir=args.root_dir,
            metadata_csv=args.metadata_csv,
            split=args.split,
            max_samples_per_method=args.max_samples,
            batch_size=args.batch_size
        )
    
    if args.mode == 'by_category' or args.mode == 'all':
        evaluator.evaluate_by_category(
            root_dir=args.root_dir,
            metadata_csv=args.metadata_csv,
            split=args.split,
            max_samples_per_category=args.max_samples,
            batch_size=args.batch_size
        )
    
    if args.mode == 'by_modality' or args.mode == 'all':
        evaluator.evaluate_by_modality(
            root_dir=args.root_dir,
            metadata_csv=args.metadata_csv,
            split=args.split,
            max_samples=args.max_samples,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()