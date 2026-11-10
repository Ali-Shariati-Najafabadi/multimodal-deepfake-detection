import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
from pathlib import Path
import kornia.augmentation as K
import numpy as np
from tqdm import tqdm
import sys
import time
import argparse
import random

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import get_model
from data.dataloaders.multi_dataset_loader import create_dataloaders, collate_fn
from training.training_utils import MetricsCalculator, CheckpointManager, AverageMeter
from torch.utils.data import DataLoader
from training.logger import TrainingLogger

class Phase1Trainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        seed = self.config['experiment'].get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"   Seed set to: {seed}")
        
        self.device = torch.device(self.config['training']['device'])
        self.use_amp = self.config['training'].get('use_amp', False)

        sys_config = self.config.get('system', {})
        torch.backends.cudnn.benchmark    = sys_config.get('cudnn_benchmark', True)
        torch.backends.cudnn.deterministic = sys_config.get('cudnn_deterministic', False)
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        if self.use_amp:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.staged_config = self.config['training'].get('staged_training', {})
        self.current_stage = None
        self.base_video_lr = self.config['training']['optimizer']['video']['lr']
        self.base_audio_lr = self.config['training']['optimizer']['audio']['lr']
        self.base_fusion_lr = self.config['training']['optimizer']['fusion']['lr']
        
        aux_config = self.config['training']['loss'].get('auxiliary', {})
        self.original_aux_alpha = aux_config.get('alpha', 0.6)
        self.original_aux_beta = aux_config.get('beta', 0.2)
        self.original_aux_gamma = aux_config.get('gamma', 0.2)
        
        self.setup_dataloaders()
        self.setup_model()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_loss()
        self.setup_augmentation()
        
        num_classes = self.config['models'].get('num_classes', 2)
        self.train_metrics = MetricsCalculator(num_classes=num_classes)
        self.val_metrics = MetricsCalculator(num_classes=num_classes)
        
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = self.config['experiment']['name']
        batch_size = self.config['training']['batch_size']
        video_lr = self.config['training']['optimizer']['video']['lr']
        audio_lr = self.config['training']['optimizer']['audio']['lr']
        
        base_log_dir = Path(self.config['logging']['tensorboard_dir'])
        run_name = f"{exp_name}_bs{batch_size}_vlr{video_lr}_alr{audio_lr}_{timestamp}"
        log_dir = base_log_dir / run_name
        
        print(f"\n TensorBoard logs: {log_dir}")
        self.writer = SummaryWriter(log_dir=log_dir)
        
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.config['checkpointing']['save_dir'],
            keep_last_n=self.config['checkpointing']['keep_last_n'],
            save_best_metric=self.config['evaluation']['save_best_metric'],
            save_best_mode=self.config['evaluation']['save_best_mode']
        )


    def setup_dataloaders(self):
        from data.dataloaders.multi_dataset_loader import create_dataloaders
        max_samples = self.config['experiment'].get('debug_samples') \
                    if self.config['experiment'].get('debug_mode') else None

        self.train_loader, self.val_loader = create_dataloaders(
            datasets_config=self.config['datasets'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers'],
            max_samples=max_samples,
            data_config=self.config.get('data', {}),
        )

        print(f"\nDataset Info:")
        print(f" Train: {len(self.train_loader.dataset)} samples")
        print(f" Val: {len(self.val_loader.dataset)} samples")

    
    def setup_model(self):
        phase = self.config['experiment'].get('phase', 'phase1')
        
        print(f"\n{'='*60}")
        print(f"Setting up model for phase: {phase}")
        print(f"{'='*60}")
        
        self.model = get_model(phase, self.config).to(self.device)
        
        model_info = self.model.get_model_info()
        
        print(f"\n MODEL ARCHITECTURE:")
        print(f"   Phase: {model_info['phase']}")
        
        if 'fusion_type' in model_info:
            print(f"   Fusion: {model_info['fusion_type']}")
        
        if 'video_architecture' in model_info:
            print(f"   Video Architecture: {model_info['video_architecture']}")
        
        if 'audio_architecture' in model_info:
            print(f"   Audio Architecture: {model_info['audio_architecture']}")
        
        print(f"   Total params: {model_info['total_params']:,}")
        print(f"   Trainable: {model_info['total_trainable']:,}")
        print(f"   Video: {model_info['video_trainable']:,}/{model_info['video_params']:,}")
        print(f"   Audio: {model_info['audio_trainable']:,}/{model_info['audio_params']:,}")
        print(f"   Fusion: {model_info['fusion_trainable']:,}/{model_info['fusion_params']:,}")
        
        if 'head_params' in model_info:
            print(f"   Head: {model_info.get('head_trainable', 0):,}/{model_info['head_params']:,}")
        
        print(f"{'='*60}\n")
        
        self.model.train()

    
    def setup_optimizer(self):
        video_params = list(self.model.video_model.parameters())
        video_opt_config = self.config['training']['optimizer']['video']
        video_opt_type = video_opt_config.get('type', 'sgd').lower()
        
        if video_opt_type == 'sgd':
            self.video_optimizer = optim.SGD(
                video_params,
                lr=video_opt_config['lr'],
                momentum=video_opt_config.get('momentum', 0.9),
                weight_decay=video_opt_config.get('weight_decay', 0.0005)
            )
            print(f"\n   Video Optimizer: SGD lr={video_opt_config['lr']}")
        elif video_opt_type == 'adamw':
            self.video_optimizer = optim.AdamW(
                video_params,
                lr=video_opt_config['lr'],
                weight_decay=video_opt_config.get('weight_decay', 0.05),
                betas=video_opt_config.get('betas', [0.9, 0.999])
            )
            print(f"\n   Video Optimizer: AdamW lr={video_opt_config['lr']}")
        else:
            raise ValueError(f"Unknown video optimizer type: {video_opt_type}")
        
        audio_params = list(self.model.audio_model.parameters())
        self.audio_optimizer = optim.AdamW(
            audio_params,
            lr=self.config['training']['optimizer']['audio']['lr'],
            weight_decay=self.config['training']['optimizer']['audio']['weight_decay'],
            betas=self.config['training']['optimizer']['audio']['betas']
        )
        print(f"   Audio Optimizer: AdamW lr={self.config['training']['optimizer']['audio']['lr']}")
        
        fusion_params = []
        if hasattr(self.model, 'fusion'):
            fusion_params.extend(list(self.model.fusion.parameters()))
        if hasattr(self.model, 'head'):
            fusion_params.extend(list(self.model.head.parameters()))
        if hasattr(self.model, 'video_aux_head'):
            fusion_params.extend(list(self.model.video_aux_head.parameters()))
        if hasattr(self.model, 'audio_aux_head'):
            fusion_params.extend(list(self.model.audio_aux_head.parameters()))

        if len(fusion_params) == 0:
            fusion_params = [torch.nn.Parameter(torch.zeros(1, requires_grad=True))]
            
        self.fusion_optimizer = optim.AdamW(
            fusion_params,
            lr=self.config['training']['optimizer']['fusion']['lr'],
            weight_decay=self.config['training']['optimizer']['fusion']['weight_decay'],
            betas=self.config['training']['optimizer']['fusion']['betas']
        )
        
        fusion_param_count = sum(p.numel() for p in self.model.fusion.parameters())
        head_param_count = sum(p.numel() for p in self.model.head.parameters()) if hasattr(self.model, 'head') else 0
        total_fusion_params = sum(p.numel() for p in fusion_params)
        
        print(f"   Fusion+Head Optimizer: AdamW lr={self.config['training']['optimizer']['fusion']['lr']}")
        print(f"     Fusion params: {fusion_param_count:,}")
        print(f"     Head params: {head_param_count:,}")
        print(f"     Total optimized: {total_fusion_params:,}")
        
        self.optimizers = {
            'video': self.video_optimizer,
            'audio': self.audio_optimizer,
            'fusion': self.fusion_optimizer
        }
    
    def setup_scheduler(self):
        scheduler_config = self.config['training']['scheduler']
        
        if 'video' in scheduler_config and 'audio' in scheduler_config:
            
            video_config = scheduler_config['video']
            if video_config['type'] == 'cosine_warmup':
                from transformers import get_cosine_schedule_with_warmup
                
                self.video_scheduler = get_cosine_schedule_with_warmup(
                    self.video_optimizer,
                    num_warmup_steps=video_config['num_warmup_steps'],
                    num_training_steps=video_config['num_training_steps']
                )
                print(f"   Video Scheduler: CosineWarmup (warmup={video_config['num_warmup_steps']}, total={video_config['num_training_steps']})")
            elif video_config['type'] == 'step':
                self.video_scheduler = optim.lr_scheduler.StepLR(
                    self.video_optimizer,
                    step_size=video_config['step_size'],
                    gamma=video_config['gamma']
                )
                print(f"   Video Scheduler: StepLR (step_size={video_config['step_size']}, gamma={video_config['gamma']})")
            
            audio_config = scheduler_config['audio']
            if audio_config['type'] == 'cosine_warmup':
                from transformers import get_cosine_schedule_with_warmup
                
                self.audio_scheduler = get_cosine_schedule_with_warmup(
                    self.audio_optimizer,
                    num_warmup_steps=audio_config['num_warmup_steps'],
                    num_training_steps=audio_config['num_training_steps']
                )
                print(f"   Audio Scheduler: CosineWarmup (warmup={audio_config['num_warmup_steps']}, total={audio_config['num_training_steps']})")
            elif audio_config['type'] == 'step':
                self.audio_scheduler = optim.lr_scheduler.StepLR(
                    self.audio_optimizer,
                    step_size=audio_config['step_size'],
                    gamma=audio_config['gamma']
                )
                print(f"   Audio Scheduler: StepLR (step_size={audio_config['step_size']}, gamma={audio_config['gamma']})")
            
            fusion_config = scheduler_config['fusion']
            if fusion_config['type'] == 'cosine_warmup':
                from transformers import get_cosine_schedule_with_warmup
                
                self.fusion_scheduler = get_cosine_schedule_with_warmup(
                    self.fusion_optimizer,
                    num_warmup_steps=fusion_config['num_warmup_steps'],
                    num_training_steps=fusion_config['num_training_steps']
                )
                print(f"   Fusion Scheduler: CosineWarmup (warmup={fusion_config['num_warmup_steps']}, total={fusion_config['num_training_steps']})")
            elif fusion_config['type'] == 'step':
                self.fusion_scheduler = optim.lr_scheduler.StepLR(
                    self.fusion_optimizer,
                    step_size=fusion_config['step_size'],
                    gamma=fusion_config['gamma']
                )
                print(f"   Fusion Scheduler: StepLR (step_size={fusion_config['step_size']}, gamma={fusion_config['gamma']})")
        
        else:
            scheduler_type = scheduler_config['type']
            
            if scheduler_type == 'step':
                step_size = scheduler_config['step_size']
                gamma = scheduler_config['gamma']
                
                self.video_scheduler = optim.lr_scheduler.StepLR(
                    self.video_optimizer,
                    step_size=step_size,
                    gamma=gamma
                )
                self.audio_scheduler = optim.lr_scheduler.StepLR(
                    self.audio_optimizer,
                    step_size=step_size,
                    gamma=gamma
                )
                self.fusion_scheduler = optim.lr_scheduler.StepLR(
                    self.fusion_optimizer,
                    step_size=step_size,
                    gamma=gamma
                )
                print(f"   Scheduler: StepLR (step_size={step_size}, gamma={gamma})")
            
            else:
                raise ValueError(f"Scheduler type {scheduler_type} not supported")
        
        self.schedulers = {
            'video': self.video_scheduler,
            'audio': self.audio_scheduler,
            'fusion': self.fusion_scheduler
        }

    
    def setup_loss(self):
        loss_type = self.config['training']['loss'].get('type', 'cross_entropy')
        class_weights_config = self.config['training']['loss'].get('class_weights', 'auto')
        label_smoothing = self.config['training']['loss'].get('label_smoothing', 0.0)
        
        weights = None
        if class_weights_config == 'auto':
            print("   Calculating automatic class weights:")
            
            train_labels = []
            dataset = self.train_loader.dataset
            
            if hasattr(dataset, 'datasets'):
                for sub_dataset in dataset.datasets:
                    if hasattr(sub_dataset, 'samples'):
                        train_labels.extend([s['label'] for s in sub_dataset.samples])
                    elif hasattr(sub_dataset, 'metadata'):
                        train_labels.extend([sub_dataset.metadata[vid_id]['label'] 
                                        for vid_id in sub_dataset.video_ids])
            elif hasattr(dataset, 'samples'):
                train_labels = [s['label'] for s in dataset.samples]
            elif hasattr(dataset, 'metadata'):
                train_labels = [dataset.metadata[vid_id]['label'] 
                            for vid_id in dataset.video_ids]
            else:
                print("   Warning: Using slow label extraction (iterating dataset)")
                for i in range(len(dataset)):
                    train_labels.append(dataset[i]['label'])

            from collections import Counter
            from sklearn.utils.class_weight import compute_class_weight
            
            label_counts = Counter(train_labels)
            total = sum(label_counts.values())
            unique_classes = sorted(label_counts.keys())
            num_classes = len(unique_classes)
            
            print(f"\n   Found {num_classes} unique classes: {unique_classes}")
            
            for cls in unique_classes:
                count = label_counts[cls]
                percentage = 100.0 * count / total
                print(f"   Class {cls}: {count:,} samples ({percentage:.2f}%)")
            
            class_weights_array = compute_class_weight(
                'balanced',
                classes=np.array(unique_classes),
                y=np.array(train_labels)
            )
            
            weights = torch.tensor(class_weights_array, dtype=torch.float32).to(self.device)
            
            print(f"\n   Computed class weights:")
            for i, (cls, weight) in enumerate(zip(unique_classes, class_weights_array)):
                print(f"     Class {cls}: {weight:.4f}")

        elif isinstance(class_weights_config, list):
            weights = torch.tensor(class_weights_config, dtype=torch.float32).to(self.device)
            print(f"\n   Using manual class weights: {class_weights_config}")
        
        if loss_type == 'focal':
            focal_config = self.config['training']['loss'].get('focal', {})
            gamma = focal_config.get('gamma', 2.0)
            alpha = focal_config.get('alpha', 0.25)
            
            from training.losses.focal_loss import FocalLoss
            self.criterion = FocalLoss(
                gamma=gamma,
                alpha=alpha,
                weight=weights,
                label_smoothing=label_smoothing
            )
            
            print(f"\n   Loss: FocalLoss")
            print(f"   Gamma: {gamma}, Alpha: {alpha}")
            print(f"   Label Smoothing: {label_smoothing}")
            
        elif loss_type == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(
                weight=weights,
                label_smoothing=label_smoothing
            )
            
            print(f"\n   Loss: CrossEntropyLoss")
            print(f"   Label Smoothing: {label_smoothing}")
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        self.aux_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"   Aux Loss: CrossEntropyLoss (binär, ungewichtet)")

    
    def setup_augmentation(self):
        aug_config = self.config.get('data', {}).get('augmentation', {})
        
        if aug_config.get('enabled', False):
            aug_list = []
            
            if 'random_horizontal_flip' in aug_config:
                aug_list.append(
                    K.RandomHorizontalFlip(p=aug_config['random_horizontal_flip'])
                )
            
            if 'color_jitter' in aug_config:
                cj = aug_config['color_jitter']
                aug_list.append(
                    K.ColorJitter(
                        brightness=cj.get('brightness', 0.2),
                        contrast=cj.get('contrast', 0.2),
                        saturation=cj.get('saturation', 0.1),
                        hue=cj.get('hue', 0.1),
                        p=cj.get('p', 0.8)
                    )
                )
            
            if 'random_rotation' in aug_config:
                aug_list.append(
                    K.RandomRotation(
                        degrees=aug_config['random_rotation'],
                        p=aug_config.get('random_rotation_p', 0.5)
                    )
                )
            
            if 'random_grayscale' in aug_config:
                aug_list.append(
                    K.RandomGrayscale(p=aug_config['random_grayscale'])
                )
            
            if 'gaussian_blur' in aug_config:
                aug_list.append(
                    K.RandomGaussianBlur(
                        kernel_size=(3, 3),
                        sigma=(0.1, 2.0),
                        p=aug_config['gaussian_blur']
                    )
                )
            
            if 'jpeg_compression' in aug_config:
                jc = aug_config['jpeg_compression']
                if jc.get('enabled', False):
                    aug_list.append(
                        K.RandomJPEG(
                            jpeg_quality=(
                                jc.get('quality_lower', 60),
                                jc.get('quality_upper', 100)
                            ),
                            p=jc.get('p', 0.5)
                        )
                    )
            
            if 'random_crop' in aug_config:
                rc = aug_config['random_crop']
                if rc.get('enabled', False):
                    scale = rc.get('scale', [0.85, 1.0])
                    aug_list.append(
                        K.RandomResizedCrop(
                            size=(224, 224),
                            scale=tuple(scale),
                            ratio=(0.95, 1.05),
                            p=rc.get('p', 0.3)
                        )
                    )
            
            if aug_list:
                self.video_augmentation = nn.Sequential(*aug_list).to(self.device)
                print(f"   Video Augmentation: {len(aug_list)} transforms enabled")
            else:
                self.video_augmentation = None
        else:
            self.video_augmentation = None
            print(f"   Video Augmentation: Disabled")

    
    def _update_training_stage(self, epoch: int):
        if not self.staged_config.get('enabled', False):
            self.current_stage = None
            return
        
        stages = self.staged_config.get('stages', [])
        new_stage = None
        
        for stage in stages:
            start_epoch, end_epoch = stage['epochs']
            if start_epoch <= epoch <= end_epoch:
                new_stage = stage
                break
        
        if new_stage != self.current_stage:
            self.current_stage = new_stage
            
            if new_stage is not None:
                print(f"\n{'='*60}")
                print(f"STAGE CHANGE: {new_stage['name'].upper()}")
                print(f"  Epochs: {new_stage['epochs'][0]}-{new_stage['epochs'][1]}")
                print(f"  Video enabled: {new_stage.get('video_enabled', True)}")
                print(f"  Audio enabled: {new_stage.get('audio_enabled', True)}")
                print(f"  Fusion weight (alpha): {new_stage.get('fusion_weight', 0.6)}")
                
                video_mult = new_stage.get('video_lr_multiplier', 1.0)
                audio_mult = new_stage.get('audio_lr_multiplier', 1.0)
                
                new_video_lr = self.base_video_lr * video_mult
                new_audio_lr = self.base_audio_lr * audio_mult
                
                for param_group in self.video_optimizer.param_groups:
                    param_group['lr'] = new_video_lr
                
                for param_group in self.audio_optimizer.param_groups:
                    param_group['lr'] = new_audio_lr
                
                print(f"  Video LR: {new_video_lr:.6f} (base={self.base_video_lr:.6f}, mult={video_mult})")
                print(f"  Audio LR: {new_audio_lr:.6f} (base={self.base_audio_lr:.6f}, mult={audio_mult})")
                
                aux_config = self.config['training']['loss'].get('auxiliary', {})
                
                if 'loss_weights' in new_stage:
                    aux_config['alpha'] = new_stage['loss_weights']['alpha']
                    aux_config['beta'] = new_stage['loss_weights']['beta']
                    aux_config['gamma'] = new_stage['loss_weights']['gamma']
                    print(f"  Loss weights: alpha={aux_config['alpha']}, beta={aux_config['beta']}, gamma={aux_config['gamma']}")
                elif 'fusion_weight' in new_stage:
                    aux_config['alpha'] = new_stage['fusion_weight']
                
                print(f"{'='*60}\n")
            else:
                aux_config = self.config['training']['loss'].get('auxiliary', {})
                aux_config['alpha'] = self.original_aux_alpha
                aux_config['beta'] = self.original_aux_beta
                aux_config['gamma'] = self.original_aux_gamma

    
    def _get_modality_flags(self):
        modality_config = self.config['training'].get('modality', {})
        video_enabled = modality_config.get('video_enabled', True)
        audio_enabled = modality_config.get('audio_enabled', True)
        use_fusion = modality_config.get('use_fusion', True)
        
        if self.current_stage is not None:
            video_enabled = self.current_stage.get('video_enabled', video_enabled)
            audio_enabled = self.current_stage.get('audio_enabled', audio_enabled)
        
        return video_enabled, audio_enabled, use_fusion
    
    
    def train_epoch(self, epoch: int):
        self._update_training_stage(epoch)
        self.model.train()
        self.train_metrics.reset()
        
        loss_meter = AverageMeter()
        fusion_loss_meter = AverageMeter()
        video_loss_meter = AverageMeter()
        audio_loss_meter = AverageMeter()
        
        aux_config = self.config['training']['loss'].get('auxiliary', {})
        use_auxiliary = aux_config.get('enabled', False)
        num_classes = self.config['models'].get('num_classes', 2)
        
        gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        gradient_clip = self.config['training']['gradient_clip']
        
        if hasattr(sys.stdout, 'terminal'):
            terminal = sys.stdout.terminal
        else:
            terminal = sys.stdout
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", 
                    file=terminal, ncols=180)
        
        for batch_idx, batch in enumerate(pbar):
            video_features = batch['video_features'].to(self.device)
            audio_features = batch['audio_features'].to(self.device)
            video_masks = batch['video_masks'].to(self.device)
            audio_masks = batch['audio_masks'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            #   0=real, 1=AudioMod, 2=VisualMod, 3=BothMod
            if num_classes == 4:
                video_labels = (labels >= 2).long()
                audio_labels = ((labels == 1) | (labels == 3)).long()
            else:
                video_labels = audio_labels = labels
            
            if self.video_augmentation is not None:
                B, C, T, H, W = video_features.shape
                video_flat = video_features.permute(0, 2, 1, 3, 4).contiguous()
                video_flat = video_flat.view(B * T, C, H, W)
                video_flat = self.video_augmentation(video_flat)
                video_features = video_flat.view(B, T, C, H, W)
                video_features = video_features.permute(0, 2, 1, 3, 4).contiguous()
            
            video_enabled, audio_enabled, use_fusion = self._get_modality_flags()
            
            if not video_enabled:
                video_features = torch.zeros_like(video_features)
                video_masks = torch.zeros_like(video_masks)
            
            if not audio_enabled:
                audio_features = torch.zeros_like(audio_features)
                audio_masks = torch.zeros_like(audio_masks)
            
            modality_dropout_config = self.config['training'].get('modality_dropout', {})
            if modality_dropout_config.get('enabled', False) and video_enabled and audio_enabled:
                video_p = modality_dropout_config.get('video_p', 0.0)
                audio_p = modality_dropout_config.get('audio_p', 0.0)
                
                if random.random() < video_p:
                    video_features = torch.zeros_like(video_features)
                    video_masks = torch.zeros_like(video_masks)
                
                if random.random() < audio_p:
                    audio_features = torch.zeros_like(audio_features)
                    audio_masks = torch.zeros_like(audio_masks)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                if not use_fusion and video_enabled:
                    video_logits = self.model.video_model(video_features, video_masks)
                    fusion_loss = self.criterion(video_logits, labels)
                    loss = fusion_loss
                    fusion_logits = video_logits
                    audio_logits = torch.zeros_like(video_logits)
                    video_loss = torch.tensor(0.0, device=self.device)
                    audio_loss = torch.tensor(0.0, device=self.device)
                
                elif use_auxiliary:
                    fusion_logits, video_logits, audio_logits = self.model(
                        video_features, audio_features, 
                        video_masks, audio_masks, 
                        return_auxiliary=True
                    )
                    
                    fusion_loss = self.criterion(fusion_logits, labels)
                    video_loss = self.aux_criterion(video_logits, video_labels)
                    audio_loss = self.aux_criterion(audio_logits, audio_labels)
                    
                    alpha = aux_config['alpha']
                    beta = aux_config['beta']
                    gamma = aux_config['gamma']
                    loss = alpha * fusion_loss + beta * video_loss + gamma * audio_loss
                
                else:
                    fusion_logits = self.model(video_features, audio_features, video_masks, audio_masks)
                    fusion_loss = self.criterion(fusion_logits, labels)
                    loss = fusion_loss
                    video_logits = torch.zeros_like(fusion_logits)
                    audio_logits = torch.zeros_like(fusion_logits)
                    video_loss = torch.tensor(0.0, device=self.device)
                    audio_loss = torch.tensor(0.0, device=self.device)
            
            if batch_idx == 0 and epoch <= 3:
                self._log_debug_info(
                    epoch, video_features, audio_features, video_masks, audio_masks,
                    fusion_logits, video_logits, audio_logits,
                    fusion_loss, video_loss, audio_loss, loss,
                    use_auxiliary, aux_config, gradient_accumulation_steps
                )
            
            scaled_loss = loss / gradient_accumulation_steps
            self.scaler.scale(scaled_loss).backward()
            
            is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == len(self.train_loader)
            
            if is_accumulation_step or is_last_batch:
                self.scaler.unscale_(self.video_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.video_model.parameters(), gradient_clip
                )
                
                if not (not use_fusion and video_enabled):
                    self.scaler.unscale_(self.audio_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.audio_model.parameters(), gradient_clip
                    )
                
                self.scaler.unscale_(self.fusion_optimizer)
                fusion_clip_params = []
                if hasattr(self.model, 'fusion'):
                    fusion_clip_params.extend(list(self.model.fusion.parameters()))
                if hasattr(self.model, 'head'):
                    fusion_clip_params.extend(list(self.model.head.parameters()))
                if hasattr(self.model, 'video_aux_head'):
                    fusion_clip_params.extend(list(self.model.video_aux_head.parameters()))
                if hasattr(self.model, 'audio_aux_head'):
                    fusion_clip_params.extend(list(self.model.audio_aux_head.parameters()))
                if fusion_clip_params:
                    torch.nn.utils.clip_grad_norm_(fusion_clip_params, gradient_clip)
                
                if batch_idx == 0 and epoch <= 3:
                    self._log_gradient_norms()
                
                self.scaler.step(self.video_optimizer)
                if not (not use_fusion and video_enabled):
                    self.scaler.step(self.audio_optimizer)
                self.scaler.step(self.fusion_optimizer)
                self.scaler.update()
                
                self.video_optimizer.zero_grad(set_to_none=True)
                self.audio_optimizer.zero_grad(set_to_none=True)
                self.fusion_optimizer.zero_grad(set_to_none=True)
                
                self.step_schedulers(per_batch=True)
            
            with torch.no_grad():
                loss_meter.update(loss.item(), labels.size(0))
                fusion_loss_meter.update(fusion_loss.item(), labels.size(0))
                if use_auxiliary:
                    video_loss_meter.update(video_loss.item(), labels.size(0))
                    audio_loss_meter.update(audio_loss.item(), labels.size(0))
            
            with torch.no_grad():
                preds = torch.argmax(fusion_logits, dim=1)
                probs = torch.softmax(fusion_logits, dim=1)
                self.train_metrics.update(
                    labels.cpu(),
                    preds.cpu(),
                    probs.cpu()
                )
            
            if batch_idx % self.config['system'].get('empty_cache_every_n_steps', 100) == 0:
                torch.cuda.empty_cache()
            
            if use_auxiliary:
                pbar.set_postfix(
                    loss=f"{loss_meter.avg:.4f}",
                    f=f"{fusion_loss_meter.avg:.4f}",
                    v=f"{video_loss_meter.avg:.4f}",
                    a=f"{audio_loss_meter.avg:.4f}"
                )
            else:
                pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")
        
        if use_auxiliary:
            print(f"\nEpoch {epoch} Training Summary:")
            print("  Average Losses:")
            print(f"    Total: {loss_meter.avg:.4f}")
            print(f"    Fusion: {fusion_loss_meter.avg:.4f} (alpha={aux_config['alpha']})")
            print(f"    Video:  {video_loss_meter.avg:.4f} (beta={aux_config['beta']})")
            print(f"    Audio:  {audio_loss_meter.avg:.4f} (gamma={aux_config['gamma']})")
        else:
            print(f"\nEpoch {epoch} - Loss: {loss_meter.avg:.4f}")
        
        self.step_schedulers(per_batch=False)
        
        print("\nLearning Rates:")
        print(f"  Video:  {self.video_optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Audio:  {self.audio_optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Fusion: {self.fusion_optimizer.param_groups[0]['lr']:.6f}")
        
        metrics = self.train_metrics.compute()
        metrics['loss'] = loss_meter.avg
        if use_auxiliary:
            metrics['fusion_loss'] = fusion_loss_meter.avg
            metrics['video_loss'] = video_loss_meter.avg
            metrics['audio_loss'] = audio_loss_meter.avg
        
        return metrics


    def _log_debug_info(self, epoch, video_features, audio_features, video_masks, audio_masks,
                        fusion_logits, video_logits, audio_logits,
                        fusion_loss, video_loss, audio_loss, total_loss,
                        use_auxiliary, aux_config, gradient_accumulation_steps):
        print("=" * 60)
        print(f"DEBUG Epoch {epoch}, Batch 0:")
        print("\n  INPUT SHAPES:")
        print(f"    Video features: {video_features.shape}")
        
        if video_features.dim() == 5:
            B, C, T, H, W = video_features.shape
            print(f"       Batch={B}, Channels={C}, Frames={T}, Height={H}, Width={W}")
        elif video_features.dim() == 3:
            B, T, D = video_features.shape
            print(f"       Batch={B}, Frames={T}, Features={D}")
        
        print(f"    Audio features: {audio_features.shape}")
        if audio_features.dim() == 3:
            B, num_segments, segment_length = audio_features.shape
            print(f"       Batch={B}, Segments={num_segments}, Samples/segment={segment_length}")
            print(f"       Segment duration: {segment_length/16000:.2f}s @ 16kHz")
        elif audio_features.dim() == 2:
            B, total_samples = audio_features.shape
            print(f"       Batch={B}, Total samples={total_samples}")
        
        print(f"    Video masks: {video_masks.shape}")
        print(f"    Audio masks: {audio_masks.shape}")
        
        print("\n  LOGITS STATISTICS:")
        if use_auxiliary:
            print(f"    Video logits: shape={video_logits.shape}, range=[{video_logits.min():.2f}, {video_logits.max():.2f}], std={video_logits.std():.4f}")
            print(f"    Audio logits: shape={audio_logits.shape}, range=[{audio_logits.min():.2f}, {audio_logits.max():.2f}], std={audio_logits.std():.4f}")
        print(f"    Fusion logits: shape={fusion_logits.shape}, range=[{fusion_logits.min():.2f}, {fusion_logits.max():.2f}], std={fusion_logits.std():.4f}")
        
        print("\n  LOSSES:")
        print(f"    Fusion loss: {fusion_loss.item():.4f}")
        if use_auxiliary:
            print(f"    Video loss:  {video_loss.item():.4f}")
            print(f"    Audio loss:  {audio_loss.item():.4f}")
            print(f"    Total loss:  {total_loss.item():.4f}")
            print(f"    Weights: alpha={aux_config['alpha']}, beta={aux_config['beta']}, gamma={aux_config['gamma']}")
        
        print("\n  FEATURE DIMENSIONS:")
        if hasattr(self.model, 'video_feature_dim'):
            print(f"    Video features: {self.model.video_feature_dim}")
        if hasattr(self.model, 'audio_feature_dim'):
            print(f"    Audio features: {self.model.audio_feature_dim}")
        if hasattr(self.model, 'fusion') and hasattr(self.model.fusion, 'output_dim'):
            print(f"    Fusion features: {self.model.fusion.output_dim}")
        
        print(f"\n  GRADIENT ACCUMULATION STEPS: {gradient_accumulation_steps}")
        print("=" * 60)
    

    def _log_gradient_norms(self):
        video_grad_norm = 0.0
        audio_grad_norm = 0.0
        fusion_grad_norm = 0.0
        
        for p in self.model.video_model.parameters():
            if p.grad is not None:
                video_grad_norm += p.grad.norm().item() ** 2
        video_grad_norm = video_grad_norm ** 0.5
        
        for p in self.model.audio_model.parameters():
            if p.grad is not None:
                audio_grad_norm += p.grad.norm().item() ** 2
        audio_grad_norm = audio_grad_norm ** 0.5
        
        if hasattr(self.model, 'head'):
            for p in self.model.head.parameters():
                if p.grad is not None:
                    fusion_grad_norm += p.grad.norm().item() ** 2
        if hasattr(self.model, 'fusion'):
            for p in self.model.fusion.parameters():
                if p.grad is not None:
                    fusion_grad_norm += p.grad.norm().item() ** 2
        fusion_grad_norm = fusion_grad_norm ** 0.5
        
        print(f"\n  Gradients (post-clip): video={video_grad_norm:.6f}, audio={audio_grad_norm:.6f}, fusion={fusion_grad_norm:.6f}")

    
    def step_schedulers(self, per_batch: bool):
        scheduler_config = self.config['training']['scheduler']
        
        if 'video' in scheduler_config and 'audio' in scheduler_config:
            if per_batch:
                if scheduler_config['video'].get('type') == 'cosine_warmup':
                    self.video_scheduler.step()
                if scheduler_config['audio'].get('type') == 'cosine_warmup':
                    self.audio_scheduler.step()
                if scheduler_config['fusion'].get('type') == 'cosine_warmup':
                    self.fusion_scheduler.step()
            else:
                if scheduler_config['video'].get('type') == 'step':
                    self.video_scheduler.step()
                if scheduler_config['audio'].get('type') == 'step':
                    self.audio_scheduler.step()
                if scheduler_config['fusion'].get('type') == 'step':
                    self.fusion_scheduler.step()
        else:
            if not per_batch:
                self.video_scheduler.step()
                self.audio_scheduler.step()
                self.fusion_scheduler.step()

    
    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        self.val_metrics.reset()
        
        val_loss = AverageMeter()
        
        pbar = tqdm(
            self.val_loader, 
            desc=f"Epoch {epoch} [Val]",
            file=sys.__stdout__,
            ncols=180
        )
        
        video_enabled, audio_enabled, use_fusion = self._get_modality_flags()
        
        for batch_idx, batch in enumerate(pbar):
            video_features = batch['video_features'].to(self.device)
            audio_features = batch['audio_features'].to(self.device)
            video_masks = batch['video_masks'].to(self.device)
            audio_masks = batch['audio_masks'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                if not use_fusion and video_enabled:
                    logits = self.model.video_model(video_features, video_masks)
                else:
                    logits = self.model(video_features, audio_features, video_masks, audio_masks)
                
                loss = self.criterion(logits, labels)
            
            val_loss.update(loss.item(), labels.size(0))
            
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            self.val_metrics.update(labels, preds, probs)
            
            pbar.set_postfix({'loss': f'{val_loss.avg:.4f}'})
        
        num_classes = self.config['models'].get('num_classes', 2)
        
        print(f"\nEpoch {epoch} Validation Summary:")
        print(f"  Fusion Logits (last batch):")
        print(f"    Range: [{logits.min():.3f}, {logits.max():.3f}]")
        print(f"    Mean: {logits.mean():.3f}, Std: {logits.std():.3f}")
        
        if num_classes == 2:
            print(f"    Class 0: {logits[:, 0].mean():.3f}, Class 1: {logits[:, 1].mean():.3f}")
        else:
            logit_means = [logits[:, i].mean().item() for i in range(num_classes)]
            logit_str = ", ".join([f"Class {i}: {mean:.3f}" for i, mean in enumerate(logit_means)])
            print(f"    {logit_str}")
        
        print(f"  Average Loss: {val_loss.avg:.4f}")
        
        metrics = self.val_metrics.compute()
        metrics['loss'] = val_loss.avg
        
        return metrics

    
    
    def train(self, num_epochs: int = None):
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        print(f"\nStarting Training")
        print(f"   Epochs: {num_epochs}")
        print(f"   Device: {self.device}")
        print(f"   AMP: {self.use_amp}\n")
        
        try:
            for epoch in range(1, num_epochs + 1):
                print(f"\n{'='*60}")
                print(f"Epoch {epoch}/{num_epochs}")
                print(f"{'='*60}")
                
                train_metrics = self.train_epoch(epoch)
                
                print(f"\nTrain:")
                self.train_metrics.print_metrics(train_metrics, prefix="  ")
                
                val_metrics = self.validate(epoch)
                
                print(f"\nValidation:")
                self.val_metrics.print_metrics(val_metrics, prefix="  ")
                
                num_classes = self.config['models'].get('num_classes', 2)
                
                self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                
                if 'fusion_loss' in train_metrics:
                    self.writer.add_scalar('Loss/train_fusion', train_metrics['fusion_loss'], epoch)
                if 'video_loss' in train_metrics:
                    self.writer.add_scalar('Loss/train_video', train_metrics['video_loss'], epoch)
                if 'audio_loss' in train_metrics:
                    self.writer.add_scalar('Loss/train_audio', train_metrics['audio_loss'], epoch)
                
                self.writer.add_scalar('Metrics/Accuracy/train', train_metrics['accuracy'], epoch)
                self.writer.add_scalar('Metrics/Accuracy/val', val_metrics['accuracy'], epoch)
                
                if num_classes == 2 and 'auc' in train_metrics:
                    self.writer.add_scalar('Metrics/AUC/train', train_metrics['auc'], epoch)
                    self.writer.add_scalar('Metrics/AUC/val', val_metrics['auc'], epoch)
                
                self.writer.add_scalar('Metrics/UAR/train', train_metrics['uar'], epoch)
                self.writer.add_scalar('Metrics/UAR/val', val_metrics['uar'], epoch)
                
                self.writer.add_scalar('Metrics/Precision/train', train_metrics['precision'], epoch)
                self.writer.add_scalar('Metrics/Precision/val', val_metrics['precision'], epoch)
                
                self.writer.add_scalar('Metrics/Recall/train', train_metrics['recall'], epoch)
                self.writer.add_scalar('Metrics/Recall/val', val_metrics['recall'], epoch)
                
                self.writer.add_scalar('Metrics/F1/train', train_metrics['f1'], epoch)
                self.writer.add_scalar('Metrics/F1/val', val_metrics['f1'], epoch)
                
                self.writer.add_scalar('Metrics/WA-F1/train', train_metrics['wa_f1'], epoch)
                self.writer.add_scalar('Metrics/WA-F1/val', val_metrics['wa_f1'], epoch)
                
                if 'recall_real' in train_metrics:
                    self.writer.add_scalar('Metrics/Recall/Real/train', train_metrics['recall_real'], epoch)
                    self.writer.add_scalar('Metrics/Recall/Real/val', val_metrics['recall_real'], epoch)
                
                if 'recall_fake' in train_metrics:
                    self.writer.add_scalar('Metrics/Recall/Fake/train', train_metrics['recall_fake'], epoch)
                    self.writer.add_scalar('Metrics/Recall/Fake/val', val_metrics['recall_fake'], epoch)
                
                if 'true_positives' in train_metrics:
                    self.writer.add_scalar('ConfusionMatrix/TP/train', train_metrics['true_positives'], epoch)
                    self.writer.add_scalar('ConfusionMatrix/TP/val', val_metrics['true_positives'], epoch)
                    self.writer.add_scalar('ConfusionMatrix/TN/train', train_metrics['true_negatives'], epoch)
                    self.writer.add_scalar('ConfusionMatrix/TN/val', val_metrics['true_negatives'], epoch)
                    self.writer.add_scalar('ConfusionMatrix/FP/train', train_metrics['false_positives'], epoch)
                    self.writer.add_scalar('ConfusionMatrix/FP/val', val_metrics['false_positives'], epoch)
                    self.writer.add_scalar('ConfusionMatrix/FN/train', train_metrics['false_negatives'], epoch)
                    self.writer.add_scalar('ConfusionMatrix/FN/val', val_metrics['false_negatives'], epoch)
                
                for i in range(num_classes):
                    if f'precision_class_{i}' in train_metrics:
                        self.writer.add_scalar(f'PerClass/Precision/class_{i}/train', train_metrics[f'precision_class_{i}'], epoch)
                        self.writer.add_scalar(f'PerClass/Precision/class_{i}/val', val_metrics[f'precision_class_{i}'], epoch)
                    
                    if f'recall_class_{i}' in train_metrics:
                        self.writer.add_scalar(f'PerClass/Recall/class_{i}/train', train_metrics[f'recall_class_{i}'], epoch)
                        self.writer.add_scalar(f'PerClass/Recall/class_{i}/val', val_metrics[f'recall_class_{i}'], epoch)
                    
                    if f'f1_class_{i}' in train_metrics:
                        self.writer.add_scalar(f'PerClass/F1/class_{i}/train', train_metrics[f'f1_class_{i}'], epoch)
                        self.writer.add_scalar(f'PerClass/F1/class_{i}/val', val_metrics[f'f1_class_{i}'], epoch)
                
                self.writer.add_scalar('LearningRate/video', self.video_optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('LearningRate/audio', self.audio_optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('LearningRate/fusion', self.fusion_optimizer.param_groups[0]['lr'], epoch)
                
                acc_gap = train_metrics['accuracy'] - val_metrics['accuracy']
                self.writer.add_scalar('Gap/Accuracy_gap', acc_gap, epoch)
                
                if num_classes == 2 and 'auc' in train_metrics:
                    auc_gap = train_metrics['auc'] - val_metrics['auc']
                    self.writer.add_scalar('Gap/AUC_gap', auc_gap, epoch)
                
                self.writer.flush()
                
                is_best = self.checkpoint_manager.should_save_best(val_metrics)
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'video_optimizer_state_dict': self.video_optimizer.state_dict(),
                    'audio_optimizer_state_dict': self.audio_optimizer.state_dict(),
                    'fusion_optimizer_state_dict': self.fusion_optimizer.state_dict(),
                    'video_scheduler_state_dict': self.video_scheduler.state_dict(),
                    'audio_scheduler_state_dict': self.audio_scheduler.state_dict(),
                    'fusion_scheduler_state_dict': self.fusion_scheduler.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'metrics': val_metrics,
                    'config': self.config
                }
                
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model=checkpoint,
                    optimizer=None,
                    scheduler=None,
                    metrics=val_metrics,
                    is_best=is_best
                )
                
                if is_best:
                    print(f"\n   New best model! {self.config['evaluation']['save_best_metric']}: {val_metrics[self.config['evaluation']['save_best_metric']]:.4f}\n")
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        
        finally:
            self.writer.close()
            print("\nTraining complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    config_path = args.config
    
    log_dir = Path('logs/training_logs')
    
    with TrainingLogger(log_dir=log_dir, config_path=config_path) as logger:
        trainer = Phase1Trainer(config_path)
        trainer.train()


if __name__ == '__main__':
    main()