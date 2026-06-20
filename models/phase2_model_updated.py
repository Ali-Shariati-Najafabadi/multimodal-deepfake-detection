import torch
import torch.nn as nn
from typing import Optional, Dict
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.video_models.x3d import X3D
from models.audio_models.audio_model import AudioModel
from models.fusion_models.concat_fusion import ConcatFusion
from models.heads.classification_head import ClassificationHead


class Phase2Model(nn.Module):
    def __init__(
        self,
        video_config: Dict,
        audio_config: Dict,
        fusion_config: Dict = None,
        head_config: Dict = None,
        num_classes: int = 2
    ):
        super(Phase2Model, self).__init__()
        
        self.num_classes = num_classes
        self.phase = 'phase2'
        self.use_auxiliary_loss = True
        
        video_arch = video_config.get('architecture', 'x3d_m')
        
        if not video_arch.startswith('x3d'):
            raise ValueError(f"Phase 2 only supports X3D architectures, got: {video_arch}")
        
        self.video_model = X3D(
            architecture=video_arch,
            num_classes=num_classes,
            pretrained=video_config.get('pretrained', True),
            freeze_backbone=video_config.get('freeze_backbone', False),
            unfreeze_layers=video_config.get('unfreeze_layers', None),
            dropout=video_config.get('dropout', 0.5),
            num_frames=video_config.get('num_frames', 16),
            output_type=video_config.get('output_type', 'video')
        )
        
        self.audio_model = AudioModel(
            architecture=audio_config.get('architecture', 'hubert'),
            pretrained_model=audio_config.get('pretrained_model', 'facebook/hubert-base-ls960'),
            num_classes=num_classes,
            freeze_feature_extractor=audio_config.get('freeze_feature_extractor', False),
            unfreeze_last_n_layers=audio_config.get('unfreeze_last_n_layers', 2),
            dropout=audio_config.get('dropout', 0.5),
            pooling=audio_config.get('pooling', 'avg')
        )
        
        self.video_feature_dim = self.video_model.get_feature_dim()
        self.audio_feature_dim = self.audio_model.get_feature_dim()
        
        if fusion_config:
            fusion_type = fusion_config.get('type', 'concat')
        else:
            fusion_type = 'concat'
        
        if fusion_type == 'concat':
            self.fusion = ConcatFusion(self.video_feature_dim, self.audio_feature_dim)
        else:
            raise ValueError(f"Phase 2 currently only supports concat fusion, got: {fusion_type}")
        
        fusion_output_dim = self.fusion.get_output_dim()
        
        if head_config:
            hidden_dims = head_config.get('hidden_dims', [512, 256])
            head_dropout = head_config.get('dropout', 0.5)
        else:
            hidden_dims = [512, 256]
            head_dropout = 0.5
        
        self.head = ClassificationHead(
            input_dim=fusion_output_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=head_dropout
        )
        
        # Binäre Aux-Köpfe pro Modalität (real/fake), unabhängig von num_classes.
        self.video_aux_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.video_feature_dim, 2)
        )
        self.audio_aux_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.audio_feature_dim, 2)
        )
        
        self.video_architecture = video_arch
    
    def extract_video_features(self, video_input: torch.Tensor, video_mask: Optional[torch.Tensor] = None):
        return self.video_model.extract_features(video_input)
    
    def extract_audio_features(self, audio_input: torch.Tensor, audio_mask: Optional[torch.Tensor] = None):
        batch_size, num_segments, segment_length = audio_input.shape
        x = audio_input.view(batch_size * num_segments, segment_length)
        
        outputs = self.audio_model.encoder(x)
        features = outputs.last_hidden_state.mean(dim=1)
        features = features.view(batch_size, num_segments, -1)
        
        if audio_mask is not None:
            mask = audio_mask.unsqueeze(-1)
            pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = features.mean(dim=1)
        
        return pooled
    
    def forward(
        self,
        video_input: torch.Tensor,
        audio_input: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        return_auxiliary: bool = False
    ):
        video_features = self.extract_video_features(video_input, video_mask)
        audio_features = self.extract_audio_features(audio_input, audio_mask)
        
        fused_features = self.fusion(video_features, audio_features)
        fusion_logits = self.head(fused_features)
        
        if return_auxiliary and self.use_auxiliary_loss:
            video_logits = self.video_aux_head(video_features)
            audio_logits = self.audio_aux_head(audio_features)
            return fusion_logits, video_logits, audio_logits
        
        return fusion_logits
    
    def forward_video_only(self, video_input: torch.Tensor, video_mask: Optional[torch.Tensor] = None):
        return self.video_model(video_input)
    
    def forward_audio_only(self, audio_input: torch.Tensor, audio_mask: Optional[torch.Tensor] = None):
        return self.audio_model(audio_input, audio_mask)
    
    def get_model_info(self):
        video_params = sum(p.numel() for p in self.video_model.parameters())
        audio_params = sum(p.numel() for p in self.audio_model.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        head_params = sum(p.numel() for p in self.head.parameters()) \
            + sum(p.numel() for p in self.video_aux_head.parameters()) \
            + sum(p.numel() for p in self.audio_aux_head.parameters())
        
        video_trainable = sum(p.numel() for p in self.video_model.parameters() if p.requires_grad)
        audio_trainable = sum(p.numel() for p in self.audio_model.parameters() if p.requires_grad)
        fusion_trainable = sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)
        head_trainable = sum(p.numel() for p in self.head.parameters() if p.requires_grad) \
            + sum(p.numel() for p in self.video_aux_head.parameters() if p.requires_grad) \
            + sum(p.numel() for p in self.audio_aux_head.parameters() if p.requires_grad)
        
        return {
            'phase': self.phase,
            'video_architecture': self.video_architecture,
            'use_auxiliary_loss': self.use_auxiliary_loss,
            'num_classes': self.num_classes,
            'video_params': video_params,
            'audio_params': audio_params,
            'fusion_params': fusion_params,
            'head_params': head_params,
            'total_params': video_params + audio_params + fusion_params + head_params,
            'video_trainable': video_trainable,
            'audio_trainable': audio_trainable,
            'fusion_trainable': fusion_trainable,
            'head_trainable': head_trainable,
            'total_trainable': video_trainable + audio_trainable + fusion_trainable + head_trainable,
            'video_feature_dim': self.video_feature_dim,
            'audio_feature_dim': self.audio_feature_dim
        }


def create_phase2_model(config: Dict) -> Phase2Model:
    model_config = config['models']
    model = Phase2Model(
        video_config=model_config['video'],
        audio_config=model_config['audio'],
        fusion_config=model_config.get('fusion', {}),
        head_config=model_config.get('head', {}),
        num_classes=model_config.get('num_classes', 2)
    )
    return model