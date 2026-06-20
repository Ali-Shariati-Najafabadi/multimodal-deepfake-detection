import torch
import torch.nn as nn
from typing import Dict, Optional
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.video_models.video_swin import VideoSwinTransformer
from models.video_models.x3d import X3D
from models.video_models.resnet_lstm import ResNetLSTM
from models.video_models.ftcn import FTCN_ResNet50
from models.audio_models.audio_model import AudioModel

from models.fusion_models.concat_fusion import ConcatFusion
from models.fusion_models.add_fusion import AddFusion
from models.fusion_models.multiply_fusion import MultiplyFusion
from models.fusion_models.avg_fusion import AvgFusion
from models.fusion_models.weighted_fusion import WeightedFusion
from models.fusion_models.cross_attention_fusion import CrossAttentionFusion
from models.fusion_models.gated_fusion import GatedFusion
from models.fusion_models.film_fusion import FiLMFusion

from models.heads.classification_head import ClassificationHead


class Phase4Model(nn.Module):
    def __init__(
        self,
        video_config: Dict,
        audio_config: Dict,
        fusion_config: Dict,
        head_config: Dict,
        num_classes: int = 2
    ):
        super(Phase4Model, self).__init__()
        
        self.num_classes = num_classes
        self.phase = 'phase4'
        self.use_auxiliary_loss = True
        
        video_arch = video_config.get('architecture', 'x3d_m')
        if video_arch.startswith('video_swin'):
            self.video_model = VideoSwinTransformer(
                architecture=video_arch,
                num_classes=num_classes,
                pretrained=video_config.get('pretrained', True),
                freeze_backbone=video_config.get('freeze_backbone', False),
                unfreeze_layers=video_config.get('unfreeze_layers', None),
                dropout=video_config.get('dropout', 0.5),
                num_frames=video_config.get('num_frames', 32),
                output_type=video_config.get('output_type', 'video')
            )
        elif video_arch.startswith('x3d'):
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
        elif video_arch == 'resnet_lstm':
            self.video_model = ResNetLSTM(
                architecture=video_config.get('base_architecture', 'resnet50'),
                num_classes=num_classes,
                pretrained=video_config.get('pretrained', True),
                freeze_backbone=video_config.get('freeze_backbone', False),
                unfreeze_layers=video_config.get('unfreeze_layers', None),
                dropout=video_config.get('dropout', 0.5),
                lstm_hidden_size=video_config.get('lstm', {}).get('hidden_size', 512),
                lstm_num_layers=video_config.get('lstm', {}).get('num_layers', 2),
                lstm_bidirectional=video_config.get('lstm', {}).get('bidirectional', True),
                lstm_dropout=video_config.get('lstm', {}).get('dropout', 0.3),
                output_type=video_config.get('output_type', 'video')
            )
        elif video_arch == 'ftcn':
            self.video_model = FTCN_ResNet50(
                num_classes=num_classes,
                dropout=video_config.get('dropout', 0.5),
                pretrained_2d=video_config.get('pretrained', True),
                freeze_backbone=video_config.get('freeze_backbone', False),
                unfreeze_layers=video_config.get('unfreeze_layers', None)
            )
        else:
            raise ValueError(f"Unknown video architecture: {video_arch}")
        
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
        
        fusion_type = fusion_config.get('type', 'concat')
        fusion_dropout = fusion_config.get('dropout', 0.5)
        fusion_hidden = fusion_config.get('hidden_dim', 512)
        
        print(f"\n{'='*60}")
        print(f"Phase 4: Testing Fusion Strategy = '{fusion_type}'")
        print(f"{'='*60}\n")
        
        if fusion_type == 'concat':
            self.fusion = ConcatFusion(self.video_feature_dim, self.audio_feature_dim)
        elif fusion_type == 'add':
            self.fusion = AddFusion(self.video_feature_dim, self.audio_feature_dim, fusion_hidden, fusion_dropout)
        elif fusion_type == 'multiply':
            self.fusion = MultiplyFusion(self.video_feature_dim, self.audio_feature_dim, fusion_hidden, fusion_dropout)
        elif fusion_type == 'avg':
            self.fusion = AvgFusion(self.video_feature_dim, self.audio_feature_dim, fusion_hidden, fusion_dropout)
        elif fusion_type == 'weighted':
            weighted_config = fusion_config.get('weighted', {})
            self.fusion = WeightedFusion(
                self.video_feature_dim, self.audio_feature_dim, fusion_hidden, fusion_dropout,
                weighted_config.get('init_video_weight', 0.5),
                weighted_config.get('init_audio_weight', 0.5)
            )
        elif fusion_type == 'cross_attention':
            ca_config = fusion_config.get('cross_attention', {})
            self.fusion = CrossAttentionFusion(
                self.video_feature_dim, self.audio_feature_dim, fusion_hidden,
                ca_config.get('num_heads', 8),
                ca_config.get('num_layers', 2),
                fusion_dropout
            )
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(self.video_feature_dim, self.audio_feature_dim, fusion_hidden, fusion_dropout)
        elif fusion_type == 'film':
            film_config = fusion_config.get('film', {})
            self.fusion = FiLMFusion(
                self.video_feature_dim, self.audio_feature_dim, fusion_hidden, fusion_dropout,
                film_config.get('conditioning', 'audio')
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        fusion_output_dim = self.fusion.get_output_dim()
        
        self.head = ClassificationHead(
            input_dim=fusion_output_dim,
            num_classes=num_classes,
            hidden_dims=head_config.get('hidden_dims', [512, 256]),
            dropout=head_config.get('dropout', 0.5)
        )
        
        self.fusion_type = fusion_type
    
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
            video_logits = self.video_model.classifier(video_features)
            audio_logits = self.audio_model.classifier(audio_features)
            return fusion_logits, video_logits, audio_logits
        
        return fusion_logits
    
    def get_model_info(self):
        video_params = sum(p.numel() for p in self.video_model.parameters())
        audio_params = sum(p.numel() for p in self.audio_model.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        
        return {
            'phase': self.phase,
            'fusion_type': self.fusion_type,
            'total_params': video_params + audio_params + fusion_params + head_params,
            'video_params': video_params,
            'audio_params': audio_params,
            'fusion_params': fusion_params,
            'head_params': head_params
        }


def create_phase4_model(config: Dict) -> Phase4Model:
    model_config = config['models']
    model = Phase4Model(
        video_config=model_config['video'],
        audio_config=model_config['audio'],
        fusion_config=model_config['fusion'],
        head_config=model_config['head'],
        num_classes=model_config.get('num_classes', 2)
    )
    return model
