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
from models.video_models.video_timesformer import TimeSformer


class Phase3Model(nn.Module):
    def __init__(
        self,
        video_config: Dict,
        audio_config: Dict,
        fusion_config: Dict,
        head_config: Dict,
        num_classes: int = 2
    ):
        super(Phase3Model, self).__init__()
        self.num_classes = num_classes
        self.phase = 'phase3'
        self.use_auxiliary_loss = True

        video_arch = video_config.get('architecture', 'video_swin_t')

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
        elif video_arch in ['timesformer_base', 'timesformer_hr']:
            self.video_model = TimeSformer(
                architecture=video_arch,
                num_classes=num_classes,
                pretrained=video_config.get('pretrained', True),
                freeze_backbone=video_config.get('freeze_backbone', False),
                unfreeze_layers=video_config.get('unfreeze_layers', None),
                dropout=video_config.get('dropout', 0.5),
                num_frames=video_config.get('num_frames', 8),
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
        self.is_timesformer = isinstance(self.video_model, TimeSformer)

        fusion_type = fusion_config.get('type', 'cross_attention')
        fusion_dropout = fusion_config.get('dropout', 0.1)
        fusion_hidden = fusion_config.get('hidden_dim', 512)

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

        # Nur Cross-Attention arbeitet auf Sequenzen, alle anderen Fusionen brauchen gepoolte Vektoren
        self.fusion_needs_sequence = isinstance(self.fusion, CrossAttentionFusion)

        fusion_output_dim = self.fusion.get_output_dim()

        self.head = ClassificationHead(
            input_dim=fusion_output_dim,
            num_classes=num_classes,
            hidden_dims=head_config.get('hidden_dims', [512, 256]),
            dropout=head_config.get('dropout', 0.5)
        )

        # Modalitaetsspezifische Aux-Koepfe, immer binaer (real/fake pro Modalitaet)
        aux_dropout = head_config.get('dropout', 0.3)
        self.video_aux_head = nn.Sequential(
            nn.Dropout(aux_dropout),
            nn.Linear(self.video_feature_dim, 2)
        )
        self.audio_aux_head = nn.Sequential(
            nn.Dropout(aux_dropout),
            nn.Linear(self.audio_feature_dim, 2)
        )

    def extract_video_features(self, video_input: torch.Tensor, video_mask: Optional[torch.Tensor] = None):
        if self.is_timesformer:
            return self.video_model.extract_features(video_input, return_sequence=True)
        return self.video_model.extract_features(video_input)

    def extract_audio_features(self, audio_input: torch.Tensor, audio_mask: Optional[torch.Tensor] = None):
        batch_size, num_segments, segment_length = audio_input.shape
        x = audio_input.view(batch_size * num_segments, segment_length)

        outputs = self.audio_model.encoder(x)
        features = outputs.last_hidden_state                            # [B*S, T, 768]
        features = features.reshape(batch_size, -1, features.shape[-1]) # [B, S*T, 768]

        return features

    @staticmethod
    def _pool(features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=1) if features.dim() == 3 else features

    def forward(self, video_input, audio_input, video_mask=None, audio_mask=None, return_auxiliary=False):
        video_features = self.extract_video_features(video_input, video_mask)
        audio_features = self.extract_audio_features(audio_input, audio_mask)

        if self.fusion_needs_sequence:
            fused_features = self.fusion(video_features, audio_features)
        else:
            fused_features = self.fusion(self._pool(video_features), self._pool(audio_features))

        fusion_logits = self.head(fused_features)

        if return_auxiliary and self.use_auxiliary_loss:
            video_logits = self.video_aux_head(self._pool(video_features))
            audio_logits = self.audio_aux_head(self._pool(audio_features))
            return fusion_logits, video_logits, audio_logits

        return fusion_logits

    def get_model_info(self):
        video_params = sum(p.numel() for p in self.video_model.parameters())
        audio_params = sum(p.numel() for p in self.audio_model.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        aux_params = sum(p.numel() for p in self.video_aux_head.parameters()) \
                   + sum(p.numel() for p in self.audio_aux_head.parameters())
        head_params += aux_params

        video_trainable = sum(p.numel() for p in self.video_model.parameters() if p.requires_grad)
        audio_trainable = sum(p.numel() for p in self.audio_model.parameters() if p.requires_grad)
        fusion_trainable = sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)
        head_trainable = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        head_trainable += sum(p.numel() for p in self.video_aux_head.parameters() if p.requires_grad)
        head_trainable += sum(p.numel() for p in self.audio_aux_head.parameters() if p.requires_grad)

        return {
            'phase': self.phase,
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
            'audio_feature_dim': self.audio_feature_dim,
            'fusion_type': self.fusion.__class__.__name__,
            'video_architecture': self.video_model.__class__.__name__,
            'audio_architecture': self.audio_model.__class__.__name__
        }


def create_phase3_model(config: Dict) -> Phase3Model:
    model_config = config['models']
    model = Phase3Model(
        video_config=model_config['video'],
        audio_config=model_config['audio'],
        fusion_config=model_config['fusion'],
        head_config=model_config['head'],
        num_classes=model_config.get('num_classes', 2)
    )
    return model