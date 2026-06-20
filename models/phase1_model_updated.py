import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.video_models.resnet_lstm import ResNetLSTM
from models.audio_models.audio_model import AudioModel
from models.fusion_models.concat_fusion import ConcatFusion
from models.fusion_models.cross_attention_fusion import CrossAttentionFusion
from models.heads.classification_head import ClassificationHead


class Phase1Model(nn.Module):
    def __init__(
        self,
        video_config: Dict,
        audio_config: Dict,
        fusion_config: Dict = None,
        head_config: Dict = None,
        num_classes: int = 2
    ):
        super(Phase1Model, self).__init__()
        
        self.num_classes = num_classes
        self.phase = 'phase1'
        self.use_auxiliary_loss = True
        
        video_arch = video_config.get('architecture', 'resnet50')
        
        if video_arch == 'resnet_lstm':
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
                output_type=video_config.get('output_type', 'video'),
                pooling_type=video_config.get('pooling_type', 'last'),
                use_gradient_checkpointing=video_config.get('use_gradient_checkpointing', True)
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
            pooling=audio_config.get('pooling', 'avg'),
            use_gradient_checkpointing=audio_config.get('use_gradient_checkpointing', True)
        )
        
        self.video_feature_dim = self.video_model.get_feature_dim()
        self.audio_feature_dim = self.audio_model.get_feature_dim()
        
        if fusion_config:
            fusion_type = fusion_config.get('type', 'concat')
        else:
            fusion_type = 'concat'
        
        if fusion_type == 'concat':
            self.fusion = ConcatFusion(self.video_feature_dim, self.audio_feature_dim)
        elif fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(
                video_feature_dim=self.video_feature_dim,
                audio_feature_dim=self.audio_feature_dim,
                hidden_dim=fusion_config.get('hidden_dim', 512),
                num_heads=fusion_config.get('num_heads', 8),
                num_layers=fusion_config.get('num_layers', 2),
                dropout=fusion_config.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
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
        # Video kann nur Video-Fakeness lernen, Audio nur Audio-Fakeness.
        self.video_aux_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.video_feature_dim, 2)
        )
        self.audio_aux_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.audio_feature_dim, 2)
        )
    
    def extract_video_features(self, video_input: torch.Tensor, video_mask: Optional[torch.Tensor] = None):
        batch_size = video_input.shape[0]
        
        if video_input.dim() == 5:
            if video_input.shape[1] == 3:
                c, num_frames, h, w = video_input.shape[1:]
                video_input = video_input.permute(0, 2, 1, 3, 4)
            else:
                num_frames, c, h, w = video_input.shape[1:]
        
        video_input = video_input.contiguous().view(batch_size * num_frames, c, h, w)
        
        if self.video_model.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            features = checkpoint(self.video_model.backbone, video_input, use_reentrant=False)
        else:
            features = self.video_model.backbone(video_input)
        
        features = features.view(batch_size * num_frames, -1)
        features = features.view(batch_size, num_frames, -1)
        
        if video_mask is not None:
            lengths = video_mask.sum(dim=1).cpu().long()
            packed_features = nn.utils.rnn.pack_padded_sequence(
                features, 
                lengths, 
                batch_first=True, 
                enforce_sorted=False
            )
            lstm_out, _ = self.video_model.lstm(packed_features)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.video_model.lstm(features)
        
        if self.video_model.output_type == 'video':
            if self.video_model.pooling_type == 'last':
                if video_mask is not None:
                    last_indices = (video_mask.sum(dim=1) - 1).long()
                    features = lstm_out[torch.arange(batch_size), last_indices]
                else:
                    features = lstm_out[:, -1, :]
            
            elif self.video_model.pooling_type == 'max':
                features = lstm_out.max(dim=1)[0]
            
            elif self.video_model.pooling_type == 'max_mean':
                if video_mask is not None:
                    mask_expanded = video_mask.unsqueeze(-1)
                    max_pool = lstm_out.max(dim=1)[0]
                    masked_lstm = lstm_out * mask_expanded
                    mean_pool = masked_lstm.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
                else:
                    max_pool = lstm_out.max(dim=1)[0]
                    mean_pool = lstm_out.mean(dim=1)
                
                features = torch.cat([max_pool, mean_pool], dim=-1)
            
            elif self.video_model.pooling_type == 'attention':
                features = self.video_model.attention(lstm_out, video_mask)
        else:
            features = lstm_out
        
        return features
    
    def extract_audio_features(self, audio_input: torch.Tensor, audio_mask: Optional[torch.Tensor] = None):
        batch_size, num_segments, segment_length = audio_input.shape
        x = audio_input.view(batch_size * num_segments, segment_length)
        
        attention_mask = None
        if audio_mask is not None:
            attention_mask = audio_mask.view(batch_size * num_segments).float()
            attention_mask = attention_mask.unsqueeze(-1).expand(-1, segment_length)
        
        outputs = self.audio_model.encoder(x, attention_mask=attention_mask)
        features = outputs.last_hidden_state.mean(dim=1)
        features = features.view(batch_size, num_segments, -1)
        
        if audio_mask is not None:
            mask_expanded = audio_mask.unsqueeze(-1)
            features = features * mask_expanded
        
        if self.audio_model.pooling == 'avg':
            if audio_mask is not None:
                pooled = features.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = features.mean(dim=1)
        
        elif self.audio_model.pooling == 'max':
            pooled = features.max(dim=1)[0]
        
        elif self.audio_model.pooling == 'max_mean':
            if audio_mask is not None:
                max_pool = features.max(dim=1)[0]
                mean_pool = features.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                max_pool = features.max(dim=1)[0]
                mean_pool = features.mean(dim=1)
            
            pooled = torch.cat([max_pool, mean_pool], dim=-1)
        
        elif self.audio_model.pooling == 'attention':
            pooled = self.audio_model.attention(features, audio_mask)
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
        features = self.extract_video_features(video_input, video_mask)
        return self.video_model.classifier(features)
    
    def forward_audio_only(self, audio_input: torch.Tensor, audio_mask: Optional[torch.Tensor] = None):
        features = self.extract_audio_features(audio_input, audio_mask)
        return self.audio_model.classifier(features)
    
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


def create_phase1_model(config: Dict) -> Phase1Model:
    model_config = config['models']
    model = Phase1Model(
        video_config=model_config['video'],
        audio_config=model_config['audio'],
        fusion_config=model_config.get('fusion', {}),
        head_config=model_config.get('head', {}),
        num_classes=model_config.get('num_classes', 2)
    )
    return model