import torch
import torch.nn as nn
from transformers import HubertModel, Wav2Vec2Model, WavLMModel
from typing import Optional

class AudioModel(nn.Module):
    def __init__(
        self,
        architecture: str = 'hubert',
        pretrained_model: str = 'facebook/hubert-base-ls960',
        num_classes: int = 2,
        freeze_feature_extractor: bool = False,
        unfreeze_last_n_layers: int = 4,
        dropout: float = 0.5,
        pooling: str = 'avg',
        use_gradient_checkpointing: bool = True
    ):
        super(AudioModel, self).__init__()
        
        self.architecture = architecture
        self.num_classes = num_classes
        self.pooling = pooling
        self.freeze_feature_extractor = freeze_feature_extractor
        
        if architecture == 'hubert':
            self.encoder = HubertModel.from_pretrained(pretrained_model)
            feature_dim = self.encoder.config.hidden_size
            
            if use_gradient_checkpointing:
                self.encoder.config.gradient_checkpointing = True
                print("   Audio: Gradient Checkpointing enabled")
        
        elif architecture == 'wav2vec2':
            self.encoder = Wav2Vec2Model.from_pretrained(pretrained_model)
            feature_dim = self.encoder.config.hidden_size
        
        elif architecture == 'wavlm':
            self.encoder = WavLMModel.from_pretrained(pretrained_model)
            feature_dim = self.encoder.config.hidden_size
        
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        if unfreeze_last_n_layers > 0 and unfreeze_last_n_layers < len(self.encoder.encoder.layers):
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            total_layers = len(self.encoder.encoder.layers)
            layers_to_unfreeze = min(unfreeze_last_n_layers, total_layers)
            
            for layer in self.encoder.encoder.layers[-layers_to_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True
            
            print(f"   Audio: Unfrozen last {layers_to_unfreeze}/{total_layers} encoder layers")
        
        elif freeze_feature_extractor:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(f"   Audio: All parameters frozen")
        
        else:
            print(f"   Audio: All parameters trainable")
        
        self.base_feature_dim = feature_dim
        if pooling == 'max_mean':
            self.feature_dim = feature_dim * 2
        else:
            self.feature_dim = feature_dim
        
        if pooling == 'attention':
            self.attention = TemporalAttention(feature_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_segments, segment_length = x.shape
        x = x.view(batch_size * num_segments, segment_length)
        
        outputs = self.encoder(x)
        features = outputs.last_hidden_state
        features = features.mean(dim=1)
        features = features.view(batch_size, num_segments, -1)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            features = features * mask_expanded
        
        if self.pooling == 'avg':
            if mask is not None:
                pooled = features.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = features.mean(dim=1)
        
        elif self.pooling == 'max':
            pooled = features.max(dim=1)[0]
        
        elif self.pooling == 'max_mean':
            # Max + Mean pooling
            if mask is not None:
                max_pool = features.max(dim=1)[0]
                mean_pool = features.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                max_pool = features.max(dim=1)[0]
                mean_pool = features.mean(dim=1)
            
            pooled = torch.cat([max_pool, mean_pool], dim=-1)
        
        elif self.pooling == 'attention':
            pooled = self.attention(features, mask)
        
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        logits = self.classifier(pooled)
        return logits
    
    def get_feature_dim(self):
        return self.feature_dim


class TemporalAttention(nn.Module):
    def __init__(self, feature_dim: int):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.Tanh(),
            nn.Linear(feature_dim // 4, 1)
        )
    
    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.attention(features)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        
        weights = torch.softmax(scores, dim=1)
        attended = (features * weights).sum(dim=1)
        
        return attended
