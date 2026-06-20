import torch
import torch.nn as nn
from typing import List


class SegmentHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.5,
        temporal_conv: bool = True,
        kernel_size: int = 3
    ):
        super(SegmentHead, self).__init__()
        
        self.num_classes = num_classes
        self.temporal_conv = temporal_conv
        
        if temporal_conv:
            self.temporal = nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(input_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        batch_size, seq_len, feat_dim = features.shape
        
        if self.temporal_conv:
            features_t = features.permute(0, 2, 1)
            features_t = self.temporal(features_t)
            features = features_t.permute(0, 2, 1)
        
        features_flat = features.reshape(batch_size * seq_len, feat_dim)
        logits = self.classifier(features_flat)
        logits = logits.reshape(batch_size, seq_len, self.num_classes)
        
        return logits
