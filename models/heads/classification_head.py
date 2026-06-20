import torch
import torch.nn as nn
from typing import List


class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.5
    ):
        super(ClassificationHead, self).__init__()
        
        self.num_classes = num_classes
        
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
        if features.dim() == 3:
            features = features.mean(dim=1)
        
        logits = self.classifier(features)
        return logits
