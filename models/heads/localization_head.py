import torch
import torch.nn as nn


class LocalizationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        method: str = 'proposal',
        num_proposals: int = 100,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super(LocalizationHead, self).__init__()
        
        self.method = method
        self.num_proposals = num_proposals
        
        if method == 'proposal':
            self.boundary_regressor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2)
            )
            
            self.confidence_scorer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        
        elif method == 'boundary_regression':
            self.boundary_regressor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2)
            )
        
        elif method == 'actionformer':
            self.temporal_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=input_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=num_layers
            )
            
            self.boundary_regressor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2)
            )
            
            self.confidence_scorer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        
        else:
            raise ValueError(f"Unknown localization method: {method}")
    
    def forward(self, features: torch.Tensor):
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        batch_size, seq_len, feat_dim = features.shape
        
        if self.method == 'proposal':
            features_flat = features.reshape(batch_size * seq_len, feat_dim)
            
            boundaries = self.boundary_regressor(features_flat)
            boundaries = boundaries.reshape(batch_size, seq_len, 2)
            
            confidence = self.confidence_scorer(features_flat)
            confidence = confidence.reshape(batch_size, seq_len)
            
            if seq_len > self.num_proposals:
                top_k_conf, top_k_idx = torch.topk(confidence, self.num_proposals, dim=1)
                batch_idx = torch.arange(batch_size, device=features.device).unsqueeze(1).expand(-1, self.num_proposals)
                boundaries = boundaries[batch_idx, top_k_idx]
                confidence = top_k_conf
            
            return boundaries, confidence
        
        elif self.method == 'boundary_regression':
            features_pooled = features.mean(dim=1)
            boundaries = self.boundary_regressor(features_pooled)
            return boundaries.unsqueeze(1), None
        
        elif self.method == 'actionformer':
            encoded = self.temporal_encoder(features)
            
            encoded_flat = encoded.reshape(batch_size * seq_len, feat_dim)
            
            boundaries = self.boundary_regressor(encoded_flat)
            boundaries = boundaries.reshape(batch_size, seq_len, 2)
            
            confidence = self.confidence_scorer(encoded_flat)
            confidence = confidence.reshape(batch_size, seq_len)
            
            if seq_len > self.num_proposals:
                top_k_conf, top_k_idx = torch.topk(confidence, self.num_proposals, dim=1)
                batch_idx = torch.arange(batch_size, device=features.device).unsqueeze(1).expand(-1, self.num_proposals)
                boundaries = boundaries[batch_idx, top_k_idx]
                confidence = top_k_conf
            
            return boundaries, confidence
