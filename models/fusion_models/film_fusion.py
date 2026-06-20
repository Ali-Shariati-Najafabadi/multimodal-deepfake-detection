import torch
import torch.nn as nn


class FiLMFusion(nn.Module):
    def __init__(
        self,
        video_feature_dim: int,
        audio_feature_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        conditioning: str = 'audio'
    ):
        super(FiLMFusion, self).__init__()
        
        self.conditioning = conditioning
        
        if conditioning == 'audio':
            self.base_proj = nn.Linear(video_feature_dim, hidden_dim)
            cond_dim = audio_feature_dim
        else:
            self.base_proj = nn.Linear(audio_feature_dim, hidden_dim)
            cond_dim = video_feature_dim
        
        self.film_generator = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )
        
        self.output_dim = hidden_dim
    
    def forward(self, video_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        if self.conditioning == 'audio':
            base_features = video_features
            cond_features = audio_features
        else:
            base_features = audio_features
            cond_features = video_features
        
        is_sequential = base_features.dim() == 3
        
        if is_sequential:
            batch_size, seq_len, _ = base_features.shape
            base_features_flat = base_features.reshape(batch_size * seq_len, -1)
            
            if cond_features.dim() == 2:
                cond_features = cond_features.unsqueeze(1).expand(-1, seq_len, -1)
            cond_features_flat = cond_features.reshape(batch_size * seq_len, -1)
        else:
            base_features_flat = base_features
            cond_features_flat = cond_features
        
        base_proj = self.base_proj(base_features_flat)
        
        film_params = self.film_generator(cond_features_flat)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        
        fused = gamma * base_proj + beta
        
        if is_sequential:
            fused = fused.reshape(batch_size, seq_len, -1)
        
        return fused
    
    def get_output_dim(self):
        return self.output_dim
