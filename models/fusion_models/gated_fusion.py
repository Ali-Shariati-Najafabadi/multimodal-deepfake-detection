import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(
        self,
        video_feature_dim: int,
        audio_feature_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.5
    ):
        super(GatedFusion, self).__init__()
        
        self.video_proj = nn.Linear(video_feature_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_feature_dim, hidden_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(video_feature_dim + audio_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.output_dim = hidden_dim
    
    def forward(self, video_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        video_orig_shape = video_features.shape
        audio_orig_shape = audio_features.shape
        
        if video_features.dim() == 3:
            batch_size, seq_len, _ = video_features.shape
            video_features_flat = video_features.reshape(batch_size * seq_len, -1)
            if audio_features.dim() == 2:
                audio_features = audio_features.unsqueeze(1).expand(-1, seq_len, -1)
            audio_features_flat = audio_features.reshape(batch_size * seq_len, -1)
            is_sequential = True
        else:
            video_features_flat = video_features
            audio_features_flat = audio_features
            is_sequential = False
        
        video_proj = self.video_proj(video_features_flat)
        audio_proj = self.audio_proj(audio_features_flat)
        
        gate_input = torch.cat([video_features_flat, audio_features_flat], dim=-1)
        gate_values = self.gate(gate_input)
        
        fused = gate_values * video_proj + (1 - gate_values) * audio_proj
        
        if is_sequential:
            fused = fused.reshape(batch_size, seq_len, -1)
        
        return fused
    
    def get_output_dim(self):
        return self.output_dim
