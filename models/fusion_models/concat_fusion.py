import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def __init__(
        self,
        video_feature_dim: int,
        audio_feature_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.5
    ):
        super(ConcatFusion, self).__init__()
        
        self.video_feature_dim = video_feature_dim
        self.audio_feature_dim = audio_feature_dim
        self.output_dim = video_feature_dim + audio_feature_dim
    
    def forward(self, video_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        if video_features.dim() == 3 and audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(1).expand(-1, video_features.size(1), -1)
        elif video_features.dim() == 2 and audio_features.dim() == 3:
            video_features = video_features.unsqueeze(1).expand(-1, audio_features.size(1), -1)
        
        fused = torch.cat([video_features, audio_features], dim=-1)
        return fused
    
    def get_output_dim(self):
        return self.output_dim
