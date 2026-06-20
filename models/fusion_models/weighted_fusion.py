import torch
import torch.nn as nn


class WeightedFusion(nn.Module):
    def __init__(
        self,
        video_feature_dim: int,
        audio_feature_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        init_video_weight: float = 0.5,
        init_audio_weight: float = 0.5
    ):
        super(WeightedFusion, self).__init__()
        
        self.video_proj = nn.Linear(video_feature_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_feature_dim, hidden_dim)
        
        self.video_weight = nn.Parameter(torch.tensor(init_video_weight))
        self.audio_weight = nn.Parameter(torch.tensor(init_audio_weight))
        
        self.output_dim = hidden_dim
    
    def forward(self, video_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        video_proj = self.video_proj(video_features)
        audio_proj = self.audio_proj(audio_features)
        
        if video_proj.dim() == 3 and audio_proj.dim() == 2:
            audio_proj = audio_proj.unsqueeze(1).expand(-1, video_proj.size(1), -1)
        elif video_proj.dim() == 2 and audio_proj.dim() == 3:
            video_proj = video_proj.unsqueeze(1).expand(-1, audio_proj.size(1), -1)
        
        weights = torch.softmax(torch.stack([self.video_weight, self.audio_weight]), dim=0)
        fused = weights[0] * video_proj + weights[1] * audio_proj
        
        return fused
    
    def get_output_dim(self):
        return self.output_dim
