import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        video_feature_dim: int,
        audio_feature_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super(CrossAttentionFusion, self).__init__()
        
        self.video_proj = nn.Linear(video_feature_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_feature_dim, hidden_dim)
        
        self.video_to_audio_attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.audio_to_video_attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.video_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.audio_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        self.output_dim = hidden_dim * 2
    
    def forward(self, video_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        video_proj = self.video_proj(video_features)
        audio_proj = self.audio_proj(audio_features)
        
        if video_proj.dim() == 2:
            video_proj = video_proj.unsqueeze(1)
        if audio_proj.dim() == 2:
            audio_proj = audio_proj.unsqueeze(1)
        
        for i in range(len(self.video_to_audio_attention)):
            video_attended, _ = self.video_to_audio_attention[i](
                video_proj, audio_proj, audio_proj
            )
            video_proj = self.video_norm[i](video_proj + video_attended)
            
            audio_attended, _ = self.audio_to_video_attention[i](
                audio_proj, video_proj, video_proj
            )
            audio_proj = self.audio_norm[i](audio_proj + audio_attended)
        
        video_pooled = video_proj.mean(dim=1)
        audio_pooled = audio_proj.mean(dim=1)
        
        fused = torch.cat([video_pooled, audio_pooled], dim=-1)
        
        return fused
    
    def get_output_dim(self):
        return self.output_dim
