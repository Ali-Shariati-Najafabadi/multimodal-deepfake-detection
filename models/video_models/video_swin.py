import torch
import torch.nn as nn
from typing import Optional, List
import warnings


class VideoSwinTransformer(nn.Module):
    def __init__(
        self,
        architecture: str = 'video_swin_t',
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        unfreeze_layers: Optional[List[str]] = None,
        dropout: float = 0.5,
        num_frames: int = 32,
        output_type: str = 'video'
    ):
        super(VideoSwinTransformer, self).__init__()
        self.architecture = architecture
        self.num_classes = num_classes
        self.output_type = output_type
        self.num_frames = num_frames
        
        if architecture == 'video_swin_t':
            from torchvision.models.video import swin3d_t, Swin3D_T_Weights
            backbone = swin3d_t(weights=None)
            self.feature_dim = 768
            if pretrained:
                try:
                    weights_dict = Swin3D_T_Weights.KINETICS400_V1.get_state_dict(progress=True)
                    backbone.load_state_dict(weights_dict, strict=False)
                    print(f"  Loaded Kinetics400 pretrained {architecture}")
                except Exception as e:
                    warnings.warn(f"Could not load pretrained weights: {e}")
        
        elif architecture == 'video_swin_s':
            from torchvision.models.video import swin3d_s, Swin3D_S_Weights
            backbone = swin3d_s(weights=None)
            self.feature_dim = 768
            if pretrained:
                try:
                    weights_dict = Swin3D_S_Weights.KINETICS400_V1.get_state_dict(progress=True)
                    backbone.load_state_dict(weights_dict, strict=False)
                    print(f"  Loaded Kinetics400 pretrained {architecture}")
                except Exception as e:
                    warnings.warn(f"Could not load pretrained weights: {e}")
        
        elif architecture == 'video_swin_b':
            from torchvision.models.video import swin3d_b, Swin3D_B_Weights
            backbone = swin3d_b(weights=None)
            self.feature_dim = 1024
            if pretrained:
                try:
                    weights_dict = Swin3D_B_Weights.KINETICS400_V1.get_state_dict(progress=True)
                    backbone.load_state_dict(weights_dict, strict=False)
                    print(f"  Loaded Kinetics400 pretrained {architecture}")
                except Exception as e:
                    warnings.warn(f"Could not load pretrained weights: {e}")
        else:
            raise ValueError(f"Unknown Swin architecture: {architecture}")
        
        self.patch_embed = backbone.patch_embed
        self.pos_drop = backbone.pos_drop
        self.features = backbone.features
        self.norm = backbone.norm
        
        print(f"  {architecture} initialized (feature_dim={self.feature_dim})")
        
        if freeze_backbone:
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.norm.parameters():
                param.requires_grad = False
            print(f"  Backbone frozen")
            
            if unfreeze_layers and len(unfreeze_layers) > 0:
                self._selective_unfreeze(unfreeze_layers)
        else:
            print(f"  Backbone TRAINABLE")
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
    
    def _selective_unfreeze(self, layers_to_unfreeze: List[str]):
        print(f"  Attempting to unfreeze: {layers_to_unfreeze}")
        total_unfrozen = 0
        
        for layer_spec in layers_to_unfreeze:
            unfrozen_this_layer = 0
            found = False
            
            if layer_spec.isdigit():
                stage_idx = int(layer_spec)
                if stage_idx < len(self.features):
                    for param in self.features[stage_idx].parameters():
                        if not param.requires_grad:
                            param.requires_grad = True
                            unfrozen_this_layer += param.numel()
                    total_unfrozen += unfrozen_this_layer
                    found = True
                    print(f"  Unfroze stage {stage_idx} ({unfrozen_this_layer:,} params)")
            
            if not found:
                print(f"  WARNING: '{layer_spec}' not found")
        
        print(f"  Total unfrozen: {total_unfrozen:,} params")
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,T,H,W), got {x.shape}")
        
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.features(x)
        x = self.norm(x)
        x = x.mean(dim=[1, 2, 3])
        
        return x
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits
    
    def get_feature_dim(self) -> int:
        return self.feature_dim
