import torch
import torch.nn as nn
from typing import Optional, List
from transformers import TimesformerModel, TimesformerConfig, AutoImageProcessor


class TimeSformer(nn.Module):
    def __init__(
        self,
        architecture: str = 'timesformer_base',
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        unfreeze_layers: Optional[List[str]] = None,
        dropout: float = 0.5,
        num_frames: int = 8,
        output_type: str = 'video'
    ):
        super(TimeSformer, self).__init__()
        self.architecture = architecture
        self.num_classes = num_classes
        self.output_type = output_type
        self.num_frames = num_frames
        self.feature_dim = 768
        
        if architecture == 'timesformer_base':
            self.image_size = 224
            pretrained_model = "facebook/timesformer-base-finetuned-k400"
        elif architecture == 'timesformer_hr':
            self.image_size = 448
            pretrained_model = "facebook/timesformer-hr-finetuned-k400"
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        if pretrained:
            self.model = TimesformerModel.from_pretrained(pretrained_model)
            print(f"  Loaded Kinetics400 pretrained {architecture} ({self.image_size}x{self.image_size})")
        else:
            config = TimesformerConfig(
                num_frames=num_frames,
                image_size=self.image_size,
                patch_size=16,
                num_attention_heads=12,
                attention_type="divided_space_time"
            )
            self.model = TimesformerModel(config)
            print(f"  Initialized {architecture} from scratch")
        
        print(f"  {architecture} initialized (feature_dim={self.feature_dim})")
        
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"  Backbone frozen")
            
            if unfreeze_layers and len(unfreeze_layers) > 0:
                self._selective_unfreeze(unfreeze_layers)
        else:
            for param in self.model.parameters():
                param.requires_grad = True
            print(f"  Backbone TRAINABLE (all layers)")
            
            if unfreeze_layers and len(unfreeze_layers) > 0:
                print(f"  Note: unfreeze_layers specified but backbone already trainable")
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
    
    def _selective_unfreeze(self, layers_to_unfreeze: List[str]):
        print(f"  Attempting to unfreeze layers: {layers_to_unfreeze}")
        
        if not hasattr(self.model, 'encoder'):
            print(f"  ERROR: model has no 'encoder' attribute!")
            return
        
        if not hasattr(self.model.encoder, 'layer'):
            print(f"  ERROR: model.encoder has no 'layer' attribute!")
            print(f"  Available attributes: {dir(self.model.encoder)}")
            return
        
        total_layers = len(self.model.encoder.layer)
        print(f"  Total transformer layers: {total_layers}")
        
        total_unfrozen = 0
        unfrozen_layers = []
        
        for layer_spec in layers_to_unfreeze:
            if not layer_spec.isdigit():
                print(f"    WARNING: '{layer_spec}' is not a digit, skipping")
                continue
            
            layer_idx = int(layer_spec)
            
            if layer_idx < 0 or layer_idx >= total_layers:
                print(f"    WARNING: Layer {layer_idx} out of range [0, {total_layers-1}]")
                continue
            
            unfrozen_count = 0
            for param in self.model.encoder.layer[layer_idx].parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen_count += param.numel()
            
            if unfrozen_count > 0:
                total_unfrozen += unfrozen_count
                unfrozen_layers.append(layer_idx)
                print(f"    ✓ Unfroze layer {layer_idx}: {unfrozen_count:,} params")
        
        if total_unfrozen > 0:
            print(f"  Successfully unfroze layers: {unfrozen_layers}")
            print(f"  Total unfrozen: {total_unfrozen:,} params")
        else:
            print(f"  WARNING: No layers were unfrozen!")
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,T,H,W), got {x.shape}")
        
        B, C, T, H, W = x.shape
        
        if H != self.image_size or W != self.image_size:
            x = torch.nn.functional.interpolate(
                x.view(B*T, C, H, W),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).view(B, C, T, self.image_size, self.image_size)
        
        if T != self.num_frames:
            indices = torch.linspace(0, T-1, self.num_frames).long().to(x.device)
            x = x[:, :, indices, :, :]
        
        x = x.permute(0, 2, 1, 3, 4)
        
        outputs = self.model(x)
        
        features = outputs.last_hidden_state[:, 0]
        
        return features
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits
    
    def get_feature_dim(self) -> int:
        return self.feature_dim
