import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, List

class ResNetLSTM(nn.Module):
    def __init__(
        self,
        architecture: str = 'resnet50',
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        unfreeze_layers: Optional[List[str]] = None,
        dropout: float = 0.5,
        lstm_hidden_size: int = 512,
        lstm_num_layers: int = 2,
        lstm_bidirectional: bool = True,
        lstm_dropout: float = 0.3,
        output_type: str = 'video',
        pooling_type: str = 'last',  # 'last', 'max', 'max_mean', 'attention'
        use_gradient_checkpointing: bool = True
    ):
        super(ResNetLSTM, self).__init__()
        
        self.architecture = architecture
        self.num_classes = num_classes
        self.output_type = output_type
        self.pooling_type = pooling_type 
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_bidirectional = lstm_bidirectional
        
        if architecture == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif architecture == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        if self.use_gradient_checkpointing:
            print("   Video: Gradient Checkpointing enabled")
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self._freeze_bn()
        elif unfreeze_layers and len(unfreeze_layers) > 0:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self._selective_unfreeze(unfreeze_layers)
            self._freeze_bn()
        
        lstm_input_size = self.feature_dim
        lstm_hidden = lstm_hidden_size
        lstm_output_size = lstm_hidden * 2 if lstm_bidirectional else lstm_hidden
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=lstm_bidirectional,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0
        )
        
        if output_type == 'video':
            if pooling_type == 'max_mean':
                self.feature_dim = lstm_output_size * 2  # Max + Mean
            elif pooling_type == 'attention':
                self.feature_dim = lstm_output_size
                self.attention = TemporalAttention(lstm_output_size)
            else:  # 'last' or 'max'
                self.feature_dim = lstm_output_size
        else:
            self.feature_dim = lstm_output_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
    
    def _freeze_bn(self):
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()
                module.weight.requires_grad = False
                module.bias.requires_grad = False
    
    def _selective_unfreeze(self, layers_to_unfreeze: List[str]):
        layer_map = {
            'layer4': ['7'],
            'layer3': ['6'],
            'layer2': ['5'],
            'layer1': ['4']
        }
        
        for layer_name in layers_to_unfreeze:
            indices = layer_map.get(layer_name, [])
            for idx in indices:
                if idx in self.backbone._modules:
                    layer = self.backbone._modules[idx]
                    for name, param in layer.named_parameters():
                        if 'bn' not in name and 'downsample.1' not in name:
                            param.requires_grad = True
                    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                    print(f"ResNetLSTM: Unfroze layer{idx} ({trainable:,} params)")
    
    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_bn()
        return self
    
    def extract_features(self, x, mask=None):
        if x.dim() == 5:
            if x.shape[1] == 3:
                batch_size, c, num_frames, h, w = x.shape
                x = x.permute(0, 2, 1, 3, 4)
            else:
                batch_size, num_frames, c, h, w = x.shape
        else:
            raise ValueError(f"Expected 5D input, got {x.dim()}D with shape {x.shape}")
        
        x = x.contiguous().view(batch_size * num_frames, c, h, w)
        
        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            features = checkpoint(self.backbone, x, use_reentrant=False)
        else:
            features = self.backbone(x)
        
        features = features.view(batch_size * num_frames, -1)
        features = features.view(batch_size, num_frames, -1)
        
        # LSTM with mask
        if mask is not None:
            lengths = mask.sum(dim=1).cpu().long()
            packed = nn.utils.rnn.pack_padded_sequence(
                features, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(features)
        
        if self.output_type == 'video':
            if self.pooling_type == 'last':
                # Original: Take last frame
                if mask is not None:
                    last_indices = (mask.sum(dim=1) - 1).long()
                    features = lstm_out[torch.arange(batch_size), last_indices]
                else:
                    features = lstm_out[:, -1, :]
            
            elif self.pooling_type == 'max':
                # Max pooling
                features = lstm_out.max(dim=1)[0]
            
            elif self.pooling_type == 'max_mean':
                # Max + Mean pooling
                if mask is not None:
                    mask_expanded = mask.unsqueeze(-1)
                    max_pool = lstm_out.max(dim=1)[0]
                    masked_lstm = lstm_out * mask_expanded
                    mean_pool = masked_lstm.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
                else:
                    max_pool = lstm_out.max(dim=1)[0]
                    mean_pool = lstm_out.mean(dim=1)
                
                features = torch.cat([max_pool, mean_pool], dim=-1)
            
            elif self.pooling_type == 'attention':
                # Attention pooling
                features = self.attention(lstm_out, mask)
            
            else:
                raise ValueError(f"Unknown pooling_type: {self.pooling_type}")
        else:
            features = lstm_out
        
        return features
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.extract_features(x, mask)
        
        if self.output_type == 'video':
            logits = self.classifier(features)
        else:
            batch_size, seq_len, feat_dim = features.shape
            features_flat = features.view(batch_size * seq_len, feat_dim)
            logits = self.classifier(features_flat)
            logits = logits.view(batch_size, seq_len, self.num_classes)
        
        return logits
    
    def get_feature_dim(self):
        return self.feature_dim


class TemporalAttention(nn.Module):
    def __init__(self, feature_dim: int):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.Tanh(),
            nn.Linear(feature_dim // 4, 1)
        )
    
    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None):
        scores = self.attention(features)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        
        weights = torch.softmax(scores, dim=1)
        attended = (features * weights).sum(dim=1)
        
        return attended
