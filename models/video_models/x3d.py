import torch
import torch.nn as nn
from typing import Optional, List


class X3D(nn.Module):
    def __init__(
        self,
        architecture: str = 'x3d_m',
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        unfreeze_layers: Optional[List[str]] = None,
        dropout: float = 0.5,
        num_frames: int = 16,
        output_type: str = 'video'
    ):
        super(X3D, self).__init__()

        self.architecture = architecture
        self.num_classes = num_classes
        self.output_type = output_type
        self.num_frames = num_frames

        model = torch.hub.load('facebookresearch/pytorchvideo', architecture, pretrained=pretrained)

        self.backbone = nn.Sequential(*list(model.blocks.children())[:-1])

        with torch.no_grad():
            dummy = torch.randn(1, 3, num_frames, 224, 224)
            feats = self.backbone(dummy)
            if feats.dim() == 5:
                feats = feats.mean(dim=[2, 3, 4])
            elif feats.dim() == 4:
                feats = feats.mean(dim=[2, 3])
            elif feats.dim() == 3:
                feats = feats.mean(dim=2)
            self.feature_dim = feats.size(1)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self._freeze_bn()
            print("X3D: Froze entire backbone")
        elif unfreeze_layers and len(unfreeze_layers) > 0:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self._selective_unfreeze(unfreeze_layers)
            self._freeze_bn()

        print(f"X3D: Loaded {architecture} {'(pretrained)' if pretrained else ''}, feature_dim={self.feature_dim}")

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

    def _freeze_bn(self):
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                if m.weight is not None:
                    m.weight.requires_grad = False
                if m.bias is not None:
                    m.bias.requires_grad = False

    def _selective_unfreeze(self, layers_to_unfreeze: List[str]):
        # backbone-Kinder heißen "0","1",... akzeptiere "blocks.3" und "3"
        wanted = {l.split('.')[-1] for l in layers_to_unfreeze}
        unfroze = {}
        for name, param in self.backbone.named_parameters():
            top = name.split('.')[0]
            if top in wanted:
                param.requires_grad = True
                unfroze[top] = unfroze.get(top, 0) + param.numel()
        for idx, n in sorted(unfroze.items()):
            print(f"X3D: Unfroze block {idx} ({n:,} params)")

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_bn()
        return self

    def extract_features(self, x):
        # Erwartet (B, C, T, H, W). Falls (B, T, C, H, W): permutieren.
        if x.dim() == 5 and x.size(1) != 3 and x.size(2) == 3:
            x = x.permute(0, 2, 1, 3, 4)

        feats = self.backbone(x)

        if feats.dim() == 5:
            feats = feats.mean(dim=[2, 3, 4])
        elif feats.dim() == 4:
            feats = feats.mean(dim=[2, 3])
        elif feats.dim() == 3:
            feats = feats.mean(dim=2)
        elif feats.dim() != 2:
            feats = feats.view(feats.size(0), -1)

        return feats

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.classifier(self.extract_features(x))

    def get_feature_dim(self):
        return self.feature_dim