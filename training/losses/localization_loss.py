import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalizationLoss(nn.Module):
    def __init__(
        self,
        boundary_loss_type: str = 'smooth_l1',
        confidence_loss_type: str = 'bce',
        boundary_weight: float = 1.0,
        confidence_weight: float = 1.0
    ):
        super(LocalizationLoss, self).__init__()
        
        self.boundary_loss_type = boundary_loss_type
        self.confidence_loss_type = confidence_loss_type
        self.boundary_weight = boundary_weight
        self.confidence_weight = confidence_weight
    
    def forward(
        self,
        pred_boundaries: torch.Tensor,
        pred_confidence: torch.Tensor,
        target_boundaries: torch.Tensor,
        target_confidence: torch.Tensor
    ):
        if self.boundary_loss_type == 'smooth_l1':
            boundary_loss = F.smooth_l1_loss(pred_boundaries, target_boundaries)
        elif self.boundary_loss_type == 'l1':
            boundary_loss = F.l1_loss(pred_boundaries, target_boundaries)
        elif self.boundary_loss_type == 'l2':
            boundary_loss = F.mse_loss(pred_boundaries, target_boundaries)
        elif self.boundary_loss_type == 'iou':
            boundary_loss = self.iou_loss(pred_boundaries, target_boundaries)
        elif self.boundary_loss_type == 'giou':
            boundary_loss = self.giou_loss(pred_boundaries, target_boundaries)
        else:
            raise ValueError(f"Unknown boundary loss type: {self.boundary_loss_type}")
        
        if self.confidence_loss_type == 'bce':
            confidence_loss = F.binary_cross_entropy(pred_confidence, target_confidence)
        elif self.confidence_loss_type == 'focal':
            confidence_loss = self.focal_loss(pred_confidence, target_confidence)
        else:
            raise ValueError(f"Unknown confidence loss type: {self.confidence_loss_type}")
        
        total_loss = (
            self.boundary_weight * boundary_loss +
            self.confidence_weight * confidence_loss
        )
        
        return total_loss, boundary_loss, confidence_loss
    
    def iou_loss(self, pred, target):
        pred_start, pred_end = pred[..., 0], pred[..., 1]
        target_start, target_end = target[..., 0], target[..., 1]
        
        inter_start = torch.max(pred_start, target_start)
        inter_end = torch.min(pred_end, target_end)
        inter = torch.clamp(inter_end - inter_start, min=0)
        
        pred_area = pred_end - pred_start
        target_area = target_end - target_start
        union = pred_area + target_area - inter
        
        iou = inter / (union + 1e-6)
        loss = 1 - iou.mean()
        
        return loss
    
    def giou_loss(self, pred, target):
        pred_start, pred_end = pred[..., 0], pred[..., 1]
        target_start, target_end = target[..., 0], target[..., 1]
        
        inter_start = torch.max(pred_start, target_start)
        inter_end = torch.min(pred_end, target_end)
        inter = torch.clamp(inter_end - inter_start, min=0)
        
        pred_area = pred_end - pred_start
        target_area = target_end - target_start
        union = pred_area + target_area - inter
        
        iou = inter / (union + 1e-6)
        
        convex_start = torch.min(pred_start, target_start)
        convex_end = torch.max(pred_end, target_end)
        convex_area = convex_end - convex_start
        
        giou = iou - (convex_area - union) / (convex_area + 1e-6)
        loss = 1 - giou.mean()
        
        return loss
    
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = alpha * (1 - pt) ** gamma * bce
        return focal.mean()
