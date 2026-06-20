import torch
import numpy as np
from typing import List, Tuple, Dict


class LocalizationMetrics:
    def __init__(
        self,
        iou_thresholds: List[float] = [0.3, 0.5, 0.7],
        proposal_counts: List[int] = [1, 5, 10, 100]
    ):
        self.iou_thresholds = iou_thresholds
        self.proposal_counts = proposal_counts
        self.reset()
    
    def reset(self):
        self.all_predictions = []
        self.all_targets = []
    
    def update(
        self,
        pred_boundaries: torch.Tensor,
        pred_confidence: torch.Tensor,
        target_boundaries: torch.Tensor,
        target_labels: torch.Tensor
    ):
        if isinstance(pred_boundaries, torch.Tensor):
            pred_boundaries = pred_boundaries.detach().cpu().numpy()
        if isinstance(pred_confidence, torch.Tensor):
            pred_confidence = pred_confidence.detach().cpu().numpy()
        if isinstance(target_boundaries, torch.Tensor):
            target_boundaries = target_boundaries.detach().cpu().numpy()
        if isinstance(target_labels, torch.Tensor):
            target_labels = target_labels.detach().cpu().numpy()
        
        batch_size = pred_boundaries.shape[0]
        
        for i in range(batch_size):
            pred = {
                'boundaries': pred_boundaries[i],
                'confidence': pred_confidence[i] if pred_confidence is not None else None
            }
            
            target = {
                'boundaries': target_boundaries[i],
                'labels': target_labels[i]
            }
            
            self.all_predictions.append(pred)
            self.all_targets.append(target)
    
    def compute_iou(self, pred_boundary, target_boundary):
        pred_start, pred_end = pred_boundary
        target_start, target_end = target_boundary
        
        inter_start = max(pred_start, target_start)
        inter_end = min(pred_end, target_end)
        inter = max(0, inter_end - inter_start)
        
        pred_length = pred_end - pred_start
        target_length = target_end - target_start
        union = pred_length + target_length - inter
        
        if union == 0:
            return 0.0
        
        iou = inter / union
        return iou
    
    def compute_ap(self, iou_threshold: float):
        all_pred_boxes = []
        all_target_boxes = []
        
        for pred, target in zip(self.all_predictions, self.all_targets):
            pred_boundaries = pred['boundaries']
            pred_confidence = pred['confidence']
            target_boundaries = target['boundaries']
            
            if pred_confidence is not None:
                sorted_indices = np.argsort(-pred_confidence)
                pred_boundaries = pred_boundaries[sorted_indices]
                pred_confidence = pred_confidence[sorted_indices]
            
            for j, pred_box in enumerate(pred_boundaries):
                conf = pred_confidence[j] if pred_confidence is not None else 1.0
                all_pred_boxes.append({
                    'boundary': pred_box,
                    'confidence': conf,
                    'video_idx': len(all_target_boxes)
                })
            
            for target_box in target_boundaries:
                all_target_boxes.append({
                    'boundary': target_box,
                    'video_idx': len(all_target_boxes)
                })
        
        if len(all_pred_boxes) == 0 or len(all_target_boxes) == 0:
            return 0.0
        
        all_pred_boxes = sorted(all_pred_boxes, key=lambda x: x['confidence'], reverse=True)
        
        tp = np.zeros(len(all_pred_boxes))
        fp = np.zeros(len(all_pred_boxes))
        
        matched_targets = set()
        
        for i, pred_box in enumerate(all_pred_boxes):
            pred_boundary = pred_box['boundary']
            pred_video_idx = pred_box['video_idx']
            
            max_iou = 0.0
            max_target_idx = -1
            
            for j, target_box in enumerate(all_target_boxes):
                if target_box['video_idx'] != pred_video_idx:
                    continue
                
                target_boundary = target_box['boundary']
                iou = self.compute_iou(pred_boundary, target_boundary)
                
                if iou > max_iou:
                    max_iou = iou
                    max_target_idx = j
            
            if max_iou >= iou_threshold and max_target_idx not in matched_targets:
                tp[i] = 1
                matched_targets.add(max_target_idx)
            else:
                fp[i] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(all_target_boxes)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        precisions = np.concatenate(([0], precisions, [0]))
        recalls = np.concatenate(([0], recalls, [1]))
        
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])
        
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        
        return ap
    
    def compute_ar(self, num_proposals: int):
        total_recalls = []
        
        for pred, target in zip(self.all_predictions, self.all_targets):
            pred_boundaries = pred['boundaries'][:num_proposals]
            pred_confidence = pred['confidence']
            target_boundaries = target['boundaries']
            
            if len(target_boundaries) == 0:
                continue
            
            if pred_confidence is not None:
                sorted_indices = np.argsort(-pred_confidence[:num_proposals])
                pred_boundaries = pred_boundaries[sorted_indices]
            
            matched = 0
            matched_targets = set()
            
            for pred_box in pred_boundaries:
                for j, target_box in enumerate(target_boundaries):
                    if j in matched_targets:
                        continue
                    
                    iou = self.compute_iou(pred_box, target_box)
                    if iou >= 0.5:
                        matched += 1
                        matched_targets.add(j)
                        break
            
            recall = matched / len(target_boundaries)
            total_recalls.append(recall)
        
        if len(total_recalls) == 0:
            return 0.0
        
        ar = np.mean(total_recalls)
        return ar
    
    def compute(self):
        if len(self.all_predictions) == 0:
            return {}
        
        metrics = {}
        
        ap_values = []
        for iou_thresh in self.iou_thresholds:
            ap = self.compute_ap(iou_thresh)
            metrics[f'AP@{iou_thresh}'] = ap
            ap_values.append(ap)
        
        metrics['mAP'] = np.mean(ap_values)
        
        for num_proposals in self.proposal_counts:
            ar = self.compute_ar(num_proposals)
            metrics[f'AR@{num_proposals}'] = ar
        
        return metrics
    
    def print_metrics(self, metrics: Dict, prefix: str = ""):
        print(f"{prefix}Localization Metrics:")
        
        if 'mAP' in metrics:
            print(f"  mAP:             {metrics['mAP']:.4f}")
        
        for iou_thresh in self.iou_thresholds:
            key = f'AP@{iou_thresh}'
            if key in metrics:
                print(f"  {key}:          {metrics[key]:.4f}")
        
        for num_proposals in self.proposal_counts:
            key = f'AR@{num_proposals}'
            if key in metrics:
                print(f"  {key}:          {metrics[key]:.4f}")


class SegmentMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.all_labels = []
        self.all_preds = []
    
    def update(self, labels, preds):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        
        self.all_labels.extend(labels.flatten().tolist())
        self.all_preds.extend(preds.flatten().tolist())
    
    def compute(self):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        labels = np.array(self.all_labels)
        preds = np.array(self.all_preds)
        
        metrics = {
            'segment_accuracy': accuracy_score(labels, preds),
            'segment_f1': f1_score(labels, preds, average='binary', zero_division=0),
            'segment_precision': precision_score(labels, preds, average='binary', zero_division=0),
            'segment_recall': recall_score(labels, preds, average='binary', zero_division=0)
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict, prefix: str = ""):
        print(f"{prefix}Segment-Level Metrics:")
        print(f"  Accuracy:        {metrics['segment_accuracy']:.4f}")
        print(f"  F1-Score:        {metrics['segment_f1']:.4f}")
        print(f"  Precision:       {metrics['segment_precision']:.4f}")
        print(f"  Recall:          {metrics['segment_recall']:.4f}")
