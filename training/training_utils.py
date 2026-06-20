import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from pathlib import Path
import json


class MetricsCalculator:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.all_labels = []
        self.all_preds = []
        self.all_probs = []
    
    def update(self, labels, preds, probs=None):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if probs is not None and isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        self.all_labels.extend(labels.tolist())
        self.all_preds.extend(preds.tolist())
        if probs is not None:
            self.all_probs.extend(probs.tolist())
    
    def compute(self):
        labels = np.array(self.all_labels)
        preds = np.array(self.all_preds)
        
        unique_classes = np.unique(labels)
        is_binary = len(unique_classes) == 2
        avg_mode = 'binary' if is_binary else 'macro'
        
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average=avg_mode, zero_division=0),
            'recall': recall_score(labels, preds, average=avg_mode, zero_division=0),
            'f1': f1_score(labels, preds, average=avg_mode, zero_division=0),
        }
        
        try:
            recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
            metrics['uar'] = float(np.mean(recall_per_class))
        except:
            metrics['uar'] = 0.0
        
        metrics['wa_f1'] = f1_score(labels, preds, average='weighted', zero_division=0)
        
        if len(self.all_probs) > 0:
            probs = np.array(self.all_probs)
            try:
                if len(np.unique(labels)) > 1:
                    if is_binary:
                        if probs.ndim == 2:
                            probs = probs[:, 1]
                        metrics['auc'] = roc_auc_score(labels, probs)
                    else:
                        from sklearn.preprocessing import label_binarize
                        
                        labels_bin = label_binarize(labels, classes=unique_classes)
                        
                        auc_scores = []
                        for i in range(len(unique_classes)):
                            try:
                                if labels_bin[:, i].sum() > 0:
                                    auc = roc_auc_score(labels_bin[:, i], probs[:, i])
                                    auc_scores.append(auc)
                            except:
                                pass
                        
                        if len(auc_scores) > 0:
                            metrics['auc'] = float(np.mean(auc_scores))
                        else:
                            metrics['auc'] = 0.0
                else:
                    metrics['auc'] = 0.0
            except Exception as e:
                print(f"  Warning: Could not compute AUC: {e}")
                metrics['auc'] = 0.0
        else:
            metrics['auc'] = 0.0
        
        cm = confusion_matrix(labels, preds)
        metrics['confusion_matrix'] = cm.tolist()
        
        if not is_binary:
            report = classification_report(labels, preds, output_dict=True, zero_division=0)
            
            for class_id in unique_classes:
                class_key = str(int(class_id))
                if class_key in report:
                    metrics[f'precision_class_{class_id}'] = report[class_key]['precision']
                    metrics[f'recall_class_{class_id}'] = report[class_key]['recall']
                    metrics[f'f1_class_{class_id}'] = report[class_key]['f1-score']
        
        if is_binary and cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            recall_real = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            recall_fake = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics['recall_real'] = recall_real
            metrics['recall_fake'] = recall_fake
        
        return metrics

    
    def print_metrics(self, metrics, prefix=""):
        print(f"{prefix}Metrics:")
        print(f"  Accuracy (ACC):  {metrics['accuracy']*100:.2f}%")
        print(f"  UAR:             {metrics.get('uar', 0.0)*100:.2f}%")
        print(f"  WA-F1:           {metrics.get('wa_f1', 0.0)*100:.2f}%")
        
        if metrics.get('auc', 0.0) > 0:
            print(f"  AUC:             {metrics['auc']:.4f}")
        
        print(f"  Precision:       {metrics['precision']:.4f}")
        print(f"  Recall:          {metrics['recall']:.4f}")
        print(f"  F1-Score:        {metrics['f1']:.4f}")
        
        if 'recall_real' in metrics and 'recall_fake' in metrics:
            print(f"  Recall (Real):   {metrics['recall_real']:.4f}")
            print(f"  Recall (Fake):   {metrics['recall_fake']:.4f}")
        
        per_class_keys = [k for k in metrics.keys() if k.startswith('recall_class_')]
        if per_class_keys:
            print(f"\n  Per-Class Metrics:")
            num_classes = len(per_class_keys)
            for i in range(num_classes):
                if f'recall_class_{i}' in metrics:
                    print(f"    Class {i}:")
                    print(f"      Precision: {metrics.get(f'precision_class_{i}', 0.0):.4f}")
                    print(f"      Recall:    {metrics.get(f'recall_class_{i}', 0.0):.4f}")
                    print(f"      F1-Score:  {metrics.get(f'f1_class_{i}', 0.0):.4f}")
        
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            print(f"\n  Confusion Matrix ({cm.shape[0]}x{cm.shape[1]}):")
            for i in range(cm.shape[0]):
                print(f"    {cm[i].tolist()}")



class CheckpointManager:
    def __init__(
        self,
        save_dir: str,
        keep_last_n: int = 3,
        save_best_metric: str = 'auc',
        save_best_mode: str = 'max'
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_last_n = keep_last_n
        self.save_best_metric = save_best_metric
        self.save_best_mode = save_best_mode
        if save_best_mode == 'max':
            self.best_value = -float('inf')
            self.is_better = lambda x, y: x > y
        else:
            self.best_value = float('inf')
            self.is_better = lambda x, y: x < y
    
    def save_checkpoint(
        self,
        epoch: int,
        model,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        metrics: dict = None,
        is_best: bool = False
    ):
        if isinstance(model, dict):
            checkpoint = model
        else:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }
            
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            metrics_path = self.save_dir / 'best_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        checkpoints = sorted(
            self.save_dir.glob('checkpoint_epoch_*.pth'),
            key=lambda x: int(x.stem.split('_')[-1])
        )
        
        if len(checkpoints) > self.keep_last_n:
            for checkpoint in checkpoints[:-self.keep_last_n]:
                checkpoint.unlink()
    
    def should_save_best(self, metrics: dict) -> bool:
        if self.save_best_metric not in metrics:
            return False
        
        current_value = metrics[self.save_best_metric]
        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            return True
        return False
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler = None
    ):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        
        return epoch, metrics


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count