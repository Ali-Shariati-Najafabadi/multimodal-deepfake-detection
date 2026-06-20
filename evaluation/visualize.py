import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict, Tuple
import json


class DeepfakeVisualizer:
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_timeline(
        self,
        video_id: str,
        duration: float,
        pred_boundaries: List[Tuple[float, float]],
        pred_confidence: List[float],
        target_boundaries: List[Tuple[float, float]] = None,
        segment_predictions: np.ndarray = None,
        segment_duration: float = 1.0,
        save_path: str = None
    ):
        fig, axes = plt.subplots(3, 1, figsize=(15, 8))
        
        ax1 = axes[0]
        ax1.set_xlim(0, duration)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Confidence')
        ax1.set_title(f'Predicted Fake Segments - {video_id}')
        ax1.grid(True, alpha=0.3)
        
        for i, ((start, end), conf) in enumerate(zip(pred_boundaries, pred_confidence)):
            color = 'red' if conf > 0.5 else 'orange'
            alpha = min(conf, 0.8)
            rect = patches.Rectangle(
                (start, 0), end - start, 1,
                linewidth=2, edgecolor=color, facecolor=color, alpha=alpha
            )
            ax1.add_patch(rect)
            
            if conf > 0.5:
                ax1.text(
                    (start + end) / 2, 0.5, f'{conf:.2f}',
                    ha='center', va='center', color='white', fontweight='bold'
                )
        
        ax2 = axes[1]
        ax2.set_xlim(0, duration)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_title('Ground Truth Fake Segments')
        ax2.grid(True, alpha=0.3)
        
        if target_boundaries is not None:
            for start, end in target_boundaries:
                rect = patches.Rectangle(
                    (start, 0), end - start, 1,
                    linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.5
                )
                ax2.add_patch(rect)
        
        ax3 = axes[2]
        ax3.set_xlim(0, duration)
        ax3.set_ylim(0, 1)
        ax3.set_xlabel('Time (seconds)')
        ax3.set_title('Segment-Level Predictions')
        ax3.grid(True, alpha=0.3)
        
        if segment_predictions is not None:
            num_segments = len(segment_predictions)
            for i, pred in enumerate(segment_predictions):
                start = i * segment_duration
                end = start + segment_duration
                
                color = 'red' if pred == 1 else 'green'
                rect = patches.Rectangle(
                    (start, 0), segment_duration, 1,
                    linewidth=1, edgecolor='black', facecolor=color, alpha=0.4
                )
                ax3.add_patch(rect)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"{video_id}_timeline.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def visualize_confidence_distribution(
        self,
        confidences: List[float],
        save_path: str = None
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(confidences, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
        
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Prediction Confidences')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = self.output_dir / "confidence_distribution.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def visualize_iou_distribution(
        self,
        iou_scores: List[float],
        save_path: str = None
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(iou_scores, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='IoU=0.5')
        
        ax.set_xlabel('IoU Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of IoU Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = self.output_dir / "iou_distribution.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def save_predictions_json(
        self,
        video_id: str,
        duration: float,
        pred_boundaries: List[Tuple[float, float]],
        pred_confidence: List[float],
        target_boundaries: List[Tuple[float, float]] = None,
        segment_predictions: np.ndarray = None,
        save_path: str = None
    ):
        data = {
            'video_id': video_id,
            'duration': duration,
            'predictions': {
                'boundaries': [
                    {'start': float(start), 'end': float(end), 'confidence': float(conf)}
                    for (start, end), conf in zip(pred_boundaries, pred_confidence)
                ],
                'segments': segment_predictions.tolist() if segment_predictions is not None else None
            }
        }
        
        if target_boundaries is not None:
            data['ground_truth'] = {
                'boundaries': [
                    {'start': float(start), 'end': float(end)}
                    for start, end in target_boundaries
                ]
            }
        
        if save_path is None:
            save_path = self.output_dir / f"{video_id}_predictions.json"
        else:
            save_path = Path(save_path)
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(save_path)
    
    def create_summary_report(
        self,
        metrics: Dict,
        num_videos: int,
        save_path: str = None
    ):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        ax1 = axes[0, 0]
        classification_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'uar']
        values = [metrics.get(m, 0) for m in classification_metrics]
        ax1.bar(classification_metrics, values, color='steelblue', alpha=0.7)
        ax1.set_ylabel('Score')
        ax1.set_title('Video-Level Classification Metrics')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(values):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        ax2 = axes[0, 1]
        ap_metrics = []
        ap_values = []
        for key, value in metrics.items():
            if key.startswith('AP@'):
                ap_metrics.append(key)
                ap_values.append(value)
        
        if ap_values:
            ax2.bar(ap_metrics, ap_values, color='coral', alpha=0.7)
            ax2.set_ylabel('Score')
            ax2.set_title('Average Precision at Different IoU Thresholds')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3, axis='y')
            
            for i, v in enumerate(ap_values):
                ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        ax3 = axes[1, 0]
        ar_metrics = []
        ar_values = []
        for key, value in metrics.items():
            if key.startswith('AR@'):
                ar_metrics.append(key)
                ar_values.append(value)
        
        if ar_values:
            ax3.bar(ar_metrics, ar_values, color='lightgreen', alpha=0.7)
            ax3.set_ylabel('Score')
            ax3.set_title('Average Recall at Different Proposal Counts')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3, axis='y')
            
            for i, v in enumerate(ar_values):
                ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        ax4 = axes[1, 1]
        segment_metrics = ['segment_accuracy', 'segment_f1', 'segment_precision', 'segment_recall']
        seg_values = [metrics.get(m, 0) for m in segment_metrics]
        labels = ['Accuracy', 'F1', 'Precision', 'Recall']
        ax4.bar(labels, seg_values, color='mediumpurple', alpha=0.7)
        ax4.set_ylabel('Score')
        ax4.set_title('Segment-Level Metrics')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(seg_values):
            ax4.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        fig.suptitle(f'Deepfake Detection Summary Report\nEvaluated on {num_videos} videos', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "summary_report.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)


def visualize_batch_predictions(
    visualizer: DeepfakeVisualizer,
    batch_data: Dict,
    num_samples: int = 10
):
    saved_paths = []
    
    for i in range(min(num_samples, len(batch_data['video_ids']))):
        video_id = batch_data['video_ids'][i]
        duration = batch_data['durations'][i]
        pred_boundaries = batch_data['pred_boundaries'][i]
        pred_confidence = batch_data['pred_confidence'][i]
        target_boundaries = batch_data.get('target_boundaries', [None] * len(batch_data['video_ids']))[i]
        segment_predictions = batch_data.get('segment_predictions', [None] * len(batch_data['video_ids']))[i]
        segment_duration = batch_data.get('segment_duration', 1.0)
        
        path = visualizer.visualize_timeline(
            video_id=video_id,
            duration=duration,
            pred_boundaries=pred_boundaries,
            pred_confidence=pred_confidence,
            target_boundaries=target_boundaries,
            segment_predictions=segment_predictions,
            segment_duration=segment_duration
        )
        saved_paths.append(path)
        
        json_path = visualizer.save_predictions_json(
            video_id=video_id,
            duration=duration,
            pred_boundaries=pred_boundaries,
            pred_confidence=pred_confidence,
            target_boundaries=target_boundaries,
            segment_predictions=segment_predictions
        )
    
    return saved_paths
