import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir):
    results_dir = Path(results_dir)
    
    all_results = {}
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        metrics_file = model_dir / "best_metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            all_results[model_name] = metrics
    
    return all_results


def create_comparison_table(results):
    video_metrics = ['accuracy', 'uar', 'f1', 'auc']
    segment_metrics = ['segment_accuracy', 'segment_f1']
    localization_metrics = ['mAP', 'AP@0.5', 'AR@10']
    
    data = []
    for model, metrics in results.items():
        row = {'Audio Model': model}
        
        for metric in video_metrics:
            row[f'Video_{metric.upper()}'] = metrics.get(metric, 0.0)
        
        for metric in segment_metrics:
            key = metric.replace('segment_', '').upper()
            row[f'Segment_{key}'] = metrics.get(metric, 0.0)
        
        for metric in localization_metrics:
            row[metric] = metrics.get(metric, 0.0)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.sort_values('mAP', ascending=False)
    
    return df


def visualize_comparison(results, output_path):
    models = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    ax1 = axes[0, 0]
    video_metrics = ['accuracy', 'uar', 'f1', 'auc']
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(video_metrics):
        values = [results[m].get(metric, 0) for m in models]
        ax1.bar(x + i*width, values, width, label=metric.upper(), alpha=0.8)
    
    ax1.set_xlabel('Audio Model')
    ax1.set_ylabel('Score')
    ax1.set_title('Video-Level Classification Metrics')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1)
    
    ax2 = axes[0, 1]
    segment_metrics = ['segment_accuracy', 'segment_f1']
    colors = ['steelblue', 'coral']
    
    x = np.arange(len(models))
    width = 0.35
    
    for i, metric in enumerate(segment_metrics):
        values = [results[m].get(metric, 0) for m in models]
        label = metric.replace('segment_', '').upper()
        ax2.bar(x + i*width, values, width, label=label, color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Audio Model')
    ax2.set_ylabel('Score')
    ax2.set_title('Segment-Level Metrics')
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)
    
    ax3 = axes[1, 0]
    ap_metrics = ['AP@0.3', 'AP@0.5', 'AP@0.7']
    x = np.arange(len(models))
    width = 0.25
    
    for i, metric in enumerate(ap_metrics):
        values = [results[m].get(metric, 0) for m in models]
        ax3.bar(x + i*width, values, width, label=metric, alpha=0.8)
    
    ax3.set_xlabel('Audio Model')
    ax3.set_ylabel('Average Precision')
    ax3.set_title('Localization: AP at Different IoU Thresholds')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1)
    
    ax4 = axes[1, 1]
    map_values = [results[m].get('mAP', 0) for m in models]
    colors_map = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = ax4.bar(models, map_values, color=colors_map, alpha=0.8, edgecolor='black')
    
    ax4.set_xlabel('Audio Model')
    ax4.set_ylabel('mAP')
    ax4.set_title('Mean Average Precision (mAP)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 1)
    
    for bar, val in zip(bars, map_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {output_path}")


def print_summary(df):
    print("\n" + "="*100)
    print("AUDIO MODEL COMPARISON - SUMMARY")
    print("="*100 + "\n")
    
    print(df.to_string(index=False))
    print("\n")
    
    best_model = df.iloc[0]['Audio Model']
    best_map = df.iloc[0]['mAP']
    
    print(f"🏆 Best Audio Model: {best_model} (mAP: {best_map:.4f})")
    print("\n" + "="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare Audio Model Results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing results from all audio models')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Directory to save comparison outputs')
    
    args = parser.parse_args()
    
    results = load_results(args.results_dir)
    
    if not results:
        print("No results found!")
        return
    
    df = create_comparison_table(results)
    
    print_summary(df)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "audio_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    plot_path = output_dir / "audio_comparison.png"
    visualize_comparison(results, plot_path)


if __name__ == '__main__':
    main()
