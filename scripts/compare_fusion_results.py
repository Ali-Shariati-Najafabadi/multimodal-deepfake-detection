import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir):
    results_dir = Path(results_dir)
    
    all_results = {}
    
    for strategy_dir in results_dir.iterdir():
        if not strategy_dir.is_dir():
            continue
        
        strategy_name = strategy_dir.name
        metrics_file = strategy_dir / "best_metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            all_results[strategy_name] = metrics
    
    return all_results


def create_comparison_table(results):
    metrics_of_interest = ['accuracy', 'uar', 'f1', 'auc', 'precision', 'recall']
    
    data = []
    for strategy, metrics in results.items():
        row = {'Strategy': strategy}
        for metric in metrics_of_interest:
            row[metric.upper()] = metrics.get(metric, 0.0)
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.sort_values('UAR', ascending=False)
    
    return df


def visualize_comparison(results, output_path):
    strategies = list(results.keys())
    metrics = ['accuracy', 'uar', 'f1', 'auc']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        values = [results[s].get(metric, 0) for s in strategies]
        colors = plt.cm.viridis(np.linspace(0, 1, len(strategies)))
        
        bars = ax.bar(strategies, values, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Score')
        ax.set_title(f'{metric.upper()} Comparison')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
        
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {output_path}")


def print_summary(df):
    print("\n" + "="*80)
    print("FUSION STRATEGY COMPARISON - SUMMARY")
    print("="*80 + "\n")
    
    print(df.to_string(index=False))
    print("\n")
    
    best_strategy = df.iloc[0]['Strategy']
    best_uar = df.iloc[0]['UAR']
    
    print(f"🏆 Best Strategy: {best_strategy} (UAR: {best_uar:.4f})")
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare Fusion Strategy Results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing results from all fusion strategies')
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
    
    csv_path = output_dir / "fusion_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    plot_path = output_dir / "fusion_comparison.png"
    visualize_comparison(results, plot_path)


if __name__ == '__main__':
    main()
