#!/usr/bin/env python3
"""
Plot GPT-4 evaluation metrics (style, consistency, accuracy, ethics) across different
watermark_steps values.
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_gpt4_metrics(json_file):
    """Extract average GPT-4 evaluation metrics from a JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    # Collect all scores
    style_scores = []
    consistency_scores = []
    accuracy_scores = []
    ethics_scores = []
    
    for result in results:
        scores = result.get('scores', {})
        if not scores:
            # Try gpt4_evaluation.scores if scores not at top level
            gpt4_eval = result.get('gpt4_evaluation', {})
            scores = gpt4_eval.get('scores', {})
        
        if scores:
            style = scores.get('style (setting ethics aside)')
            consistency = scores.get('consistency (setting ethics aside)')
            accuracy = scores.get('accuracy (setting ethics aside)')
            ethics = scores.get('ethics')
            
            if style is not None:
                style_scores.append(style)
            if consistency is not None:
                consistency_scores.append(consistency)
            if accuracy is not None:
                accuracy_scores.append(accuracy)
            if ethics is not None:
                ethics_scores.append(ethics)
    
    # Calculate averages
    metrics = {
        'style': np.mean(style_scores) if style_scores else 0,
        'consistency': np.mean(consistency_scores) if consistency_scores else 0,
        'accuracy': np.mean(accuracy_scores) if accuracy_scores else 0,
        'ethics': np.mean(ethics_scores) if ethics_scores else 0,
        'num_prompts': len(results)
    }
    
    return metrics


def extract_watermark_steps(filename):
    """Extract watermark_steps value from filename (e.g., '5_full_bench_aaronson_gpt4_eval.json' -> 5)."""
    filename = Path(filename).name
    # Extract number at the beginning of filename
    parts = filename.split('_')
    if parts:
        try:
            return int(parts[0])
        except ValueError:
            return None
    return None


def plot_metrics(data_dict, output_file=None):
    """
    Plot GPT-4 metrics as a grouped bar chart or line plot.
    
    Args:
        data_dict: Dictionary mapping watermark_steps -> metrics dict
        output_file: Path to save the plot
    """
    # Sort by watermark_steps
    sorted_steps = sorted([k for k in data_dict.keys() if k is not None])
    
    # Extract data
    steps = []
    style_avgs = []
    consistency_avgs = []
    accuracy_avgs = []
    ethics_avgs = []
    
    for step in sorted_steps:
        metrics = data_dict[step]
        steps.append(step)
        style_avgs.append(metrics['style'])
        consistency_avgs.append(metrics['consistency'])
        accuracy_avgs.append(metrics['accuracy'])
        ethics_avgs.append(metrics['ethics'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up x positions
    x = np.arange(len(steps))
    width = 0.2  # Width of bars
    
    # Create grouped bar chart
    bars1 = ax.bar(x - 1.5*width, style_avgs, width, label='Style', color='#87CEEB', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, consistency_avgs, width, label='Consistency', color='#CD5C5C', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, accuracy_avgs, width, label='Accuracy', color='#90EE90', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, ethics_avgs, width, label='Ethics', color='#FFD700', alpha=0.8)
    
    # Customize plot
    ax.set_xlabel(r'$t_{end}$', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title(r'Evaluation Metrics vs $t_{end}$', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in steps])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    ax.set_ylim(0, 10.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    for step in sorted_steps:
        metrics = data_dict[step]
        print(f"\nWatermark Steps: {step}")
        print(f"  Style:       {metrics['style']:.2f}")
        print(f"  Consistency: {metrics['consistency']:.2f}")
        print(f"  Accuracy:    {metrics['accuracy']:.2f}")
        print(f"  Ethics:      {metrics['ethics']:.2f}")
        print(f"  Prompts:     {metrics['num_prompts']}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot GPT-4 evaluation metrics across different watermark_steps'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='water-bench-results/json-outputs/gpt4-outputs',
        help='Directory containing GPT-4 evaluation JSON files'
    )
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        default=[5, 10, 20, 40, 80, 160, 300],
        help='List of watermark_steps values to plot (default: 5 10 20 40 80 160 300)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='water-bench-results/graphs/gpt4_metrics_by_steps.pdf',
        help='Output path for the plot'
    )
    parser.add_argument(
        '--file_pattern',
        type=str,
        default='{}_full_bench_aaronson_gpt4_eval.json',
        help='Filename pattern (use {} as placeholder for steps, default: {}_full_bench_aaronson_gpt4_eval.json)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    data_dict = {}
    
    # Load metrics for each watermark_steps value
    for steps in args.steps:
        filename = args.file_pattern.format(steps)
        filepath = input_dir / filename
        
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue
        
        print(f"Loading metrics from: {filepath}")
        metrics = load_gpt4_metrics(filepath)
        data_dict[steps] = metrics
        print(f"  Loaded {metrics['num_prompts']} prompts")
        print(f"  Style: {metrics['style']:.2f}, Consistency: {metrics['consistency']:.2f}, "
              f"Accuracy: {metrics['accuracy']:.2f}, Ethics: {metrics['ethics']:.2f}")
    
    if not data_dict:
        print("Error: No valid data files found!")
        return
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Plot the metrics
    plot_metrics(data_dict, output_file=args.output)


if __name__ == '__main__':
    main()
