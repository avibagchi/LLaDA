#!/usr/bin/env python3
"""
Create a box plot comparing normalized watermark detection scores for 
watermarked vs non-watermarked text, similar to the reference figure.
"""
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_normalized_scores(json_file, max_prompts=None):
    """Extract normalized scores from a JSON evaluation file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    if max_prompts is not None:
        results = results[:max_prompts]
    
    scores = []
    for result in results:
        watermark_metrics = result.get('watermark_metrics', {})
        # Try both 'normalized_score' and 'z_score' for compatibility
        score = watermark_metrics.get('normalized_score') or watermark_metrics.get('z_score')
        if score is not None:
            scores.append(score)
    
    return scores


def create_boxplot(no_watermark_file, watermark_file, output_file=None, title=None, max_prompts=None):
    """
    Create a box plot comparing normalized scores for watermarked vs non-watermarked text.
    
    Args:
        no_watermark_file: Path to JSON file with non-watermarked results
        watermark_file: Path to JSON file with watermarked results
        output_file: Path to save the plot (default: watermark_boxplot.png)
        title: Custom title for the plot (default: auto-generated)
        max_prompts: Maximum number of prompts to consider (default: 1000)
    """
    # Load scores from both files
    no_watermark_scores = load_normalized_scores(no_watermark_file, max_prompts=max_prompts)
    watermark_scores = load_normalized_scores(watermark_file, max_prompts=max_prompts)
    
    print(f"Loaded {len(no_watermark_scores)} scores from no-watermark file")
    print(f"Loaded {len(watermark_scores)} scores from watermark file")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare data for box plot
    data_to_plot = [no_watermark_scores, watermark_scores]
    labels = ['No Watermark', 'Watermarked']
    
    # Create box plot
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                    showmeans=False, showfliers=True)
    
    # Customize box plot colors
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize median line
    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2)
    
    # Customize whiskers and caps
    for element in ['whiskers', 'caps', 'fliers']:
        for item in bp[element]:
            item.set_color('black')
            item.set_linewidth(1)
    
    # Set labels and title
    ax.set_ylabel('Normalized Score', fontsize=16)
    ax.set_xlabel('')
    ax.tick_params(axis='both', labelsize=14)

    if title is None:
        num_prompts = len(watermark_scores)
        title = f'Watermark Detection Scores for {num_prompts} Open-Ended Prompts'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    # Set y-axis limits with some padding
    all_scores = no_watermark_scores + watermark_scores
    y_min = min(all_scores) * 0.95
    y_max = max(all_scores) * 1.05
    ax.set_ylim(y_min, y_max)
    
    # Add minor grid lines
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.grid(True, which='minor', alpha=0.2, linestyle=':', axis='y')
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Box plot saved to {output_file}")
    else:
        plt.show()
    
    # Print statistics
    print("\nStatistics:")
    print(f"No Watermark - Mean: {np.mean(no_watermark_scores):.4f}, "
          f"Median: {np.median(no_watermark_scores):.4f}, "
          f"Std: {np.std(no_watermark_scores):.4f}")
    print(f"Watermarked - Mean: {np.mean(watermark_scores):.4f}, "
          f"Median: {np.median(watermark_scores):.4f}, "
          f"Std: {np.std(watermark_scores):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Create a box plot comparing normalized watermark detection scores'
    )
    parser.add_argument(
        '--no_watermark_file',
        type=str,
        default='/work/nvme/bemc/abagchi2/LLaDA/water-bench-results/json-outputs/no_2000_aaronson.json',
        help='Path to non-watermarked evaluation JSON file'
    )
    parser.add_argument(
        '--watermark_file',
        type=str,
        default='/work/nvme/bemc/abagchi2/LLaDA/water-bench-results/json-outputs/2000_aaronson.json',
        help='Path to watermarked evaluation JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='water-bench-results/graphs/watermark_boxplot.pdf',
        help='Output path for the box plot image'
    )
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Custom title for the plot (default: auto-generated)'
    )
    parser.add_argument(
        '--max_prompts',
        type=int,
        default=1000,
        help='Maximum number of prompts to consider (default: 1000)'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_boxplot(
        args.no_watermark_file,
        args.watermark_file,
        output_file=args.output,
        title=args.title,
        max_prompts=args.max_prompts
    )


if __name__ == '__main__':
    main()
