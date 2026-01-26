#!/usr/bin/env python3
"""
Plot percentage of prompts with normalized_score > tau vs threshold tau
for watermarked and non-watermarked outputs.
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_normalized_scores(json_file):
    """Load normalized scores from a JSON evaluation file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    normalized_scores = []
    for result in data.get('results', []):
        if 'watermark_metrics' in result and 'normalized_score' in result['watermark_metrics']:
            score = result['watermark_metrics']['normalized_score']
            normalized_scores.append(score)
    
    return np.array(normalized_scores)


def calculate_percentage_above_threshold(scores, threshold):
    """Calculate percentage of scores above a given threshold."""
    if len(scores) == 0:
        return 0.0
    return 100.0 * np.sum(scores > threshold) / len(scores)


def main():
    parser = argparse.ArgumentParser(
        description='Plot normalized score threshold analysis'
    )
    parser.add_argument(
        '--no_watermark_file',
        type=str,
        default='water-bench-results/json-outputs/gpt4-outputs/no-watermark/no_watermark_gpt4_eval.json',
        help='Path to non-watermarked evaluation JSON file'
    )
    parser.add_argument(
        '--watermark_file',
        type=str,
        default='water-bench-results/json-outputs/gpt4-outputs/with-watermark/aaronson/aaronson_gpt4_eval.json',
        help='Path to watermarked evaluation JSON file'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='water-bench-results/graphs/normalized_score_threshold_analysis.png',
        help='Path to output graph file'
    )
    parser.add_argument(
        '--tau_min',
        type=float,
        default=0.0,
        help='Minimum threshold value (default: 0.0)'
    )
    parser.add_argument(
        '--tau_max',
        type=float,
        default=None,
        help='Maximum threshold value (default: auto-determined from data)'
    )
    parser.add_argument(
        '--tau_step',
        type=float,
        default=0.1,
        help='Step size for threshold values (default: 0.1)'
    )
    
    args = parser.parse_args()
    
    # Load normalized scores
    print(f"Loading non-watermarked scores from: {args.no_watermark_file}")
    no_watermark_scores = load_normalized_scores(args.no_watermark_file)
    print(f"  Loaded {len(no_watermark_scores)} scores")
    print(f"  Min: {np.min(no_watermark_scores):.4f}, Max: {np.max(no_watermark_scores):.4f}, Mean: {np.mean(no_watermark_scores):.4f}")
    
    print(f"\nLoading watermarked scores from: {args.watermark_file}")
    watermark_scores = load_normalized_scores(args.watermark_file)
    print(f"  Loaded {len(watermark_scores)} scores")
    print(f"  Min: {np.min(watermark_scores):.4f}, Max: {np.max(watermark_scores):.4f}, Mean: {np.mean(watermark_scores):.4f}")
    
    # Determine threshold range
    if args.tau_max is None:
        all_scores = np.concatenate([no_watermark_scores, watermark_scores])
        tau_max = np.max(all_scores) + 0.1
    else:
        tau_max = args.tau_max
    
    # Generate threshold values
    tau_values = np.arange(args.tau_min, tau_max + args.tau_step, args.tau_step)
    
    # Calculate percentages for each threshold
    no_watermark_percentages = []
    watermark_percentages = []
    differences = []
    
    for tau in tau_values:
        no_wm_pct = calculate_percentage_above_threshold(no_watermark_scores, tau)
        wm_pct = calculate_percentage_above_threshold(watermark_scores, tau)
        no_watermark_percentages.append(no_wm_pct)
        watermark_percentages.append(wm_pct)
        differences.append(wm_pct - no_wm_pct)  # Difference: watermarked - no_watermark
    
    # Find threshold where difference is maximized
    differences = np.array(differences)
    max_diff_idx = np.argmax(differences)
    max_diff_tau = tau_values[max_diff_idx]
    max_diff_value = differences[max_diff_idx]
    
    print(f"\nMaximum difference found at threshold τ = {max_diff_tau:.4f}")
    print(f"  Difference: {max_diff_value:.2f}%")
    print(f"  No Watermark % above threshold: {no_watermark_percentages[max_diff_idx]:.2f}%")
    print(f"  Watermarked % above threshold: {watermark_percentages[max_diff_idx]:.2f}%")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(tau_values, no_watermark_percentages, 'b-', linewidth=2, label='No Watermark', marker='o', markersize=3)
    plt.plot(tau_values, watermark_percentages, 'r-', linewidth=2, label='Aaronson Watermark', marker='s', markersize=3)
    
    # Add vertical line at maximum difference threshold
    plt.axvline(x=max_diff_tau, color='green', linestyle='--', linewidth=2, 
                label=f'Max Difference (τ={max_diff_tau:.3f})', alpha=0.7)
    
    # Add annotation at the maximum difference point
    max_diff_y = (no_watermark_percentages[max_diff_idx] + watermark_percentages[max_diff_idx]) / 2
    plt.annotate(f'τ={max_diff_tau:.3f}\nΔ={max_diff_value:.1f}%', 
                xy=(max_diff_tau, max_diff_y), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.xlabel('Threshold τ (normalized score)', fontsize=12)
    plt.ylabel('Percentage of Prompts with Normalized Score > τ (%)', fontsize=12)
    plt.title('Normalized Score Threshold Analysis', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(args.tau_min, tau_max)
    plt.ylim(0, 105)
    
    # Add some statistics as text
    stats_text = f'No Watermark: Mean={np.mean(no_watermark_scores):.3f}, Std={np.std(no_watermark_scores):.3f}\n'
    stats_text += f'Aaronson: Mean={np.mean(watermark_scores):.3f}, Std={np.std(watermark_scores):.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    plt.savefig(args.output_file, dpi=300, bbox_inches='tight')
    print(f"\nGraph saved to: {args.output_file}")
    
    # Print some key statistics
    print("\n" + "="*60)
    print("Key Statistics:")
    print("="*60)
    print(f"No Watermark:")
    print(f"  Total prompts: {len(no_watermark_scores)}")
    print(f"  Mean normalized score: {np.mean(no_watermark_scores):.4f}")
    print(f"  Std normalized score: {np.std(no_watermark_scores):.4f}")
    print(f"  Median normalized score: {np.median(no_watermark_scores):.4f}")
    print(f"\nAaronson Watermark:")
    print(f"  Total prompts: {len(watermark_scores)}")
    print(f"  Mean normalized score: {np.mean(watermark_scores):.4f}")
    print(f"  Std normalized score: {np.std(watermark_scores):.4f}")
    print(f"  Median normalized score: {np.median(watermark_scores):.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
