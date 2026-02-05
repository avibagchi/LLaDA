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


def load_normalized_scores(json_file, max_prompts=None):
    """Load normalized scores from a JSON evaluation file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    if max_prompts is not None:
        results = results[:max_prompts]
    
    normalized_scores = []
    for result in results:
        watermark_metrics = result.get('watermark_metrics', {})
        # Try both 'normalized_score' and 'z_score' for compatibility
        score = watermark_metrics.get('normalized_score') or watermark_metrics.get('z_score')
        if score is not None:
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
        default='/work/nvme/bemc/abagchi2/LLaDA/water-bench-results/json-outputs/gpt4-outputs/2000_no_seed43_gpt4_eval.json',
        help='Path to non-watermarked evaluation JSON file'
    )
    parser.add_argument(
        '--watermark_file',
        type=str,
        default='/work/nvme/bemc/abagchi2/LLaDA/water-bench-results/json-outputs/gpt4-outputs/2000_tau_abl_seed43_gpt4_eval.json',
        help='Path to watermarked evaluation JSON file'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='water-bench-results/graphs/normalized_score_threshold_analysis.pdf',
        help='Path to output graph file'
    )
    parser.add_argument(
        '--tau_min',
        type=float,
        default=0.75,
        help='Minimum threshold value (default: 0.75)'
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
        default=0.05,
        help='Step size for threshold values (default: 0.05)'
    )
    parser.add_argument(
        '--tau',
        type=float,
        default=None,
        help='Fixed threshold value τ to mark on the plot (default: auto-find 1% soundness threshold)'
    )
    parser.add_argument(
        '--max_prompts',
        type=int,
        default=100,
        help='Maximum number of prompts to consider (default: 1000)'
    )
    
    args = parser.parse_args()
    
    # Load normalized scores
    print(f"Loading non-watermarked scores from: {args.no_watermark_file} (max {args.max_prompts} prompts)")
    no_watermark_scores = load_normalized_scores(args.no_watermark_file, max_prompts=args.max_prompts)
    print(f"  Loaded {len(no_watermark_scores)} scores")
    print(f"  Min: {np.min(no_watermark_scores):.4f}, Max: {np.max(no_watermark_scores):.4f}, Mean: {np.mean(no_watermark_scores):.4f}")
    
    print(f"\nLoading watermarked scores from: {args.watermark_file} (max {args.max_prompts} prompts)")
    watermark_scores = load_normalized_scores(args.watermark_file, max_prompts=args.max_prompts)
    print(f"  Loaded {len(watermark_scores)} scores")
    print(f"  Min: {np.min(watermark_scores):.4f}, Max: {np.max(watermark_scores):.4f}, Mean: {np.mean(watermark_scores):.4f}")
    
    # Determine threshold range
    if args.tau_max is None:
        all_scores = np.concatenate([no_watermark_scores, watermark_scores])
        tau_max = min(np.max(all_scores) + 0.1, 2.1)  # Cap at ~2.1 to match reference
    else:
        tau_max = args.tau_max
    
    # Generate threshold values for plotting (normal resolution)
    tau_values = np.arange(args.tau_min, tau_max + args.tau_step, args.tau_step)
    
    # Calculate percentages for each threshold
    no_watermark_percentages = []
    watermark_percentages = []
    
    for tau in tau_values:
        no_wm_pct = calculate_percentage_above_threshold(no_watermark_scores, tau)
        wm_pct = calculate_percentage_above_threshold(watermark_scores, tau)
        no_watermark_percentages.append(no_wm_pct)
        watermark_percentages.append(wm_pct)
    
    # Determine which threshold to use
    if args.tau is not None:
        # Use fixed tau value
        tau_at_1pct = args.tau
        no_wm_pct_at_1pct = calculate_percentage_above_threshold(no_watermark_scores, tau_at_1pct)
        watermark_pct_at_1pct = calculate_percentage_above_threshold(watermark_scores, tau_at_1pct)
    else:
        # Find exact threshold where no watermark is at 1% using fine resolution
        fine_tau_step = 0.001  # Fine step for finding exact 1% threshold
        fine_tau_values = np.arange(args.tau_min, tau_max + fine_tau_step, fine_tau_step)
        
        # Find the threshold closest to exactly 1%
        best_tau = None
        best_diff = float('inf')
        best_no_wm_pct = None
        best_wm_pct = None
        
        for tau in fine_tau_values:
            no_wm_pct = calculate_percentage_above_threshold(no_watermark_scores, tau)
            diff = abs(no_wm_pct - 1.0)
            if diff < best_diff:
                best_diff = diff
                best_tau = tau
                best_no_wm_pct = no_wm_pct
                best_wm_pct = calculate_percentage_above_threshold(watermark_scores, tau)
        
        tau_at_1pct = best_tau
        no_wm_pct_at_1pct = best_no_wm_pct
        watermark_pct_at_1pct = best_wm_pct
    
    # Create the plot with reference style
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors matching the reference
    light_blue = '#87CEEB'  # Light blue for No Watermark
    brick_red = '#CD5C5C'   # Brick red/reddish-brown for Watermarked
    
    # Plot lines with markers matching reference style
    ax.plot(tau_values, no_watermark_percentages, color=light_blue, linewidth=2, 
            label='No Watermark', marker='o', markersize=6, markeredgecolor='black', 
            markeredgewidth=0.5, markerfacecolor=light_blue)
    ax.plot(tau_values, watermark_percentages, color=brick_red, linewidth=2, 
            label='Watermarked', marker='s', markersize=6, markeredgecolor='black', 
            markeredgewidth=0.5, markerfacecolor=brick_red)
    
    # Add vertical dotted line at the selected threshold
    ax.axvline(x=tau_at_1pct, color='green', linestyle=':', linewidth=4.5, alpha=0.7)
    
    # Add annotation for this threshold (positioned near x-axis)
    ax.text(tau_at_1pct, 5, 
            f'τ = {tau_at_1pct:.2f}\n({no_wm_pct_at_1pct:.2f}% FP)', 
            fontsize=16, ha='left', va='bottom', rotation=0,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='grey'))
    
    if args.tau is not None:
        print(f"\nUsing fixed threshold: τ = {tau_at_1pct:.4f}")
    else:
        print(f"\nThreshold where no watermark is exactly 1% (fine resolution): τ = {tau_at_1pct:.4f}")
    print(f"  No Watermark % above this threshold: {no_wm_pct_at_1pct:.2f}%")
    print(f"  Watermarked % above this threshold: {watermark_pct_at_1pct:.2f}%")
    
    # Set labels matching reference style
    ax.set_xlabel('Detection Threshold (τ)', fontsize=16)
    ax.set_ylabel('Percentage of Prompts with Normalized Score > τ', fontsize=15)
    
    # Auto-generate title with number of prompts used
    num_prompts = len(watermark_scores)
    
    ax.set_title(f'Detection Rate vs Threshold for {num_prompts} Open-Ended Prompts', 
                fontsize=16, fontweight='bold')
    
    # Legend in top-right with white background and border
    legend = ax.legend(loc='upper right', fontsize=11, frameon=True, 
                      fancybox=False, framealpha=1.0, edgecolor='black', 
                      facecolor='white')
    legend.get_frame().set_linewidth(1.0)
    
    # Light grey grid (both horizontal and vertical)
    ax.grid(True, alpha=0.3, color='lightgrey', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set axis limits
    ax.set_xlim(args.tau_min, tau_max)
    ax.set_ylim(0, 100)
    
    # Set axis ticks to match reference style (adjust if tau_max is different)
    if tau_max >= 2.0:
        ax.set_xticks(np.arange(0.8, 2.1, 0.2))
    else:
        # Auto-generate ticks if range is smaller
        ax.set_xticks(np.arange(args.tau_min, tau_max + 0.1, 0.2))
    ax.set_yticks(np.arange(0, 101, 20))
    
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
