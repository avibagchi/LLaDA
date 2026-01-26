#!/usr/bin/env python3
"""
Plot ROC curve for watermark detection using normalized scores.
Shows the trade-off between True Positive Rate (TPR) and False Positive Rate (FPR).
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc


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


def main():
    parser = argparse.ArgumentParser(
        description='Plot ROC curve for watermark detection'
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
        default='water-bench-results/graphs/roc_curve.png',
        help='Path to output graph file'
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
    
    # Prepare data for ROC curve
    # y_true: 0 for no watermark, 1 for watermark
    # y_scores: normalized scores
    y_true = np.concatenate([
        np.zeros(len(no_watermark_scores)),  # 0 = no watermark
        np.ones(len(watermark_scores))        # 1 = watermark
    ])
    y_scores = np.concatenate([no_watermark_scores, watermark_scores])
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (Youden's J statistic: maximizes TPR - FPR)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    
    print(f"\n{'='*60}")
    print("ROC Curve Statistics:")
    print(f"{'='*60}")
    print(f"AUC (Area Under Curve): {roc_auc:.4f}")
    print(f"\nOptimal Threshold (Youden's J):")
    print(f"  Threshold (τ): {optimal_threshold:.4f}")
    print(f"  True Positive Rate (TPR): {optimal_tpr:.4f} ({optimal_tpr*100:.2f}%)")
    print(f"  False Positive Rate (FPR): {optimal_fpr:.4f} ({optimal_fpr*100:.2f}%)")
    print(f"  Youden's J (TPR - FPR): {youden_j[optimal_idx]:.4f}")
    print(f"{'='*60}")
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, 'b-', linewidth=2, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier (AUC = 0.50)')
    
    # Mark optimal threshold point
    plt.plot(optimal_fpr, optimal_tpr, 'ro', markersize=10, 
             label=f'Optimal Threshold (τ={optimal_threshold:.3f})')
    
    # Add annotation for optimal point
    plt.annotate(f'τ={optimal_threshold:.3f}\nTPR={optimal_tpr:.3f}\nFPR={optimal_fpr:.3f}',
                xy=(optimal_fpr, optimal_tpr),
                xytext=(20, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.xlabel('False Positive Rate (FPR)\n(Percentage of Non-Watermarked Classified as Watermarked)', 
               fontsize=12)
    plt.ylabel('True Positive Rate (TPR)\n(Percentage of Watermarked Correctly Detected)', 
               fontsize=12)
    plt.title('ROC Curve for Aaronson Watermark Detection', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Add some statistics as text
    stats_text = f'No Watermark: Mean={np.mean(no_watermark_scores):.3f}, Std={np.std(no_watermark_scores):.3f}\n'
    stats_text += f'Aaronson: Mean={np.mean(watermark_scores):.3f}, Std={np.std(watermark_scores):.3f}\n'
    stats_text += f'AUC: {roc_auc:.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    plt.savefig(args.output_file, dpi=300, bbox_inches='tight')
    print(f"\nROC curve saved to: {args.output_file}")
    
    # Print detailed statistics
    print("\n" + "="*60)
    print("Detailed Statistics:")
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
    print(f"\nAt optimal threshold τ={optimal_threshold:.4f}:")
    print(f"  True Positives (TP): {int(optimal_tpr * len(watermark_scores))} / {len(watermark_scores)}")
    print(f"  False Positives (FP): {int(optimal_fpr * len(no_watermark_scores))} / {len(no_watermark_scores)}")
    print(f"  True Negatives (TN): {int((1 - optimal_fpr) * len(no_watermark_scores))} / {len(no_watermark_scores)}")
    print(f"  False Negatives (FN): {int((1 - optimal_tpr) * len(watermark_scores))} / {len(watermark_scores)}")
    print("="*60)


if __name__ == "__main__":
    main()
