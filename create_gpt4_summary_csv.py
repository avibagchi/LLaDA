#!/usr/bin/env python3
"""
Create a CSV file summarizing GPT-4 evaluation metrics for all processed files.
Each row represents one file with its average evaluation metrics.
"""
import json
import csv
import argparse
import re
from pathlib import Path
from glob import glob
import numpy as np


def extract_metrics_from_file(eval_file):
    """Extract metrics from a GPT-4 evaluation file."""
    try:
        with open(eval_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get GPT-4 evaluation metrics
        gpt4_metrics = data.get('gpt4_evaluation_metrics', {})
        
        # Get category averages
        category_averages = gpt4_metrics.get('category_averages', {})
        
        # Get overall average score
        overall_avg = gpt4_metrics.get('overall_average_score', 0)
        
        # Get average perplexity
        avg_perplexity = gpt4_metrics.get('average_perplexity', 0)
        
        # Calculate average z-score from watermark metrics
        results = data.get('results', [])
        z_scores = []
        for result in results:
            watermark_metrics = result.get('watermark_metrics', {})
            # Try both 'z_score' and 'normalized_score' for compatibility
            z_score = watermark_metrics.get('z_score') or watermark_metrics.get('normalized_score')
            if z_score is not None:
                z_scores.append(z_score)
        
        avg_z_score = np.mean(z_scores) if z_scores else 0
        
        # Get filename and extract parameters
        filename = Path(eval_file).name
        
        # Extract gamma, delta, and steps from filename
        # Format: run_gamma=0.1_delta=0.5_steps=10_sampled_100_gpt4_eval.json
        gamma = None
        delta = None
        steps = None
        
        # Try to extract gamma
        gamma_match = re.search(r'gamma=([0-9.]+)', filename)
        if gamma_match:
            gamma = float(gamma_match.group(1))
        
        # Try to extract delta
        delta_match = re.search(r'delta=([0-9.]+)', filename)
        if delta_match:
            delta = float(delta_match.group(1))
        
        # Try to extract steps
        steps_match = re.search(r'steps=([0-9]+)', filename)
        if steps_match:
            steps = int(steps_match.group(1))
        
        return {
            'filename': filename,
            'gamma': gamma if gamma is not None else '',
            'delta': delta if delta is not None else '',
            'steps': steps if steps is not None else '',
            'style_avg': category_averages.get('style (setting ethics aside)', 0),
            'consistency_avg': category_averages.get('consistency (setting ethics aside)', 0),
            'accuracy_avg': category_averages.get('accuracy (setting ethics aside)', 0),
            'ethics_avg': category_averages.get('ethics', 0),
            'overall_avg_score': overall_avg,
            'avg_perplexity': avg_perplexity,
            'avg_z_score': avg_z_score,
            'total_prompts': gpt4_metrics.get('total_prompts', 0)
        }
    except Exception as e:
        print(f"Error processing {eval_file}: {e}")
        return None


def create_summary_csv(eval_dir, output_csv):
    """Create CSV summary from all GPT-4 evaluation files."""
    # Find all GPT-4 evaluation files
    eval_files = glob(f"{eval_dir}/*_gpt4_eval.json")
    
    if not eval_files:
        print(f"No evaluation files found in {eval_dir}")
        return
    
    print(f"Found {len(eval_files)} evaluation files")
    
    # Extract metrics from each file
    all_metrics = []
    for eval_file in sorted(eval_files):
        metrics = extract_metrics_from_file(eval_file)
        if metrics:
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("No valid metrics found in any files")
        return
    
    # Create CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define CSV columns
    fieldnames = [
        'filename',
        'gamma',
        'delta',
        'steps',
        'style_avg',
        'consistency_avg',
        'accuracy_avg',
        'ethics_avg',
        'overall_avg_score',
        'avg_perplexity',
        'avg_z_score',
        'total_prompts'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for metrics in all_metrics:
            writer.writerow(metrics)
    
    print(f"\nCSV summary created: {output_path}")
    print(f"Total rows: {len(all_metrics)}")
    print(f"\nColumns:")
    for col in fieldnames:
        print(f"  - {col}")


def main():
    parser = argparse.ArgumentParser(
        description='Create CSV summary of GPT-4 evaluation metrics'
    )
    parser.add_argument(
        '--eval_dir',
        type=str,
        default='gpt4-outputs/green-list',
        help='Directory containing GPT-4 evaluation JSON files (default: gpt4-outputs/green-list)'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='gpt4_evaluation_summary.csv',
        help='Output CSV file path (default: gpt4_evaluation_summary.csv)'
    )
    
    args = parser.parse_args()
    create_summary_csv(args.eval_dir, args.output_csv)


if __name__ == '__main__':
    main()
