#!/usr/bin/env python3
"""
Benchmark testing with watermarking for LLaDA.
Tests standard benchmarks (MMLU, BBH, ARC-C, Hellaswag, TruthfulQA, WinoGrande, PIQA) 
with and without watermarking to measure performance impact.
"""

import sys
import os
import torch
import subprocess
import json
import csv
from pathlib import Path

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate import generate, calculate_green_matches
from transformers import AutoTokenizer, AutoModel


class WatermarkBenchmarkTester:
    """Test benchmarks with watermarking to measure performance impact."""
    
    def __init__(self, model_path='GSAI-ML/LLaDA-8B-Instruct', device='cuda'):
        self.model_path = model_path
        self.device = device
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).to(device).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        # Benchmark configurations
        self.benchmarks = {
            'mmlu': {
                'task': 'mmlu',
                'num_fewshot': 5,
                'cfg': 0.0,
                'mc_num': 1,
                'batch_size': 1
            },
            'bbh': {
                'task': 'bbh',
                'num_fewshot': 3,
                'cfg': 0.0,
                'gen_length': 1024,
                'steps': 1024,
                'block_length': 1024,
                'batch_size': 8
            },
            'arc_challenge': {
                'task': 'arc_challenge',
                'num_fewshot': 0,
                'cfg': 0.5,
                'mc_num': 128,
                'batch_size': 8
            },
            'hellaswag': {
                'task': 'hellaswag',
                'num_fewshot': 0,
                'cfg': 0.5,
                'mc_num': 128,
                'batch_size': 8
            },
            'truthfulqa_mc2': {
                'task': 'truthfulqa_mc2',
                'num_fewshot': 0,
                'cfg': 2.0,
                'mc_num': 128,
                'batch_size': 8
            },
            'winogrande': {
                'task': 'winogrande',
                'num_fewshot': 5,
                'cfg': 0.0,
                'mc_num': 128,
                'batch_size': 8
            },
            'piqa': {
                'task': 'piqa',
                'num_fewshot': 0,
                'cfg': 0.5,
                'mc_num': 128,
                'batch_size': 8
            }
        }
    
    def run_benchmark_with_watermark(self, benchmark_name, gamma=0.5, amplification=2.0, 
                                   watermark_steps=None, model_seed=1):
        """Run a benchmark with watermarking enabled."""
        print(f"\n{'='*80}")
        print(f"RUNNING {benchmark_name.upper()} WITH WATERMARKING")
        print(f"Gamma: {gamma}, Amplification: {amplification}, Steps: {watermark_steps}")
        print(f"{'='*80}")
        
        # Set random seed
        torch.manual_seed(model_seed)
        
        # Get benchmark config
        config = self.benchmarks[benchmark_name]
        
        # Prepare command
        cmd = [
            'accelerate', 'launch', 'eval_llada.py',
            '--tasks', config['task'],
            '--model', 'llada_dist',
            '--model_args', f"model_path='{self.model_path}',cfg={config['cfg']},is_check_greedy=False"
        ]
        
        # Add task-specific parameters
        if 'num_fewshot' in config:
            cmd.extend(['--num_fewshot', str(config['num_fewshot'])])
        
        if 'mc_num' in config:
            cmd.extend(['--model_args', f"mc_num={config['mc_num']}"])
        
        if 'gen_length' in config:
            cmd.extend(['--model_args', f"gen_length={config['gen_length']},steps={config['steps']},block_length={config['block_length']}"])
        
        cmd.extend(['--batch_size', str(config['batch_size'])])
        
        # Add watermarking parameters
        cmd.extend(['--model_args', f"gamma={gamma},amplification={amplification}"])
        if watermark_steps is not None:
            cmd.extend(['--model_args', f"watermark_steps={watermark_steps}"])
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Run the benchmark
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                print("✅ Benchmark completed successfully")
                # Parse results from stdout
                return self._parse_benchmark_results(result.stdout)
            else:
                print(f"❌ Benchmark failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("⏰ Benchmark timed out")
            return None
        except Exception as e:
            print(f"❌ Error running benchmark: {e}")
            return None
    
    def run_benchmark_without_watermark(self, benchmark_name, model_seed=1):
        """Run a benchmark without watermarking (baseline)."""
        print(f"\n{'='*80}")
        print(f"RUNNING {benchmark_name.upper()} WITHOUT WATERMARKING (BASELINE)")
        print(f"{'='*80}")
        
        # Set random seed
        torch.manual_seed(model_seed)
        
        # Get benchmark config
        config = self.benchmarks[benchmark_name]
        
        # Prepare command (no watermarking parameters)
        cmd = [
            'accelerate', 'launch', 'eval_llada.py',
            '--tasks', config['task'],
            '--model', 'llada_dist',
            '--model_args', f"model_path='{self.model_path}',cfg={config['cfg']},is_check_greedy=False"
        ]
        
        # Add task-specific parameters
        if 'num_fewshot' in config:
            cmd.extend(['--num_fewshot', str(config['num_fewshot'])])
        
        if 'mc_num' in config:
            cmd.extend(['--model_args', f"mc_num={config['mc_num']}"])
        
        if 'gen_length' in config:
            cmd.extend(['--model_args', f"gen_length={config['gen_length']},steps={config['steps']},block_length={config['block_length']}"])
        
        cmd.extend(['--batch_size', str(config['batch_size'])])
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Run the benchmark
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                print("✅ Baseline benchmark completed successfully")
                return self._parse_benchmark_results(result.stdout)
            else:
                print(f"❌ Baseline benchmark failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("⏰ Baseline benchmark timed out")
            return None
        except Exception as e:
            print(f"❌ Error running baseline benchmark: {e}")
            return None
    
    def _parse_benchmark_results(self, stdout):
        """Parse benchmark results from stdout."""
        try:
            # Look for JSON results in the output
            lines = stdout.split('\n')
            for line in lines:
                if line.strip().startswith('{') and 'acc' in line:
                    try:
                        result = json.loads(line.strip())
                        return result
                    except json.JSONDecodeError:
                        continue
            
            # If no JSON found, try to extract accuracy from text
            for line in lines:
                if 'acc' in line.lower() and any(char.isdigit() for char in line):
                    # Extract number from line
                    import re
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        return {'acc': float(numbers[0])}
            
            print(f"Warning: Could not parse results from: {stdout[:200]}...")
            return None
            
        except Exception as e:
            print(f"Error parsing results: {e}")
            return None
    
    def test_watermark_impact(self, benchmark_name, gamma=0.5, amplification=2.0, 
                            watermark_steps=None, model_seed=1):
        """Test the impact of watermarking on a specific benchmark."""
        print(f"\n🧪 TESTING WATERMARK IMPACT ON {benchmark_name.upper()}")
        print(f"Parameters: gamma={gamma}, amplification={amplification}, steps={watermark_steps}")
        
        # Run without watermarking (baseline)
        baseline_results = self.run_benchmark_without_watermark(benchmark_name, model_seed)
        
        # Run with watermarking
        watermarked_results = self.run_benchmark_with_watermark(
            benchmark_name, gamma, amplification, watermark_steps, model_seed
        )
        
        # Compare results
        if baseline_results and watermarked_results:
            baseline_acc = baseline_results.get('acc', 0)
            watermarked_acc = watermarked_results.get('acc', 0)
            impact = watermarked_acc - baseline_acc
            
            print(f"\n📊 RESULTS COMPARISON:")
            print(f"  Baseline (no watermark): {baseline_acc:.2f}%")
            print(f"  With watermark: {watermarked_acc:.2f}%")
            print(f"  Impact: {impact:+.2f}%")
            
            return {
                'benchmark': benchmark_name,
                'gamma': gamma,
                'amplification': amplification,
                'watermark_steps': watermark_steps,
                'model_seed': model_seed,
                'baseline_acc': baseline_acc,
                'watermarked_acc': watermarked_acc,
                'impact': impact
            }
        else:
            print("❌ Could not complete comparison")
            return None
    
    def run_full_watermark_evaluation(self, gamma_list=[0.5], amplification_list=[2.0], 
                                    watermark_steps_list=[None], model_seed_list=[1]):
        """Run full evaluation across multiple benchmarks and parameters."""
        print("🚀 STARTING FULL WATERMARK EVALUATION")
        print(f"Benchmarks: {list(self.benchmarks.keys())}")
        print(f"Gamma values: {gamma_list}")
        print(f"Amplification values: {amplification_list}")
        print(f"Watermark steps: {watermark_steps_list}")
        print(f"Model seeds: {model_seed_list}")
        
        results = []
        
        for model_seed in model_seed_list:
            for gamma in gamma_list:
                for amplification in amplification_list:
                    for watermark_steps in watermark_steps_list:
                        for benchmark_name in self.benchmarks.keys():
                            try:
                                result = self.test_watermark_impact(
                                    benchmark_name, gamma, amplification, 
                                    watermark_steps, model_seed
                                )
                                if result:
                                    results.append(result)
                            except Exception as e:
                                print(f"❌ Error testing {benchmark_name}: {e}")
                                continue
        
        # Save results
        self._save_results(results)
        self._print_summary(results)
        
        return results
    
    def _save_results(self, results):
        """Save results to CSV file."""
        filename = 'benchmark_watermark_results.csv'
        with open(filename, 'w', newline='') as csvfile:
            if results:
                fieldnames = results[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
                print(f"\n💾 Results saved to {filename}")
    
    def _print_summary(self, results):
        """Print summary of results."""
        if not results:
            print("❌ No results to summarize")
            return
        
        print(f"\n📈 SUMMARY OF {len(results)} TESTS")
        print("="*80)
        
        # Group by benchmark
        by_benchmark = {}
        for result in results:
            benchmark = result['benchmark']
            if benchmark not in by_benchmark:
                by_benchmark[benchmark] = []
            by_benchmark[benchmark].append(result)
        
        for benchmark, benchmark_results in by_benchmark.items():
            print(f"\n{benchmark.upper()}:")
            avg_impact = sum(r['impact'] for r in benchmark_results) / len(benchmark_results)
            print(f"  Average impact: {avg_impact:+.2f}%")
            
            # Show individual results
            for result in benchmark_results:
                print(f"    γ={result['gamma']}, amp={result['amplification']}, steps={result['watermark_steps']}: "
                      f"{result['baseline_acc']:.1f}% → {result['watermarked_acc']:.1f}% ({result['impact']:+.1f}%)")


def main():
    """Main function to run benchmark tests."""
    # Test parameters
    gamma_list = [0.5]  # Fraction of green tokens
    amplification_list = [0.0, 2.0, 5.0]  # Watermark strength
    watermark_steps_list = [None, 10, 50]  # Steps to watermark (None = all steps)
    model_seed_list = [1]  # Random seeds
    
    # Initialize tester
    tester = WatermarkBenchmarkTester()
    
    # Run evaluation
    results = tester.run_full_watermark_evaluation(
        gamma_list=gamma_list,
        amplification_list=amplification_list,
        watermark_steps_list=watermark_steps_list,
        model_seed_list=model_seed_list
    )
    
    print(f"\n✅ Evaluation completed with {len(results)} tests")


if __name__ == '__main__':
    main()
