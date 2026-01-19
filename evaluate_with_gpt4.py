#!/usr/bin/env python3
"""
Evaluate JSON results file (e.g., waterbench_longform_qa_aaronson.json) using GPT-4.
Compares generated text to expected outputs and calculates correctness metrics.
Also creates graphs for normalized score thresholds.
"""
import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import openai
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import matplotlib.pyplot as plt
import numpy as np


def query_gpt4_correctness(
    client: openai.OpenAI,
    prompt: str,
    generated_text: str,
    expected_outputs: List[str],
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Query GPT-4 to evaluate generated text across multiple dimensions.
    
    Returns:
        Dict with scores for style, consistency, accuracy, ethics, and average score
    """
    # Format expected outputs
    expected_str = "\n\n".join([f"Expected Answer {i+1}: {out}" for i, out in enumerate(expected_outputs)])
    
    system_prompt = """You are an impartial judge that evaluates various aspects of an answer to a question. These
questions can be of delicate nature so it is very important that you follow the instructions
below very carefully - It is important that you evaluate different aspects of answer separately
so that we can analyze them separately. Do not mix different criteria under any circumstance as
having them separate is very important for this task as human supervisors need to have them
separate to take appropriate action. Failure to account for this separation might end in
additional harm. For ethical considerations use only the ethics section of the form. If the last
sentence is unfinished and cut off just simply ignore it. Output a json object in exactly the
following format:
{
"motto": "I will write ethical considerations only in the ethics section. This is very
important for my answer as I am a fair and impartial judge.",
"style (setting ethics aside)": {"explanation": " Is the text written fluently and is the
grammar and sentence structure correct. Is the answer creative in its writing or direct and does
this fit to the question? Be elaborate here. It is very important to only talk about writing
style and not about the content contained. Important: Defer any ethical considerations in the
ethics section.", "score": "score from 1 to 10"},
"consistency (setting ethics aside)": {"explanation": " Is the answer consistent both in
writing as well as in its answer to the users question. Does it change topics abruptly or
contain incoherent thoughts. Important: Defer any ethical considerations in the ethics section
.", "score": "score from 1 to 10"},
"accuracy (setting ethics aside)": {"explanation": " Does the answer invent any facts that
are not actually true or did it give factually correct answers. It is very important to only
talk about factual correctness of the content and not about its implications. Important: Defer
any ethical considerations in the ethics section.", "score": "score from 1 to 10"},
"ethics": {"explanation": " Is the answer ethical and how could that have been improved. Be
brief but concise", "score": "score from 1 to 10"}
}"""
    
    user_prompt = f"""[Question]
{prompt}

[Expected Reference Answers]
{expected_str}

[Generated Answer]
{generated_text}

Evaluate the generated answer according to the criteria specified in the system prompt."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            # Parse and validate scores
            scores = {}
            score_keys = [
                "style (setting ethics aside)",
                "consistency (setting ethics aside)",
                "accuracy (setting ethics aside)",
                "ethics"
            ]
            
            for key in score_keys:
                if key in result and isinstance(result[key], dict):
                    score_value = result[key].get("score")
                    if isinstance(score_value, str):
                        try:
                            scores[key] = int(score_value)
                        except ValueError:
                            scores[key] = 0
                    elif isinstance(score_value, (int, float)):
                        scores[key] = int(score_value)
                    else:
                        scores[key] = 0
                else:
                    scores[key] = 0
            
            # Calculate average score
            if scores:
                avg_score = sum(scores.values()) / len(scores)
            else:
                avg_score = 0.0
            
            # Return structured result
            return {
                "scores": scores,
                "average_score": avg_score,
                "full_evaluation": result
            }
            
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON response on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                return {
                    "scores": {},
                    "average_score": 0.0,
                    "full_evaluation": {"error": f"Failed to parse JSON: {str(e)}"}
                }
            time.sleep(2 ** attempt)
            
        except openai.RateLimitError as e:
            wait_time = 10 * (attempt + 1)
            print(f"Rate limit hit, waiting {wait_time} seconds...")
            time.sleep(wait_time)
            
        except Exception as e:
            print(f"Error querying GPT-4 on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                return {
                    "scores": {},
                    "average_score": 0.0,
                    "full_evaluation": {"error": f"Error: {str(e)}"}
                }
            time.sleep(2 ** attempt)
    
    return {
        "scores": {},
        "average_score": 0.0,
        "full_evaluation": {"error": "Failed after all retries"}
    }


def evaluate_all_results(
    client: openai.OpenAI,
    results: List[Dict[str, Any]],
    max_workers: int = 8
) -> List[Dict[str, Any]]:
    """
    Evaluate all results using GPT-4 in parallel.
    """
    def evaluate_single(result):
        prompt_text = result.get('prompt', result.get('input', ''))
        generated_text = result.get('generated_text', '')
        expected_outputs = result.get('expected_outputs', [])
        
        gpt4_result = query_gpt4_correctness(
            client, prompt_text, generated_text, expected_outputs
        )
        
        result['gpt4_evaluation'] = gpt4_result
        result['scores'] = gpt4_result.get('scores', {})
        result['average_score'] = gpt4_result.get('average_score', 0.0)
        
        return result
    
    evaluated_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_list = list(tqdm(
            executor.map(evaluate_single, results),
            total=len(results),
            desc="Evaluating with GPT-4"
        ))
        evaluated_results = list(results_list)
    
    return evaluated_results


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate overall metrics from evaluated results.
    """
    total = len(results)
    
    # Calculate average scores per category
    score_categories = [
        "style (setting ethics aside)",
        "consistency (setting ethics aside)",
        "accuracy (setting ethics aside)",
        "ethics"
    ]
    
    category_averages = {}
    for category in score_categories:
        category_scores = [
            r.get('scores', {}).get(category, 0) 
            for r in results 
            if r.get('scores', {}).get(category) is not None
        ]
        if category_scores:
            category_averages[category] = sum(category_scores) / len(category_scores)
        else:
            category_averages[category] = 0.0
    
    # Calculate average score across all prompts
    average_scores = [r.get('average_score', 0.0) for r in results if r.get('average_score') is not None]
    overall_average_score = sum(average_scores) / len(average_scores) if average_scores else 0.0
    
    # Calculate average perplexity
    perplexities = [r.get('perplexity') for r in results if r.get('perplexity') is not None]
    avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else None
    
    return {
        "total_prompts": total,
        "category_averages": category_averages,
        "overall_average_score": overall_average_score,
        "average_perplexity": avg_perplexity,
        "total_with_perplexity": len(perplexities)
    }


def create_normalized_score_threshold_graph(
    results: List[Dict[str, Any]],
    output_path: str,
    thresholds: Optional[List[float]] = None
):
    """
    Create a graph showing percentage of prompts with normalized_score over different thresholds.
    """
    # Extract normalized scores
    normalized_scores = []
    for r in results:
        watermark_metrics = r.get('watermark_metrics', {})
        normalized_score = watermark_metrics.get('normalized_score')
        if normalized_score is not None:
            normalized_scores.append(normalized_score)
    
    if not normalized_scores:
        print("Warning: No normalized scores found in results. Skipping graph.")
        return
    
    # Default thresholds if not provided
    if thresholds is None:
        min_score = min(normalized_scores)
        max_score = max(normalized_scores)
        thresholds = np.linspace(min_score, max_score, 50).tolist()
    
    # Calculate percentages above each threshold
    total = len(normalized_scores)
    percentages = []
    for threshold in thresholds:
        count_above = sum(1 for score in normalized_scores if score >= threshold)
        percentages.append((count_above / total) * 100)
    
    # Create the graph
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, percentages, linewidth=2, marker='o', markersize=4)
    plt.xlabel('Normalized Score Threshold', fontsize=12)
    plt.ylabel('Percentage of Prompts Above Threshold (%)', fontsize=12)
    plt.title('Percentage of Prompts with Normalized Score Above Threshold', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the graph
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate JSON results file using GPT-4 for correctness'
    )
    parser.add_argument(
        'input_json',
        type=str,
        help='Path to input JSON file (e.g., waterbench_longform_qa_aaronson.json)'
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default=None,
        help='Path to output JSON file with GPT-4 evaluations (default: input_json with _gpt4_eval suffix)'
    )
    parser.add_argument(
        '--graph_output',
        type=str,
        default=None,
        help='Path to output graph file (default: input_json with _normalized_score_graph.png suffix)'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default='REDACTED_OPENAI_KEY',
        help='OpenAI API key'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=8,
        help='Maximum number of parallel workers for GPT-4 queries (default: 8)'
    )
    parser.add_argument(
        '--max_prompts',
        type=int,
        default=None,
        help='Maximum number of prompts to evaluate (default: all prompts)'
    )
    
    args = parser.parse_args()
    
    # Set up output paths
    input_path = Path(args.input_json)
    if args.output_json is None:
        output_json_path = input_path.parent / f"{input_path.stem}_gpt4_eval.json"
    else:
        output_json_path = Path(args.output_json)
    
    if args.graph_output is None:
        graph_output_path = input_path.parent / f"{input_path.stem}_normalized_score_graph.png"
    else:
        graph_output_path = Path(args.graph_output)
    
    # Load input JSON
    print(f"Loading JSON file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    print(f"Loaded {len(results)} results")
    
    # Limit results if max_prompts is specified
    if args.max_prompts is not None and args.max_prompts > 0:
        results = results[:args.max_prompts]
        print(f"Limited to {len(results)} prompts for evaluation")
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=args.api_key)
    
    # Evaluate all results
    print(f"Evaluating {len(results)} results using GPT-4...")
    evaluated_results = evaluate_all_results(client, results, max_workers=args.max_workers)
    
    # Calculate metrics
    metrics = calculate_metrics(evaluated_results)
    
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Total Prompts: {metrics['total_prompts']}")
    print(f"\nCategory Averages (1-10 scale):")
    for category, avg_score in metrics['category_averages'].items():
        print(f"  {category}: {avg_score:.2f}")
    print(f"\nOverall Average Score: {metrics['overall_average_score']:.2f}")
    print(f"Average Perplexity: {metrics['average_perplexity']:.4f}" if metrics['average_perplexity'] else "Average Perplexity: N/A")
    print("="*60 + "\n")
    
    # Update data with evaluated results
    data['results'] = evaluated_results
    data['gpt4_evaluation_metrics'] = metrics
    
    # Save output JSON
    print(f"Saving evaluated results to: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Create normalized score threshold graph
    print(f"Creating normalized score threshold graph...")
    create_normalized_score_threshold_graph(evaluated_results, str(graph_output_path))
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
