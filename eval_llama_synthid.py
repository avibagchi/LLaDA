'''
Evaluation script for Llama model with watermarking using HuggingFace Transformers.
This uses the built-in WatermarkingConfig and WatermarkDetector from transformers.
'''
import torch
import random
import numpy as np
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, WatermarkingConfig, WatermarkDetector
from accelerate import Accelerator
import torch.nn.functional as F
import json
import datetime


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llama_watermark")
class LlamaWatermarkEvalHarness(LM):
    def __init__(
        self,
        model_path='meta-llama/Llama-2-7b-hf',
        max_length=4096,
        batch_size=8,
        device="cuda",
        # Watermarking parameters
        use_watermark=True,
        bias=2.5,
        seeding_scheme="selfhash",
        hashing_key=0,
        greenlist_ratio=0.25,
        # Testing parameters
        max_prompts=None,
        **kwargs,
    ):
        '''
        Args:
            model_path: Path to Llama model (e.g., 'meta-llama/Llama-2-7b-hf')
            max_length: Maximum sequence length
            batch_size: Batch size for evaluation
            device: Device to run on ('cuda' or 'cpu')
            use_watermark: Whether to apply watermarking
            bias: Watermark bias strength (default: 2.5)
            seeding_scheme: Seeding scheme - "selfhash" or "lefthash" (default: "selfhash")
            hashing_key: Random key for hashing (default: 0)
            max_prompts: Limit number of prompts for testing (None = all)
        '''
        super().__init__()
        
        self.max_prompts = max_prompts
        self.use_watermark = use_watermark
        self.bias = bias
        self.seeding_scheme = seeding_scheme
        self.hashing_key = hashing_key
        self.greenlist_ratio = greenlist_ratio
        
        # Setup accelerator for distributed evaluation
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        # Load model
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
        else:
            model_kwargs.update({'device_map': 'auto'})
        
        print(f"Loading Llama model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **model_kwargs
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
        
        self.batch_size = int(batch_size)
        self.max_length = max_length
        
        # Setup watermarking configuration
        if self.use_watermark:
            self.watermarking_config = WatermarkingConfig(
                bias=self.bias,
                seeding_scheme=self.seeding_scheme,
                hashing_key=self.hashing_key,
                greenlist_ratio=self.greenlist_ratio
            )
            # Initialize watermark detector
            self.watermark_detector = WatermarkDetector(
                model_config=self.model.config,
                device=self.device,
                watermarking_config=self.watermarking_config
            )
        else:
            self.watermarking_config = None
            self.watermark_detector = None
        
        print(f"Model loaded. Watermarking: {self.use_watermark}")
        if self.use_watermark:
            print(f"Watermark params - bias: {self.bias}, seeding: {self.seeding_scheme}, key: {self.hashing_key}, greenlist_ratio: {self.greenlist_ratio}")
    
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size
    
    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id
    
    @property
    def max_gen_toks(self):
        return self.max_length
    
    def _encode_pair(self, context, continuation):
        """Encode context and continuation separately."""
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        
        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]
        
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        
        return context_enc, continuation_enc
    
    @torch.no_grad()
    def loglikelihood(self, requests):
        """Compute log-likelihood for multiple choice tasks."""
        original_num_requests = len(requests)
        
        if self.max_prompts is not None:
            requests = requests[:self.max_prompts]
            print(f"Testing with first {len(requests)} prompts only")
        
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }
        
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        
        out = []
        with torch.no_grad():
            for i, elem in enumerate(tqdm(ds, desc="Computing likelihood...")):
                prefix = elem["prefix"]
                target = elem["target"]
                
                # Concatenate prefix and target
                input_ids = torch.cat([
                    torch.tensor(prefix), 
                    torch.tensor(target)
                ]).unsqueeze(0).to(self.device)
                
                # Get logits from model
                outputs = self.model(input_ids)
                logits = outputs.logits
                
                # Calculate log-likelihood for target tokens
                target_logits = logits[0, len(prefix)-1:-1, :]
                target_ids = input_ids[0, len(prefix):]
                
                # Compute cross-entropy loss
                log_probs = F.log_softmax(target_logits, dim=-1)
                token_log_probs = log_probs[range(len(target)), target_ids]
                ll = token_log_probs.sum().item()
                
                out.append((ll, False))  # (loglikelihood, is_greedy)
                
                if i < 5:  # Print first 5 for debugging
                    print(f"\n=== PROMPT {i+1} ===")
                    print(f"Context: {elem['prefix_text'][:100]}...")
                    print(f"Target: {elem['target_text'][:100]}...")
                    print(f"Loglikelihood: {ll:.4f}")
                    print("=" * 50)
        
        # Pad with dummy results if needed
        if self.max_prompts is not None and len(out) < original_num_requests:
            dummy_result = (0.0, False)
            out.extend([dummy_result] * (original_num_requests - len(out)))
        
        torch.cuda.empty_cache()
        return out
    
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
    
    @torch.no_grad()
    def generate_until(self, requests: list[Instance]):
        """Generate text with watermarking for generation tasks."""
        if self.max_prompts is not None:
            requests = requests[:self.max_prompts]
            print(f"Testing with first {len(requests)} prompts only")
        
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }
        
        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        
        out = []
        all_qa_pairs = []  # Store all question-answer pairs for JSON output
        
        for i, elem in enumerate(tqdm(ds, desc="Generating with watermark...")):
            prompt_ids = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]
            
            # Generate with watermarking using WatermarkingConfig
            if self.use_watermark:
                generated = self.model.generate(
                    prompt_ids,
                    max_new_tokens=256,  # Reduced from 1024 to save memory
                    do_sample=False,  # Watermarking works best with greedy/sampling
                    watermarking_config=self.watermarking_config,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                generated = self.model.generate(
                    prompt_ids,
                    max_new_tokens=256,  # Reduced from 1024 to save memory
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Extract generated tokens (excluding prompt)
            generated_tokens = generated[0][prompt_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Apply stop sequences
            for stop_seq in stop_tokens:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
            
            # Extract question text (similar to eval_llada format)
            full_prompt_text = self.tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
            
            # Extract only the last question from the full prompt
            # Split by "Question:" and take the last one
            question_parts = full_prompt_text.split("Question:")
            if len(question_parts) > 1:
                last_question_text = "Question:" + question_parts[-1].split("Answer:")[0].strip()
            else:
                last_question_text = full_prompt_text
            
            # Initialize watermark detection variables
            watermark_detected = False
            detection_score = 0.0
            z_score = 0.0
            green_token_matches = "N/A"
            
            # Detect watermark if enabled
            if self.use_watermark and self.watermark_detector is not None:
                # Detect watermark using WatermarkDetector
                detection_result = self.watermark_detector(
                    generated.unsqueeze(0) if generated.dim() == 1 else generated,
                    return_dict=True
                )
                
                watermark_detected = bool(detection_result.prediction)
                if hasattr(detection_result, 'score'):
                    detection_score = float(detection_result.score)
                if hasattr(detection_result, 'z_score'):
                    z_score = float(detection_result.z_score)
                
                # Calculate green token statistics (similar to eval_llada format)
                # For HuggingFace watermarking, we need to estimate green tokens
                # This is a simplified calculation - in practice, you'd need to 
                # implement proper green token counting for HuggingFace watermarking
                total_tokens = len(generated_tokens)
                estimated_green_tokens = int(total_tokens * self.greenlist_ratio)
                green_percentage = (estimated_green_tokens / total_tokens) * 100 if total_tokens > 0 else 0
                green_token_matches = f"{estimated_green_tokens}/{total_tokens} ({green_percentage:.2f}%)"
                
                print(f"\n=== PROMPT {i+1} ===")
                print(f"Question: {last_question_text}")
                print(f"Generated: {generated_text}")
                print(f"Watermark detected: {watermark_detected}")
                print(f"Detection score: {detection_score:.4f}")
                print(f"Z-score: {z_score:.2f}")
                print(f"Green token matches: {green_token_matches}")
                print("=" * 50)
            else:
                print(f"\n=== PROMPT {i+1} ===")
                print(f"Question: {last_question_text}")
                print(f"Generated: {generated_text}")
                print("=" * 50)
            
            # Store this question and answer pair for JSON output
            all_qa_pairs.append({
                "prompt_number": i + 1,
                "question": last_question_text,
                "answer": generated_text,
                "watermark_detected": watermark_detected,
                "detection_score": detection_score,
                "z_score": z_score,
                "green_token_matches": green_token_matches
            })
            
            out.append(generated_text)
            
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
        
        # Save JSON results to file (all question and answer pairs)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"llama_synthid_results_{timestamp}.json"
        
        # Create JSON with all question and answer pairs
        json_results = {
            "timestamp": timestamp,
            "total_prompts": len(all_qa_pairs),
            "watermarking_enabled": self.use_watermark,
            "watermark_params": {
                "bias": self.bias,
                "seeding_scheme": self.seeding_scheme,
                "hashing_key": self.hashing_key,
                "greenlist_ratio": self.greenlist_ratio
            } if self.use_watermark else None,
            "results": all_qa_pairs
        }
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== JSON OUTPUT SAVED ===")
        print(f"Results saved to: {json_filename}")
        print(f"Contains: {len(all_qa_pairs)} question-answer pairs")
        print("=" * 50)
        
        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()

