'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import json

from transformers import AutoTokenizer, AutoModel
from generate import generate, calculate_green_matches
import math
from accelerate import Accelerator


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        device="cuda",
        # Watermarking parameters
        gamma=0.5,
        amplification=0.0,
        watermark_steps=None,
        # Testing parameters
        max_prompts=None,  # Set to limit number of prompts for testing
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()

        # Store max_prompts for testing
        self.max_prompts = max_prompts

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(device)

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        
        # Watermarking parameters
        self.gamma = gamma
        self.amplification = amplification
        self.watermark_steps = watermark_steps    
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        # Store original number of requests for proper result handling
        original_num_requests = len(requests)
        
        # Limit prompts for testing if max_prompts is set
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

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for i, elem in enumerate(tqdm(ds, desc="Computing likelihood...")):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
                
                # Print detailed information for each prompt (only for non-dummy prompts)
                if self.max_prompts is None or i < self.max_prompts:
                    print(f"\n=== PROMPT {i+1} ===")
                    print(f"Question: {elem['prefix_text']}")
                    print(f"Target Answer: {elem['target_text']}")
                    print(f"Loglikelihood: {ll:.4f}")
                    print(f"Greedy Prediction: {is_target_greedy_dec}")
                    print("=" * 50)
        
        # If we limited prompts, pad the results with dummy values to match original request count
        if self.max_prompts is not None and len(out) < original_num_requests:
            dummy_result = (0.0, 0.0)  # Dummy loglikelihood and greedy prediction
            out.extend([dummy_result] * (original_num_requests - len(out)))
            print(f"Padded results with {original_num_requests - len(requests)} dummy values to match original request count")
        
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        # Limit prompts for testing if max_prompts is set BEFORE tokenization
        if self.max_prompts is not None:
            requests = requests[:self.max_prompts]
            print(f"Testing with first {len(requests)} prompts only")

        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        all_qa_pairs = []  # Store all question-answer pairs
        
        for i, elem in enumerate(tqdm(ds, desc="Generating...")):
            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]
 
            generated_answer = generate(self.model, prompt, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                        temperature=0, cfg_scale=self.cfg, remasking=self.remasking, mask_id=self.mask_id,
                                        gamma=self.gamma, amplification=self.amplification, watermark_steps=self.watermark_steps)
            
            # Extract generated tokens for green token analysis
            generated_tokens = generated_answer[0][prompt.shape[1]:]
            
            # Decode generated text first
            generated_answer_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
            for stop_seq in stop_tokens:
                    if stop_seq in generated_answer_text:
                        generated_answer_text = generated_answer_text.split(stop_seq)[0]

            # Extract ALL answer portions (after "A:") for green token analysis
            if "A:" in generated_answer_text:
                # Split by "A:" to get all answers
                parts = generated_answer_text.split("A:")
                all_answers = []
                
                for i, part in enumerate(parts):
                    if i == 0:
                        continue  # Skip the first part (before first "A:")
                    
                    # Extract the answer text (up to next "Q:" or end)
                    if "Q:" in part:
                        answer_text = part.split("Q:")[0].strip()
                    else:
                        answer_text = part.strip()
                    
                    if answer_text:  # Only add non-empty answers
                        all_answers.append(answer_text)
                
                print(f"Found {len(all_answers)} answers:")
                for j, answer in enumerate(all_answers):
                    print(f"  Answer {j+1}: {answer}")
                
                # Combine all answers into one text for analysis
                combined_answers = " ".join(all_answers)
                
                # Tokenize all answers combined
                answer_tokens = self.tokenizer(combined_answers)["input_ids"]
                answer_tokens_tensor = torch.tensor(answer_tokens).unsqueeze(0)
                
                # Calculate green token matches for all answers
                max_match_percent, actual_length, max_num_matches, best_start, match_arr = calculate_green_matches(
                    answer_tokens_tensor, gamma=self.gamma
                )
            else:
                # Fallback: use full generated text if no "A:" found
                max_match_percent, actual_length, max_num_matches, best_start, match_arr = calculate_green_matches(
                    generated_tokens.unsqueeze(0), gamma=self.gamma
                )
            
            # Calculate Z-score
            true_num_green = self.gamma * actual_length
            if math.sqrt(true_num_green * (1-self.gamma)) == 0:
                z_score = 0
            else:
                z_score = (max_num_matches - true_num_green) / math.sqrt(true_num_green * (1-self.gamma))

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer_text)["input_ids"]
            generated_answer_text = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            
            # Print detailed information for each prompt
            full_prompt_text = self.tokenizer.decode(prompt[0], skip_special_tokens=True)
            
            # Extract only the last question from the full prompt
            # Split by "Question:" and take the last one
            question_parts = full_prompt_text.split("Question:")
            if len(question_parts) > 1:
                last_question_text = "Question:" + question_parts[-1].split("Answer:")[0].strip()
            else:
                last_question_text = full_prompt_text
            
            print(f"\n=== PROMPT {i+1} ===")
            print(f"Question: {last_question_text}")
            print(f"Generated: {generated_answer_text}")
            print(f"Green token matches: {max_num_matches}/{actual_length} ({max_match_percent:.2%})")
            print(f"Z-score: {z_score:.2f}")
            print("=" * 50)
            
            # Store this question and answer pair for JSON output
            all_qa_pairs.append({
                "prompt_number": i + 1,
                "question": last_question_text,
                "answer": generated_answer_text,
                "green_token_matches": f"{max_num_matches}/{actual_length} ({max_match_percent:.2%})",
                "z_score": z_score
            })
            
            out.append(generated_answer_text)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        # Save JSON results to file (all question and answer pairs)
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"llada_results_{timestamp}.json"
        
        # Create JSON with all question and answer pairs
        json_results = {
            "timestamp": timestamp,
            "total_prompts": len(all_qa_pairs),
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
    