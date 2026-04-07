from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .attention import (
    aggregate_over_steps,
    compute_step_scores,
    compute_top_k,
    extract_step_input_attention,
    find_chat_input_end,
    find_context_start,
)
from .config import GenerationConfig, MacsConfig, OutputConfig
from .schemas import InferenceStats, StepTrace, TextMacsResult
from .visualization import render_text_scores_html


class MACSTextPipeline:
    def __init__(self, model, tokenizer, device: str = "cuda") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str | None = None,
        torch_dtype: torch.dtype | str = torch.bfloat16,
        attn_implementation: str = "eager",
    ) -> "MACSTextPipeline":
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if resolved_device == "cuda" else None,
            attn_implementation=attn_implementation,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model.eval()
        if resolved_device != "cuda":
            model.to(resolved_device)
        return cls(model=model, tokenizer=tokenizer, device=resolved_device)

    def _reset_cuda_stats(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def _build_inputs(self, prompt: str) -> tuple[str, dict[str, torch.Tensor], int, int]:
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt")
        model_inputs = {key: value.to(self.device) for key, value in model_inputs.items()}
        input_ids = model_inputs["input_ids"][0].tolist()
        input_start = find_context_start(input_ids, self.tokenizer)
        input_end = find_chat_input_end(input_ids, self.tokenizer)
        return text, model_inputs, input_start, input_end

    def generate(
        self,
        prompt: str,
        macs_config: MacsConfig | None = None,
        generation_config: GenerationConfig | None = None,
        output_config: OutputConfig | None = None,
        gold_answer: str | None = None,
    ) -> TextMacsResult:
        macs = macs_config or MacsConfig()
        generation = generation_config or GenerationConfig()
        outputs = output_config or OutputConfig()

        self._reset_cuda_stats()
        prompt_text, model_inputs, input_start, input_end = self._build_inputs(prompt)
        input_ids = model_inputs["input_ids"][0].tolist()
        input_length = len(input_ids)
        masking_portion = max(1, int(macs.top_fraction * input_length))

        if macs.exclude_special_tokens:
            seq_ids = input_ids[input_start:input_end]
        else:
            seq_ids = input_ids
        seq_tokens = [self.tokenizer.decode([token_id]) for token_id in seq_ids]

        if gold_answer:
            print("Gold response:\n", gold_answer)

        current_input_ids = model_inputs["input_ids"].clone()
        eos_token_id = self.tokenizer.eos_token_id
        masked_positions: list[int] | None = None

        generated_token_ids: list[int] = []
        step_traces: list[StepTrace] = []
        step_tensors: list[np.ndarray] = []
        cumulative_logits = 0.0
        total_negative_log_prob = 0.0
        generated_token_count = 0

        start_time = time.time()

        while current_input_ids.shape[1] - input_length < generation.max_new_tokens:
            sequence_length = current_input_ids.size(1)
            if masked_positions:
                causal_mask = torch.tril(torch.ones((sequence_length, sequence_length), device=self.device))
                attention_mask = causal_mask.unsqueeze(0)
                for masked_index in masked_positions:
                    if masked_index < sequence_length:
                        attention_mask[:, :, masked_index] = 0
                model_outputs = self.model(
                    input_ids=current_input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_attentions=True,
                )
            else:
                model_outputs = self.model(
                    input_ids=current_input_ids,
                    return_dict=True,
                    output_attentions=True,
                )

            next_token_logits = model_outputs.logits[:, -1, :]
            cumulative_logits += float(torch.max(next_token_logits).detach().to(torch.float32).cpu().item())

            next_token_id = torch.argmax(next_token_logits, dim=-1)
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            total_negative_log_prob += float(-log_probs[0, next_token_id.item()].item())

            generated_token = self.tokenizer.decode(next_token_id).replace("\n", "\\n")
            previous_token = (
                self.tokenizer.decode(current_input_ids[:, -1]).replace("\n", "\\n")
                if generated_token_count > 0
                else "First token, no previous token"
            )
            generated_token_ids.append(int(next_token_id.item()))

            if generation.verbose_generation:
                print(f"Generation step {generated_token_count}: {generated_token}")
                print(f"Conditioned on: {previous_token}")
                print("=" * 60)

            generated_token_count += 1

            if next_token_id.item() == eos_token_id:
                break

            input_attention = extract_step_input_attention(
                attentions=model_outputs["attentions"][: macs.layer_count],
                input_length=input_length,
                input_start_index=input_start,
                input_end_index=input_end,
                exclude_special_tokens=macs.exclude_special_tokens,
                only_context=macs.only_context,
            )
            joint_tensor, score_vector = compute_step_scores(
                step_attention=input_attention,
                alpha=macs.alpha,
                return_raw_scores=macs.return_raw_scores,
            )
            top_indices, bottom_indices = compute_top_k(score_vector, macs.top_fraction)
            top_tokens = [seq_tokens[index] for index in top_indices]
            bottom_tokens = [seq_tokens[index] for index in bottom_indices]

            step_trace = StepTrace(
                step=generated_token_count - 1,
                generated_token=generated_token,
                previous_token=previous_token,
                top_indices=top_indices,
                bottom_indices=bottom_indices,
                top_tokens=top_tokens,
                bottom_tokens=bottom_tokens,
                z_scores=score_vector,
                attention_tensor=joint_tensor,
            )
            step_traces.append(step_trace)
            step_tensors.append(joint_tensor)

            if outputs.save_step_heatmaps and seq_tokens:
                step_output = outputs.output_root / "text" / f"genstep_{generated_token_count}_{generated_token.strip() or 'blank'}.html"
                render_text_scores_html(
                    scores=score_vector[1:] if seq_tokens[0].startswith(":") or "\n" in seq_tokens[0] else score_vector,
                    text=seq_tokens[1:] if seq_tokens[0].startswith(":") or "\n" in seq_tokens[0] else seq_tokens,
                    output_path=step_output,
                    method=outputs.step_html_normalization,
                )

            if generation.masking_generation:
                masked_positions = top_indices if generation.mask_type == "max" else bottom_indices

            current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(1)], dim=1)

        inference_time = time.time() - start_time
        generated_ids = current_input_ids[:, input_length:]
        token_total = int(generated_ids.shape[1])
        tokens_per_second = token_total / inference_time if inference_time > 0 else 0.0
        vram_peak = (
            float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            if torch.cuda.is_available()
            else None
        )
        perplexity = math.exp(total_negative_log_prob / generated_token_count) if generated_token_count else None
        mean_max_logit = cumulative_logits / generated_token_count if generated_token_count else 0.0
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)

        overall_scores = aggregate_over_steps(step_tensors, return_raw_scores=macs.return_raw_scores)
        overall_top_indices, _ = compute_top_k(overall_scores, macs.top_fraction) if len(overall_scores) else ([], [])
        overall_top_tokens = [seq_tokens[index] for index in overall_top_indices]

        if outputs.save_overall_heatmap and len(overall_scores):
            overall_output = outputs.output_root / "text" / "overall" / "gen_max_step.html"
            render_text_scores_html(
                scores=overall_scores[1:] if seq_tokens and (seq_tokens[0].startswith(":") or "\n" in seq_tokens[0]) else overall_scores,
                text=seq_tokens[1:] if seq_tokens and (seq_tokens[0].startswith(":") or "\n" in seq_tokens[0]) else seq_tokens,
                output_path=overall_output,
                method=outputs.step_html_normalization,
            )

        result = TextMacsResult(
            response=response,
            prompt_text=prompt_text,
            input_tokens=seq_tokens,
            generated_token_ids=generated_token_ids,
            steps=step_traces,
            overall_scores=overall_scores,
            overall_top_indices=overall_top_indices,
            overall_top_tokens=overall_top_tokens,
            stats=InferenceStats(
                inference_time=inference_time,
                num_tokens=token_total,
                tokens_per_second=tokens_per_second,
                vram_peak_mb=vram_peak,
                mean_max_logit=mean_max_logit,
                perplexity=perplexity,
            ),
        )

        if outputs.save_step_json:
            self.save_result(result, outputs.output_root / "text" / "result.json")

        print("Top tokens across all steps:", overall_top_tokens)
        print("Model response:\n", response)
        return result

    def save_result(self, result: TextMacsResult, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "response": result.response,
            "prompt_text": result.prompt_text,
            "input_tokens": result.input_tokens,
            "generated_token_ids": result.generated_token_ids,
            "overall_scores": result.overall_scores.tolist() if result.overall_scores is not None else None,
            "overall_top_indices": result.overall_top_indices,
            "overall_top_tokens": result.overall_top_tokens,
            "steps": [
                {
                    "step": step.step,
                    "generated_token": step.generated_token,
                    "previous_token": step.previous_token,
                    "top_indices": step.top_indices,
                    "bottom_indices": step.bottom_indices,
                    "top_tokens": step.top_tokens,
                    "bottom_tokens": step.bottom_tokens,
                    "z_scores": step.z_scores.tolist(),
                }
                for step in result.steps
            ],
            "stats": {
                "inference_time": result.stats.inference_time if result.stats else None,
                "num_tokens": result.stats.num_tokens if result.stats else None,
                "tokens_per_second": result.stats.tokens_per_second if result.stats else None,
                "vram_peak_mb": result.stats.vram_peak_mb if result.stats else None,
                "mean_max_logit": result.stats.mean_max_logit if result.stats else None,
                "perplexity": result.stats.perplexity if result.stats else None,
            },
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
