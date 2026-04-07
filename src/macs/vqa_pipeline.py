from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from .attention import compute_macs_tensor
from .config import MacsConfig, VQAConfig
from .schemas import VQAGenerationResult
from .visualization import generate_smooth_heatmap


class MACSVQAPipeline:
    def __init__(self, model, tokenizer, processor, device: str = "cuda") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        config: VQAConfig | None = None,
        device: str | None = None,
    ) -> "MACSVQAPipeline":
        vqa_config = config or VQAConfig()
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vqa_config.model_id,
            torch_dtype="auto",
            device_map="auto" if resolved_device == "cuda" else None,
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(vqa_config.model_id)
        processor = AutoProcessor.from_pretrained(
            vqa_config.model_id,
            min_pixels=vqa_config.min_pixels,
            max_pixels=vqa_config.max_pixels,
        )
        model.eval()
        if resolved_device != "cuda":
            model.to(resolved_device)
        return cls(model=model, tokenizer=tokenizer, processor=processor, device=resolved_device)

    @staticmethod
    def find_image_index(
        input_ids: list[int] | torch.Tensor,
        start_token_id: int,
        end_token_id: int,
    ) -> tuple[int, int]:
        values = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
        start = 0
        end = 0
        for index, token_id in enumerate(values):
            if token_id == start_token_id:
                start = index + 1
            if token_id == end_token_id:
                end = index
        return start, end

    def generate(
        self,
        image: str | Path,
        question: str,
        macs_config: MacsConfig | None = None,
        vqa_config: VQAConfig | None = None,
        max_new_tokens: int = 256,
    ) -> VQAGenerationResult:
        from qwen_vl_utils import process_vision_info

        macs = macs_config or MacsConfig()
        cfg = vqa_config or VQAConfig()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image)},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_attentions=True,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated.sequences)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        input_start_index, input_end_index = self.find_image_index(
            inputs["input_ids"][0],
            start_token_id=cfg.image_start_token_id,
            end_token_id=cfg.image_end_token_id,
        )
        converted_attentions = self.convert_generation_attentions(
            generated_attentions=generated["attentions"],
            input_length=len(inputs["input_ids"][-1]),
            input_start_index=input_start_index,
            input_end_index=input_end_index,
        )
        step_attention_scores = [compute_macs_tensor(step, alpha=macs.alpha) for step in converted_attentions]

        return VQAGenerationResult(
            output_text=output_text,
            generated_token_ids=generated_ids_trimmed[0].tolist(),
            converted_attentions=converted_attentions,
            step_attention_scores=step_attention_scores,
            metadata={
                "input_start_index": input_start_index,
                "input_end_index": input_end_index,
                "image_size": image_inputs[0].size,
            },
        )

    @staticmethod
    def convert_generation_attentions(
        generated_attentions,
        input_length: int,
        input_start_index: int,
        input_end_index: int,
    ) -> list[np.ndarray]:
        converted: list[np.ndarray] = []

        first_step_layers = []
        for layer_attention in generated_attentions[0]:
            first_step_layers.append(layer_attention[:, :, -1, :].unsqueeze(2))
        attention_steps = list(generated_attentions)
        attention_steps[0] = tuple(first_step_layers)

        for step_attentions in attention_steps:
            layer_tensor = torch.cat([layer for layer in step_attentions], dim=0)
            step_to_input = layer_tensor[:, :, :, :input_length]
            step_to_output = layer_tensor[:, :, :, input_length:]

            if step_to_output.shape[-1] > 0:
                redistributed = step_to_output.sum(dim=-1) / max(1, (input_end_index - input_start_index))
                step_to_input = step_to_input + redistributed.unsqueeze(-1)

            step_to_input = step_to_input[:, :, :, input_start_index:input_end_index]
            converted.append(step_to_input.detach().clone().cpu().to(torch.float32).numpy())

        return converted

    @staticmethod
    def span_mean_attention(step_attention_scores: list[np.ndarray], start: int, end: int) -> np.ndarray:
        mean_attention = np.array(step_attention_scores[start:end], dtype=np.float32).mean(axis=0)
        return mean_attention[-1, 0, :]

    @staticmethod
    def save_span_heatmap(
        original_image: Image.Image,
        step_attention_scores: list[np.ndarray],
        start: int,
        end: int,
        output_path: str | Path,
    ) -> None:
        last_layer_scores = MACSVQAPipeline.span_mean_attention(step_attention_scores, start=start, end=end)
        heatmap, _ = generate_smooth_heatmap(original_image=original_image, attention_scores=last_layer_scores)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        heatmap.save(output_path)
