from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def stable_zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    mean = float(array.mean())
    std = float(array.std())
    if std == 0.0:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - mean) / std).astype(np.float32)


def compute_top_k(scores: np.ndarray, fraction: float) -> tuple[list[int], list[int]]:
    total = max(1, int(len(scores) * fraction))
    descending = np.argsort(scores)
    bottom = descending[:total].tolist()
    top = descending[-total:][::-1].tolist()
    return top, bottom


def find_context_start(input_ids: list[int], tokenizer, context_marker: str = "context") -> int:
    marker = context_marker.lower()
    for index, token_id in enumerate(input_ids):
        decoded = tokenizer.decode([token_id]).lower()
        if marker in decoded:
            return index + 1
    return 0


def find_chat_input_end(input_ids: list[int], tokenizer) -> int:
    assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")
    if isinstance(assistant_token_id, int) and assistant_token_id in input_ids:
        return max(0, input_ids.index(assistant_token_id) - 3)
    return len(input_ids)


def extract_step_input_attention(
    attentions: Iterable[torch.Tensor],
    input_length: int,
    input_start_index: int,
    input_end_index: int,
    exclude_special_tokens: bool,
    only_context: bool,
) -> np.ndarray:
    stacked = torch.stack([layer.detach().cpu() for layer in attentions]).squeeze(1)
    row = stacked[:, :, -1, :]
    input_row = row[:, :, :input_length]

    generated_row = row[:, :, input_length:]
    if generated_row.shape[-1] > 0:
        denom = (input_end_index - input_start_index) if exclude_special_tokens else input_length
        denom = max(1, denom)
        redistributed = generated_row.sum(dim=-1) / denom
        input_row = input_row + redistributed[:, :, np.newaxis]

    if exclude_special_tokens and only_context:
        input_row = input_row[:, :, input_start_index:input_end_index]

    return input_row.to(torch.float32).numpy()


def compute_macs_tensor(step_attention: np.ndarray, alpha: float) -> np.ndarray:
    attention = np.asarray(step_attention, dtype=np.float32)
    if attention.ndim == 4:
        attention = attention.squeeze(2)
    if attention.ndim != 3:
        raise ValueError(f"Expected [layers, heads, tokens], got shape {attention.shape}")

    pooled = attention.max(axis=1)[:, np.newaxis, :]
    floor = np.ones((1, 1, pooled.shape[-1]), dtype=np.float32)
    blended = alpha * pooled + (1.0 - alpha) * floor

    joint = np.zeros_like(blended)
    joint[0] = blended[0]
    for layer_index in range(1, blended.shape[0]):
        joint[layer_index] = blended[layer_index] * joint[layer_index - 1]
    return joint


def compute_step_scores(step_attention: np.ndarray, alpha: float, return_raw_scores: bool) -> tuple[np.ndarray, np.ndarray]:
    joint = compute_macs_tensor(step_attention=step_attention, alpha=alpha)
    raw = joint[-1, 0, :]
    if return_raw_scores:
        return joint, raw.astype(np.float32)
    return joint, stable_zscore(raw)


def aggregate_over_steps(step_tensors: list[np.ndarray], return_raw_scores: bool) -> np.ndarray:
    if not step_tensors:
        return np.zeros((0,), dtype=np.float32)
    last_layer_scores = np.array([step[-1, 0, :] for step in step_tensors], dtype=np.float32)
    aggregated = last_layer_scores.max(axis=0)
    if return_raw_scores:
        return aggregated
    return stable_zscore(aggregated)
