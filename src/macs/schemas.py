from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class InferenceStats:
    inference_time: float
    num_tokens: int
    tokens_per_second: float
    vram_peak_mb: float | None
    mean_max_logit: float
    perplexity: float | None


@dataclass(slots=True)
class StepTrace:
    step: int
    generated_token: str
    previous_token: str
    top_indices: list[int]
    bottom_indices: list[int]
    top_tokens: list[str]
    bottom_tokens: list[str]
    z_scores: np.ndarray
    attention_tensor: np.ndarray


@dataclass(slots=True)
class TextMacsResult:
    response: str
    prompt_text: str
    input_tokens: list[str]
    generated_token_ids: list[int]
    steps: list[StepTrace] = field(default_factory=list)
    overall_scores: np.ndarray | None = None
    overall_top_indices: list[int] = field(default_factory=list)
    overall_top_tokens: list[str] = field(default_factory=list)
    stats: InferenceStats | None = None


@dataclass(slots=True)
class VQAGenerationResult:
    output_text: list[str]
    generated_token_ids: list[int]
    converted_attentions: list[np.ndarray]
    step_attention_scores: list[np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
