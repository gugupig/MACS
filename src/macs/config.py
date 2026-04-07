from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class MacsConfig:
    alpha: float = 0.8
    layer_count: int = 28
    context_marker: str = "context"
    exclude_special_tokens: bool = True
    only_context: bool = True
    top_fraction: float = 0.15
    return_raw_scores: bool = False


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 256
    verbose_generation: bool = False
    masking_generation: bool = False
    mask_type: str = "max"


@dataclass(slots=True)
class OutputConfig:
    output_root: Path = Path("outputs")
    save_step_heatmaps: bool = False
    save_overall_heatmap: bool = False
    save_step_json: bool = True
    step_html_normalization: str = "z-score"


@dataclass(slots=True)
class VQAConfig:
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    image_start_token_id: int = 151652
    image_end_token_id: int = 151653
