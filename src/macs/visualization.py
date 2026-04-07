from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, zoom

from .attention import stable_zscore


def _normalize_scores(scores: np.ndarray, method: str, window_size: int = 10) -> np.ndarray:
    data = np.asarray(scores, dtype=np.float32)
    epsilon = 1e-9

    if method == "none":
        return data
    if method == "min-max":
        delta = float(data.max() - data.min())
        if delta < epsilon:
            return np.zeros_like(data)
        return (data - data.min()) / delta
    if method == "mean":
        return data - float(data.mean())
    if method == "moving-average":
        if len(data) < window_size:
            return data - float(data.mean())
        moving = np.convolve(data, np.ones(window_size) / window_size, mode="same")
        return data - moving
    if method == "z-score":
        return stable_zscore(data)
    if method == "z-score_exclude_max":
        if len(data) <= 1:
            return np.zeros_like(data)
        temp = np.delete(data, int(np.argmax(data)))
        std = float(temp.std()) + epsilon
        return (data - float(temp.mean())) / std
    if method == "max_average":
        denominator = float(np.max(np.abs(data))) + epsilon
        return data / denominator
    raise ValueError(f"Unsupported normalization method: {method}")


def render_text_scores_html(
    scores: np.ndarray,
    text: list[str] | str,
    output_path: str | Path,
    normalize: bool = True,
    method: str = "z-score",
    window_size: int = 10,
) -> None:
    joined_text = "".join(text) if isinstance(text, list) else text
    vector = np.asarray(scores, dtype=np.float32)
    if vector.ndim == 2:
        vector = vector[0]
    if len(vector) != len(joined_text):
        raise ValueError("Score length must match rendered text length.")

    display_scores = _normalize_scores(vector, method=method, window_size=window_size) if normalize else vector

    min_color = float(display_scores.min())
    max_color = float(display_scores.max())
    if max_color == min_color:
        color_scores = np.full_like(display_scores, 0.5)
    else:
        color_scores = (display_scores - min_color) / (max_color - min_color)

    def score_to_color(score: float) -> str:
        clipped = float(np.clip(score, 0.0, 1.0))
        red = int(255 * clipped)
        blue = 255 - red
        return f"rgb({red}, 0, {blue})"

    chars_per_line = 50
    line_count = len(joined_text) // chars_per_line + (1 if len(joined_text) % chars_per_line else 0)
    html = [
        "<html>",
        "<head>",
        "  <title>MACS Text Scores</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f6f8; }",
        "    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 24px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }",
        "    .line { display: flex; align-items: flex-start; margin-bottom: 12px; }",
        "    .line-number { min-width: 40px; color: #6b7280; text-align: right; padding-right: 12px; font-family: monospace; }",
        "    .line-content { display: flex; flex-wrap: wrap; gap: 4px; font-family: monospace; }",
        "    .char-unit { display: inline-flex; flex-direction: column; align-items: center; padding: 4px 6px; border-radius: 4px; min-width: 2.3em; }",
        "    .boxed { background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.25); }",
        "    .score { font-size: 0.65em; margin-bottom: 2px; }",
        "    .token { font-size: 1.05em; font-weight: 700; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <div class='container'>",
        "    <h2>MACS token attribution</h2>",
    ]

    for line_index in range(line_count):
        start = line_index * chars_per_line
        end = start + chars_per_line
        html.append("    <div class='line'>")
        html.append(f"      <div class='line-number'>{line_index + 1}</div>")
        html.append("      <div class='line-content'>")
        for offset, character in enumerate(joined_text[start:end]):
            score = float(display_scores[start + offset])
            color = score_to_color(float(color_scores[start + offset]))
            safe_char = character.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;")
            css_class = "char-unit boxed" if score > 0 else "char-unit"
            html.append(
                f"        <div class='{css_class}'><span class='score' style='color:{color}'>{score:.2f}</span>"
                f"<span class='token' style='color:{color}'>{safe_char}</span></div>"
            )
        html.append("      </div>")
        html.append("    </div>")

    html.extend(["  </div>", "</body>", "</html>"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(html), encoding="utf-8")


def map_token_to_image_position(
    image_grid_thw,
    merge_size: int,
    patch_size: int,
    orig_width: int,
    orig_height: int,
    start_pos: int,
    end_pos: int,
) -> list[tuple[float, float, float, float]]:
    time_steps, patch_h, patch_w = image_grid_thw[0].tolist()
    if time_steps != 1:
        raise ValueError("Only single-image inference is supported.")

    resized_h = patch_h * patch_size
    resized_w = patch_w * patch_size
    scale_w = orig_width / resized_w
    scale_h = orig_height / resized_h

    token_count = (patch_h * patch_w) // (merge_size**2)
    if end_pos - start_pos != token_count:
        raise ValueError("Image token count does not match image grid.")

    positions: list[tuple[float, float, float, float]] = []
    patches_per_row = patch_w // merge_size

    for token_index in range(token_count):
        row = token_index // patches_per_row
        col = token_index % patches_per_row
        left = col * merge_size * patch_size
        top = row * merge_size * patch_size
        right = (col + 1) * merge_size * patch_size
        bottom = (row + 1) * merge_size * patch_size
        positions.append((left * scale_w, top * scale_h, right * scale_w, bottom * scale_h))
    return positions


def generate_heatmap(
    original_image: Image.Image,
    mapping: list[tuple[float, float, float, float]],
    attention_scores: list[float] | np.ndarray,
    use_zscore: bool = False,
) -> Image.Image:
    normalized = np.asarray(attention_scores, dtype=np.float32)
    if use_zscore:
        normalized = 1.0 / (1.0 + np.exp(-stable_zscore(normalized)))
    else:
        min_score = float(normalized.min())
        max_score = float(normalized.max())
        if max_score == min_score:
            normalized = np.full_like(normalized, 0.5)
        else:
            normalized = (normalized - min_score) / (max_score - min_score)

    base = original_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    color_map = cm.get_cmap("jet")

    for (left, top, right, bottom), score in zip(mapping, normalized):
        red, green, blue, _ = color_map(float(score))
        fill = (int(red * 255), int(green * 255), int(blue * 255), 128)
        draw.rectangle([round(left), round(top), round(right), round(bottom)], fill=fill)

    return Image.alpha_composite(base, overlay)


def generate_smooth_heatmap(original_image: Image.Image, attention_scores: list[float] | np.ndarray) -> tuple[Image.Image, np.ndarray]:
    width, height = original_image.size
    aspect_ratio = height / width

    scores = np.asarray(attention_scores, dtype=np.float32)
    patch_w = int(np.round(np.sqrt(len(scores) / aspect_ratio)))
    patch_h = int(np.round(patch_w * aspect_ratio))
    grid_size = patch_h * patch_w

    if grid_size > len(scores):
        scores = np.pad(scores, (0, grid_size - len(scores)), constant_values=float(scores.mean()))
    elif grid_size < len(scores):
        scores = scores[:grid_size]

    grid = scores.reshape((patch_h, patch_w))
    interpolated = zoom(grid, (height / patch_h, width / patch_w), order=1)
    raw = interpolated.copy()

    min_score = float(interpolated.min())
    max_score = float(interpolated.max())
    if max_score == min_score:
        interpolated = np.full_like(interpolated, 0.5)
    else:
        interpolated = (interpolated - min_score) / (max_score - min_score)

    color_map = cm.get_cmap("jet")
    heatmap = Image.fromarray((color_map(interpolated)[:, :, :3] * 255).astype(np.uint8)).convert("RGBA")
    heatmap.putalpha(128)
    combined = Image.alpha_composite(original_image.convert("RGBA"), heatmap)
    return combined, raw


def generate_smooth_heatmap_mapping(
    original_image: Image.Image,
    mapping: list[tuple[float, float, float, float]],
    attention_scores: list[float] | np.ndarray,
    sigma: float = 10.0,
) -> Image.Image:
    if len(mapping) != len(attention_scores):
        raise ValueError("Mapping length must match attention score length.")

    width, height = original_image.size
    grid = np.zeros((height, width), dtype=np.float32)

    for (left, top, right, bottom), score in zip(mapping, attention_scores):
        grid[max(0, round(top)):min(height, round(bottom)), max(0, round(left)):min(width, round(right))] = float(score)

    smoothed = gaussian_filter(grid, sigma=sigma)
    min_score = float(smoothed.min())
    max_score = float(smoothed.max())
    if max_score == min_score:
        normalized = np.full_like(smoothed, 0.5)
    else:
        normalized = (smoothed - min_score) / (max_score - min_score)

    color_map = cm.get_cmap("jet")
    heatmap = Image.fromarray((color_map(normalized)[:, :, :3] * 255).astype(np.uint8)).convert("RGBA")
    heatmap.putalpha(128)
    return Image.alpha_composite(original_image.convert("RGBA"), heatmap)


def pad_mask(mask: np.ndarray, original_size: tuple[int, int], padded_size: tuple[int, int]) -> np.ndarray:
    original_h, original_w = original_size
    padded_h, padded_w = padded_size

    pad_left = max(0, (padded_w - original_w) // 2)
    pad_top = max(0, (padded_h - original_h) // 2)
    pad_right = max(0, padded_w - original_w - pad_left)
    pad_bottom = max(0, padded_h - original_h - pad_top)

    working = mask[: min(original_h, padded_h), : min(original_w, padded_w)]
    padded = np.zeros((padded_h, padded_w), dtype=working.dtype)
    padded[pad_top:padded_h - pad_bottom, pad_left:padded_w - pad_right] = working
    return padded
