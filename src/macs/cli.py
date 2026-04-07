from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .config import GenerationConfig, MacsConfig, OutputConfig, VQAConfig
from .prompting import build_qa_prompt
from .text_pipeline import MACSTextPipeline
from .vqa_pipeline import MACSVQAPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MACS modular pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    text_parser = subparsers.add_parser("text", help="Run the text MACS pipeline")
    text_parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B")
    text_parser.add_argument("--question")
    text_parser.add_argument("--context")
    text_parser.add_argument("--dataset", type=Path)
    text_parser.add_argument("--sample-index", type=int)
    text_parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    text_parser.add_argument("--max-new-tokens", type=int, default=256)
    text_parser.add_argument("--save-step-heatmaps", action="store_true")
    text_parser.add_argument("--save-overall-heatmap", action="store_true")

    vqa_parser = subparsers.add_parser("vqa", help="Run the VQA MACS pipeline")
    vqa_parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    vqa_parser.add_argument("--image", required=True, type=Path)
    vqa_parser.add_argument("--question", required=True)
    vqa_parser.add_argument("--output", type=Path, default=Path("outputs/vqa/heatmap.png"))
    vqa_parser.add_argument("--span-start", type=int, required=True)
    vqa_parser.add_argument("--span-end", type=int, required=True)
    vqa_parser.add_argument("--max-new-tokens", type=int, default=256)

    return parser


def _load_qa_sample(dataset_path: Path, sample_index: int) -> tuple[str, str, str | None]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    sample = payload[sample_index]
    answers = sample.get("answers")
    if isinstance(answers, list):
        gold = answers[0] if answers else None
    else:
        gold = answers
    return sample["question"], sample["context"], gold


def _run_text(args: argparse.Namespace) -> None:
    if args.dataset is not None:
        if args.sample_index is None:
            raise ValueError("--sample-index is required when --dataset is provided.")
        question, context, gold = _load_qa_sample(args.dataset, args.sample_index)
    else:
        if not args.question or not args.context:
            raise ValueError("Either provide --dataset/--sample-index or both --question and --context.")
        question, context, gold = args.question, args.context, None

    prompt = build_qa_prompt(question=question, context=context)
    pipeline = MACSTextPipeline.from_pretrained(model_id=args.model_id)
    pipeline.generate(
        prompt=prompt,
        gold_answer=gold,
        macs_config=MacsConfig(),
        generation_config=GenerationConfig(max_new_tokens=args.max_new_tokens),
        output_config=OutputConfig(
            output_root=args.output_root,
            save_step_heatmaps=args.save_step_heatmaps,
            save_overall_heatmap=args.save_overall_heatmap,
        ),
    )


def _run_vqa(args: argparse.Namespace) -> None:
    vqa_config = VQAConfig(model_id=args.model_id)
    pipeline = MACSVQAPipeline.from_pretrained(config=vqa_config)
    result = pipeline.generate(
        image=args.image,
        question=args.question,
        vqa_config=vqa_config,
        macs_config=MacsConfig(),
        max_new_tokens=args.max_new_tokens,
    )
    original_image = Image.open(args.image)
    pipeline.save_span_heatmap(
        original_image=original_image,
        step_attention_scores=result.step_attention_scores,
        start=args.span_start,
        end=args.span_end,
        output_path=args.output,
    )
    print("Generated answer:", result.output_text)
    print("Heatmap saved to:", args.output)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "text":
        _run_text(args)
        return
    if args.command == "vqa":
        _run_vqa(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
