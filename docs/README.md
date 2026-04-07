# MACS

Implementation of MACS from the EMNLP Findings 2025 paper [Attention Consistency for LLMs Explanation](https://aclanthology.org/2025.findings-emnlp.91/).

Two main tasks:
- Text QA attribution for decoder-only LLMs such as Llama 3.1.
- VQA-style attention extraction and heatmap generation for Qwen2.5-VL.

## Paper Summary

MACS(Multi-Layer Attention Consistency Score), is a lightweight attribution heuristic for decoder-based Transformers. Instead of aggregating every attention path like Attention Rollout, MACS focuses on whether a token keeps receiving strong attention across the model depth.

## Algorithm:

1. At each generation step, extract the last query row from every layer/head.
2. Split attention into input-side attention and previously generated output attention.
3. Redistribute output attention back to input tokens with uniform averaging.
4. Max-pool across heads at each layer.
5. Add a floor vector with `alpha = 0.8`.
6. Multiply layer scores with a Hadamard product to obtain the final consistency vector.
7. Normalize per step with Z-score to highlight salient tokens.

MACS achieves faithfulness comparable to more complex methods while reducing VRAM usage by 22% and latency by 30%.


- Paper page: https://aclanthology.org/2025.findings-emnlp.91/

## Repository Layout

```text
.
├── MACS_implementation.ipynb     # original notebook
├── stqa_350.json                 # demo QA subset used in the notebook
├── src/
│   └── macs/
│       ├── attention.py          # MACS math and attention extraction helpers
│       ├── cli.py                # command line entrypoint
│       ├── config.py             # dataclass configs
│       ├── prompting.py          # QA prompt builder
│       ├── schemas.py            # result containers
│       ├── text_pipeline.py      # text LLM pipeline
│       ├── visualization.py      # HTML and image heatmap utilities
│       └── vqa_pipeline.py       # Qwen2.5-VL pipeline
└── docs/
    └── index.md                 
```

## Installation

```bash
pip install -e .
```

Optional VQA dependency:

```bash
pip install -e ".[vl]"
```

## Usage

Run the text attribution pipeline on a dataset sample:

```bash
macs text --dataset stqa_350.json --sample-index 212 --model-id meta-llama/Llama-3.1-8B --save-step-heatmaps --save-overall-heatmap
```

Run the text attribution pipeline with manual inputs:

```bash
macs text --question "In what country is Normandy located?" --context "..." --model-id meta-llama/Llama-3.1-8B
```

Run the VQA pipeline and render a span heatmap:

```bash
macs vqa --image example.jpg --question "What is in the image?" --span-start 3 --span-end 6
```

## Notebook-to-Package Mapping

- Notebook cell `9` became the main logic in `src/macs/text_pipeline.py`.
- Notebook cells `16-21` became the VQA utilities in `src/macs/vqa_pipeline.py` and `src/macs/visualization.py`.
- Notebook cell `3` HTML rendering logic became `render_text_scores_html`.
- Prompt and configuration constants were separated into dedicated modules so they can be reused from code or CLI.

## Notes

- The VQA path remains an engineering utility layer; it is separated from the text pipeline so text-only usage does not depend on visual tooling.
- This refactor focuses on structure and compile-time correctness. It does not include re-running the experiments from the paper.
