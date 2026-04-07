# MACS: Attention Consistency for LLM Explanation

Implementation of MACS from the EMNLP Findings 2025 paper [Attention Consistency for LLMs Explanation](https://aclanthology.org/2025.findings-emnlp.91/)

## What MACS Does

MACS (Multi-Layer Attention Consistency Score), explains decoder-based language models by measuring whether an input token keeps receiving strong attention across layers during generation.

Compared with full attention aggregation methods such as Attention Rollout, MACS uses a stricter signal:

- keep the strongest attention link per layer;
- keep weak layers from collapsing to zero with a floor vector;
- reward tokens whose attention remains strong through the whole stack.

The result is a token attribution score that is simple enough to compute during inference, while still staying faithful enough to be useful for debugging and analysis.

### MACS can do text QA attribution
<img src="heatmap/QA attr.png" alt="Text QA attribution">

### MACS can do VQA attribution
<img src="heatmap/VQA attr.png" alt="Text QA attribution">

## Algorithm

For each generation step:

1. Extract the last attention row from every layer and head.
2. Separate attention to original inputs and attention to already generated outputs.
3. Redistribute output attention back to input tokens with uniform averaging.
4. Max-pool over heads to keep the strongest signal per token.
5. Blend the result with a floor vector using `alpha = 0.8`.
6. Multiply layer scores with a Hadamard product to measure cross-layer consistency.
7. Convert the final vector into Z-scores to surface salient tokens.

This implementation follows the structure described in Sections 3.2 and 3.3 of the paper.

## Repository Layout
- `MACS_implementation`: All-in-one implementation
- `src/macs/text_pipeline.py`: text-only MACS pipeline for LLM attribution.
- `src/macs/vqa_pipeline.py`: VQA-oriented attention extraction and heatmap workflow.
- `src/macs/attention.py`: reusable MACS algorithm and attention helpers.
- `src/macs/visualization.py`: HTML token heatmaps and image overlays.
- `src/macs/cli.py`: command-line entrypoint for running the pipeline.

## Supported Workflows

### Text QA Attribution

Use the text pipeline when you want to explain how a decoder-only model answers a question from context. This is the closest path to the experiments described in the paper.

Example:

```bash
macs text --dataset stqa_350.json --sample-index 212 --model-id meta-llama/Llama-3.1-8B
```

### Visual QA Heatmaps

Use the VQA path when you want to inspect which image regions are emphasized while the model generates a span of output tokens.

Example:

```bash
macs vqa --image example.jpg --question "What is in the image?" --span-start 3 --span-end 6
```
## Compatibility
- Currently support models like Qwen2.5,Qwen-VL and Llama3.1-8B.
- More model support can be easily added since MACS only need attention score from each layer/head.
- Set `attn_implementation="eager`.


## Notebook-to-Package Mapping

- Notebook cell `9` became the main logic in `src/macs/text_pipeline.py`.
- Notebook cells `16-21` became the VQA utilities in `src/macs/vqa_pipeline.py` and `src/macs/visualization.py`.
- Notebook cell `3` HTML rendering logic became `render_text_scores_html`.
- Prompt and configuration constants were separated into dedicated modules so they can be reused from code or CLI.


## References

- ACL Anthology page: https://aclanthology.org/2025.findings-emnlp.91/
- SQuAD2.0: https://rajpurkar.github.io/SQuAD-explorer/
