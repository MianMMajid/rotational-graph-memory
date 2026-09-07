# Rotational Graph Memory

Exploratory experiments in symbolic grounding, graph memory, and rotational data representation.

**Status: discontinued research prototype.** This repository preserves the experiments and research assessment. It is not a working rotational database, a production memory system, or a demonstrated context-window extension.

## Original idea

Explore whether reusable symbols could be stored once, with their arrangements, graph connections, and changing viewpoints representing different information. A backend would retrieve or reconstruct the relevant representation, potentially as graphics, to reduce duplication and improve usable AI memory.

The idea developed into questions about dynamic graph retrieval, geometric associative memory, visual context compression, and learning from verified experiences. The scripts below explore symbolic grounding and feedback loops; they do not implement the complete proposed architecture.

## Contents

| File or directory | Purpose |
| --- | --- |
| `neuro_symbolic_loop.py` | LLM-to-Lean proof generation with compiler feedback and retries. |
| `semantic_grounding_loop.py` | English-to-action translation in a five-position sandbox. |
| `glyph_grounding_loop.py` | Visual glyph sandbox with local vision-language inference and data collection. |
| `curriculum_generator.py` | Synthetic image/instruction/action curriculum generation. |
| `english_to_glyph_compiler.py` | Hand-written English parser producing three-character commands. |
| `autonomous_topological_agent.py` | Candidate generation, associative scoring, and sandbox verification. |
| `sleep_cycle_training.sh` | MLX LoRA training and model fusion from collected command pairs. |
| `data/` | Saved datasets and 3,000 generated images. |
| `adapters/` | Saved LoRA adapter checkpoints, including step 500. |
| `model/` | Model configurations, tokenizers, and upstream model cards. Full model weight files are absent. |
| `verified_patterns/` | A saved even-sum Lean proof and an earlier failed attempt. |
| [RESEARCH.md](RESEARCH.md) | Research findings, related implementations, limitations, and the decision to stop. |

## Small example

The command language represents “Move the red cube to position 2” as `XMC`:

- `X`: red cube
- `M`: move
- `C`: position 2

The English compiler can be explored without loading a model:

```bash
python3 english_to_glyph_compiler.py
```

It prints the compiled command; its displayed handoff does not invoke a separate AI agent.

## Environment and limitations

These are standalone experimental scripts, with no pinned dependency set or integrated application. Depending on the script, dependencies include Python, Pillow, the OpenAI Python package, Lean 4, MLX-LM, and MLX-VLM. Local model paths and API model names reflect the original experiments and may need adjustment. API credentials are read from environment variables.

The training script installs or upgrades MLX-LM and **overwrites `data/train.jsonl` and `data/valid.jsonl`** with `data/topological_train.jsonl`. The curriculum generator also overwrites the training dataset. Preserve the historical data before rerunning these scripts.

The saved artifacts have several important limitations:

- The sandbox has only 16 successful command sequences from its reset state.
- Candidate generation has access to the sandbox's valid character pools; discovery is not fully unconstrained.
- Training and validation contain the same 16 records, so they are not an independent evaluation.
- The larger topological dataset contains 5,002 records, many inconsistent with the current command mappings and reset state.
- Both exported model directories reference missing `model.safetensors` files. Adapter checkpoints are included, but do not constitute complete standalone models.
- No end-to-end benchmark demonstrates storage savings, improved reasoning, or a larger usable context window.

All original project files are preserved. Existing third-party model cards and notices remain with the model metadata; their applicable terms are not replaced by this README.
