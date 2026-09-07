# Research assessment

Assessment date: September 7, 2026.

This document summarizes the project's investigation and design discussion. Related papers, documentation, and selected source paths were inspected; their model benchmarks were not independently reproduced. Reported results should not be interpreted as measured performance of this repository.

## Question investigated

Could a database store reusable English symbols once, represent documents or knowledge as spatial graphs, and change viewpoints, connections, or geometric arrangements to increase usable memory and reduce model token consumption?

The investigation separated four mechanisms:

1. Shared symbols and references: dictionary encoding and deduplication.
2. Changing graph viewpoints: question-dependent retrieval.
3. Evolving relationships: temporal or event-based memory.
4. Geometric representation: vector-symbolic composition, visual encodings, or rotation-assisted numerical compression.

## Findings

Shared elements do not remove the need to represent their occurrences, ordering, relationships, or history. Those arrangements carry information and consume storage. Unlimited exact information cannot be preserved in fixed finite storage. Rotation by itself does not remove this constraint. See [Shannon's information theory](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf).

Database bytes, transfer bytes, model input tokens, and inference working memory are different budgets. A technique may improve one while increasing another. Reconstructing a compressed paragraph and sending the original text to a model ordinarily preserves its original token cost. Retrieving only relevant evidence can reduce input tokens without losslessly compressing the entire source history.

## Related implementations

### MAGMA: multiple relational views

[Repository](https://github.com/FredJiang0324/MAGMA) · [Published paper](https://aclanthology.org/2026.acl-long.1709.pdf)

MAGMA connects memories through semantic, temporal, causal, and entity relationships, and adapts traversal to the question. It is the closest match to changing viewpoints over an existing memory. Its evaluation supports adaptive retrieval, while acknowledging graph-construction errors and added storage and engineering complexity. The implementation retains narrative text and embeddings; it does not replace documents with a fixed alphabet.

### Graphiti / Zep: changing relationships

[Repository](https://github.com/getzep/graphiti) · [Zep paper](https://arxiv.org/abs/2501.13956)

Graphiti maintains facts, source episodes, and temporal relationships. Zep's reported evaluation demonstrates that selective memory retrieval can substantially reduce context size on long-conversation tasks. This is not equivalent to encoding all source information into the retrieved tokens, and Zep's reported results are not guaranteed for an arbitrary Graphiti deployment.

### TorchHD: geometric symbolic composition

[Repository](https://github.com/hyperdimensional-computing/torchhd) · [Paper](https://arxiv.org/abs/2205.09208)

TorchHD implements vector-symbolic architectures, including holographic representations. Its graph structure binds endpoint vectors, uses permutations to represent direction, and bundles edges into aggregate vectors. This provides a concrete interpretation of reusable symbols combined through transformations. Retrieval capacity and interference must be measured, including the cost of the symbol dictionary. It is not a lossless paragraph database.

### DeepSeek-OCR: optical context compression

[Repository](https://github.com/deepseek-ai/DeepSeek-OCR) · [Paper](https://arxiv.org/abs/2510.18234) · [Follow-up study](https://arxiv.org/abs/2512.03643)

DeepSeek-OCR explores representing text-rich images with fewer visual tokens. Its compression experiment reported high but imperfect text recovery. A follow-up found that strong reconstruction did not necessarily translate into better language modelling, within its tested English Wikipedia setting. Using this for reasoning requires a compatible model to consume the compressed representation; decoding everything back to text for another model restores the expanded input.

### TurboQuant: rotation-assisted numerical compression

[Paper](https://arxiv.org/abs/2504.19874) · [Google explanation](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) · [Community MLX implementation](https://github.com/yzamari/mlx-turboquant) · [MLX-VLM implementation PR](https://github.com/Blaizzy/mlx-vlm/pull/858)

TurboQuant uses rotations and quantization to reduce the representation size of numerical vectors, including attention-cache data. Its paper explicitly treats quantization as lossy, even when downstream benchmark performance is preserved. This can reduce bytes per cached token, not the number of text tokens. Community implementations require separate validation; the inspected MLX repository documents retrieval failures and configurations with little total-memory benefit.

## Human-inspired memory direction

A later proposed design used event records, a small active workspace, selective retrieval, predictions compared against observed outcomes, and background consolidation. These mechanisms also have existing implementations:

- [Nemori](https://github.com/nemori-ai/nemori): episode segmentation, semantic knowledge, and predict-calibrate learning.
- [Letta context repositories](https://www.letta.com/blog/context-repositories/): persistent editable memory and background reflection.
- [EverMemOS paper](https://arxiv.org/abs/2601.02163): encoding, consolidation, and context reconstruction; the original repository URL now redirects to [EverOS](https://github.com/EverMind-AI/EverOS).
- [Generative Agents](https://github.com/joonspk-research/generative_agents): observation, retrieval, reflection, and planning.

These are engineering analogies to cognitive processes, not demonstrations of human-equivalent memory. Useful inspiration includes [complementary learning systems](https://www.nature.com/articles/s41593-023-01382-9) and [human replay research](https://www.nature.com/articles/s41467-024-51582-5).

## X discussion leads

X access was partial. These were discovery leads rather than independent evidence of performance:

- [Akshay Pachaar on shared Graphiti memory](https://x.com/akshay_pachaar/status/1969740898021085664), readable through a [Thread Reader copy](https://threadreaderapp.com/thread/1969740898021085664.html).
- [Prince Canuma on TurboQuant in MLX](https://x.com/Prince_Canuma/status/2036611007523512397), linked from GitHub discussion; the original X page could not be retrieved.
- [Avi Chawla on Graphiti MCP memory](https://x.com/_avichawla/status/1986160137363660961), an indexed lead whose original page could not be retrieved.

## What a meaningful experiment would require

The defensible remaining hypothesis was whether a compact geometric index over an evolving graph could retrieve evidence more cheaply than strong text and graph baselines, at comparable accuracy. Exact source records would remain available.

A comparison would need identical models, sources, and questions across ordinary hybrid retrieval, temporal graph retrieval, adaptive graph traversal, and a geometric-index variant. Numerical cache compression should be tested separately. Measures should include correctness, temporal updates, retrieval precision/recall, total storage, ingestion and query costs, peak memory, and latency.

Any feasibility percentages discussed during exploration were subjective engineering estimates, not calculated probabilities. They should not guide investment decisions. Likewise, suggested 30% cost savings or 10-percentage-point accuracy gains were proposed acceptance thresholds, not predictions.

## Decision

The project owner decided not to pursue further development and requested public preservation of the work. The broad architecture overlaps substantially with existing systems, and the current sandbox does not establish a differentiated advantage. No claim of novel storage capacity, human-like cognition, or improved model context is made.
