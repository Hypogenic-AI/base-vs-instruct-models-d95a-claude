# Datasets

This directory holds datasets for studying the **token-by-token divergence between base
and instruction-tuned LLMs**. Data files are NOT committed to git (see `.gitignore`);
follow the download instructions to reproduce. Small `samples.json` files are committed
for reference.

The "data" for this project is really two things:
1. **Instruction/prompt corpora** — text on which we run *both* a base and an instruct
   model to measure per-token divergence (KL, base-rank, log-prob delta).
2. **Base/instruct model pairs** — the two distributions being compared. These are
   model weights on the HuggingFace Hub (see the table at the bottom). They are NOT
   downloaded here because they are multi-GB; the experiment runner should pull the
   chosen pair on demand.

---

## Dataset 1: just-eval-instruct  ⭐ (primary prompt set)

### Overview
- **Source**: HuggingFace `re-align/just-eval-instruct`
- **Size**: 1,000 instructions (single `test` split)
- **Format**: HuggingFace `Dataset` (Arrow), saved under `just-eval-instruct/hf/`
- **Task**: Open-domain instruction following / response generation
- **Fields**: `id`, `instruction`, `source_id`, `dataset`, `category`, `task`, `topic`,
  `difficulty`, `output`, `generator`
- **Why this one**: This is the *exact* evaluation set used by URIAL (Lin et al., 2023),
  the paper that introduced the base-vs-aligned token-distribution-shift analysis this
  project builds on. It aggregates 1,000 diverse instructions from 9 datasets
  (AlpacaEval, MT-bench, LIMA, etc.) and is tagged by `category` (regular vs. safety),
  `task`, and `topic` — which makes it ideal for slicing the divergence-predictor's
  residuals (e.g., "where is the predictor most wrong, by topic/task?").

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("re-align/just-eval-instruct")
ds.save_to_disk("datasets/just-eval-instruct/hf")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/just-eval-instruct/hf")["test"]
prompts = ds["instruction"]            # 1,000 instructions
# the first 800 are "regular", the last 200 are safety-related (see `category`)
```

### Notes
- `output` / `generator` are null — you generate responses yourself with the model pair.
- Safety prompts (`category != "regular"`) are where alignment shifts tokens most
  (refusals/disclaimers), so they are a natural stress test for a divergence predictor.

---

## Dataset 2: dolly-15k  (larger held-out instruction set)

### Overview
- **Source**: HuggingFace `databricks/databricks-dolly-15k`
- **Size**: 15,011 instruction-context-response triples (single `train` split)
- **Format**: HuggingFace `Dataset` (Arrow), saved under `dolly-15k/hf/` (~12 MB)
- **Task**: Instruction following across 8 categories (open QA, closed QA,
  brainstorming, classification, summarization, info-extraction, creative, general QA)
- **Fields**: `instruction`, `context`, `response`, `category`
- **License**: CC BY-SA 3.0
- **Why this one**: Provides enough volume to *train* the per-token divergence predictor
  on one corpus and hold out just-eval-instruct (or a category slice) for residual
  analysis, avoiding train/test leakage. The `category` field gives natural domain
  splits.

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("databricks/databricks-dolly-15k")
ds.save_to_disk("datasets/dolly-15k/hf")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/dolly-15k/hf")["train"]
```

### Notes
- Many examples have a non-empty `context`; for clean instruction-following prompts you
  may want to filter to `context == ""` or fold context into the prompt.

---

## Base / Instruct Model Pairs (download on demand, NOT stored here)

Per-token KL/divergence requires the base and instruct model to **share the same
tokenizer/vocabulary**. All pairs below satisfy this. Verified present on the Hub
(2026-06).

| Base model | Instruct/chat model | Size | Gated? | Tokenizer shared | Recommendation |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-0.5B` | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | No | Yes | ⭐ Best for limited compute; fast, ungated |
| `Qwen/Qwen2.5-1.5B` | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | No | Yes | Good quality/compute trade-off |
| `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | No | Yes | Ungated Llama-arch pair |
| `google/gemma-2-2b` | `google/gemma-2-2b-it` | 2B | Yes* | Yes | Strong small pair (*license accept) |
| `mistralai/Mistral-7B-v0.1` | `mistralai/Mistral-7B-Instruct-v0.1` | 7B | Yes* | Yes | Used in URIAL (SFT-only instruct) |
| `meta-llama/Llama-2-7b-hf` | `meta-llama/Llama-2-7b-chat-hf` | 7B | Yes* | Yes | Used in URIAL (SFT+RLHF instruct) |

\* Gated pairs need `huggingface-cli login` and one-time license acceptance on the model page.

### Load a pair
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
name_base, name_inst = "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"
tok  = AutoTokenizer.from_pretrained(name_inst)
base = AutoModelForCausalLM.from_pretrained(name_base, torch_dtype=torch.float16, device_map="auto")
inst = AutoModelForCausalLM.from_pretrained(name_inst, torch_dtype=torch.float16, device_map="auto")
# assert base and inst share vocab size before computing per-token KL
assert base.config.vocab_size == inst.config.vocab_size
```

### Per-token divergence (sketch — what the experiment computes)
For a prompt, generate a response with the **instruct** model (greedy), then for each
position `t` feed the same prefix to **both** models and compare next-token distributions:
- `KL(P_inst || P_base)` at position t
- base-rank of the instruct's top token under `P_base` (URIAL's η)
- log-prob delta `log P_inst(o_t) - log P_base(o_t)` (the EFT/proxy "behavior delta")

These per-token signals are the **targets** for the divergence predictor in the
hypothesis; the residuals (predictor errors) are the object of study.
