# Resources Catalog

## Summary
Resources gathered for **"What's surprisingly different between base and instruct
models?"** — predicting per-token base↔instruct divergence (KL) and studying where the
predictor errs. Covers 11 papers, 2 instruction datasets (+ 6 documented model pairs),
and 4 code repositories.

## Papers
Total downloaded: **11** (all arXiv PDFs, verified valid). See `papers/README.md`.

| Title | Authors | Year | File | Key info |
|---|---|---|---|---|
| URIAL: Unlocking Spell on Base LLMs ⭐ | Lin et al. (AI2/UW) | 2023 | papers/2312.01552_*.pdf | Token distribution shift; KL/base-rank per position; just-eval-instruct |
| LLM Probability Concentration (Branching Factor) ⭐ | Yang, Li, Holtzman (UChicago) | 2026 | papers/2506.17871_*.pdf | Alignment sharpens dist.; entropy/BF metric & codebase |
| Tuning LMs by Proxy ⭐ | Liu et al. (UW/AI2) | 2024 | papers/2401.08565_*.pdf | logit difference = behavior signal; §6.1 token analysis |
| Emulated Fine-Tuning (EFT) ⭐ | Mitchell et al. (Stanford) | 2023 | papers/2310.12962_*.pdf | behavior delta = log π_inst − log π_base; KL-RL view |
| LIMA | Zhou et al. (Meta) | 2023 | papers/2305.11206_*.pdf | Superficial Alignment Hypothesis origin |
| Revisiting Superficial Alignment | — | 2024 | papers/2410.03717_*.pdf | Counter-evidence: post-training adds capability |
| Is ICL Sufficient for Instruction Following? | — | 2024 | papers/2405.19874_*.pdf | Bounds how superficial the gap is |
| Shadow-FT | — | 2025 | papers/2505.12716_*.pdf | base/instruct weight delta is transferable |
| Contrastive Decoding Improves Reasoning | O'Brien & Lewis | 2023 | papers/2309.09117_*.pdf | expert−amateur log-prob diff decoding |
| DoLa | Chuang et al. | 2023 | papers/2309.03883_*.pdf | layer-contrast; where divergence localizes |
| LogitLens4LLMs | — | 2025 | papers/2503.11667_*.pdf | logit-lens tooling for residual attribution |

## Datasets
Total downloaded: **2** (data git-ignored; samples + READMEs committed). See `datasets/README.md`.

| Name | Source | Size | Task | Location | Notes |
|---|---|---|---|---|---|
| just-eval-instruct ⭐ | HF `re-align/just-eval-instruct` | 1,000 | instruction following | datasets/just-eval-instruct/hf | URIAL's set; tagged category/task/topic; 200 safety prompts |
| dolly-15k | HF `databricks/databricks-dolly-15k` | 15,011 | instruction following | datasets/dolly-15k/hf | Larger train set; category-tagged; CC BY-SA 3.0 |

**Model pairs (documented, not downloaded — multi-GB, fetch on demand):**
Qwen2.5-0.5B/-Instruct ⭐, Qwen2.5-1.5B/-Instruct, TinyLlama-1.1B base/Chat,
gemma-2-2b/-it, Mistral-7B-v0.1/-Instruct-v0.1, Llama-2-7b/-chat. All share a tokenizer
within the pair (required for per-token KL). Table + load snippet in `datasets/README.md`.

## Code Repositories
Total cloned: **4**. See `code/README.md`.

| Name | URL | Purpose | Location | Notes |
|---|---|---|---|---|
| URIAL ⭐ | github.com/Re-Align/URIAL | dist-shift method + just-eval-instruct + base inference | code/urial | analysis code reimplementable per paper §2 |
| proxy-tuning ⭐ | github.com/alisawuffles/proxy-tuning | logit-difference (DExperts) + token-level analysis | code/proxy-tuning | dexperts.py, notebooks/S6.1 |
| LLMBranchingFactor | github.com/yangalan123/LLMBranchingFactor | entropy/BF + nudging | code/branching-factor | installable; UChicago |
| emulated-fine-tuning | github.com/eric-mitchell/emulated-fine-tuning | EFT pointer | code/emulated-fine-tuning | **stub repo** (no code released); method trivial to reproduce |

## Resource Gathering Notes

### Search Strategy
Extracted concepts from the hypothesis (base vs instruct, per-token KL/distribution shift,
predicting divergence, residuals). Primary literature search via the **arXiv API**
(`arxiv_search.py`, ~12 batched queries) after the paper-finder service was unavailable.
GitHub/HF resources located by probing canonical org/repo URLs and one targeted web search
for the Branching-Factor codebase.

### Selection Criteria
Prioritized papers that (a) measure base↔instruct divergence directly (URIAL, BF),
(b) operationalize the per-token signal (proxy-tuning, EFT, contrastive decoding), or
(c) frame/challenge the superficial-alignment premise (LIMA, Revisiting, ICL-sufficiency).
Datasets chosen to match URIAL's protocol (just-eval-instruct) plus a larger held-out set
(dolly-15k) for train/test separation.

### Challenges Encountered
- **paper-finder service down**: returned HTTP 500 with backend `RateLimitError` on every
  attempt (fast and diligent, cache on/off). Fell back to the arXiv API, which found all
  the key papers including the foundational URIAL.
- arXiv relevance search is noisy (returns off-topic "proxy/alignment" hits); filtered by
  reading titles/abstracts.
- EFT code is an unreleased stub — noted; the method is a one-line log-prob identity.
- Model weights not downloaded (size); the experiment runner pulls the chosen pair.

### Gaps and Workarounds
- No existing repo implements a **divergence predictor + residual analysis** (the project's
  novelty) — to be built. URIAL §2 + proxy-tuning give all the measurement primitives.
- URIAL's exact per-position plotting code isn't in the public repo; reimplementation sketch
  provided in `datasets/README.md`.

## Recommendations for Experiment Design
1. **Primary dataset(s)**: train predictor on `dolly-15k`; analyze residuals on
   `just-eval-instruct` (rich tags, safety slice).
2. **Model pair(s)**: `Qwen2.5-0.5B`/`-Instruct` first (cheap, ungated, shared tokenizer);
   confirm on Mistral-7B (SFT) and, if access permits, Llama-2-7b/-chat (SFT+RLHF).
3. **Targets**: KL(P_inst‖P_base) primary; cross-check base-rank and log-prob delta.
4. **Predictable baseline to subtract**: position + base entropy/BF + base-prob of the
   instruct token + stylistic-token/POS flags. Study the residual after this.
5. **Code to reuse**: `proxy-tuning/modeling/dexperts.py` (logit combination),
   `branching-factor` (entropy/BF features), URIAL `src` (generation), LogitLens4LLMs
   (residual attribution).
