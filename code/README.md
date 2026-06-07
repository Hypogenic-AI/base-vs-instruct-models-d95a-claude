# Cloned Repositories

Four repositories supporting the base-vs-instruct divergence study. Cloned with
`--depth 1`. Large repo contents are git-ignored at the workspace root.

---

## 1. URIAL  (`urial/`)  ⭐
- **URL**: https://github.com/Re-Align/URIAL
- **Paper**: Lin et al. 2023, "The Unlocking Spell on Base LLMs" (arXiv 2312.01552)
- **Purpose**: Source of the token-distribution-shift analysis (base-rank / unshifted /
  marginal / shifted) and of the `just-eval-instruct` prompt set. Tuning-free alignment
  via in-context examples for base models.
- **Key files**:
  - `src/unified_infer.py`, `src/hf_models.py` — batched HF/vLLM inference; useful for
    generating base- and instruct-model responses to score.
  - `urial_prompts/` — the URIAL system prompt + restyled ICL examples (lets you make a
    base model behave like an instruct model **without** weight changes — a useful third
    condition for divergence analysis).
  - `evaluate/` — multi-aspect evaluation utilities.
- **Note**: The published repo focuses on *inference + alignment*; the per-position
  KL/base-rank plotting from §2 of the paper is straightforward to reimplement (see
  `datasets/README.md` "Per-token divergence (sketch)").

## 2. proxy-tuning  (`proxy-tuning/`)  ⭐
- **URL**: https://github.com/alisawuffles/proxy-tuning
- **Paper**: Liu et al. 2024, "Tuning Language Models by Proxy" (arXiv 2401.08565)
- **Purpose**: Reference implementation of combining models at the logit level via the
  instruct−base difference — the cleanest existing code for manipulating/measuring the
  per-token behavior delta.
- **Key files**:
  - `modeling/dexperts.py` — the DExperts logit-combination class
    (`base + (expert − anti_expert)`); adapt directly to compute per-token KL /
    log-prob delta between a base and instruct model sharing a vocab.
  - `notebooks/S6.1_analysis.ipynb` — **token-level analysis of which tokens the offset
    most affects** (reasoning + stylistic). This is the closest existing analysis to the
    project's "which tokens diverge" question.
  - `notebooks/S6.2_alpha_analysis.ipynb` — scaling the offset strength α.
  - `data.zip` (~part of the 111 MB), `results/`, `eval/` — eval harness & cached results.
- **Deps**: see `requirements.txt` (transformers, torch, etc.).

## 3. branching-factor / LLMBranchingFactor  (`branching-factor/`)
- **URL**: https://github.com/yangalan123/LLMBranchingFactor
- **Paper**: Yang, Li, Holtzman 2026, "LLM Probability Concentration" (arXiv 2506.17871)
- **Purpose**: Compute the Branching Factor (entropy-based concentration) and run the
  "nudging" experiments (prefix a base model with aligned-style tokens). Provides an
  information-theoretic complement to KL: an alternative *target* the predictor could use,
  and a strong baseline feature (entropy/BF) for predicting divergence.
- **Key files**:
  - `language_modeling/main.py`, `language_modeling/nudging_probabilistic_computation.py`
    — BF + nudging pipeline.
  - `uncertainty_quantification/` — shared entropy/UQ utilities.
  - `visualization/`, `pyproject.toml`, `setup.py` — installable package + plots.

## 4. emulated-fine-tuning  (`emulated-fine-tuning/`)
- **URL**: https://github.com/eric-mitchell/emulated-fine-tuning
- **Paper**: Mitchell et al. 2023, EFT (arXiv 2310.12962)
- **Purpose / status**: **Stub only** — the README says "Work in progress, check back
  soon!" and no code is released. Kept as a pointer. The EFT operation is trivial to
  reproduce, though: emulated logits = `base_logprobs + (instruct_logprobs − base_logprobs)`,
  i.e. the behavior delta *is* the per-token divergence signal. No code dependency.

---

## How these map to the hypothesis
- **Compute the per-token divergence target** (KL / base-rank / log-prob delta): adapt
  `proxy-tuning/modeling/dexperts.py`; methodology from URIAL §2.
- **Alternative target / strong feature** (entropy, Branching Factor): `branching-factor/`.
- **Prompts to run on**: `datasets/just-eval-instruct` (URIAL's set) and `dolly-15k`.
- **"Where the predictor is wrong"**: slice residuals by `just-eval-instruct` `category`/
  `task`/`topic`; inspect tokens with `proxy-tuning` S6.1-style analysis and
  `LogitLens4LLMs` for layer attribution.
