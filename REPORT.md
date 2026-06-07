# What's Surprisingly Different Between Base and Instruct Models?
### Predicting per-token base↔instruct divergence and studying where the predictor is wrong

---

## 1. Executive Summary

**Research question (one sentence).** Where does instruction tuning change a base model's
next-token distribution in ways a learned predictor *cannot* anticipate from simple
base-side context — and what kinds of tokens are those?

**Key finding (one sentence).** Most of the per-token KL between a base and its instruct
model is *predictable* from the base model's own uncertainty (entropy and token-surprise) —
a gradient-boosted predictor reaches cross-corpus R² ≈ 0.60–0.68 — and the **residual**
(where the predictor is wrong) is not noise: it concentrates, with high statistical
significance, on **safety prompts** and on **discourse/structural pivots and
alignment-injected content** (`If`, `However`, `Additionally`, `Safety`, `guidelines`,
`assistant`, `Alibaba`, `provide`), i.e. exactly the behaviors post-training installs.

**Practical implication.** The "surprising" part of alignment is *localizable and cheap to
find*: rank tokens by divergence-prediction residual and you recover refusal-recovery
pivots, persona/identity injections, and helpfulness framing — without labels. This refines
the Superficial Alignment Hypothesis: the *average* shift is superficial and predictable
(base entropy), but the *unpredictable residual* is substantive and alignment-specific.

---

## 2. Research Question & Motivation

The dominant view (Superficial Alignment Hypothesis; LIMA, URIAL) is that
instruction/alignment tuning mostly changes **style/format**, leaving knowledge intact.
URIAL's token-distribution-shift analysis supports this: base and aligned models agree on
the top-1 token ~78% of the time, and the positions that shift are dominated by
stylistic/discourse tokens and decay with position.

The user's hypothesis goes one step further: **train a model to predict the per-token
divergence, then look where the predictor is wrong** — the intuition that *systematic,
learnable* divergence is "boring stylistic alignment," while *unpredictable* divergence is
where something genuinely surprising happens. **No prior work trains an explicit divergence
predictor and studies its residuals** (URIAL/proxy-tuning/EFT/Branching-Factor all
characterize the *average* shift). That residual is this project's object of study.

**Hypotheses.**
- **H1 (predictability).** Per-token KL(P_inst‖P_base) is largely predictable from base-side
  context → high predictor R².
- **H2 (residual is structured).** After subtracting the prediction, the residual is not
  uniform — it concentrates on identifiable slices.
- **H3 (residual ≠ generic style).** High-residual tokens are disproportionately
  safety/refusal/persona/helpfulness pivots, not the generic stylistic tokens that dominate
  the *predictable* shift.

---

## 3. Experimental Setup

### Models (real model pairs, run locally on GPU)
Three ungated base/instruct pairs sharing a tokenizer within each pair:

| Pair | Base | Instruct | Params | Vocab |
|---|---|---|---|---|
| Qwen2.5-0.5B (primary) | `Qwen/Qwen2.5-0.5B` | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | 151,936 |
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | 151,936 |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` | `Qwen/Qwen2.5-7B-Instruct` | 7B | 152,064 |

Loaded in fp16, base on `cuda:0`, instruct on `cuda:1` (4× RTX A6000 48 GB available).

### Per-token divergence measurement
For each instruction we **generate a greedy response with the instruct model**
(chat-formatted prompt, `do_sample=False`, ≤80 new tokens). We then **teacher-force the
identical full token sequence (chat-formatted prompt + response) through *both* models** and,
at each response position `t`, compute over the full vocabulary:

- **target** `kl = KL(P_inst‖P_base)` (primary); also `base_rank` (URIAL's η) and
  `logprob_delta = log P_inst(o_t) − log P_base(o_t)` (the EFT/proxy "behavior delta");
- **base-side features** (the candidate *predictable* structure): response position `pos`,
  base entropy, base top-1 prob, base log-prob of the emitted token, plus token-class flags
  (stylistic/punct/number/word-leading) and token length.

Giving **both models the identical chat-formatted context** isolates the weight difference —
the cleanest measure of distributional divergence (cf. EFT/proxy-tuning, which treat the
logit difference under a shared context as the behavior signal).

### Predictor & residual protocol (no train/test leakage)
- **Train** the divergence predictor on **dolly-15k** (clean instructions, `context==""`;
  1,200–1,500 instructions per pair).
- **Analyze residuals** on **just-eval-instruct** (URIAL's 1,000-instruction set, tagged by
  `category` incl. **200 safety prompts**, `task`, `topic`) — a disjoint corpus.
- **Predictor:** sklearn `HistGradientBoostingRegressor` (gradient-boosted trees, 600 iters,
  early stopping) on the base-side features. **Baselines:** linear regression on
  position-only, entropy-only, base-logprob-only.
- **Residual** = `actual_kl − predicted_kl` on just-eval. Sliced by category (safety vs
  regular), token class, and position bin.

### Reproducibility
Seed 42 everywhere; greedy decoding (deterministic). Python 3.12, torch 2.12+cu130,
transformers 5.10.2, scikit-learn 1.9, datasets 5.0. Hardware: RTX A6000. Total wall-clock
≈ 35 min (0.5B+1.5B) plus ≈ 15 min (7B incl. download). Per-token tables in
`results/*_{dolly,justeval}.parquet`; metrics in `results/*_metrics.json`.

---

## 4. Results

### 4.0 Measurement validation (reproduces URIAL stylized facts)
Top-1 base/instruct agreement is **0.83–0.85** (URIAL reports ~78% on Llama/Mistral);
mean per-token KL is small (0.21–0.24 nats) and **decays sharply with position** (e.g. 0.5B:
mean KL 0.68 at positions 0–4 → 0.12 at 40+). See `figures/*_kl_vs_pos.png`. Our measurement
matches the literature.

### 4.1 H1 — Per-token KL is largely predictable (from base uncertainty, *not* position)

Cross-pair summary (`results/summary_table.csv`):

| Pair | top-1 agree | mean KL | R² pos-only | R² entropy-only | R² base-logp-only | **R² GBT (val)** | **R² GBT (x-corpus)** | Spearman |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.829 | 0.207 | 0.015 | 0.196 | 0.398 | **0.704** | **0.599** | 0.821 |
| Qwen2.5-1.5B | 0.852 | 0.214 | 0.046 | 0.394 | 0.382 | **0.693** | **0.680** | 0.849 |
| Qwen2.5-7B | 0.826 | 0.240 | 0.022 | 0.281 | 0.510 | **0.621** | **0.626** | 0.885 |

- The gradient-boosted predictor explains **~60–68%** of per-token KL variance on a **disjoint
  corpus**, with Spearman ρ(pred, actual) **0.82–0.89**. **H1 supported.**
- **Position alone is nearly useless** (R² 0.02–0.05). The predictable structure is the
  base model's own uncertainty: permutation importance (drop in R²) is dominated by
  `base_tok_logprob` and `base_entropy` (≈0.4–0.6 each), with `pos` ≈0.02–0.08 and all
  stylistic/lexical flags <0.02. This **refines URIAL**: the famous "KL decays with
  position" is mostly *mediated by entropy decay*, not position per se.
  (`figures/*_featimp.png`, `*_calibration.png`.)

### 4.2 H2 — The residual is structured, not noise
Kruskal–Wallis across token classes is hugely significant for every pair (p < 1e-22).
Residual also varies systematically with position bin even though position is a predictor
feature (the predictor under-predicts early-position divergence). **H2 supported.**

### 4.3 H3 — The residual is alignment-specific (safety + pivots), not generic style

**Safety vs regular prompts** (Mann–Whitney one-sided, residual on just-eval):

| Pair | mean resid (safety) | mean resid (regular) | rank-biserial | p |
|---|---|---|---|---|
| Qwen2.5-0.5B | **+0.138** | −0.012 | 0.322 | ≈ 0 |
| Qwen2.5-1.5B | **+0.085** | −0.015 | 0.302 | 3e-284 |
| Qwen2.5-7B | **+0.061** | −0.012 | 0.141 | 2e-164 |

Safety prompts are **where the predictor most under-estimates divergence**, in all three
pairs (small-to-medium effect; the effect shrinks with scale — larger models' safety
behavior is somewhat *more* predictable, but remains highly significant).
See `figures/*_resid_category.png` and `figures/summary_across_pairs.png`.

**Which token strings are most over-divergent** (mean residual, n≥5; `results/*_residual_tokens.csv`):

- **0.5B:** `Additionally`, `politics`, `sex`, `inquiries`, `If`, `answers`, `violence`,
  `encourage`, `topics`, `I`, `However`, `political`, `distress` …
- **7B:** `Safety`, `Such`, `assistant`, `guidelines`, `If`, `discuss`, `planning`,
  `Alibaba`, `Specifically`, `October`, `Instead`, `provide`, `recommend` …

Two interpretable families emerge — **neither is the generic stylistic token the
*predictable* shift is made of:**

1. **Discourse/structural pivots at clause boundaries** — the aligned model commits to a
   structured, helpful continuation far more sharply than base entropy/surprise predicts:
   - 0.5B: *"…won't assist with that request. **[If]** you have…"* (KL 19.9, base ranks `If`
     only 3rd) — a **refusal-recovery pivot**.
   - 0.5B: *"…about the weather in London. **[Additionally]**, you…"*; *"…academic context.
     **[The]** phrase…"* (sentence/list openers).
   - 7B: *"…built-in clock functionality. **[If]** you need…"*; *"…distinctive appearance.
     **[However]**, it…"*; *"…the reaction. **[For]** example:"* — helpful follow-up pivots.
2. **Safety / persona / identity / helpfulness injections** — explicit post-training content:
   `Safety`, `guidelines`, `assistant`, **`Alibaba`** (Qwen's maker — *identity injection*),
   `October` (knowledge-cutoff framing), `provide`/`recommend`/`discuss` (helpfulness verbs),
   and sensitive-topic words (`sex`, `violence`, `politics`, `distress`).

**H3 supported:** the unpredictable residual recovers exactly the behaviors alignment
installs (refusal recovery, helpful structure, persona/identity, safety framing), not the
generic `However/Here` style tokens that the predictor already accounts for.

Raw artifacts: `results/*_residual_top.csv` (top individual tokens with context-locatable
`example_id`/`pos`), `results/*_residual_tokens.csv` (aggregated by token string).

---

## 5. Analysis & Discussion

- **The hypothesis holds.** "Where the divergence predictor is wrong" is indeed "where the
  interesting stuff is": refusal-recovery, persona/identity, and helpfulness-structure
  pivots, plus a significant safety-prompt concentration — surfaced with **no labels**,
  purely from prediction residuals.
- **Refinement of the Superficial Alignment Hypothesis.** The *bulk* of the base→instruct
  KL is explained by the base model's own uncertainty (entropy/token-surprise): where the
  base is unsure, alignment sharpens — consistent with Branching-Factor's "alignment
  surfaces latent low-entropy paths." Position is almost irrelevant once entropy is known.
  The *residual* is the genuinely non-superficial part, and it is alignment-specific.
- **Refusal-recovery pivots** are a clean qualitative discovery: right after a refusal, the
  instruct model deterministically emits boilerplate ("If you have any other questions…")
  that the base model assigns low rank — a high-KL, high-residual signature of RLHF/SFT
  templating.
- **Scale trend.** The safety residual effect shrinks 0.32 → 0.30 → 0.14 from 0.5B → 7B:
  larger instruct models integrate safety behavior more smoothly (more predictable from
  context), though it remains overwhelmingly significant. Predictor R² is stable (~0.6–0.7).
- **Effect sizes & significance.** All headline effects are statistically significant with
  reported effect sizes (rank-biserial), on a corpus disjoint from training, replicated
  across three model scales — not a single-model or p-hacked artifact.

---

## 6. Limitations

- **Sequence choice.** We score the *instruct model's own greedy decode*; that text is
  in-distribution for the instruct model, which can inflate KL on instruct-preferred tokens
  (a known URIAL confound). We mitigate by using a shared context and by analyzing
  *residuals* (position/entropy-adjusted), but scoring base-decoded or reference sequences
  would be a stronger control (left to future work).
- **`base_tok_logprob` as a feature** is informative-by-construction (a token the base finds
  unlikely tends to diverge); we keep it because it *is* legitimate base-side predictable
  structure, and the residual is defined relative to the *best* such predictor. Excluding it
  only raises the residual, never the reverse.
- **Token-class proxy for POS.** We use lightweight token-class flags rather than true POS
  tagging (subword tagging is noisy); the qualitative families are read from token strings.
- **Model family.** All pairs are Qwen2.5 (RLHF/DPO-aligned). Gated SFT-only pairs
  (Mistral-7B-v0.1, Llama-2-chat) were not downloadable without an HF token, so we could not
  cleanly separate SFT-only vs RLHF effects; scale (0.5B→7B) is covered instead.
- **Greedy, ≤80 tokens.** Short greedy responses; longer/sampled generations and other
  decoding settings are untested.

---

## 7. Conclusions & Next Steps

**Answer to the research question.** Per-token base→instruct divergence is mostly a
*predictable* function of the base model's own uncertainty (R²≈0.6–0.68; position barely
matters). The **unpredictable residual** — where a divergence predictor is wrong — is *not*
noise: it concentrates, significantly and across three model scales, on **safety prompts**
and on **discourse/structural pivots and alignment-injected content** (refusal-recovery
boilerplate, persona/identity such as `Alibaba`, helpfulness verbs, safety framing). The
user's intuition is confirmed: the residual is where the genuinely alignment-specific,
non-superficial behavior lives.

**Next steps.** (1) Score base-decoded and reference sequences to remove the
which-model-generated confound; (2) add an SFT-only pair (with HF access) to separate SFT vs
RLHF residual signatures; (3) train a *residual* probe on base hidden states + logit-lens
attribution (LogitLens4LLMs) to mechanistically localize where residual tokens are decided;
(4) use the residual as an *unsupervised detector* of refusal/persona/safety tokens and
validate against labels.

---

## References (used)
- Lin et al. 2023, **URIAL: The Unlocking Spell on Base LLMs**, arXiv:2312.01552 (token
  distribution shift; just-eval-instruct).
- Yang, Li, Holtzman 2026, **LLM Probability Concentration / Branching Factor**,
  arXiv:2506.17871 (alignment sharpens distribution).
- Liu et al. 2024, **Tuning Language Models by Proxy**, arXiv:2401.08565 (logit-difference
  behavior signal).
- Mitchell et al. 2023, **Emulated Fine-Tuning**, arXiv:2310.12962 (behavior delta = log
  π_inst − log π_base).
- Zhou et al. 2023, **LIMA**, arXiv:2305.11206 (Superficial Alignment Hypothesis).
- **Revisiting the Superficial Alignment Hypothesis** 2024, arXiv:2410.03717 (counter-evidence).
- Datasets: `re-align/just-eval-instruct`; `databricks/databricks-dolly-15k`.
- Models: `Qwen/Qwen2.5-{0.5B,1.5B,7B}` and `-Instruct`.
