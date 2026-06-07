# Literature Review: What's Surprisingly Different Between Base and Instruct Models?

**Research hypothesis.** Take a base and an instruct model, measure their per-token
distributional divergence (e.g., KL) across positions, train a model to *predict* that
divergence from context, and study the positions where the **predictor is wrong** — the
intuition being that systematic, learnable divergence is "boring stylistic alignment,"
while *unpredictable* divergence is where something genuinely surprising happens.

---

## Research Area Overview

There is a now-substantial line of work characterizing how instruction/alignment tuning
(SFT and/or RLHF) changes a pretrained base model. The dominant empirical finding is the
**Superficial Alignment Hypothesis**: alignment mostly changes *style/format and
safety behavior*, while knowledge and reasoning come from pre-training. The strongest
direct evidence is **token distribution shift** analysis (URIAL): base and aligned models
agree on the top token at ~78% of positions (~92% within top-3), and the positions that
*do* shift are dominated by stylistic/discourse/refusal tokens, concentrated early in the
response, and shrinking in KL as decoding proceeds.

Three methodological threads make the per-token divergence *operational*:
1. **Distribution-shift analysis** (URIAL) — KL / base-rank / base-prob per position.
2. **Logit-difference methods** (proxy-tuning, EFT, contrastive decoding) — treat
   `log π_instruct − log π_base` as a usable signal (a "behavior delta" / implicit reward).
3. **Information-theoretic concentration** (Branching Factor) — alignment *sharpens* the
   distribution (lower entropy/BF), surfacing low-entropy paths already latent in the base.

The hypothesis sits squarely on top of thread (1)+(2): the per-token KL/delta is the
prediction *target*; the open question — under-explored in the literature — is what the
**residual** (unpredictable divergence) looks like.

---

## Key Papers

### Paper 1: URIAL — The Unlocking Spell on Base LLMs (Lin et al., 2023; arXiv 2312.01552)  ⭐
- **Key contribution**: Direct evidence for superficial alignment via *token distribution
  shift*; plus a tuning-free alignment method (URIAL) using ICL.
- **Methodology**: Decode the **aligned** model greedily to get response `o`. At each
  position `t`, feed the same prefix to the **base** model and rank `o_t` under `P_base`
  (the *base-rank* η). Buckets: **unshifted** (η=1), **marginal** (1<η≤3), **shifted**
  (η>3). Metrics per position: KL(P_align‖P_base), base-rank, base-prob of the aligned
  token.
- **Datasets**: `just-eval-instruct` (1,000 instructions from 9 sources), introduced here.
- **Models**: Llama-2-7b/-chat (RLHF), Llama-2-7b/Vicuna-7b (SFT), Mistral-7b/-Instruct.
- **Results**: ~77.7% unshifted, ~92.2% within top-3; shifted tokens are stylistic
  (`However`, `cannot`, `Here`, `Thank`, `Remember`, refusals). KL and base-rank **decay
  with position**. URIAL (3 ICL examples + system prompt) matches/beats SFT(+RLHF).
- **Relevance**: Defines the exact divergence measurement and the prompt set. The
  decay-with-position and the stylistic-token concentration are precisely the *predictable*
  structure a divergence predictor will learn — so residuals are "everything not explained
  by position + stylistic-token identity."

### Paper 2: LLM Probability Concentration / Branching Factor (Yang, Li, Holtzman, UChicago, 2026; arXiv 2506.17871)  ⭐
- **Key contribution**: **Branching Factor (BF)** = length-normalized exponentiated entropy
  (effective # of plausible next tokens); a token-invariant concentration metric.
- **Methodology**: Estimate BF from naturally sampled sequences (entropy-rate). Compare
  base vs aligned; track BF over position; "nudging" = prefix a base model with an
  aligned-style token (e.g., "Sure,").
- **Results**: Alignment cuts BF 2–5× (up to 12→1.2 at start); BF declines over a
  generation; nudging a base model reproduces the drop ⇒ alignment *surfaces latent
  low-entropy paths* rather than reshaping the manifold.
- **Relevance**: (a) BF/entropy is a strong **feature** and an alternative **target** for
  the predictor; (b) explains *why* divergence is largest early (high base entropy →
  room to shift); (c) same lab/affiliation as the user — methods and codebase are directly
  reusable.

### Paper 3: Tuning Language Models by Proxy (Liu et al., COLM 2024; arXiv 2401.08565)  ⭐
- **Key contribution**: `proxy-tuning` — steer a base model with the **logit difference**
  `(s_smallchat − s_smallbase)` added to `s_base` (DExperts form).
- **Results**: Closes 88–91% of base→chat gap; **§6.1**: the offset most promotes
  *reasoning and stylistic* tokens, not knowledge.
- **Relevance**: Cleanest code/operationalization of the per-token divergence as a logit
  offset; §6.1 is the closest existing "which tokens diverge" analysis.

### Paper 4: Emulated Fine-Tuning (Mitchell et al., 2023; arXiv 2310.12962)  ⭐
- **Key contribution**: Factorize a fine-tuned model as `base log-probs + behavior delta`,
  with **behavior delta = log π_instruct − log π_base**; framed as KL-constrained RL so the
  delta is an implicit reward / log-importance weight.
- **Results**: Pre-training scale → factuality; fine-tuning scale → helpfulness; "up-scaling"
  ensembles small-instruct + large-base.
- **Relevance**: Gives a principled identity for the prediction target and a reason the
  delta should be *low-complexity and predictable* most of the time.

### Paper 5: LIMA (Zhou et al., 2023; arXiv 2305.11206)
- Origin of the Superficial Alignment Hypothesis (1k SFT examples suffice). The claim
  URIAL tests; sets the prior that base↔instruct differences are mostly format/style.

### Paper 6: Revisiting the Superficial Alignment Hypothesis (2024; arXiv 2410.03717)
- **Counter-evidence**: post-training adds capability and scales with data, not purely
  style. Important caveat: the *surprising* (residual) divergence may be substantive, not
  noise — supports the hypothesis's premise that residuals are where the interesting stuff is.

### Paper 7: Is ICL Sufficient for Instruction Following? (2024; arXiv 2405.19874)
- ICL-only alignment (URIAL-style) still lags real SFT on harder cases ⇒ the divergence
  is not *entirely* superficial; bounds what a "predictable" model can capture.

### Paper 8: Shadow-FT (2025; arXiv 2505.12716)
- The base↔instruct **weight delta** is transferable across paired models ⇒ the divergence
  is structured/low-rank-ish, reinforcing predictability of the bulk of it.

### Paper 9: Contrastive Decoding Improves Reasoning (O'Brien & Lewis, 2023; arXiv 2309.09117)
- Decode on expert−amateur log-prob difference; methodological sibling for using divergence.

### Paper 10: DoLa — Decoding by Contrasting Layers (Chuang et al., 2023; arXiv 2309.03883)
- Within-model layer contrast; relevant to *where* in the network divergence localizes.

### Paper 11: LogitLens4LLMs (2025; arXiv 2503.11667)
- Logit-lens tooling for modern LLMs — for interpreting residual (predictor-wrong) tokens
  via layer-wise attribution.

---

## Common Methodologies
- **Per-position distribution comparison** between paired models (URIAL): KL, base-rank,
  base-prob. *Used in:* URIAL; reusable as the prediction target.
- **Logit/log-prob difference** as a behavior signal (EFT, proxy-tuning, contrastive
  decoding). *Used in:* Papers 3, 4, 9.
- **Entropy / concentration metrics** (Branching Factor, semantic entropy). *Used in:* Paper 2.
- **Token-bucket / token-identity analysis** of *which* tokens shift. *Used in:* URIAL §2,
  proxy-tuning §6.1.

## Standard Baselines (for a divergence predictor)
- **Position-only baseline**: predict KL from token index (URIAL shows KL ≈ monotone decay).
- **Entropy/BF baseline**: base-model entropy at the position strongly predicts how much
  room there is to shift (Paper 2).
- **Token-identity / POS baseline**: stylistic vs content token features (URIAL, proxy §6.1).
- **Surprisal of the instruct token under the base model** (base-prob) — near-trivial
  predictor of base-rank.
A good predictor combines position + base-entropy + local lexical/POS features; the
*residual after these* is the project's object of study.

## Evaluation Metrics
- For the divergence itself: **KL(P_inst‖P_base)** (primary), symmetric/JS KL, **base-rank**
  (η), base-prob, top-1 agreement rate, Branching-Factor ratio.
- For the predictor: MSE / R² on held-out tokens, calibration of predicted vs actual KL.
- For residual analysis: rank tokens by |actual − predicted|; aggregate residuals by
  `category`/`task`/`topic` (just-eval-instruct), by POS/token class, by position bin.

## Datasets in the Literature
- **just-eval-instruct** (URIAL) — 1,000 tagged instructions; the canonical set for this
  exact analysis. *(Downloaded.)*
- **AlpacaEval, MT-bench, LIMA** — components of just-eval-instruct; used broadly.
- **databricks-dolly-15k** — larger, category-tagged instruction set. *(Downloaded.)*
- Benchmark suites (knowledge/reasoning/safety: MMLU, GSM8K, TruthfulQA, ToxiGen) appear
  in proxy-tuning/EFT for *downstream* evaluation, not per-token divergence.

## Gaps and Opportunities
- **The residual is unexplored.** Existing work characterizes the *average/predictable*
  shift (decay with position, stylistic tokens, entropy drop). No paper trains an explicit
  divergence *predictor* and studies its **errors**. This is the project's novel contribution.
- **Surprising ≠ stylistic.** Paper 6/7 show real capability changes; the residual may
  concentrate on (a) safety/refusal trigger points, (b) reasoning-pivot tokens, (c)
  factual commitments where instruct/base disagree — all candidates for "interesting stuff."
- **Mechanistic attribution** of residual tokens (logit lens / layer contrast) is open.

## Recommendations for Our Experiment
- **Datasets**: train the predictor on `dolly-15k` (volume), analyze residuals on
  `just-eval-instruct` (rich tags incl. safety) — avoids leakage and gives slicing handles.
- **Model pairs**: start with `Qwen2.5-0.5B` / `-0.5B-Instruct` (ungated, shared tokenizer,
  cheap); confirm trends on `Mistral-7B-v0.1`/`-Instruct-v0.1` (SFT, used by URIAL) and, if
  access permits, `Llama-2-7b`/`-chat` (SFT+RLHF) to separate SFT vs RLHF effects.
- **Targets**: primary KL(P_inst‖P_base); also base-rank and log-prob delta (cross-check).
- **Predictor features (the "predictable" baseline to subtract out)**: position index,
  base-model entropy/BF at the position, base-prob of the instruct token, local n-gram /
  POS / is-stylistic-token flags, maybe a small probe on base hidden states.
- **Residual analysis**: rank by |actual−predicted| KL; aggregate by category/task/topic,
  POS/token-class, position bin; inspect top-residual tokens with logit-lens
  (LogitLens4LLMs) and proxy-tuning S6.1-style token attribution.
- **Methodological care**: ensure identical tokenizer/vocab; decide whose response to score
  (URIAL scores the *instruct* greedy decode — consider also scoring base decodes and a
  shared third sequence to avoid which-model-generated confounds); KL decays strongly with
  position, so *always* control for position before claiming a token is "surprising."
