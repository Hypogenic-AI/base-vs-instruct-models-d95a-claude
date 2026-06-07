# Planning: What's Surprisingly Different Between Base and Instruct Models?

## Motivation & Novelty Assessment

### Why This Research Matters
Understanding *what* instruction/alignment tuning actually changes in a pretrained model is
central to interpretability, alignment safety, and cheap-alignment methods (proxy-tuning,
URIAL). The dominant view (Superficial Alignment Hypothesis) says alignment mostly changes
*style/format*. If true, the base→instruct distributional shift should be highly
**predictable** from simple context features. The scientifically interesting question is
therefore not the average shift but its **unpredictable residual**: the tokens where
alignment does something a simple model of "stylistic shift" cannot anticipate. Those are
candidates for genuine, non-superficial behavioral change (safety pivots, reasoning forks,
factual commitments).

### Gap in Existing Work
URIAL, proxy-tuning, EFT, and Branching-Factor all characterize the *average / predictable*
shift: KL decays with position, shifts concentrate on stylistic/discourse/refusal tokens,
alignment lowers entropy. **No prior work trains an explicit divergence predictor and
studies its errors.** The residual is unexplored territory — exactly what the user asked
for.

### Our Novel Contribution
1. Operationalize per-token base↔instruct divergence (KL, base-rank, log-prob delta) at
   scale on real model pairs.
2. Train an explicit **divergence predictor** from base-side context features and quantify
   how predictable the shift is (R²).
3. Define and characterize the **residual** (|actual − predicted| KL): where the predictor
   is wrong, sliced by category (incl. safety), POS/token class, and position.
4. Cross-check across model scale (0.5B → 7B) and alignment type (RLHF/DPO vs SFT-only).

### Experiment Justification
- **Exp 1 — Measure divergence.** Reproduce the URIAL-style per-token KL/base-rank profile
  for a real pair (validates our measurement against known results: top-1 agreement ≈78%,
  KL decays with position). Without this we cannot trust the predictor target.
- **Exp 2 — Train predictor & quantify predictability.** Fit baselines (position-only,
  entropy-only) and a gradient-boosted predictor on base-side features. Tells us *how much*
  of the shift is "boring/predictable." A strong-but-imperfect R² is the precondition for
  the residual being meaningful.
- **Exp 3 — Residual analysis (the core).** Rank tokens by residual; test whether residuals
  concentrate on specific categories (safety), token classes, and positions. This directly
  answers "where is the interesting stuff."
- **Exp 4 — Robustness.** Replicate the predictor + residual structure on a larger Qwen
  pair and an SFT-only pair (Mistral-7B-v0.1) to show findings are not a 0.5B / RLHF
  artifact.

## Research Question
Where does instruction tuning change a base model's next-token distribution in ways that a
learned predictor of the divergence *cannot* anticipate from simple context features — and
what kinds of tokens/contexts are those residuals?

## Hypothesis Decomposition
- **H1 (predictability).** Per-token KL(P_inst‖P_base) is largely predictable from base-side
  context (position, base entropy, base-prob of the token): predictor R² is high.
- **H2 (residual is structured, not noise).** After subtracting the prediction, the residual
  is *not* uniformly distributed: it concentrates on identifiable slices.
- **H3 (residual ≠ stylistic).** High-residual tokens are disproportionately at
  safety/refusal pivots and content-commitment points, not the generic stylistic tokens
  (`However`, `Here`) that dominate the *predictable* shift.

## Proposed Methodology

### Approach
For each instruction, generate a response with the **instruct** model (greedy). Teacher-force
that identical token sequence through **both** base and instruct models under the **same
chat-formatted context** (both models see identical tokens — isolates the weight difference,
the cleanest measure of distributional divergence). At each response position compute:
- target: `KL(P_inst‖P_base)` (primary), plus base-rank η and log-prob delta (cross-check);
- base-side features: response-position index, base entropy, base top-1 prob, base-prob &
  base-rank of the emitted token, token-identity/POS/is-stylistic flags.
Train a predictor of KL from base-side features on **dolly-15k**; analyze residuals on
**just-eval-instruct** (rich tags incl. 200 safety prompts) → no train/test leakage.

### Experimental Steps
1. Validate measurement (Exp 1) on Qwen2.5-0.5B pair: top-1 agreement, KL-vs-position decay.
2. Build per-token feature table for dolly (train) and just-eval (analysis).
3. Fit predictors: position-only, entropy-only, base-prob-only (baselines) + LightGBM
   (all base-side features). Report held-out R²/MSE (Exp 2).
4. Residual = actual − LightGBM-predicted KL on just-eval. Rank, slice by category/POS/
   position; statistical tests (Exp 3).
5. Replicate Exp 2–3 on Qwen2.5-7B (scale) and Mistral-7B-v0.1 (SFT-only) pairs (Exp 4).

### Baselines
- Predictor baselines: position-only, base-entropy-only, base-prob-only regressors (these
  encode the "known" predictable structure from URIAL/Branching-Factor). LightGBM must beat
  them for the features to matter; the residual is defined relative to the best predictor.
- Divergence baseline sanity: URIAL's reported ~78% top-1 agreement / KL-position decay.

### Evaluation Metrics
- Divergence: KL(P_inst‖P_base), top-1 agreement, base-rank, log-prob delta.
- Predictor: held-out R², MSE, Spearman ρ(pred, actual).
- Residual: distribution of |actual−predicted|; mean residual per slice; Kruskal–Wallis
  across categories; bootstrap 95% CIs; effect sizes (rank-biserial / Cliff's δ).

### Statistical Analysis Plan
α = 0.05. Train/analysis on disjoint corpora. Predictor evaluated on held-out tokens.
For residual-by-slice: Kruskal–Wallis omnibus + Mann–Whitney (safety vs regular) with
bootstrap CIs and effect size. Control for position (residual is already position-adjusted
because position is a predictor feature, but we also report residual within position bins).
Seeds fixed (42). Greedy decoding (temperature 0) for determinism.

## Expected Outcomes
- H1 supported if LightGBM R² ≳ 0.5 (much of KL is predictable).
- H2 supported if residual variance differs significantly across slices.
- H3 supported if safety prompts / specific token classes show significantly higher residual
  than generic stylistic tokens.
- Refutation: residual is uniform noise (no slice structure) → "surprising" divergence is
  just estimation noise.

## Timeline and Milestones
1. Env + model load + measurement validation (Exp 1) — 30 min.
2. Feature extraction over corpora — 45 min.
3. Predictor + residual analysis + figures (Exp 2–3) — 45 min.
4. Robustness pairs (Exp 4) — 45 min.
5. Documentation (REPORT.md, README.md) — 30 min.

## Potential Challenges
- **Vocab/tokenizer mismatch** → assert equal vocab per pair; KL only over shared vocab.
- **Position confound** (KL decays with position) → include position as a predictor feature
  and report residuals within position bins.
- **Which sequence to score** (confound: instruct-generated text is in-distribution for
  instruct) → primary uses instruct greedy decode under identical context; robustness check
  scores dolly reference responses.
- **Compute / storage** → compute KL on-GPU per position, store only scalars+features.
- **POS tagging** → NLTK averaged_perceptron_tagger on decoded token strings (approx).

## Success Criteria
A working, validated per-token divergence measurement (matches URIAL stylized facts); a
predictor with reported R² and baselines; and a residual analysis with statistically tested
slice structure and concrete example tokens — delivered in REPORT.md across ≥2 model pairs.
