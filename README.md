# What's Surprisingly Different Between Base and Instruct Models?

Predict the **per-token KL divergence** between a base LLM and its instruct version, then
study **where the predictor is wrong** — the hypothesis being that *unpredictable* divergence
is where the genuinely interesting, non-superficial alignment behavior lives.

## Key findings
- **Most divergence is predictable.** A gradient-boosted predictor explains **R² ≈ 0.60–0.68**
  of per-token KL on a held-out corpus (Spearman 0.82–0.89), replicated on Qwen2.5-0.5B/1.5B/7B.
- **It's driven by base uncertainty, not position.** Permutation importance is dominated by
  base entropy + base token-surprise; **position-only R² ≈ 0.02–0.05**. This refines URIAL's
  "KL decays with position" — the decay is mostly mediated by entropy.
- **The residual is alignment-specific, not noise.** Where the predictor is wrong concentrates
  significantly (p≈0, all 3 pairs) on **safety prompts** (rank-biserial 0.14–0.32) and on
  **discourse/structural pivots + injected content**: refusal-recovery boilerplate
  (*"…can't assist. **If** you have…"*), helpfulness framing (`provide`, `recommend`,
  `However`, `Additionally`), and **persona/identity** tokens (`assistant`, `Safety`,
  `guidelines`, `Alibaba`).
- **Measurement validated:** top-1 base/instruct agreement 0.83–0.85 (matches URIAL's ~78%).

→ Full write-up with tables, figures, and statistics: **[REPORT.md](REPORT.md)**.

## Reproduce
```bash
uv venv && source .venv/bin/activate
uv sync                                   # installs torch, transformers, datasets, sklearn, ...

# 1. Extract per-token divergence (base vs instruct) — train + analysis corpora
python src/extract_divergence.py --base Qwen/Qwen2.5-0.5B --inst Qwen/Qwen2.5-0.5B-Instruct \
       --dataset dolly     --n 1500 --out results/qwen05_dolly.parquet
python src/extract_divergence.py --base Qwen/Qwen2.5-0.5B --inst Qwen/Qwen2.5-0.5B-Instruct \
       --dataset just-eval --n 1000 --out results/qwen05_justeval.parquet

# 2. Train divergence predictor + residual analysis (figures + metrics)
python src/analyze.py --train results/qwen05_dolly.parquet \
       --analysis results/qwen05_justeval.parquet --pair qwen05

# 3. Cross-pair summary (after running pairs qwen05/qwen15/qwen7)
python src/summarize.py
```
GPU recommended (fp16; base + instruct on separate devices). Seed 42, greedy decoding.

## File structure
```
planning.md                 # Phase-0/1 motivation, hypothesis decomposition, design
REPORT.md                   # full research report (primary deliverable)
src/extract_divergence.py   # per-token KL/base-rank/logprob-delta + base-side features
src/analyze.py              # predictor (baselines + GBT), residual stats, figures
src/summarize.py            # cross-pair comparison table + figure
results/                    # *_{dolly,justeval}.parquet, *_metrics.json, *_residual_*.csv, summary_table.csv
figures/                    # kl_vs_pos, calibration, featimp, resid_category, summary_across_pairs
datasets/                   # just-eval-instruct, dolly-15k (pre-gathered)
literature_review.md, resources.md, papers/, code/   # pre-gathered resources
```

## Datasets & models
- **just-eval-instruct** (`re-align/just-eval-instruct`, 1k, tagged incl. 200 safety) — residual analysis.
- **dolly-15k** (`databricks/databricks-dolly-15k`) — predictor training (disjoint corpus).
- **Qwen2.5-{0.5B,1.5B,7B}** base/instruct pairs (ungated, shared tokenizer per pair).
