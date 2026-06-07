# Downloaded Papers

11 papers, all from arXiv (open access), directly relevant to characterizing the
divergence between base and instruction-tuned LLMs. Tier 1 = deep-read (core to the
hypothesis); Tier 2 = skimmed (supporting methods/context). PDFs chunked under
`papers/pages/` for the Tier-1 set.

## Tier 1 — Core (deep read)

1. **The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning (URIAL)**
   — `2312.01552_urial_unlocking_spell_base_llm_icl.pdf`
   - Authors: Lin, Ravichander, Lu, Dziri, Sclar, Chandu, Bhagavatula, Choi (AI2 / UW), 2023
   - arXiv: 2312.01552
   - **The foundational paper for this project.** Introduces *token distribution shift*
     analysis between base and aligned models: decode the aligned model, then at each
     position categorize the aligned top token by its **base-rank** (unshifted η=1 ≈78%,
     marginal 1<η≤3, shifted η>3 ≈5–7%). Shifted positions are overwhelmingly *stylistic*
     (discourse markers, refusals/disclaimers); knowledge tokens are unshifted. KL and
     base-rank **decay over position**. Provides `just-eval-instruct` (our primary dataset).

2. **LLM Probability Concentration: How Alignment Shrinks the Generative Horizon**
   — `2506.17871_alignment_probability_concentration.pdf`
   - Authors: Yang, Li, Holtzman (**University of Chicago**), 2026 (v3)
   - arXiv: 2506.17871 · Code: github.com/yangalan123/LLMBranchingFactor
   - Introduces the **Branching Factor (BF)** = length-normalized exponentiated entropy.
     Alignment cuts BF 2–5× overall (up to 12→1.2 at the start); BF declines over a
     generation; alignment "surfaces latent low-entropy paths" — nudging a base model with
     an aligned-style prefix (e.g. "Sure,") reproduces the drop. Complements URIAL with an
     information-theoretic, distribution-sharpening view of the same shift.

3. **Tuning Language Models by Proxy (proxy-tuning)**
   — `2401.08565_proxy_tuning_logit_difference.pdf`
   - Authors: Liu, Han, Wang, Tsvetkov, Choi, Smith (UW / AI2), COLM 2024
   - arXiv: 2401.08565 · Code: github.com/alisawuffles/proxy-tuning
   - Operationalizes the base↔instruct gap as a **logit difference**:
     `s_base + (s_smallchat − s_smallbase)`. Closes ~88–91% of the base→chat gap.
     §6.1 token-level analysis: the logit offset most promotes **reasoning and stylistic
     tokens**, not knowledge — direct evidence about *what* the divergence encodes.

4. **An Emulator for Fine-Tuning LLMs using Small Language Models (EFT / up-scaling)**
   — `2310.12962_emulator_finetuning_eft.pdf`
   - Authors: Mitchell, Rafailov, Sharma, Finn, Manning (Stanford), 2023
   - arXiv: 2310.12962
   - Factorizes a fine-tuned model's logits as `base log-probs + behavior delta`, where
     **behavior delta = log π_instruct − log π_base** is exactly the per-token signal the
     hypothesis wants to predict. Frames fine-tuning as KL-constrained RL, so the delta is
     an (implicit) reward/log-importance weight. Finds pre-training scale → factuality,
     fine-tuning scale → helpfulness.

## Tier 2 — Supporting (skimmed via abstract)

5. **LIMA: Less Is More for Alignment** — `2305.11206_lima_superficial_alignment.pdf`
   - Zhou et al. (Meta), 2023 · arXiv 2305.11206. Origin of the **Superficial Alignment
     Hypothesis**: 1,000 SFT examples suffice; alignment teaches format/style, knowledge is
     from pre-training. The hypothesis URIAL tests directly.

6. **Revisiting the Superficial Alignment Hypothesis** — `2410.03717_revisiting_superficial_alignment.pdf`
   - 2024 · arXiv 2410.03717. Pushes back: post-training scales with more finetuning data
     and adds capabilities beyond style — a useful counter-view (the "surprising"
     differences may not all be stylistic).

7. **Is In-Context Learning Sufficient for Instruction Following in LLMs?**
   — `2405.19874_icl_sufficient_instruction_following.pdf`
   - 2024 · arXiv 2405.19874. Re-examines URIAL; finds ICL alignment still lags real SFT on
     hard cases — bounds how "superficial" the difference is.

8. **Shadow-FT: Tuning Instruct Model via Training on Paired Base Model**
   — `2505.12716_shadow_ft_paired_base_model.pdf`
   - 2025 · arXiv 2505.12716. Exploits the base/instruct **weight delta** being transferable;
     evidence the divergence is a low-complexity, structured offset.

9. **Contrastive Decoding Improves Reasoning in Large Language Models**
   — `2309.09117_contrastive_decoding_reasoning.pdf`
   - O'Brien & Lewis, 2023 · arXiv 2309.09117. Decoding on the log-prob *difference* between
     a strong (expert) and weak (amateur) model — methodological sibling of computing/using
     base↔instruct divergence.

10. **DoLa: Decoding by Contrasting Layers** — `2309.03883_dola_decoding_contrasting_layers.pdf`
    - Chuang et al., 2023 · arXiv 2309.03883. Contrasts later vs. earlier layers within one
      model; relevant for *where* in the network the divergence localizes.

11. **LogitLens4LLMs** — `2503.11667_logitlens4llms.pdf`
    - 2025 · arXiv 2503.11667. Tooling to apply the logit lens to modern LLMs — useful for
      interpreting *why* the predictor is wrong at specific tokens (layer-wise attribution).

## Notes
- `papers/pages/` contains 3-pages-per-chunk PDF splits + manifests for the Tier-1 papers.
- The paper-finder service was rate-limited (HTTP 500 / backend RateLimitError) during this
  session; papers were gathered via the arXiv API instead (see `resources.md`).
