"""Aggregate metrics across model pairs into a comparison table + summary figure."""
import json, glob, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAIRS = {"qwen05": "Qwen2.5-0.5B", "qwen15": "Qwen2.5-1.5B", "qwen7": "Qwen2.5-7B"}

rows = []
for tag, name in PAIRS.items():
    p = f"results/{tag}_metrics.json"
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    sr = d.get("residual_safety_vs_regular", {})
    rows.append(dict(
        pair=name,
        top1_agree=round(d["top1_agree_analysis"], 3),
        mean_kl=round(d["mean_kl_analysis"], 3),
        r2_pos_only=round(d["baselines"]["position_only"]["val_r2"], 3),
        r2_entropy_only=round(d["baselines"]["entropy_only"]["val_r2"], 3),
        r2_baselogp_only=round(d["baselines"]["base_logprob_only"]["val_r2"], 3),
        r2_gbt_val=round(d["lgbm"]["val_r2"], 3),
        r2_gbt_xcorpus=round(d["lgbm"]["analysis_r2"], 3),
        spearman=round(d["lgbm"]["analysis_spearman"], 3),
        safety_resid=round(sr.get("mean_resid_safety", float("nan")), 3),
        regular_resid=round(sr.get("mean_resid_regular", float("nan")), 3),
        safety_rank_biserial=round(sr.get("rank_biserial", float("nan")), 3),
        safety_p=sr.get("mannwhitney_p_greater", float("nan")),
    ))
df = pd.DataFrame(rows)
df.to_csv("results/summary_table.csv", index=False)
print(df.to_string(index=False))

# summary figure: predictability + safety effect across scale
if len(df) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    x = range(len(df))
    axes[0].plot(x, df.r2_pos_only, "o-", label="position-only")
    axes[0].plot(x, df.r2_entropy_only, "s-", label="entropy-only")
    axes[0].plot(x, df.r2_gbt_xcorpus, "^-", label="GBT (all feats)")
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(df.pair, rotation=20)
    axes[0].set_ylabel("cross-corpus R^2"); axes[0].set_title("KL predictability"); axes[0].legend(fontsize=8)
    w = 0.35
    axes[1].bar([i - w/2 for i in x], df.regular_resid, w, label="regular")
    axes[1].bar([i + w/2 for i in x], df.safety_resid, w, label="safety")
    axes[1].axhline(0, color="gray", lw=0.7)
    axes[1].set_xticks(list(x)); axes[1].set_xticklabels(df.pair, rotation=20)
    axes[1].set_ylabel("mean residual KL"); axes[1].set_title("Residual: safety vs regular"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/summary_across_pairs.png", dpi=130)
    print("wrote figures/summary_across_pairs.png")
