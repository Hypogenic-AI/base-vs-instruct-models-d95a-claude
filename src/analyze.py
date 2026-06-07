"""
Train a per-token divergence predictor and analyze its residuals.

Train predictor of KL(P_inst||P_base) on dolly (held-out instruction corpus);
analyze residuals on just-eval-instruct (rich tags incl. 200 safety prompts).

Outputs: results/<pair>_metrics.json, results/<pair>_residual_top.csv,
figures/<pair>_*.png
"""
import argparse, json, os, re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42

# URIAL-style discourse / refusal / stylistic markers (the *predictable* shift).
STYLISTIC = {
    "however", "here", "sure", "certainly", "thank", "thanks", "remember", "note",
    "please", "additionally", "overall", "of", "as", "to", "first", "i", "it",
    "there", "some", "moreover", "furthermore", "importantly", "however,", "well",
    "cannot", "can't", "sorry", "unfortunately", "however", "but", "while",
}


def token_class(s):
    t = s.strip()
    if t == "":
        return "space"
    if re.fullmatch(r"[^\w\s]+", t):
        return "punct"
    if re.fullmatch(r"\d[\d,.]*", t):
        return "number"
    if t.lower() in STYLISTIC:
        return "stylistic"
    if s[:1] == " ":
        return "word_lead"        # space-leading word piece (new word)
    return "subword"              # mid-word continuation


def add_features(df):
    df = df.copy()
    df["tcls"] = df["token_str"].map(token_class)
    df["is_stylistic"] = (df["tcls"] == "stylistic").astype(int)
    df["is_punct"] = (df["tcls"] == "punct").astype(int)
    df["is_number"] = (df["tcls"] == "number").astype(int)
    df["is_wordlead"] = (df["tcls"] == "word_lead").astype(int)
    df["tok_len"] = df["token_str"].str.len()
    return df


FEATS = ["pos", "base_entropy", "base_top1_prob", "base_tok_logprob",
         "is_stylistic", "is_punct", "is_number", "is_wordlead", "tok_len"]


def fit_baseline(Xtr, ytr, Xte, cols):
    m = LinearRegression().fit(Xtr[cols], ytr)
    return m, m.predict(Xte[cols])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)      # dolly parquet
    ap.add_argument("--analysis", required=True)   # just-eval parquet
    ap.add_argument("--pair", required=True)        # tag for outputs
    args = ap.parse_args()
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    tr = add_features(pd.read_parquet(args.train))
    an = add_features(pd.read_parquet(args.analysis))
    y_tr, y_an = tr["kl"].values, an["kl"].values
    print(f"[{args.pair}] train tokens={len(tr)} analysis tokens={len(an)}")
    print(f"  top-1 agreement train={tr.agree.mean():.3f} analysis={an.agree.mean():.3f}")

    # internal dolly train/val split for honest predictor R^2
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(tr))
    n_val = len(tr) // 5
    val_idx, fit_idx = perm[:n_val], perm[n_val:]
    Xfit, Xval = tr.iloc[fit_idx], tr.iloc[val_idx]
    yfit, yval = y_tr[fit_idx], y_tr[val_idx]

    metrics = {"pair": args.pair, "n_train": len(tr), "n_analysis": len(an),
               "top1_agree_train": float(tr.agree.mean()),
               "top1_agree_analysis": float(an.agree.mean()),
               "mean_kl_analysis": float(an.kl.mean()),
               "median_kl_analysis": float(an.kl.median())}

    # ---- baselines ----
    baselines = {"position_only": ["pos"],
                 "entropy_only": ["base_entropy"],
                 "base_logprob_only": ["base_tok_logprob"]}
    metrics["baselines"] = {}
    for name, cols in baselines.items():
        _, pv = fit_baseline(Xfit, yfit, Xval, cols)
        metrics["baselines"][name] = {"val_r2": float(r2_score(yval, pv))}

    # ---- gradient-boosted-tree full predictor (HistGradientBoosting) ----
    gbm = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
        validation_fraction=0.1, early_stopping=True, n_iter_no_change=40,
        random_state=SEED)
    gbm.fit(Xfit[FEATS], yfit)
    pv = gbm.predict(Xval[FEATS])
    metrics["lgbm"] = {
        "val_r2": float(r2_score(yval, pv)),
        "val_rmse": float(np.sqrt(mean_squared_error(yval, pv))),
        "val_spearman": float(stats.spearmanr(yval, pv).statistic),
        "best_iter": int(gbm.n_iter_)}

    # cross-corpus: predict on just-eval (the analysis set)
    pred_an = gbm.predict(an[FEATS])
    metrics["lgbm"]["analysis_r2"] = float(r2_score(y_an, pred_an))
    metrics["lgbm"]["analysis_spearman"] = float(stats.spearmanr(y_an, pred_an).statistic)
    an = an.copy()
    an["pred_kl"] = pred_an
    an["resid"] = an["kl"] - an["pred_kl"]          # signed residual
    an["abs_resid"] = an["resid"].abs()

    # permutation importance on a val subsample (drop in R^2 when feature shuffled)
    sub = Xval.sample(min(5000, len(Xval)), random_state=SEED)
    ysub = yval[sub.index.map(lambda x: Xval.index.get_loc(x))] if False else \
        tr.loc[sub.index, "kl"].values
    pi = permutation_importance(gbm, sub[FEATS], ysub, n_repeats=5,
                                random_state=SEED, scoring="r2")
    fi = dict(zip(FEATS, pi.importances_mean))
    metrics["lgbm"]["feature_importance_gain"] = {k: float(v) for k, v in fi.items()}

    # ---- H3: residual by category (safety vs regular) ----
    if "category" in an.columns:
        an["is_safety"] = (an["category"] != "regular").astype(int)
        reg = an.loc[an.is_safety == 0, "resid"].values
        saf = an.loc[an.is_safety == 1, "resid"].values
        if len(saf) > 20:
            u = stats.mannwhitneyu(saf, reg, alternative="greater")
            # rank-biserial effect size
            rbc = 2 * u.statistic / (len(saf) * len(reg)) - 1
            metrics["residual_safety_vs_regular"] = {
                "mean_resid_safety": float(saf.mean()),
                "mean_resid_regular": float(reg.mean()),
                "median_resid_safety": float(np.median(saf)),
                "median_resid_regular": float(np.median(reg)),
                "mannwhitney_p_greater": float(u.pvalue),
                "rank_biserial": float(rbc),
                "n_safety_tokens": int(len(saf)), "n_regular_tokens": int(len(reg))}

    # ---- residual by token class (Kruskal-Wallis) ----
    groups = [g["resid"].values for _, g in an.groupby("tcls")]
    if len(groups) > 2:
        kw = stats.kruskal(*groups)
        metrics["residual_by_tclass"] = {
            "kruskal_p": float(kw.pvalue),
            "mean_resid": {k: float(v) for k, v in
                           an.groupby("tcls").resid.mean().items()},
            "count": {k: int(v) for k, v in an.groupby("tcls").size().items()}}

    # ---- residual within position bins (control for position) ----
    an["posbin"] = pd.cut(an["pos"], [-1, 4, 9, 19, 39, 999],
                          labels=["0-4", "5-9", "10-19", "20-39", "40+"])
    metrics["resid_by_posbin"] = {
        str(k): {"mean_resid": float(v), "mean_kl": float(an.groupby("posbin", observed=True).kl.mean()[k])}
        for k, v in an.groupby("posbin", observed=True).resid.mean().items()}

    # ---- top residual tokens (the "surprising" ones) ----
    top = an.sort_values("resid", ascending=False).head(40)
    cols_out = ["category", "pos", "token_str", "kl", "pred_kl", "resid",
                "base_rank", "base_entropy", "example_id"]
    cols_out = [c for c in cols_out if c in top.columns]
    top[cols_out].to_csv(f"results/{args.pair}_residual_top.csv", index=False)

    # aggregate which token strings are most over-divergent on average (min count)
    g = an.groupby("token_str").agg(n=("resid", "size"), mean_resid=("resid", "mean"),
                                    mean_kl=("kl", "mean")).query("n >= 5")
    g.sort_values("mean_resid", ascending=False).head(30).to_csv(
        f"results/{args.pair}_residual_tokens.csv")

    with open(f"results/{args.pair}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---------- figures ----------
    # 1. KL vs position (validation: decay)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    pp = an.groupby("pos").kl.mean()
    ax.plot(pp.index[:80], pp.values[:80])
    ax.set(xlabel="response position", ylabel="mean KL(inst||base)",
           title=f"{args.pair}: KL vs position")
    fig.tight_layout(); fig.savefig(f"figures/{args.pair}_kl_vs_pos.png", dpi=130)

    # 2. predicted vs actual (calibration)
    fig, ax = plt.subplots(figsize=(4, 4))
    s = an.sample(min(4000, len(an)), random_state=SEED)
    ax.scatter(s.pred_kl, s.kl, s=4, alpha=0.25)
    m = max(s.kl.max(), s.pred_kl.max())
    ax.plot([0, m], [0, m], "r--", lw=1)
    ax.set(xlabel="predicted KL", ylabel="actual KL",
           title=f"{args.pair}: R2={metrics['lgbm']['analysis_r2']:.2f}")
    fig.tight_layout(); fig.savefig(f"figures/{args.pair}_calibration.png", dpi=130)

    # 3. feature importance
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fis = pd.Series(fi).sort_values()
    ax.barh(fis.index, fis.values)
    ax.set(title=f"{args.pair}: LGBM feature importance (gain)")
    fig.tight_layout(); fig.savefig(f"figures/{args.pair}_featimp.png", dpi=130)

    # 4. residual by category
    if "is_safety" in an.columns:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        data = [an.loc[an.is_safety == 0, "resid"], an.loc[an.is_safety == 1, "resid"]]
        ax.boxplot(data, labels=["regular", "safety"], showfliers=False)
        ax.axhline(0, color="gray", lw=0.7)
        ax.set(ylabel="residual (actual-pred KL)",
               title=f"{args.pair}: residual by category")
        fig.tight_layout(); fig.savefig(f"figures/{args.pair}_resid_category.png", dpi=130)

    plt.close("all")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
