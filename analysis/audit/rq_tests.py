"""The pre-registered hypothesis tests (plan §1 and §7) on the D2 scorecards.

Inputs (all produced by flavoraudit.audit.scorecard_row, so every number is computed the same way):
  data/interim/audit/<tag>/scorecard.parquet        LLM cuisines (analysis/audit/llm_audit.py)
  data/interim/audit/real/subsamples_n160.parquet   equal-N real subsamples (analysis/audit/real_subsamples.py)
  data/interim/audit/copy_mutate/scorecard.parquet  copy-mutate baseline (analysis/audit/copy_mutate_baseline.py)

Tests (one row each in tables/rq_tests_<tag>.csv; BH-FDR at q = 0.05 within each hypothesis across models, and
across the 22 regions for region-level tests):
  H1  per model: Fisher's exact test (one-sided, greater) of sign errors in the 6 contrasting regions vs the 16
      uniform ones. A sign error is a ΔNs bootstrap CI that excludes 0 with the sign opposite to the real
      cuisine's Z. Also reported: the rule's false-positive rate on genuine 160-recipe samples, and a
      size-aware FDR variant (p from z = sqrt(n) * d, BH across regions).
  H2b per model: Wilcoxon signed-rank of log SAI > 0 across regions (E[SAI] = 1 for genuine samples), and of
      log diversity ratio < 0.
  H2c per model and region: the LLM's fingerprint Spearman vs the distribution over real 160-recipe subsamples
      (empirical lower-tail p, BH across regions), and a Wilcoxon of (LLM rho - median real-subsample rho) < 0.
  H4  per model: paired Wilcoxon persona vs plain across regions (two-sided) for |d - d_real|, prevalence JSD,
      fingerprint Spearman and |log SAI|; across models: proprietary vs open and Qwen3.5-9B vs -4B.
  CM  per model: paired Wilcoxon LLM vs copy-mutate (CM-R/ranked, n = 160, mean of 10 instances) per metric.
Exploratory (labelled as such): the direction of the pairing shift, d_LLM - d_real.

Usage (repo root): .venv/bin/python analysis/audit/rq_tests.py [--tag full]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_io import AUDIT, CONTRASTING, LLM_N_PER_REGION, REAL, REPO  # noqa: E402

N_RAND = 100_000  # Z_SB = sqrt(N_RAND) * d
PROPRIETARY = {"gemini-3.5-flash-lite", "openai-gpt-6-luna", "openai-gpt-6.2-sol"}
METRICS_H4 = {  # name -> (column builder, direction: +1 = higher is more faithful)
    "abs_d_gap": (lambda d: (d["z_sb"] - d["z_real"]).abs() / np.sqrt(N_RAND), -1),
    "prevalence_jsd": (lambda d: d["prevalence_jsd"], -1),
    "fingerprint_spearman": (lambda d: d["fingerprint_spearman"], +1),
    "abs_log_sai": (lambda d: np.log(d["sai"]).abs(), -1),
}


def bh(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values (NaNs kept)."""
    ok = p.dropna()
    if ok.empty:
        return p
    adj = st.false_discovery_control(ok.to_numpy(), method="bh")
    return pd.Series(adj, index=ok.index).reindex(p.index)


def wilcoxon(x: pd.Series, alternative: str) -> tuple[float, float, int]:
    x = x.dropna()
    x = x[x != 0]
    if len(x) < 5:
        return float("nan"), float("nan"), len(x)
    r = st.wilcoxon(x, alternative=alternative)
    return float(r.statistic), float(r.pvalue), len(x)


def size_aware_p(d: pd.DataFrame) -> pd.Series:
    """Two-sided p for ΔNs != 0 from z = sqrt(n) * d (the size-aware significance of plan §6.1, normal approx.)."""
    z = np.sqrt(d["n_pairing_recipes"]) * d["d"]
    return pd.Series(2 * st.norm.sf(np.abs(z)), index=d.index)


def run(tag: str) -> pd.DataFrame:
    llm = pd.read_parquet(AUDIT / tag / "scorecard.parquet")
    real = pd.read_parquet(REAL / f"subsamples_n{LLM_N_PER_REGION}.parquet")
    cm_path = AUDIT / "copy_mutate" / "scorecard.parquet"
    cm = pd.read_parquet(cm_path) if cm_path.exists() else None
    pooled = llm[llm["scope"] == "pooled"].copy()
    pooled["contrasting"] = pooled["region"].isin(CONTRASTING)
    rows: list[dict] = []

    def add(hyp, model, test, stat, p, effect, n, note=""):
        rows.append(
            {"hypothesis": hyp, "model": model, "test": test, "statistic": stat, "p": p, "effect": effect, "n": n}
            | {"note": note}
        )

    # ---------------- H1: sign errors cluster in contrasting regions
    fp_rate = float(real["sign_error"].mean())
    add("H1", "real@160", "sign-error rule false-positive rate", np.nan, np.nan, fp_rate, len(real), "genuine samples")
    for model, g in pooled.groupby("model"):
        e_c = int(g.loc[g["contrasting"], "sign_error"].sum())
        e_u = int(g.loc[~g["contrasting"], "sign_error"].sum())
        n_c, n_u = int(g["contrasting"].sum()), int((~g["contrasting"]).sum())
        res = st.fisher_exact([[e_c, n_c - e_c], [e_u, n_u - e_u]], alternative="greater")
        add(
            "H1",
            model,
            "Fisher exact, errors contrasting > uniform",
            res.statistic,
            res.pvalue,
            f"{e_c}/{n_c} vs {e_u}/{n_u}",
            n_c + n_u,
        )
        p_fdr = bh(size_aware_p(g))
        opposite = np.sign(g["d"]) != np.sign(g["z_real"])
        add(
            "H1",
            model,
            "sign errors, size-aware p with BH across regions",
            np.nan,
            np.nan,
            int(((p_fdr < 0.05) & opposite).sum()),
            len(g),
            "count of FDR-significant opposite-sign regions",
        )
        stat, p, n = wilcoxon(g["d"] - g["z_real"] / np.sqrt(N_RAND), "greater")
        add(
            "H1-exploratory",
            model,
            "Wilcoxon d_LLM - d_real > 0 (push toward positive pairing)",
            stat,
            p,
            float((g["d"] - g["z_real"] / np.sqrt(N_RAND)).median()),
            n,
            "exploratory",
        )

    # ---------------- H1, equal-N amendment (decision log 2026-11-22): genuine samples of the same region and size
    calib = real.groupby("region").agg(
        sign_match_at_160=("sign_match", "mean"), sign_rule_false_positive=("sign_error", "mean")
    )
    calib.to_csv(REPO / "tables" / "sign_rule_calibration_n160.csv")
    eq_path = REAL / "equal_n_pairing.parquet"
    if eq_path.exists():
        eq = pd.read_parquet(eq_path)
        grid = np.array(sorted(eq["N"].unique()))
        recs = []
        for r in pooled.itertuples():
            n_ref = int(grid[np.argmin(np.abs(grid - r.n_pairing_recipes))])
            ref_d = eq.loc[(eq["region"] == r.region) & (eq["N"] == n_ref), "d"].to_numpy()
            lo, hi = (1 + (ref_d <= r.d).sum()) / (1 + len(ref_d)), (1 + (ref_d >= r.d).sum()) / (1 + len(ref_d))
            recs.append({"model": r.model, "region": r.region, "contrasting": r.contrasting, "n": r.n_pairing_recipes,
                         "n_ref": n_ref, "d": r.d, "d_ref_median": float(np.median(ref_d)),
                         "d_ref_lo": float(np.quantile(ref_d, 0.025)), "d_ref_hi": float(np.quantile(ref_d, 0.975)),
                         "p": min(1.0, 2 * min(lo, hi))})  # fmt: skip
        h = pd.DataFrame(recs)
        h["p_adj"] = h.groupby("model", group_keys=False)["p"].apply(bh)
        h["verdict"] = np.where(
            h["p_adj"] >= 0.05, "typical", np.where(h["d"] > h["d_ref_median"], "too uniform", "too contrasting")
        )
        h.to_csv(REPO / "tables" / f"h1_equal_n_{tag}.csv", index=False)
        for model, g in h.groupby("model"):
            shift = g["d"] - g["d_ref_median"]
            stat, p, n = wilcoxon(shift, "greater")
            add("H1-equalN", model, "Wilcoxon d_LLM - median d_genuine(same N) > 0", stat, p, float(shift.median()), n,
                f"{(g['verdict'] == 'too uniform').sum()} too uniform, {(g['verdict'] == 'too contrasting').sum()} "
                "too contrasting (BH)")  # fmt: skip
            up = shift > 0
            a_c, n_c = int((up & g["contrasting"]).sum()), int(g["contrasting"].sum())
            a_u, n_u = int((up & ~g["contrasting"]).sum()), int((~g["contrasting"]).sum())
            res = st.fisher_exact([[a_c, n_c - a_c], [a_u, n_u - a_u]], alternative="greater")
            add("H1-equalN", model, "Fisher: shift toward uniform, contrasting vs uniform regions", res.statistic,
                res.pvalue, f"{a_c}/{n_c} vs {a_u}/{n_u}", n_c + n_u)  # fmt: skip

    # ---------------- H2b: stereotype amplification and lower diversity
    for model, g in pooled.groupby("model"):
        stat, p, n = wilcoxon(np.log(g["sai"]), "greater")
        add("H2b", model, "Wilcoxon log SAI > 0", stat, p, float(np.exp(np.log(g["sai"]).median())), n)
        stat, p, n = wilcoxon(np.log(g["diversity_ratio"]), "less")
        add(
            "H2b-diversity", model, "Wilcoxon log diversity ratio < 0", stat, p, float(g["diversity_ratio"].median()), n
        )
    real_sai = real.groupby("region")["sai"]
    add(
        "H2b",
        "real@160",
        "median SAI of genuine samples (expected 1)",
        np.nan,
        np.nan,
        float(real_sai.median().median()),
        len(real),
    )

    # ---------------- H2c: fingerprints vs resampling noise
    rfp = real.dropna(subset=["fingerprint_spearman"]).groupby("region")["fingerprint_spearman"]
    ref_med = rfp.median()
    for model, g in pooled.groupby("model"):
        g = g.dropna(subset=["fingerprint_spearman"])
        p_reg = pd.Series(
            {
                r.region: (1 + (rfp.get_group(r.region) <= r.fingerprint_spearman).sum())
                / (1 + rfp.get_group(r.region).size)
                for r in g.itertuples()
            }
        )
        n_sig = int((bh(p_reg) < 0.05).sum())
        stat, p, n = wilcoxon(g.set_index("region")["fingerprint_spearman"] - ref_med, "less")
        add(
            "H2c",
            model,
            "Wilcoxon rho_LLM - median rho_real@160 < 0",
            stat,
            p,
            float((g.set_index("region")["fingerprint_spearman"] - ref_med).median()),
            n,
            f"{n_sig} regions below the resampling distribution (BH)",
        )

    # ---------------- H4: persona vs plain, proprietary vs open, scale
    cond = llm[llm["scope"].isin(["plain", "persona"])]
    for model, g in cond.groupby("model"):
        for name, (fn, _sign) in METRICS_H4.items():
            w = g.assign(v=fn(g)).pivot_table(index="region", columns="scope", values="v")
            if {"plain", "persona"} <= set(w.columns):
                stat, p, n = wilcoxon(w["persona"] - w["plain"], "two-sided")
                add(
                    "H4-persona",
                    model,
                    f"Wilcoxon persona - plain: {name}",
                    stat,
                    p,
                    float((w["persona"] - w["plain"]).median()),
                    n,
                )
    by_model = {
        name: pooled.assign(v=fn(pooled)).pivot_table(index="region", columns="model", values="v")
        for name, (fn, _s) in METRICS_H4.items()
    }
    for name, w in by_model.items():
        prop = [m for m in w.columns if m in PROPRIETARY]
        open_ = [m for m in w.columns if m not in PROPRIETARY]
        if prop and open_:
            diff = w[prop].mean(axis=1) - w[open_].mean(axis=1)
            stat, p, n = wilcoxon(diff, "two-sided")
            add(
                "H4-proprietary",
                "+".join(prop),
                f"Wilcoxon proprietary - open: {name}",
                stat,
                p,
                float(diff.median()),
                n,
            )
        if {"qwen3.5-9b", "qwen3.5-4b"} <= set(w.columns):
            diff = w["qwen3.5-9b"] - w["qwen3.5-4b"]
            stat, p, n = wilcoxon(diff, "two-sided")
            add("H4-scale", "qwen3.5-9b vs 4b", f"Wilcoxon 9B - 4B: {name}", stat, p, float(diff.median()), n)

    # ---------------- LLM vs the copy-mutate baseline (equal N)
    if cm is not None:
        base = cm[(cm["source"] == "CM-R/ranked") & (cm["scope"] == f"n={LLM_N_PER_REGION}")]
        for name, (fn, _s) in METRICS_H4.items():
            cm_v = base.assign(v=fn(base)).groupby("region")["v"].mean()
            for model in by_model[name].columns:
                diff = by_model[name][model] - cm_v.reindex(by_model[name].index)
                stat, p, n = wilcoxon(diff, "two-sided")
                add("CM", model, f"Wilcoxon LLM - CM-R/ranked: {name}", stat, p, float(diff.median()), n)

    out = pd.DataFrame(rows)
    out["p_adj"] = out.groupby(["hypothesis", "test"], group_keys=False)["p"].apply(bh)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    out = run(args.tag)
    path = REPO / "tables" / f"rq_tests_{args.tag}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(out.round(4).to_string(index=False))
    print(f"-> {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
