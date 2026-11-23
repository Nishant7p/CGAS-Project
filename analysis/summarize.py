"""Collect every headline number into results/summary.json: the single source for the report, slides, poster and app.

Every value is computed from the pipeline's outputs (nothing typed by hand). Rerun after new data arrive (e.g. the
Llama run or more Gemini calls); the documents are rebuilt from this file. Missing inputs are recorded as null with a
"pending" note, so a document can show what is still running.

Usage (repo root): .venv/bin/python analysis/summarize.py
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

REPO = Path(__file__).resolve().parents[1]
INTERIM = REPO / "data" / "interim"
AUDIT = INTERIM / "audit"
OUT = REPO / "results" / "summary.json"
CONTRASTING = ["British Isles", "DACH Countries", "Eastern Europe", "Japan", "Korea", "Scandinavia"]
CALLS_PER_MODEL = 704
MODEL_NAMES = {
    "openai-gpt-6-luna": "GPT-6-Luna",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "qwen3.5-9b": "Qwen3.5-9B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "qwen3.5-4b": "Qwen3.5-4B",
    "ministral-3-8b": "Ministral-3-8B",
}


def r(x, k=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


def dataset() -> dict:
    calls = pd.read_parquet(REPO / "data/generated/parsed/calls.parquet")
    recipes = pd.read_parquet(REPO / "data/generated/parsed/recipes.parquet")
    lines = pd.read_parquet(REPO / "data/generated/aliased/lines.parquet")
    calls, recipes = calls[calls["mode"] == "full"], recipes[(recipes["mode"] == "full") & (recipes["recipe_idx"] < 5)]
    lines = lines[lines["recipe_uid"].isin(recipes["recipe_uid"])]
    per = {}
    for m, c in calls.groupby("model_key"):
        rec, ln = recipes[recipes["model_key"] == m], lines[lines["model_key"] == m]
        per[m] = {
            "name": MODEL_NAMES.get(m, m),
            "family": str(c["family"].iloc[0]),
            "calls": int(len(c)),
            "planned_calls": CALLS_PER_MODEL,
            "complete": bool(len(c) >= CALLS_PER_MODEL),
            "recipes": int(len(rec)),
            "regions": int(rec["region_code"].nunique()),
            "exactly5_share": r((c["n_recipes"] == 5).mean()),
            "parse_ok_share": r((c["parse_status"] == "ok").mean()),
            "parse_any_share": r(c["parse_status"].isin(["ok", "repaired", "salvaged"]).mean()),
            "lines_per_recipe": r(rec["n_ingredient_lines"].mean(), 2),
            "alias_full_share": r((ln["how"] == "full").mean()),
            "alias_matched_share": r(ln["entity_id"].notna().mean()),
        }
    proprietary = {"openai-gpt-6-luna", "gemini-3.5-flash-lite"}
    families = {v["family"] for k, v in per.items() if k not in proprietary}
    return {
        "models": per,
        "total_recipes": int(sum(v["recipes"] for v in per.values())),
        "total_calls": int(sum(v["calls"] for v in per.values())),
        "n_models": len(per),
        "n_proprietary": len(proprietary & set(per)),
        "n_open_families": len(families),
        "meets_proposal": bool(
            sum(v["recipes"] for v in per.values()) >= 10_000 and len(per) >= 3 and len(families) >= 2
        ),
        "pending": [MODEL_NAMES.get(m, m) for m, v in per.items() if not v["complete"]],
    }


def replication() -> dict:
    tg = json.loads((REPO / "data/reference/singh_bagler_2018_targets.json").read_text())
    res = pd.read_csv(INTERIM / "replication/results_v1_keep_empty.csv")
    out = {}
    for null, pub in tg["z_vs_null"].items():
        sub = res[res["null_model"] == null].set_index("code")
        codes = [c for c in pub if c in sub.index]
        x, y = np.array([pub[c] for c in codes]), sub.loc[codes, "z_sb"].to_numpy()
        out[null] = {"n": len(codes), "sign_agreement": int((np.sign(x) == np.sign(y)).sum()),
                     "spearman": r(st.spearmanr(x, y).statistic)}  # fmt: skip
    rand = res[res["null_model"] == "random"].set_index("region")
    korea = pd.read_csv(INTERIM / "replication/contributions_ns_ingredient_noadditive.csv")
    korea = korea[korea["region"] == "Korea"].sort_values("chi_percent", ascending=False).head(3)
    return {
        "by_null": out,
        "z_random": {k: r(v, 1) for k, v in rand["z_sb"].items()},
        "d_random": {k: r(v) for k, v in rand["d"].items()},
        "korea_top3": [
            {"name": n, "chi_percent": r(c, 2)} for n, c in zip(korea["name"], korea["chi_percent"], strict=True)
        ],
        "n_recipes": {k: int(v) for k, v in rand["n_recipes"].items()},
    }


def equal_n() -> dict:
    out = {}
    m = AUDIT / "real/equal_n_pairing_matched290.parquet"
    if m.exists():
        mm = pd.read_parquet(m)
        share = mm.groupby("region")["d"].apply(lambda s: (s > 0).mean())
        out["matched290_positive_share"] = {k: r(v) for k, v in share.items()}
        out["contrasting_robust"] = int(sum(share[c] == 0 for c in CONTRASTING))
        uniform = [k for k in share.index if k not in CONTRASTING]
        out["uniform_robust"] = int(sum(share[k] >= 0.95 for k in uniform))
        out["uniform_flipped"] = sorted(k for k in uniform if share[k] < 0.5)
    sub = AUDIT / "real/subsamples_n160.parquet"
    if sub.exists():
        s = pd.read_parquet(sub)
        out["n160_sign_rule_false_positive_rate"] = r(s["sign_error"].mean(), 4)
        g = s.groupby("region")
        out["n160_sign_match"] = {k: r(v) for k, v in g["sign_match"].mean().items()}
        out["n160_reference_medians"] = {
            c: r(s[c].median(), 4)
            for c in [
                "prevalence_jsd",
                "category_jsd",
                "sai",
                "fingerprint_spearman",
                "freq_rank_mae",
                "near_copy_rate",
            ]
        }
        out["n160_sai_band"] = [r(s["sai"].quantile(0.025)), r(s["sai"].quantile(0.975))]
    eq = AUDIT / "real/equal_n_pairing.parquet"
    if eq.exists():
        e = pd.read_parquet(eq)
        usa = e[(e["region"] == "USA") & (e["N"] == 160)]
        out["usa_pool_effect"] = {"delta_ns_own_pool_n160": r(usa["delta_ns"].median(), 2),
                                  "delta_ns_full_pool_n160": r(usa["delta_ns_fullpool"].median(), 2),
                                  "pool_n160": r(usa["pool"].mean(), 0)}  # fmt: skip
    return out


def copy_mutate() -> dict:
    p = AUDIT / "copy_mutate/scorecard.parquet"
    if not p.exists():
        return {"pending": True}
    cm = pd.read_parquet(p)
    cols = [
        "sign_match",
        "sign_error",
        "prevalence_jsd",
        "sai",
        "freq_rank_mae",
        "fingerprint_spearman",
        "near_copy_rate",
    ]
    out = {}
    for (src, scope), g in cm.groupby(["source", "scope"]):
        out[f"{src}|{scope}"] = {c: r(g[c].mean() if c.startswith("sign") else g[c].median()) for c in cols}
    return out


def llm_audit() -> dict:
    p = AUDIT / "full/scorecard.parquet"
    if not p.exists():
        return {"pending": True}
    sc = pd.read_parquet(p)
    pooled = sc[sc["scope"] == "pooled"]
    out = {}
    for m, g in pooled.groupby("model"):
        out[m] = {
            "regions": int(len(g)),
            "recipes": int(g["n_recipes"].sum()),
            "sign_agreement": r(g["sign_match"].mean()),
            "sign_errors_preregistered": int(g["sign_error"].sum()),
            "sign_errors_in_contrasting": int(g[g["region"].isin(CONTRASTING)]["sign_error"].sum()),
            "median_prevalence_jsd": r(g["prevalence_jsd"].median()),
            "median_category_jsd": r(g["category_jsd"].median()),
            "median_sai": r(g["sai"].median()),
            "median_fingerprint_spearman": r(g["fingerprint_spearman"].median()),
            "median_diversity_ratio": r(g["diversity_ratio"].median()),
            "median_freq_rank_mae": r(g["freq_rank_mae"].median()),
            "mean_near_copy_rate": r(g["near_copy_rate"].mean()),
            "median_novelty": r(g["novelty_mean"].median()),
        }
    h1 = REPO / "tables/h1_equal_n_full.csv"
    if h1.exists():
        h = pd.read_csv(h1)
        for m, g in h.groupby("model"):
            if m in out:
                out[m]["equal_n_too_uniform"] = int((g["verdict"] == "too uniform").sum())
                out[m]["equal_n_too_contrasting"] = int((g["verdict"] == "too contrasting").sum())
                out[m]["equal_n_typical"] = int((g["verdict"] == "typical").sum())
                out[m]["equal_n_median_shift_d"] = r((g["d"] - g["d_ref_median"]).median())
    return out


def rq_tests() -> list[dict]:
    p = REPO / "tables/rq_tests_full.csv"
    if not p.exists():
        return []
    t = pd.read_csv(p)
    return [
        {k: (r(v, 5) if isinstance(v, float) else v) for k, v in row.items()} for row in t.to_dict(orient="records")
    ]


def steering() -> dict:
    out = {}
    m = INTERIM / "steering/scorer_metrics.json"
    if m.exists():
        s = json.loads(m.read_text())
        c = s["classifier"]
        out["scorer"] = {
            "classifier": c["selected"],
            "test_macro_f1": r(c["test_macro_f1"]),
            "test_top1": r(c["test_top1"]),
            "test_top3": r(c["test_top3"]),
            "ece_raw": r(c["test_ece_raw"]),
            "ece_calibrated": r(c["test_ece_calibrated"]),
            "comparator_val_macro_f1": r(c["selection_on_val"]["hist_gbdt"]["val_macro_f1"]),
            "weights": s["weights"],
            "weighting": s.get("weights_selected"),
            "auc_test": {
                row["negatives"]: {k: r(v) for k, v in row.items() if k != "negatives"} for row in s["scorer_auc_test"]
            },
            "eval_classifier_macro_f1": r(s["eval_classifier"]["macro_f1_on_A_val"]),
        }
    for name in ("bon_main_summary.json", "revise_main_summary.json"):
        p = INTERIM / "steering" / name
        if p.exists():
            out[name.replace("_summary.json", "")] = json.loads(p.read_text())
    return out


def kaggle_runs() -> dict:
    """Per Kaggle run: engine and generation minutes, parsed from the downloaded kernel logs."""
    import re

    out = {}
    for log in sorted((REPO / "data/generated/raw").glob("*/*.log")):
        raw = log.read_text(errors="replace")
        try:
            txt = "".join(e.get("data", "") for e in json.loads(raw))
        except Exception:
            txt = raw
        eng = re.findall(r"Engine: (\S+)", txt)
        mins = re.findall(r"Generation finished in ([\d.]+) min", txt)
        out[log.stem] = {"engine": eng[-1] if eng else None, "minutes": float(mins[-1]) if mins else None,
                         "vllm_failed": "vLLM could not start" in txt}  # fmt: skip
    return out


def resources() -> dict:
    from dotenv import dotenv_values

    ledger = json.loads((REPO / "data/generated/spend_ledger.json").read_text())
    budget = dotenv_values(REPO / ".env").get("BUDGET_USD")
    runs = kaggle_runs()
    return {"openai_usd_spent": r(sum(e.get("actual_usd", 0) for e in ledger), 2),
            "budget_usd": float(budget) if budget else None, "kaggle_runs": runs,
            "kaggle_qwen_full_minutes": runs.get("llmcuisine22-qwen3-5-9b-full", {}).get("minutes"),
            "kaggle_llama_full_minutes": runs.get("llmcuisine22-llama-3-1-8b-full", {}).get("minutes")}  # fmt: skip


def drivers() -> dict:
    """RQ2a: overlap of top-5 chi% drivers with the real cuisine, per model, and the genuine-sample reference."""
    p, ref = AUDIT / "full/drivers.parquet", AUDIT / "real/drivers_subsamples_n160.parquet"
    if not p.exists():
        return {"pending": True}
    d = pd.read_parquet(p)
    out = {m: {"regions": int(len(g)), "median_jaccard5": r(g["jaccard5"].median()),
               "median_spearman": r(g["spearman"].median()),
               "korea_top5": list(g.loc[g["region"] == "Korea", "top5_names"].iloc[0]) if (g["region"] == "Korea").any() else None}
           for m, g in d.groupby("model")}  # fmt: skip
    if ref.exists():
        rr = pd.read_parquet(ref)
        out["real@160"] = {"median_jaccard5": r(rr["jaccard5"].median()), "median_spearman": r(rr["spearman"].median()),
                           "reps_per_region": int(rr.groupby("region").size().median())}  # fmt: skip
    out["korea_real_top5"] = (
        list(d.loc[d["region"] == "Korea", "real_top5_names"].iloc[0]) if (d["region"] == "Korea").any() else None
    )
    return out


def networks() -> dict:
    """Co-occurrence networks (plan §6.5; analysis/audit/networks.py): per metric and source, the median over regions,
    and for each model the number of regions below the equal-size genuine-sample reference (BH-FDR 5%)."""
    p = AUDIT / "full/networks.parquet"
    if not p.exists():
        return {"pending": True}
    d = pd.read_parquet(p)
    out: dict = {"primary_metric": "npmi_top100", "reference": "real@160 (held out; LLM values use their own n)",
                 "n_draws": 100}  # fmt: skip
    for metric, g in d.groupby("metric"):
        out[metric] = {}
        for source, gs in g.groupby("source"):
            row = {"regions": int(len(gs)), "median": r(gs["value"].median())}
            if source != "real@160":
                row["below_reference_fdr05"] = int((gs["q_low"] < 0.05).sum())
            out[metric][source] = row
    return out


def main() -> int:
    summary = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset(),
        "replication": replication(),
        "equal_n": equal_n(),
        "copy_mutate": copy_mutate(),
        "llm_audit": llm_audit(),
        "rq_tests": rq_tests(),
        "drivers": drivers(),
        "networks": networks(),
        "steering": steering(),
        "resources": resources(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    d = summary["dataset"]
    print(
        f"{d['total_recipes']:,} recipes from {d['n_models']} models; pending: {d['pending']} -> {OUT.relative_to(REPO)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
