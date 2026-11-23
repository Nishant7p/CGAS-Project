"""D2 audit: score every LLM cuisine against the real CulinaryDB cuisine it claims to be.

Each (model, region) cuisine pools both prompt conditions (N up to 160 = 16 calls x 5 recipes x 2 conditions), and
each (model, region, condition) cuisine has N up to 80 (plain vs persona, RQ4). Every cuisine goes through
flavoraudit.audit.scorecard_row, the same function that scores the real equal-N subsamples
(analysis/audit/real_subsamples.py) and the copy-mutate baseline (analysis/audit/copy_mutate_baseline.py).

Rules: only full-mode calls (the pilot only tested the pipeline); at most the first 5 recipes of a call (the design
asks for exactly 5); unmatched lines are dropped, as in CulinaryDB; recipes with no matched ingredient are dropped.

Input:  data/generated/aliased/lines.parquet (generation/alias_lines.py)
Usage:  .venv/bin/python analysis/audit/llm_audit.py [--mode full] [--tag full] [--models a b]
Output: data/interim/audit/<tag>/scorecard.parquet (+ .csv) and summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_io import AUDIT, CONTRASTING, REPO, load_reference  # noqa: E402

from flavoraudit.audit import scorecard_row  # noqa: E402
from flavoraudit.pairing import Cuisine  # noqa: E402

LINES = REPO / "data" / "generated" / "aliased" / "lines.parquet"
RECIPES_PER_CALL = 5


def cuisines_from_lines(lines: pd.DataFrame, idx: dict[int, int], categories: dict[int, str], name: str) -> Cuisine:
    """A Cuisine from aliased lines: one recipe per recipe_uid, entities mapped to S rows and de-duplicated."""
    ok = lines.dropna(subset=["entity_id"])
    recs = []
    for _, g in ok.groupby("recipe_uid", sort=True):
        rows = np.unique(np.array([idx[int(e)] for e in g["entity_id"] if int(e) in idx], dtype=np.int64))
        if len(rows):
            recs.append(rows)
    return Cuisine(name, recs, categories)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--n-null", type=int, default=20_000)
    args = ap.parse_args()
    tag = args.tag or args.mode
    out = AUDIT / tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ref, _fp, idx, _cdb = load_reference()
    categories = next(iter(ref.real.values())).categories
    labels = {k: v["label"] for k, v in yaml.safe_load((REPO / "configs" / "regions.yaml").read_text()).items()}
    lines = pd.read_parquet(LINES)
    lines = lines[(lines["mode"] == args.mode) & (lines["recipe_idx"] < RECIPES_PER_CALL)]
    if args.models:
        lines = lines[lines["model_key"].isin(args.models)]
    rows = []
    for (mk, code), g in lines.groupby(["model_key", "region_code"], sort=True):
        region = labels[code]
        family = g["family"].iloc[0]
        scopes = [("pooled", g)] + [(cond, gc) for cond, gc in g.groupby("condition", sort=True)]
        for scope, gs in scopes:
            c = cuisines_from_lines(gs, idx, categories, f"{mk}:{code}:{scope}")
            row = scorecard_row(c, region, ref, n_null=args.n_null, seed=7)
            rows.append({"model": mk, "family": family, "scope": scope, "code": code, **row})
        p = rows[-3] if len(scopes) == 3 else rows[-1]
        print(
            f"{mk:24s} {region:20s} n={p['n_recipes']:3d} Z={p['z_sb']:7.1f} (real {p['z_real']:7.1f}) "
            f"err={int(p['sign_error'])} SAI={p['sai']:.2f} JSD={p['prevalence_jsd']:.3f} | {time.time() - t0:.0f}s",
            flush=True,
        )
    sc = pd.DataFrame(rows)
    sc.to_parquet(out / "scorecard.parquet", index=False)
    sc.to_csv(out / "scorecard.csv", index=False)
    pooled = sc[sc["scope"] == "pooled"].assign(contrasting=lambda d: d["region"].isin(CONTRASTING))
    summary = (
        pooled.groupby("model")
        .agg(
            regions=("region", "size"),
            recipes=("n_recipes", "sum"),
            sign_agreement=("sign_match", "mean"),
            sign_errors=("sign_error", "sum"),
            sign_errors_contrasting=("sign_error", lambda s: int(s[pooled.loc[s.index, "contrasting"]].sum())),
            median_sai=("sai", "median"),
            median_prev_jsd=("prevalence_jsd", "median"),
            median_fp_spearman=("fingerprint_spearman", "median"),
            median_diversity_ratio=("diversity_ratio", "median"),
            near_copy=("near_copy_rate", "mean"),
        )
        .round(3)
    )
    print("\n" + summary.to_string())
    (out / "summary.json").write_text(json.dumps(summary.reset_index().to_dict(orient="records"), indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
