"""RQ2a: do LLM cuisines rely on the same flavor drivers as the real cuisines?

Driver = χ%, Singh & Bagler's contribution: the % change in the cuisine's mean Ns when the ingredient is removed
from its recipes, oriented by the sign of Z, over the 30 most frequent non-additive ingredients (the definition
calibrated in the replication: published top-1 in our top-3 for 22/22 regions).

Per model and region (pooled LLM cuisine, N up to 160):
  jaccard5  overlap of the top-5 χ% ingredients with the real cuisine's top-5
  spearman  rank correlation of χ% over the union of both top-30 lists (an absent ingredient has χ% = 0)
The same two numbers for 50 random 160-recipe subsamples of each real cuisine give the overlap expected from
sampling alone. The plan sets no threshold for RQ2a; the per-region distributions are reported (plan §1).

Usage (repo root): .venv/bin/python analysis/audit/drivers.py [--tag full] [--reps 50]
Output: data/interim/audit/<tag>/drivers.parquet, data/interim/audit/real/drivers_subsamples_n160.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_io import AUDIT, LLM_N_PER_REGION, REAL, REPO, load_reference  # noqa: E402
from llm_audit import LINES, RECIPES_PER_CALL, cuisines_from_lines  # noqa: E402

from flavoraudit.audit import real_subsample  # noqa: E402
from flavoraudit.contribution import chi_percent  # noqa: E402
from flavoraudit.pairing import Cuisine  # noqa: E402

TOP_K, TOP_N = 30, 5


def compare(candidate: Cuisine, real_top: pd.DataFrame, real: Cuisine, S: np.ndarray, seed: int) -> dict:
    """Top-5 overlap and χ% rank correlation of a candidate cuisine with the real cuisine's drivers."""
    cand = chi_percent(candidate, S, top_k=TOP_K, n_sd=2000, seed=seed)
    top_r, top_c = set(real_top["row"].head(TOP_N)), set(cand["row"].head(TOP_N))
    union = sorted(set(real_top["row"]) | set(cand["row"]))
    a = chi_percent(real, S, rows=union, n_sd=2000, seed=seed).set_index("row")["chi_percent"]
    b = chi_percent(candidate, S, rows=union, n_sd=2000, seed=seed).set_index("row")["chi_percent"]
    rho = st.spearmanr(a.reindex(union), b.reindex(union)).statistic
    return {
        "jaccard5": len(top_r & top_c) / len(top_r | top_c),
        "spearman": float(rho),
        "top5_rows": [int(r) for r in cand["row"].head(TOP_N)],
        "top5_chi": [round(float(v), 3) for v in cand["chi_percent"].head(TOP_N)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--mode", default="full")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--skip-real", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    ref, fp, idx, _ = load_reference()
    S = ref.S
    inv = {r: e for e, r in idx.items()}

    def names(rows):
        return [fp.names.get(inv.get(int(r)), str(r)) for r in rows]

    labels = {k: v["label"] for k, v in yaml.safe_load((REPO / "configs" / "regions.yaml").read_text()).items()}
    real_top = {r: chi_percent(c, S, top_k=TOP_K, n_sd=20_000, seed=0) for r, c in ref.real.items()}
    print(f"real drivers: {time.time() - t0:.0f}s | Korea top-3: {names(real_top['Korea']['row'].head(3))}")

    if not args.skip_real:
        rows = []
        for region, c in ref.real.items():
            rng = np.random.default_rng([1, zlib.crc32(region.encode())])
            for k in range(args.reps):
                sub, _ = real_subsample(ref, region, LLM_N_PER_REGION, rng)
                res = compare(sub, real_top[region], c, S, seed=k)
                rows.append({"region": region, "rep": k, "jaccard5": res["jaccard5"], "spearman": res["spearman"]})
            print(f"  real subsamples {region:22s} | {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_parquet(REAL / f"drivers_subsamples_n{LLM_N_PER_REGION}.parquet", index=False)

    lines = pd.read_parquet(LINES)
    lines = lines[(lines["mode"] == args.mode) & (lines["recipe_idx"] < RECIPES_PER_CALL)]
    categories = next(iter(ref.real.values())).categories
    out = []
    for (mk, code), g in lines.groupby(["model_key", "region_code"], sort=True):
        region = labels[code]
        c = cuisines_from_lines(g, idx, categories, f"{mk}:{code}")
        res = compare(c, real_top[region], ref.real[region], S, seed=0)
        out.append(
            {
                "model": mk,
                "region": region,
                **res,
                "top5_names": names(res["top5_rows"]),
                "real_top5_names": names(real_top[region]["row"].head(TOP_N)),
            }
        )
    df = pd.DataFrame(out)
    path = AUDIT / args.tag / "drivers.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(df.groupby("model")[["jaccard5", "spearman"]].median().round(3).to_string())
    print(f"done in {time.time() - t0:.0f}s -> {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
