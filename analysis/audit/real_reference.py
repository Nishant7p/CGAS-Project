"""Reference measurements for the 22 REAL CulinaryDB cuisines (the yardstick for every LLM cuisine in D2).

Outputs (data/interim/audit/real/):
  prevalence.parquet          P_ic per ingredient x region
  relative_prevalence.parquet p_ic (authenticity, Jain & Bagler 2018)
  signatures.json             top-10 signature ingredients per region (by relative prevalence)
  fingerprints.parquet        ΔNs_cat / Z_cat per region x category (Jain et al. 2015), 20,000 null recipes each
  diversity.parquet           diversity at full N and at N = 160 (the per-region LLM sample size)
  sizes.parquet               recipe-size distribution per region

Usage (repo root): .venv/bin/python analysis/audit/real_reference.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from flavoraudit.composition import diversity, prevalence_table, relative_prevalence, signature_set
from flavoraudit.data import load_ahn, load_culinarydb, load_flavordb
from flavoraudit.fingerprint import fingerprint
from flavoraudit.flavor import build_profiles
from flavoraudit.pairing import build_cuisines

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "interim" / "audit" / "real"
LLM_N_PER_REGION = 160


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cdb, fdb, ahn = load_culinarydb(), load_flavordb(), load_ahn()
    fp = build_profiles(cdb, fdb, ahn)
    S, idx = fp.shared_matrix()
    inv = {r: e for e, r in idx.items()}
    cus = build_cuisines(cdb.recipe_entities, cdb.recipes, idx, fp.categories, labels=cdb.regions)
    prev = prevalence_table(cus)
    rel = relative_prevalence(prev)
    prev.index = [inv[r] for r in prev.index]
    rel.index = [inv[r] for r in rel.index]
    prev.to_parquet(OUT / "prevalence.parquet")
    rel.to_parquet(OUT / "relative_prevalence.parquet")
    sig = {}
    for region in cus:
        ents = [int(e) for e in rel[region].sort_values(ascending=False).head(10).index]
        sig[region] = [
            {
                "entity_id": e,
                "name": fp.names[e],
                "relative_prevalence": round(float(rel.loc[e, region]), 4),
                "prevalence": round(float(prev.loc[e, region]), 4),
            }
            for e in ents
        ]
    (OUT / "signatures.json").write_text(json.dumps(sig, indent=1))
    rows, div, sizes = [], [], []
    for k, (region, c) in enumerate(cus.items()):
        f = fingerprint(c, S, n_null=20_000, seed=k)
        f["region"] = region
        rows.append(f.reset_index())
        div.append({"region": region, "scope": "full", **diversity(c)})
        div.append(
            {"region": region, "scope": f"n={LLM_N_PER_REGION}", **diversity(c, n=LLM_N_PER_REGION, reps=200, seed=k)}
        )
        sizes.append(pd.DataFrame({"region": region, "size": c.sizes}))
        print(f"{region:22s} fingerprint {len(f)} categories | {time.time() - t0:.0f}s", flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(OUT / "fingerprints.parquet", index=False)
    pd.DataFrame(div).to_parquet(OUT / "diversity.parquet", index=False)
    pd.concat(sizes, ignore_index=True).to_parquet(OUT / "sizes.parquet", index=False)
    # sanity print: signature ingredients for a few regions
    for region in ["Korea", "Italy", "Indian Subcontinent", "Mexico", "Japan"]:
        print(region, "->", [s["name"] for s in sig[region][:8]])
    _ = signature_set  # exported helper used by the LLM audit
    print(f"done in {time.time() - t0:.0f}s -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
