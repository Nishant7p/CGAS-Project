"""Replicate Singh & Bagler (2018) Fig. 5 (top-3 contributors per region) with both removal procedures.

Writes data/interim/replication/contributions_<procedure>.csv and contributions_summary.csv, and prints the
agreement with data/reference/singh_bagler_2018_targets.json.

Usage (repo root): .venv/bin/python analysis/replication/replicate_contributions.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from flavoraudit.contribution import chi_percent
from flavoraudit.data import load_ahn, load_culinarydb, load_flavordb
from flavoraudit.flavor import build_profiles
from flavoraudit.pairing import build_cuisines

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "interim" / "replication"
T = json.loads((REPO / "data" / "reference" / "singh_bagler_2018_targets.json").read_text())
CODE = {v: k for k, v in T["region_codes"].items()}


def main() -> int:
    cdb, fdb, ahn = load_culinarydb(), load_flavordb(), load_ahn()
    fp = build_profiles(cdb, fdb, ahn)  # canonical configuration = v1_keep_empty (no entities dropped)
    S, idx = fp.shared_matrix()
    inv = {r: e for e, r in idx.items()}
    cuisines = build_cuisines(cdb.recipe_entities, cdb.recipes, idx, fp.categories, labels=cdb.regions)
    published = {**T["top3_contribution_percent"]["positive"], **T["top3_contribution_percent"]["negative"]}
    summary = []
    settings = {
        "ns_ingredient_noadditive": dict(basis="ns", procedure="ingredient", exclude_categories=("Additive",)),
        "z_ingredient_all": dict(basis="z", procedure="ingredient", exclude_categories=()),
        "z_recipes_all": dict(basis="z", procedure="recipes", exclude_categories=()),
    }
    for procedure, kw in settings.items():
        t, rows = time.time(), []
        for region, cu in cuisines.items():
            df = chi_percent(cu, S, top_k=30, seed=7, **kw)
            df["name"] = [fp.names[inv[r]] for r in df["row"]]
            df["region"], df["code"] = region, CODE[region]
            rows.append(df)
        allc = pd.concat(rows, ignore_index=True)
        allc.to_csv(OUT / f"contributions_{procedure}.csv", index=False)
        top1_hits = top3_overlap = 0
        detail = []
        for code, pub in published.items():
            ours = allc[allc["code"] == code].head(3)
            names = list(ours["name"])
            pub_names = list(pub)
            top1_hits += pub_names[0] in names
            top3_overlap += len(set(names) & set(pub_names))
            detail.append(
                f"{code}: ours {[(n, round(v, 2)) for n, v in zip(names, ours['chi_percent'], strict=True)]} | published {pub}"
            )
        summary.append(
            {
                "procedure": procedure,
                "published_top1_in_our_top3": f"{top1_hits}/22",
                "top3_overlap": f"{top3_overlap}/66",
                "seconds": round(time.time() - t),
            }
        )
        print(
            f"\n=== removal procedure: {procedure} -> top-1 in our top-3: {top1_hits}/22, top-3 overlap {top3_overlap}/66"
        )
        print("\n".join(detail))
    pd.DataFrame(summary).to_csv(OUT / "contributions_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
