"""D3 before/after example for the talk and the app: one real critique-and-revise pair, chosen by a fixed rule.

Rule (fixed before looking at any pair): among GPT-6-Luna's round-1 revisions (the model run for both rounds), keep
those that raised the composite fidelity score and pick the one at the median gain (a typical successful revision, not
the best one). Over all revisions it also reports the share that raised the score and the median gain, so the example
can be read in context.

Scores are percentiles against held-out genuine recipes of the claimed region, the scale the FlavorPrint app uses. The
independent classifier (never used for steering) gives the cuisine it assigns before and after.

Usage (repo root): .venv/bin/python steering/example.py
Output: results/steering_example.json
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis" / "audit"))
from audit_io import load_reference  # noqa: E402

from steering.run_revise import STEER, round_inputs, rows_by_uid  # noqa: E402

MODEL = "openai-gpt-6-luna"
OUT = REPO / "results" / "steering_example.json"


def percentile(sorted_values: np.ndarray, x: float) -> float:
    """Mid-rank percentile, as in app/streamlit_app.py."""
    lo, hi = np.searchsorted(sorted_values, x, "left"), np.searchsorted(sorted_values, x, "right")
    return 100.0 * (lo + hi) / (2 * len(sorted_values))


def as_list(x) -> list[str]:
    return [str(v) for v in (x.tolist() if hasattr(x, "tolist") else list(x))]


def main() -> int:
    ref, fp, idx, _ = load_reference()
    inv = {r: e for e, r in idx.items()}
    names = {r: str(fp.names.get(inv.get(r), r)) for r in range(ref.S.shape[0])}
    with open(STEER / "scorer.pkl", "rb") as fh:
        scorer = pickle.load(fh)
    with open(STEER / "eval_classifier.pkl", "rb") as fh:
        ev = pickle.load(fh)
    splits = json.loads((STEER / "splits.json").read_text())["positions"]
    labels = {k: v["label"] for k, v in yaml.safe_load((REPO / "configs" / "regions.yaml").read_text()).items()}
    parents, plines = round_inputs(MODEL, 1, "main")
    revised = pd.read_parquet(STEER / f"recipes_rev1_{MODEL}.parquet")
    rlines = pd.read_parquet(STEER / f"lines_rev1_{MODEL}.parquet")
    prow, rrow = rows_by_uid(plines, idx), rows_by_uid(rlines, idx)
    pinfo = parents.drop_duplicates("recipe_uid").set_index("recipe_uid")
    ref_sorted: dict[str, np.ndarray] = {}

    def pct(score: float, region: str) -> float:
        if region not in ref_sorted:
            test = [ref.real[region].recipes[i] for i in splits[region]["test"]]
            ref_sorted[region] = np.sort(scorer.score(test, region))
        return percentile(ref_sorted[region], score)

    def indep(rows: np.ndarray) -> str:
        Xe = ev["featurizer"].transform([rows])
        pred = ev["model"].predict(Xe.toarray() if ev["name"] == "hist_gbdt" else Xe)
        return str(np.array(ev["regions"])[pred][0])

    rows = []
    for rec in revised.itertuples(index=False):
        meta = json.loads(rec.meta)
        p, u = meta.get("parent_uid"), rec.recipe_uid
        if p not in prow or u not in rrow or len(prow[p]) < 2 or len(rrow[u]) < 2:
            continue
        region = labels[rec.region_code]
        before, after = float(scorer.score([prow[p]], region)[0]), float(scorer.score([rrow[u]], region)[0])
        rows.append({"uid": u, "parent": p, "region": region, "gain": after - before,
                     "pct_before": pct(before, region), "pct_after": pct(after, region)})  # fmt: skip
    df = pd.DataFrame(rows)
    pos = df[df["gain"] > 0].sort_values(["gain", "uid"]).reset_index(drop=True)
    pick = pos.iloc[len(pos) // 2]
    rec = revised.set_index("recipe_uid").loc[pick["uid"]]
    meta = json.loads(rec["meta"])
    par = pinfo.loc[pick["parent"]]
    region = pick["region"]
    a, b = set(prow[pick["parent"]].tolist()), set(rrow[pick["uid"]].tolist())

    def zs(r: np.ndarray) -> dict[str, float]:
        z = scorer.zscores([r], region).iloc[0]
        return {k: (None if pd.isna(v) else round(float(v), 2)) for k, v in z.items()}

    example = {
        "rule": "GPT-6-Luna round-1 revisions that raised the composite fidelity score; the one at the median gain",
        "model": MODEL,
        "region": region,
        "context": {
            "revisions_scored": int(len(df)),
            "share_score_raised": round(float((df["gain"] > 0).mean()), 3),
            "median_gain_all": round(float(df["gain"].median()), 3),
            "median_percentile_before": round(float(df["pct_before"].median()), 1),
            "median_percentile_after": round(float(df["pct_after"].median()), 1),
        },
        "before": {
            "title": str(par["title"]),
            "ingredients": as_list(par["ingredients"]),
            "fidelity_percentile": round(float(pick["pct_before"]), 1),
            "components_z": zs(prow[pick["parent"]]),
            "independent_classifier": indep(prow[pick["parent"]]),
        },
        "feedback_given": meta.get("feedback", {}),
        "after": {
            "title": str(rec["title"]),
            "ingredients": as_list(rec["ingredients"]),
            "fidelity_percentile": round(float(pick["pct_after"]), 1),
            "components_z": zs(rrow[pick["uid"]]),
            "independent_classifier": indep(rrow[pick["uid"]]),
        },
        "entities_added": sorted(names[r] for r in b - a),
        "entities_removed": sorted(names[r] for r in a - b),
        "identity_jaccard": round(len(a & b) / len(a | b), 3),
        "uids": {"parent": str(pick["parent"]), "revised": str(pick["uid"])},
    }
    OUT.write_text(json.dumps(example, indent=1, ensure_ascii=False))
    c = example["context"]
    print(f"{c['revisions_scored']} revisions; score raised in {c['share_score_raised']:.0%}; median percentile "
          f"{c['median_percentile_before']} -> {c['median_percentile_after']}")  # fmt: skip
    print(f"example ({region}): {example['before']['title']!r} {example['before']['fidelity_percentile']} -> "
          f"{example['after']['title']!r} {example['after']['fidelity_percentile']}; +{example['entities_added']} "
          f"-{example['entities_removed']}; classifier {example['before']['independent_classifier']} -> "
          f"{example['after']['independent_classifier']}")  # fmt: skip
    print(f"-> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
