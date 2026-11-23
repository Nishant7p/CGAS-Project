"""Precompute everything FlavorPrint (D5) needs, so the app runs on a CPU without the source databases.

Output: app/artifacts/
  core.pkl        S matrix, entity names and categories per S row, the 22 regions (labels, signature sets, real Z,
                  real recipe Ns, signature counts per genuine recipe, composite fidelity of genuine test recipes),
                  the production aliaser and the fidelity scorer
  dashboard/*.parquet  D2 results for the dashboard (whatever exists: LLM scorecard, copy-mutate, real reference)
  dashboard/summary.json  the headline numbers the dashboard quotes (replication, per-model audit, reference
                  medians, copy-mutate baseline, scorer accuracy), copied verbatim from results/summary.json

The artifacts derive from CulinaryDB and FlavorDB (CC BY-NC-SA 3.0) and are themselves CC BY-NC-SA 4.0 (see
DATA_LICENSE.md); the app shows this notice.

Usage (repo root, after steering/train_scorer.py): .venv/bin/python app/build_artifacts.py
  --dashboard-only   refresh dashboard/ only (reads data/interim/audit and results/summary.json; core.pkl untouched)
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis" / "audit"))
from audit_io import AUDIT, REAL, load_reference  # noqa: E402

from flavoraudit.aliasing import Aliaser  # noqa: E402
from flavoraudit.pairing import ns_values  # noqa: E402

OUT = REPO / "app" / "artifacts"
STEERING = REPO / "data" / "interim" / "steering"
SUMMARY = REPO / "results" / "summary.json"


def dashboard_summary(summary: dict) -> dict:
    """The subset of results/summary.json that the dashboard shows (values copied, never recomputed)."""
    ds, eq = summary.get("dataset", {}), summary.get("equal_n", {})
    keep = ("name", "family", "recipes", "regions", "complete")
    return {
        "generated_utc": summary.get("generated_utc"),
        "models": {m: {k: v.get(k) for k in keep} for m, v in ds.get("models", {}).items()},
        "total_recipes": ds.get("total_recipes"),
        "replication": summary.get("replication", {}).get("by_null", {}),
        "llm_audit": summary.get("llm_audit", {}),
        "copy_mutate": summary.get("copy_mutate", {}),
        "real_n160": {
            "medians": eq.get("n160_reference_medians", {}),
            "sai_band": eq.get("n160_sai_band"),
            "sign_match": eq.get("n160_sign_match", {}),
        },
        "scorer": {k: summary.get("steering", {}).get("scorer", {}).get(k) for k in ("test_top1", "test_top3")},
    }


def build_dashboard() -> None:
    dash = OUT / "dashboard"
    dash.mkdir(parents=True, exist_ok=True)
    for src, dst in (
        (AUDIT / "full" / "scorecard.parquet", "llm_scorecard.parquet"),
        (AUDIT / "full" / "drivers.parquet", "llm_drivers.parquet"),
        (AUDIT / "copy_mutate" / "scorecard.parquet", "copy_mutate_scorecard.parquet"),
        (REAL / "subsamples_n160.parquet", "real_subsamples.parquet"),
    ):
        if src.exists():
            shutil.copy(src, dash / dst)
            print("dashboard:", dst)
    if SUMMARY.exists():
        sub = dashboard_summary(json.loads(SUMMARY.read_text()))
        (dash / "summary.json").write_text(json.dumps(sub, indent=1) + "\n")
        print("dashboard: summary.json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dashboard-only", action="store_true", help="refresh dashboard/ only; core.pkl is untouched")
    args = ap.parse_args(argv)
    if args.dashboard_only:
        build_dashboard()
        return 0
    ref, fp, idx, cdb = load_reference()
    with open(STEERING / "scorer.pkl", "rb") as fh:
        scorer = pickle.load(fh)
    splits = json.loads((STEERING / "splits.json").read_text())["positions"]
    cfg = yaml.safe_load((REPO / "configs" / "regions.yaml").read_text())
    code_of = {v["label"]: k for k, v in cfg.items()}
    inv = {r: e for e, r in idx.items()}
    n_rows = ref.S.shape[0]
    names = [fp.names.get(inv.get(r), str(r)) for r in range(n_rows)]
    cats = [ref.real[next(iter(ref.real))].categories.get(r, "Unknown") for r in range(n_rows)]
    regions = {}
    for region, c in ref.real.items():
        sig = ref.signatures[region]
        sig_counts = np.array([np.isin(r, sig).sum() for r in c.recipes])
        test = [c.recipes[i] for i in splits[region]["test"]]
        regions[region] = {
            "code": code_of[region],
            "label": region,
            "phrase": cfg[code_of[region]]["phrase"],
            "n_recipes": len(c),
            "signature": sig,
            "z_real": float(ref.real_z[region]),
            "ns_sorted": np.sort(ns_values(c, ref.S)),
            "signature_counts": np.bincount(sig_counts, minlength=11),
            "composite_sorted": np.sort(scorer.score(test, region)),
        }
        print(
            f"{region:22s} n={len(c):5d} composite of genuine test recipes: median {np.median(regions[region]['composite_sorted']):.2f}"
        )
    core = {
        "S": ref.S.astype(np.int32),
        "entity_of_row": np.array([inv.get(r, -1) for r in range(n_rows)]),
        "row_of_entity": dict(idx),
        "names": names,
        "categories": cats,
        "regions": regions,
        "aliaser": Aliaser.reference(cdb),
        "scorer": scorer,
        "palette": yaml.safe_load((REPO / "configs" / "models.yaml").read_text())["palette"],
        "versions": {"flavoraudit": "0.1.0.dev0"},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "core.pkl", "wb") as fh:
        pickle.dump(core, fh, protocol=pickle.HIGHEST_PROTOCOL)
    build_dashboard()
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"artifacts: {size:.1f} MB -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
