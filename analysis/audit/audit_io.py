"""Shared inputs for the D2 audit scripts: the real cuisines, the shared-compound matrix and the scorecard Reference.

The Reference uses the canonical replication outputs (variant v1_keep_empty; docs/replication_report.md) for the real
cuisines' Z against the random null, and the real category fingerprints from analysis/audit/real_reference.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from flavoraudit.audit import Reference
from flavoraudit.data import load_ahn, load_culinarydb, load_flavordb
from flavoraudit.flavor import build_profiles
from flavoraudit.pairing import build_cuisines

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "data" / "interim" / "audit"
REAL = AUDIT / "real"
REPLICATION = REPO / "data" / "interim" / "replication" / "results_v1_keep_empty.csv"
LLM_N_PER_REGION = 160  # 16 calls x 5 recipes x 2 prompt conditions (plan, Phase 4)
# The 6 regions with negative (contrasting) food pairing in Singh & Bagler (2018), Fig. 4; hypothesis H1.
CONTRASTING = ("British Isles", "DACH Countries", "Eastern Europe", "Japan", "Korea", "Scandinavia")


def load_reference() -> tuple[Reference, object, dict[int, int], object]:
    """(Reference, FlavorProfiles, entity_id -> S row, CulinaryDB)."""
    cdb, fdb, ahn = load_culinarydb(), load_flavordb(), load_ahn()
    fp = build_profiles(cdb, fdb, ahn)
    S, idx = fp.shared_matrix()
    real = build_cuisines(cdb.recipe_entities, cdb.recipes, idx, fp.categories, labels=cdb.regions)
    rep = pd.read_csv(REPLICATION)
    z = rep[rep["null_model"] == "random"].set_index("region")["z_sb"].to_dict()
    fps = pd.read_parquet(REAL / "fingerprints.parquet")
    fingerprints = {r: g.drop(columns="region").set_index("category") for r, g in fps.groupby("region")}
    negative = tuple(sorted(r for r, v in z.items() if v < 0))
    assert negative == CONTRASTING, f"replication signs changed: {negative}"
    return Reference.build(real, S, real_z=z, fingerprints=fingerprints), fp, idx, cdb


def worker_count(default: int = 4) -> int:
    """Processes for per-region parallelism (the Mac has 8 cores and 8 GB RAM; each worker holds the data once)."""
    return max(1, min(default, (os.cpu_count() or 2) - 2))
