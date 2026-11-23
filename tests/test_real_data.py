"""Tests on the real CulinaryDB / FlavorDB / Ahn data (skipped when data/raw is absent, e.g. in CI)."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as st

from flavoraudit.data import default_data_dir

HAVE_DATA = (default_data_dir() / "culinarydb" / "extracted" / "01_Recipe_Details.csv").exists()
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="raw data not downloaded")
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def world():
    from flavoraudit.data import load_ahn, load_culinarydb, load_flavordb
    from flavoraudit.flavor import build_profiles

    cdb, fdb, ahn = load_culinarydb(), load_flavordb(), load_ahn()
    fp = build_profiles(cdb, fdb, ahn)
    S, idx = fp.shared_matrix()
    return cdb, fdb, fp, S, idx


def test_culinarydb_matches_paper_counts(world):
    cdb = world[0]
    assert len(cdb.recipes) == 45772 and len(cdb.regions) == 22
    misc = cdb.recipes[cdb.recipes["cuisine"].str.startswith("Misc.")]
    assert len(misc) == 207  # aggregate-only recipes, as stated in Singh & Bagler (2018)


def test_profiles_follow_reconstruction_rules(world):
    _, fdb, fp, S, idx = world
    assert len(fdb.entity_molecules) == 936
    assert fp.unmatched_constituents == {}
    prov = fp.table()["source"]
    assert (prov == "ahn2011").sum() == 4 and (prov == "no-profile (paper)").sum() == 4
    assert prov.str.startswith("flavordb-parent").sum() == 13
    assert (S == S.T).all() and all(S[idx[e], idx[e]] == fp.size(e) for e in fp.profiles)
    assert fp.size(984) == fp.size(240)  # Lemon Juice inherits Lemon


def test_korea_contrasting_and_italy_uniform(world):
    from flavoraudit.pairing import build_cuisines
    from flavoraudit.stats import food_pairing

    cdb, _, fp, S, idx = world
    cus = build_cuisines(cdb.recipe_entities, cdb.recipes, idx, fp.categories, labels=["Korea", "Italy"])
    assert food_pairing(cus["Korea"], S, "random", n_null=20000, seed=1).z_sb < 0
    assert food_pairing(cus["Italy"], S, "random", n_null=20000, seed=1).z_sb > 0


@pytest.mark.slow
def test_fig4_replication_random_null_signs_and_rank(world):
    """Regression guard for the Phase 2 gate: 22/22 signs and rho >= 0.95 against the published Fig. 4."""
    from flavoraudit.pairing import build_cuisines
    from flavoraudit.stats import food_pairing

    cdb, _, fp, S, idx = world
    t = json.loads((REPO / "data" / "reference" / "singh_bagler_2018_targets.json").read_text())
    code = {v: k for k, v in t["region_codes"].items()}
    cus = build_cuisines(cdb.recipe_entities, cdb.recipes, idx, fp.categories, labels=cdb.regions)
    ours = {code[r]: food_pairing(c, S, "random", n_null=20000, seed=2).z_sb for r, c in cus.items()}
    pub = t["z_vs_null"]["random"]
    x = np.array([pub[c] for c in pub])
    y = np.array([ours[c] for c in pub])
    assert (np.sign(x) == np.sign(y)).all()
    assert st.spearmanr(x, y).statistic >= 0.95
