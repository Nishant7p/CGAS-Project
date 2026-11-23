"""Tests for the scorecard (flavoraudit.audit) on a toy flavor world with two compound 'blocks'.

Ingredients 0-19 (block A) share 10 compounds with each other, as do 20-39 (block B); across blocks they share 1.
The real cuisine 'Mixed' always combines 3 A + 3 B ingredients, so it avoids flavor sharing (negative pairing).
A 'Clustered' candidate keeps each recipe inside one block (positive pairing): a sign error by the pre-registered
rule.
"""

import numpy as np
import pytest

from flavoraudit.audit import Reference, real_subsample, scorecard_row
from flavoraudit.composition import novelty
from flavoraudit.fingerprint import fingerprint
from flavoraudit.pairing import Cuisine

N_ING = 40


@pytest.fixture(scope="module")
def world():
    block = np.arange(N_ING) < 20
    S = np.where(block[:, None] == block[None, :], 10, 1).astype(np.int32)
    np.fill_diagonal(S, 20)
    cats = {i: f"c{i % 4}" for i in range(N_ING)}
    rng = np.random.default_rng(0)
    mixed = [
        np.sort(np.concatenate([rng.choice(20, 3, replace=False), 20 + rng.choice(20, 3, replace=False)]))
        for _ in range(300)
    ]
    other = [np.sort(rng.choice(N_ING, size=int(rng.integers(3, 9)), replace=False)) for _ in range(300)]
    real = {"Mixed": Cuisine("Mixed", mixed, cats), "Other": Cuisine("Other", other, cats)}
    fps = {k: fingerprint(c, S, n_null=1000, seed=0, min_recipes=20) for k, c in real.items()}
    ref = Reference.build(real, S, fingerprints=fps, n_null=5000)
    return S, cats, ref


FAST = dict(n_null=4000, n_boot=300, fingerprint_null=500)


def test_reference_pairing_signs(world):
    _, _, ref = world
    assert ref.real_z["Mixed"] < 0  # mixing blocks avoids shared compounds
    assert ref.offsets == {"Mixed": 0, "Other": 300} and len(ref.recipes) == 600
    assert len(ref.signatures["Mixed"]) == 10


def test_the_real_cuisine_scores_as_a_perfect_match(world):
    _, _, ref = world
    row = scorecard_row(ref.real["Mixed"], "Mixed", ref, **FAST)
    assert row["prevalence_jsd"] == pytest.approx(0, abs=1e-12) and row["category_jsd"] == pytest.approx(0, abs=1e-12)
    assert row["sai"] == pytest.approx(1) and row["freq_rank_mae"] == 0 and row["size_ks"] == 0
    assert row["sign_match"] and not row["sign_error"] and row["delta_ns_hi"] < 0
    assert row["diversity_ratio"] == pytest.approx(1)
    assert row["near_copy_rate"] == 1.0  # its recipes are in the reference set (no leave-one-out here)
    # The toy's 4 categories (i % 4) are statistically identical, so their Z_cat ranks are noise: only check that
    # the fingerprint comparison ran. On real data, H2c compares against real-subsample noise for this reason.
    assert (
        row["fingerprint_n"] >= 3
        and np.isfinite(row["fingerprint_spearman"])
        and np.isfinite(row["fingerprint_cosine"])
    )


def test_clustered_pairing_is_a_sign_error(world):
    _, cats, ref = world
    rng = np.random.default_rng(1)
    clustered = [np.sort((20 if k % 2 else 0) + rng.choice(20, 6, replace=False)) for k in range(160)]
    row = scorecard_row(Cuisine("LLM", clustered, cats), "Mixed", ref, **FAST)
    assert row["z_sb"] > 0 and row["delta_ns_lo"] > 0
    assert not row["sign_match"] and row["sign_error"]
    assert row["d_lo"] <= row["d"] <= row["d_hi"]


def test_real_subsamples_use_leave_one_out_novelty(world):
    _, _, ref = world
    sub, self_idx = real_subsample(ref, "Mixed", 50, np.random.default_rng(2))
    assert len(sub) == 50
    assert all(np.array_equal(ref.recipes[i], r) for i, r in zip(self_idx, sub.recipes, strict=True))
    loo = scorecard_row(sub, "Mixed", ref, self_index=self_idx, with_fingerprint=False, **FAST)
    naive = scorecard_row(sub, "Mixed", ref, with_fingerprint=False, **FAST)
    assert naive["novelty_mean"] == 1.0 and loo["novelty_mean"] < 1.0
    assert "fingerprint_spearman" not in loo


def test_novelty_self_index_skips_only_the_recipe_itself():
    ref = [np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([5, 6])]
    cand = Cuisine("c", [ref[0], ref[2]])
    assert novelty(cand, ref, 10).tolist() == [1.0, 1.0]
    # recipe 0 still has an exact duplicate (recipe 1); recipe 2 has no other match
    assert novelty(cand, ref, 10, self_index=np.array([0, 2])).tolist() == [1.0, 0.0]
