"""D3 steering design and best-of-N selection (steering/design.py, steering/run_bon.py)."""

import json

import numpy as np
import pandas as pd

from steering.design import (
    N_SELECTED,
    POOL_TARGET,
    STEER_REGIONS,
    bon_pool_prompts,
    default_extra_calls,
    feedback_text,
    revision_prompt,
)
from steering.run_bon import select


def test_pool_design_reaches_the_preregistered_size():
    assert POOL_TARGET == N_SELECTED * 16 == 1600
    extra = default_extra_calls()
    assert extra * 5 + 80 >= POOL_TARGET
    prompts = bon_pool_prompts(3)
    assert len(prompts) == 3 * len(STEER_REGIONS)
    assert {p["rep"] for p in prompts} == {16, 17, 18} and {p["condition"] for p in prompts} == {"plain"}
    assert all(p["meta"] == {"purpose": "bon-pool"} for p in prompts)
    assert {"BRI", "DACH", "EE", "JPN", "KOR", "SCND"} <= set(STEER_REGIONS)  # all 6 contrasting regions


def _pool(n):
    return pd.DataFrame({"uid": [f"r{i}" for i in range(n)], "rows": [np.array([i, i + 1000]) for i in range(n)]})


def test_best_of_n_picks_the_top_admissible_recipe_of_each_group():
    pool = _pool(40)
    scores = np.arange(40, dtype=float)
    adm = np.ones(40, dtype=bool)
    chosen, examined = select(pool, scores, adm, n=4, k=5, seed=0)
    order = np.random.default_rng(0).permutation(40)
    expected = [int(max(order[g : g + 4], key=lambda t: scores[t])) for g in range(0, 20, 4)]
    assert chosen == expected and examined == 20
    one, _ = select(pool, scores, adm, n=1, k=5, seed=0)
    assert one == [int(i) for i in order[:5]]  # N = 1: no selection, the shuffled order


def test_best_of_n_guards_skip_near_copies_and_duplicates():
    pool = _pool(8)
    pool.loc[3, "rows"] = pool.loc[2, "rows"]  # recipe 3 duplicates recipe 2's ingredient set
    scores = np.array([0, 1, 9, 8, 0, 0, 0, 7], dtype=float)
    adm = np.ones(8, dtype=bool)
    adm[7] = False  # a near-copy of a CulinaryDB recipe
    chosen, _ = select(pool, scores, adm, n=8, k=1, seed=0)
    assert chosen == [2]
    chosen_all, _ = select(pool, scores, adm, n=1, k=8, seed=0)
    assert 7 not in chosen_all and not ({2, 3} <= set(chosen_all))


def test_revision_prompt_carries_feedback_and_the_one_recipe_format():
    items = {"atypical": ["Cheddar Cheese"], "missing": ["Soybean Sauce", "Sesame"], "ns_verdict": "high"}
    text = feedback_text(items, "Korea")
    assert "Cheddar Cheese" in text and "Soybean Sauce, Sesame" in text and "more flavor compounds" in text
    recipe = {"title": "Bibimbap", "ingredients": ["2 cups rice", "1 cup cheddar"], "instructions": ["Cook."]}
    p = revision_prompt(recipe, items, "KOR", 1, "fabc-0")
    assert p["key"] == "rev1|fabc-0" and p["region_label"] == "Korea" and p["meta"]["parent_uid"] == "fabc-0"
    assert "exactly 1 recipe" in p["user"] and '"Bibimbap"' in p["user"]
    assert json.loads(p["user"][p["user"].index("{") : p["user"].index("}") + 1])["title"] == "Bibimbap"
    quiet = feedback_text({"atypical": [], "missing": [], "ns_verdict": "typical"}, "Italy")
    assert "already close" in quiet
