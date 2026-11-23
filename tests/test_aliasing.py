"""The CulinaryDB aliasing protocol (flavoraudit.aliasing) on a toy vocabulary: every rule in the module docstring."""

import types

import numpy as np
import pandas as pd
import pytest

from flavoraudit.aliasing import (
    CULINARY_STOPWORDS,
    ENGLISH_STOPWORDS,
    Aliaser,
    Entry,
    decode,
    singular,
    tokens,
)
from flavoraudit.contribution import chi_percent, random_null_z
from flavoraudit.pairing import Cuisine

VOCAB = {  # entity id: CulinaryDB-style synonyms
    362: ["capsicum", "pepper-bell"],
    364: ["tomato"],
    500: ["soy-sauce", "soybean sauce"],
    600: ["corn=starch"],
    700: ["garlic"],
    701: ["garlic powder"],
    800: ["parmesan-cheese"],
}


@pytest.fixture(scope="module")
def al() -> Aliaser:
    entries = [Entry(e, decode(s), s) for e, syns in VOCAB.items() for s in syns]
    return Aliaser(entries)


def test_tokens_normalize_singularize_and_drop_stopwords():
    assert singular("tomatoes") == "tomato" and singular("asparagus") == "asparagu"  # CulinaryDB did the same
    assert tokens("2 Cups of Chopped Onions!") == ["cup", "chopped", "onion"]
    assert tokens("2 Cups of Chopped Onions!", ENGLISH_STOPWORDS | CULINARY_STOPWORDS) == ["onion"]


def test_decode_culinarydb_synonym_notation():
    assert decode("pepper-bell") == (("pepper",), ("bell",))  # '-': parts in any order
    assert decode("corn=starch") == (("corn", "starch"),)  # '=': words written apart or joined
    assert decode("#garlic#") == (("garlic",),)  # '#' marks a disabled entry (skipped by from_culinarydb)
    assert decode("black-eyed peas", split_parts=False) == (("black", "eyed", "pea"),)


@pytest.mark.parametrize(
    "line,entity,how",
    [
        ("1 green bell pepper, diced", 362, "full"),  # any-order parts: "bell pepper" matches 'pepper-bell'
        ("2 tbsp cornstarch", 600, "full"),  # joined spelling of 'corn=starch'
        ("2 tbsp corn starch", 600, "full"),
        ("1 tsp garlic powder", 701, "full"),  # longest entry first: not plain Garlic
        ("3 cloves garlic, minced", 700, "full"),
        ("1/4 cup teriyaki sauce", 500, "partial"),  # partial match of 'soy-sauce'
        ("grated parmesan", 800, "partial"),
        ("unicorn tears", None, "none"),
    ],
)
def test_alias_rules(al, line, entity, how):
    m = al.alias(line)
    assert (m.entity_id, m.how) == (entity, how)


def test_ties_use_learned_preferences_then_the_rightmost_match(al):
    m = al.alias("tomato and garlic")
    assert set(m.candidates) == {364, 700} and m.entity_id == 700  # rightmost
    learner = Aliaser(list(al.entries))
    assert learner.fit_preferences(["tomato garlic", "garlic tomato"], [364, 364]) == 1
    assert learner.alias("tomato and garlic").entity_id == 364  # the expert preferred tomato


def test_overrides_and_mined_synonyms(al):
    ov = Aliaser(list(al.entries), overrides={"mystery spice": 700})  # keyed by the whole normalized line
    m = ov.alias("Mystery Spices")
    assert (m.entity_id, m.how) == (700, "override")
    assert ov.alias("2 tsp mystery spices").how == "none"  # a different normalized line ('tsp mystery spice')
    lines = [f"{k} tbsp gochugaru flakes" for k in range(6)]
    mined = al.mine_synonyms(lines, [900] * 6, min_support=5)
    assert any(e.parts == (("gochugaru",),) and e.entity_id == 900 for e in mined)
    assert al.with_entries(mined).alias("1 tbsp gochugaru").entity_id == 900
    assert al.alias("1 tbsp gochugaru").entity_id is None  # the original aliaser is unchanged


def test_from_culinarydb_skips_disabled_and_unused_entities():
    cdb = types.SimpleNamespace(
        ingredients=pd.DataFrame(
            {
                "entity_id": [1, 2, 3, 4],
                "name": ["Garlic", "Garlic Clove", "Basil", "Dodo Egg"],
                "synonyms": [["garlic"], ["#garlic#"], ["basil", "sweet-basil"], ["dodo egg"]],
            }
        ),
        aliases=pd.DataFrame(
            {"entity_id": [1, 1, 2, 3], "original": ["garlic", "2 garlic cloves", "garlic", "fresh basil leaves"]}
        ),
    )
    al = Aliaser.from_culinarydb(cdb)
    assert {e.entity_id for e in al.entries} == {1, 3}  # 2 is disabled ('#'), 4 is never used by CulinaryDB
    assert al.alias("2 sweet basil leaves").entity_id == 3
    ref = Aliaser.reference(cdb)  # also learns tie-breaks from the expert lines
    assert ref.alias("garlic").entity_id == 1


def test_chi_percent_bases_and_procedures():
    rng = np.random.default_rng(0)
    S = rng.integers(0, 10, size=(12, 12))
    S = (np.triu(S, 1) + np.triu(S, 1).T + np.diag(rng.integers(10, 20, 12))).astype(np.int32)
    cats = {i: ("Additive" if i == 11 else "Spice") for i in range(12)}
    cu = Cuisine("toy", [np.sort(rng.choice(12, 5, replace=False)) for _ in range(120)], cats)
    z = random_null_z(cu, S, n_sd=2000)
    ns_basis = chi_percent(cu, S, top_k=5, n_sd=2000)
    assert len(ns_basis) == 5 and 11 not in ns_basis["row"].tolist()  # additives excluded by default
    assert np.allclose(ns_basis["z"], z)
    z_basis = chi_percent(cu, S, rows=[0, 1], basis="z", procedure="recipes", n_sd=2000)
    assert set(z_basis["row"]) == {0, 1} and np.isfinite(z_basis["chi_percent"]).all()
    with pytest.raises(ValueError):
        chi_percent(cu, S, rows=[0], basis="nope", n_sd=500)
