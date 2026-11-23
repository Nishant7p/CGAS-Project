"""The high-level API (flavoraudit.World, load_world) and the command line (flavoraudit.cli)."""

import numpy as np
import pandas as pd
import pytest

import flavoraudit as fa
from flavoraudit import cli
from flavoraudit.pairing import Cuisine


def toy_world() -> fa.World:
    """Two cuisines over 30 ingredients; entity id = S row; 'ing<k>' lines alias to entity k."""
    rng = np.random.default_rng(0)
    S = rng.integers(0, 10, size=(30, 30))
    S = (np.triu(S, 1) + np.triu(S, 1).T + np.diag(rng.integers(10, 20, 30))).astype(np.int32)
    cats = {i: f"c{i % 3}" for i in range(30)}
    cuisines = {
        name: Cuisine(name, [np.sort(rng.choice(np.arange(lo, lo + 15), 5, replace=False)) for _ in range(80)], cats)
        for name, lo in (("Alpha", 0), ("Beta", 15))
    }
    world = fa.World(cdb=None, profiles=None, S=S, index={i: i for i in range(30)}, cuisines=cuisines)
    world.alias = lambda line: int(line[3:]) if line.startswith("ing") and line[3:].isdigit() else None
    return world


def test_world_builds_cuisines_from_raw_lines():
    w = toy_world()
    assert w.regions == ["Alpha", "Beta"]
    assert w.recipe_from_lines(["ing3", "ing1", "ing3", "salt??"]).tolist() == [1, 3]
    c = w.cuisine_from_lines("mine", [["ing1", "ing2"], ["nothing"], ["ing4"]])
    assert len(c) == 2 and c.categories[4] == "c1"  # the recipe with no matched line is dropped
    with pytest.raises(KeyError):
        w.cuisine("Gamma")


def test_cli_audit_end_to_end(tmp_path, monkeypatch):
    w = toy_world()
    monkeypatch.setattr(cli, "load_world", lambda data_dir=None: w)
    rng = np.random.default_rng(1)
    rows = [
        {"recipe_id": f"r{k}", "cuisine": "Alpha", "line": f"ing{i}"}
        for k in range(30)
        for i in rng.choice(15, 4, replace=False)
    ] + [{"recipe_id": "b0", "cuisine": "Beta", "line": "ing20"}]  # only 1 Beta recipe: skipped (< 20)
    path = tmp_path / "recipes.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    assert cli.main(["audit", str(path), "--out", str(tmp_path / "out"), "--n-boot", "200"]) == 0
    sc = pd.read_csv(tmp_path / "out" / "scorecard.csv")
    assert sc["region"].tolist() == ["Alpha"] and sc["n_recipes"].iloc[0] == 30
    assert {"z_sb", "sign_error", "prevalence_jsd", "sai", "near_copy_rate"} <= set(sc.columns)
    lines = pd.read_csv(tmp_path / "out" / "aliased_lines.csv")
    assert lines["entity_id"].notna().all()


def test_cli_rejects_bad_input(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_world", lambda data_dir=None: toy_world())
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"recipe_id": ["a"], "line": ["x"]}).to_csv(bad, index=False)
    assert cli.main(["audit", str(bad)]) == 2
    unknown = tmp_path / "unknown.csv"
    pd.DataFrame({"recipe_id": ["a"], "cuisine": ["Atlantis"], "line": ["x"]}).to_csv(unknown, index=False)
    assert cli.main(["audit", str(unknown)]) == 2


@pytest.mark.slow
def test_load_world_reproduces_korean_contrasting_pairing():
    w = fa.load_world()
    assert len(w.regions) == 22
    res = fa.food_pairing(w.cuisine("Korea"), w.S, n_null=20_000)
    assert res.z_sb < 0  # Singh & Bagler 2018: Korea is one of the 6 contrasting cuisines
    r = w.recipe_from_lines(["2 tbsp soy sauce", "1 tbsp sesame oil", "3 cloves garlic, minced"])
    assert len(r) == 3


def test_data_dir_accepted_before_and_after_subcommand(monkeypatch):
    from flavoraudit import cli

    seen = []
    monkeypatch.setattr(cli, "cmd_audit", lambda a: seen.append(a.data_dir) or 0)
    cli.main(["--data-dir", "/A", "audit", "r.csv"])
    cli.main(["audit", "r.csv", "--data-dir", "/B"])
    cli.main(["audit", "r.csv"])
    assert seen == ["/A", "/B", None]
