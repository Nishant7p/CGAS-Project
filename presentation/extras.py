"""Data-derived material for deck v2 that is not in results/summary.json, computed from the data at build time.

* CulinaryDB and FlavorDB counts (data/raw), for the data slide.
* A rule-based (not cherry-picked) pair of recipes for the "traditional vs LLM" slide: in the showcase region, the
  real CulinaryDB recipe and the LLM recipe (complete models, plain prompt, pooled) whose flavor sharing Ns is
  closest to the median Ns of its own source; ties go to the lowest id.
* A small flavor-sharing network: the showcase region's top-5 signature ingredients, with the number of flavor
  compounds each pair shares (FlavorDB, via the app's shared-compound matrix S).

Everything degrades to None when an input is missing, so the deck shows a neutral status instead.
"""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CDB = REPO / "data" / "raw" / "culinarydb" / "extracted"
FDB = REPO / "data" / "raw" / "flavordb_v1"
CORE = REPO / "app" / "artifacts" / "core.pkl"
LINES = REPO / "data" / "generated" / "aliased" / "lines.parquet"
RECIPES = REPO / "data" / "generated" / "parsed" / "recipes.parquet"
SHOWCASE = "Korea"  # the lab's showcase contrasting cuisine (Soybean Sauce is its top contributor)


def _ns(rows: np.ndarray, S: np.ndarray) -> float:
    """Ns(R) = 2/(n(n-1)) sum_{i<j} |C_i ∩ C_j| (Ahn et al. 2011; Jain et al. 2015, eq. 2)."""
    n = len(rows)
    if n < 2:
        return float("nan")
    sub = S[np.ix_(rows, rows)].astype(float)
    return float((sub.sum() - np.trace(sub)) / (n * (n - 1)))


def data_counts(labels: list[str]) -> dict | None:
    try:
        r = pd.read_csv(CDB / "01_Recipe_Details.csv")
        a = pd.read_csv(CDB / "04_Recipe-Ingredients_Aliases.csv")
        ing = pd.read_csv(CDB / "02_Ingredients.csv")
        comp = pd.read_csv(CDB / "03_Compound_Ingredients.csv")
    except OSError:
        return None
    out = {
        "recipes": len(r),
        "recipes_22": int(r["Cuisine"].isin(labels).sum()),
        "sources": r["Source"].value_counts().to_dict(),
        "basic_ingredients": len(ing),
        "compound_ingredients": len(comp),
        "rows": len(a),
        "phrases": int(a["Original Ingredient Name"].nunique()),
        "categories": int(ing["Category"].nunique()) if "Category" in ing else None,
    }
    files = sorted(FDB.glob("*.json.gz"))
    mols, links = set(), 0
    for p in files:
        try:
            with gzip.open(p) as fh:
                d = json.loads(fh.read())
        except (OSError, json.JSONDecodeError):
            continue
        m = d.get("molecules") or []
        links += len(m)
        mols.update(x.get("pubchem_id") for x in m)
    out.update({"fdb_entities": len(files), "fdb_molecules": len(mols), "fdb_links": links})
    return out


def _core():
    with open(CORE, "rb") as fh:
        return pickle.load(fh)  # noqa: S301 - artifact built by app/build_artifacts.py in this repo


def recipe_pair(models: list[str], model_names: dict[str, str]) -> dict | None:
    """Rule-based real vs LLM recipe for SHOWCASE (see the module docstring)."""
    try:
        core = _core()
        r = pd.read_csv(CDB / "01_Recipe_Details.csv")
        a = pd.read_csv(CDB / "04_Recipe-Ingredients_Aliases.csv")
        lines = pd.read_parquet(LINES)
        recipes = pd.read_parquet(RECIPES, columns=["recipe_uid", "title", "mode", "model_key"])
    except (OSError, KeyError, ValueError):
        return None
    S, row_of, names, cats = core["S"], core["row_of_entity"], core["names"], core["categories"]
    reg = core["regions"].get(SHOWCASE)
    if reg is None:
        return None
    ns_sorted = np.asarray(reg["ns_sorted"])
    signature = set(reg["signature"])

    def describe(rows: np.ndarray, raw: list[str]) -> dict:
        ns = _ns(rows, S)
        lo, hi = np.searchsorted(ns_sorted, ns, "left"), np.searchsorted(ns_sorted, ns, "right")
        cat_counts = pd.Series([cats[i] for i in rows]).value_counts().to_dict()
        return {"lines": raw, "entities": [names[i] for i in rows], "n": len(rows), "ns": ns,
                "ns_pct": 100.0 * (lo + hi) / (2 * len(ns_sorted)),
                "signature": [names[i] for i in rows if i in signature], "categories": cat_counts}  # fmt: skip

    # real recipes of the showcase region
    ids = set(r.loc[r["Cuisine"] == SHOWCASE, "Recipe ID"])
    sub = a[a["Recipe ID"].isin(ids)]
    real = []
    for rid, g in sub.groupby("Recipe ID", sort=True):
        rows = np.unique([row_of[e] for e in g["Entity ID"] if e in row_of]).astype(np.int64)
        if len(rows) >= 2:
            real.append((rid, _ns(rows, S), rows, g["Original Ingredient Name"].astype(str).tolist()))
    if not real:
        return None
    med_real = float(np.median([t[1] for t in real]))
    rid, _, rrows, rraw = min(real, key=lambda t: (abs(t[1] - med_real), t[0]))
    rmeta = r.set_index("Recipe ID").loc[rid]

    # LLM recipes: complete models, full mode, plain prompt, first 5 recipes per call
    sel = lines[(lines["mode"] == "full") & (lines["region_label"] == SHOWCASE) & (lines["condition"] == "plain")
                & (lines["model_key"].isin(models)) & (lines["recipe_idx"] < 5)]  # fmt: skip
    llm = []
    for uid, g in sel.groupby("recipe_uid", sort=True):
        ents = [int(e) for e in g["entity_id"].dropna()]
        rows = np.unique([row_of[e] for e in ents if e in row_of]).astype(np.int64)
        if len(rows) >= 2:
            llm.append((uid, _ns(rows, S), rows, g.sort_values("line_idx")["line"].astype(str).tolist(),
                        g["model_key"].iloc[0]))  # fmt: skip
    if not llm:
        return None
    med_llm = float(np.median([t[1] for t in llm]))
    uid, _, lrows, lraw, lmodel = min(llm, key=lambda t: (abs(t[1] - med_llm), t[0]))
    title = recipes.loc[recipes["recipe_uid"] == uid, "title"]
    return {
        "region": SHOWCASE,
        "real": {"title": str(rmeta["Title"]), "source": str(rmeta["Source"]).title(), "id": int(rid),
                 "median_ns": med_real, "pool": len(real), **describe(rrows, rraw)},
        "llm": {"title": str(title.iloc[0]) if len(title) else uid, "model": model_names.get(lmodel, lmodel),
                "uid": uid, "median_ns": med_llm, "pool": len(llm), **describe(lrows, lraw)},
    }  # fmt: skip


def signature_network() -> dict | None:
    """The showcase region's top-5 signature ingredients and their pairwise shared compounds."""
    try:
        core = _core()
    except OSError:
        return None
    reg = core["regions"].get(SHOWCASE)
    if reg is None:
        return None
    rows = list(reg["signature"])[:5]
    S, names = core["S"], core["names"]
    edges = [(names[a], names[b], int(S[a, b])) for i, a in enumerate(rows) for b in rows[i + 1 :]]
    return {"region": SHOWCASE, "nodes": [names[i] for i in rows],
            "sizes": {names[i]: int(S[i, i]) for i in rows}, "edges": edges,
            "ns": _ns(np.asarray(rows, dtype=np.int64), S), "z_real": float(reg["z_real"])}  # fmt: skip
