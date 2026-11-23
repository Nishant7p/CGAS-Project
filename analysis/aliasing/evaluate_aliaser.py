"""Evaluate flavoraudit.aliasing against CulinaryDB's expert aliasing (Phase 3).

Gold standard: 04_Recipe-Ingredients_Aliases.csv maps each raw line to exactly one entity. Unique lowercase lines
are split by a stable hash into TRAIN (80%) and TEST (20%); TRAIN is further split into FIT (80%) and VAL (20%).
Variants are compared on VAL using things learned on FIT (preferences, mined synonyms, the vocabulary of used
entities). The chosen variant is refit on all of TRAIN and reported once on TEST.

Outputs: data/interim/aliasing/{metrics.json, errors_test.csv}.
Usage (repo root): .venv/bin/python analysis/aliasing/evaluate_aliaser.py
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from flavoraudit.aliasing import CULINARY_STOPWORDS, ENGLISH_STOPWORDS, Aliaser
from flavoraudit.data import load_culinarydb

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "interim" / "aliasing"


def bucket(phrase: str) -> int:
    return int(hashlib.sha256(phrase.encode()).hexdigest()[:8], 16) % 25


def gold_table(cdb) -> pd.DataFrame:
    a = cdb.aliases.assign(phrase=cdb.aliases["original"].astype(str).str.strip().str.lower())
    a = a.merge(cdb.recipes[["recipe_id", "source"]], on="recipe_id", how="left")
    g = a.groupby("phrase").agg(
        entity_id=("entity_id", lambda s: Counter(s).most_common(1)[0][0]),
        n=("entity_id", "size"),
        source=("source", lambda s: Counter(s).most_common(1)[0][0]),
    )
    g = g.reset_index()
    b = g["phrase"].map(bucket)
    g["split"] = ["test" if x < 5 else ("val" if x < 9 else "fit") for x in b]  # 20% test, 16% val, 64% fit
    return g


def score(al: Aliaser, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    res = [al.alias(p) for p in df["phrase"]]
    df = df.assign(pred=[r.entity_id for r in res], how=[r.how for r in res])
    ok = df["pred"] == df["entity_id"]
    w = df["n"]
    m = {
        "lines": len(df),
        "accuracy_lines": round(float(ok.mean()), 4),
        "accuracy_rows_weighted": round(float((ok * w).sum() / w.sum()), 4),
        "coverage_lines": round(float(df["pred"].notna().mean()), 4),
        "share_partial": round(float((df["how"] == "partial").mean()), 4),
    }
    return m, df


def build(cdb, fit: pd.DataFrame, variant: str) -> Aliaser:
    used = set(int(e) for e in fit["entity_id"])
    if variant == "A_literal":
        return Aliaser.from_culinarydb(
            cdb, allowed_entities=set(cdb.ingredients["entity_id"].astype(int)), partial=False
        )
    al = Aliaser.from_culinarydb(cdb, allowed_entities=used, partial=variant != "B_vocab")
    if variant in ("B_vocab", "C_partial"):
        return al
    if variant == "D_prefs_culinary_sw":
        al = Aliaser(al.entries, stopwords=ENGLISH_STOPWORDS | CULINARY_STOPWORDS, partial=True)
    if variant in ("E_mined", "F_mined_strict"):
        strict = variant == "F_mined_strict"
        extra = al.mine_synonyms(
            fit["phrase"], fit["entity_id"], min_support=10 if strict else 5, min_precision=0.95 if strict else 0.9
        )
        al = al.with_entries(extra)
    al.fit_preferences(fit["phrase"], fit["entity_id"])
    return al


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cdb = load_culinarydb()
    g = gold_table(cdb)
    fit, val, test = (g[g["split"] == s] for s in ("fit", "val", "test"))
    train = g[g["split"] != "test"]
    print(f"unique lines: {len(g)} | fit {len(fit)}, val {len(val)}, test {len(test)}")
    variants = ["A_literal", "B_vocab", "C_partial", "D_prefs", "D_prefs_culinary_sw", "E_mined", "F_mined_strict"]
    val_results = {}
    for v in variants:
        t = time.time()
        m, _ = score(build(cdb, fit, v), val)
        val_results[v] = m
        print(f"VAL  {v:22s} {m} ({time.time() - t:.0f}s)", flush=True)
    best = max(val_results, key=lambda k: val_results[k]["accuracy_rows_weighted"])
    al = build(cdb, train, best)
    m_test, dft = score(al, test)
    print(f"\nTEST {best} (refit on all train): {m_test}")
    by_src = {s: round(float((d["pred"] == d["entity_id"]).mean()), 4) for s, d in dft.groupby("source")}
    print("TEST accuracy by source:", by_src)
    names = dict(zip(cdb.ingredients["entity_id"], cdb.ingredients["name"], strict=True))
    err = dft[dft["pred"] != dft["entity_id"]].copy()
    err["gold_name"], err["pred_name"] = err["entity_id"].map(names), err["pred"].map(names)
    err.sort_values("n", ascending=False).to_csv(OUT / "errors_test.csv", index=False)
    print("\ntop remaining confusions (gold -> predicted):")
    for (a, b), c in Counter(zip(err["gold_name"], err["pred_name"].fillna("—"), strict=True)).most_common(12):
        print(f"  {a} -> {b}: {c}")
    metrics = {
        "val": val_results,
        "chosen": best,
        "test": m_test,
        "test_by_source": by_src,
        "n_preferences": len(al.preferences),
        "n_mined_entries": sum(e.label.startswith("mined:") for e in al.entries),
    }
    with open(OUT / "metrics.json", "w") as fh:
        json.dump(
            metrics,
            fh,
            indent=1,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
