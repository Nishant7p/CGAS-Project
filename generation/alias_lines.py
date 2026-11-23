"""Alias every ingredient line of the parsed LLM recipes to CulinaryDB entities (D1, Phase 5).

It uses the production aliaser, flavoraudit.aliasing.Aliaser.reference: CulinaryDB's vocabulary plus tie-break
preferences learned from its expert-aliased lines (held-out agreement 97.9% of unique lines, 99.0% of line
occurrences). Reviewed curation decisions in curation/alias_overrides.csv are added on top: every row with
decision == "accept" becomes an extra synonym (term -> entity_id), applied to LLM text only, like the manual n-gram
curation step of the CulinaryDB protocol. A line whose result changes because of a curated synonym is reported
with how = "curated". Unmatched lines keep entity_id = NA and are reported, not guessed.

Input:  data/generated/parsed/recipes.parquet (generation/parse_outputs.py)
Output: data/generated/aliased/lines.parquet  one row per ingredient line
        data/generated/aliased/coverage.csv   per mode x model x region: lines, share matched by how, recipes
Usage (repo root): .venv/bin/python generation/alias_lines.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from flavoraudit.aliasing import Aliaser, Entry, decode  # noqa: E402
from flavoraudit.data import load_culinarydb  # noqa: E402

PARSED = REPO / "data" / "generated" / "parsed" / "recipes.parquet"
OVERRIDES = REPO / "curation" / "alias_overrides.csv"
OUT = REPO / "data" / "generated" / "aliased"
KEEP = ["mode", "model_key", "family", "region_code", "region_label", "condition", "rep", "call_id"]


def curated_entries(path: Path = OVERRIDES) -> list[Entry]:
    """Accepted curation decisions as extra synonyms (term -> entity)."""
    if not path.exists():
        return []
    df = pd.read_csv(path, dtype={"entity_id": "Int64"})
    ok = df[(df["decision"].str.lower() == "accept") & df["entity_id"].notna()]
    return [
        Entry(int(e), decode(str(t), split_parts=False), f"curated:{t}")
        for t, e in zip(ok["term"], ok["entity_id"], strict=True)
    ]


def make_aliaser(cdb, extra: list[Entry]):
    """alias(line) -> (entity_id, how); how is full | partial | none | curated."""
    base = Aliaser.reference(cdb)
    cur = base.with_entries(extra) if extra else None

    def alias(line: str) -> tuple[int | None, str]:
        m = base.alias(line)
        if cur is None:
            return m.entity_id, m.how
        c = cur.alias(line)
        return (c.entity_id, "curated") if (c.entity_id, c.how) != (m.entity_id, m.how) else (m.entity_id, m.how)

    return alias


def coverage(lines: pd.DataFrame) -> pd.DataFrame:
    keys = ["mode", "model_key", "region_code"]
    cov = (
        lines.assign(n=1)
        .pivot_table(index=keys, columns="how", values="n", aggfunc="sum", fill_value=0)
        .rename_axis(columns=None)
    )
    cov.insert(0, "lines", cov.sum(axis=1))
    for c in [c for c in cov.columns if c != "lines"]:
        cov[f"share_{c}"] = (cov[c] / cov["lines"]).round(4)
    cov["recipes"] = lines.groupby(keys)["recipe_uid"].nunique()
    return cov


def main() -> int:
    t0 = time.time()
    recipes = pd.read_parquet(PARSED)
    cdb = load_culinarydb()
    extra = curated_entries()
    alias = make_aliaser(cdb, extra)
    ing = cdb.ingredients.set_index("entity_id")
    names, cats = ing["name"].to_dict(), ing["category"].to_dict()
    cache: dict[str, tuple[int | None, str]] = {}
    rows = []
    for r in recipes.itertuples(index=False):
        base = {k: getattr(r, k) for k in KEEP}
        for j, line in enumerate(r.ingredients):
            key = str(line).strip().lower()
            if key not in cache:
                cache[key] = alias(key)
            ent, how = cache[key]
            rows.append(
                {
                    **base,
                    "recipe_uid": r.recipe_uid,
                    "recipe_idx": r.recipe_idx,
                    "line_idx": j,
                    "line": line,
                    "entity_id": ent,
                    "entity": names.get(ent),
                    "category": cats.get(ent),
                    "how": how,
                }
            )
    lines = pd.DataFrame(rows)
    lines["entity_id"] = lines["entity_id"].astype("Int64")
    OUT.mkdir(parents=True, exist_ok=True)
    lines.to_parquet(OUT / "lines.parquet", index=False)
    coverage(lines).to_csv(OUT / "coverage.csv")
    by_model = lines.groupby(["mode", "model_key"])["how"].value_counts(normalize=True).unstack(fill_value=0)
    print(by_model.round(4).to_string())
    print(
        f"{len(lines)} lines ({len(cache)} distinct), {lines['recipe_uid'].nunique()} recipes, "
        f"{len(extra)} curated synonyms | {time.time() - t0:.0f}s -> {OUT.relative_to(REPO)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
