"""Parse and alias D3 steering outputs exactly like the D1 recipes (same parser, same aliaser).

Kaggle outputs of steering modes (bon, rev1, rev2, ...) are downloaded by
`generation/kaggle/push.py output --model M --mode MODE` into data/interim/steering/raw/<model>/.

    recipes = load_outputs("qwen3.5-9b", "rev1")   # one row per parsed recipe, with the prompt's meta
    lines = alias_recipes(recipes)                 # one row per ingredient line, as generation/alias_lines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from generation.alias_lines import curated_entries, make_aliaser  # noqa: E402
from generation.common import extract_json, validate_recipes  # noqa: E402
from generation.parse_outputs import iter_records  # noqa: E402

RAW = REPO / "data" / "interim" / "steering" / "raw"


def load_outputs(model: str, mode: str) -> pd.DataFrame:
    paths = sorted((RAW / model).glob(f"*_{mode}.jsonl")) or sorted((RAW / model).glob(f"*_{mode}.jsonl.gz"))
    rows, seen = [], set()
    for _path, r in iter_records(paths):
        if r["call_id"] in seen:
            continue
        seen.add(r["call_id"])
        obj, status = extract_json(r.get("output_text", ""))
        recs, _issues = validate_recipes(obj) if obj is not None else ([], [status])
        for i, rec in enumerate(recs):
            rows.append(
                {
                    "mode": mode,
                    "model_key": model,
                    "family": r.get("family"),
                    "region_code": r["region_code"],
                    "region_label": r["region_label"],
                    "condition": r["condition"],
                    "rep": r["rep"],
                    "call_id": r["call_id"],
                    "recipe_uid": f"{mode}-{r['call_id']}-{i}",
                    "recipe_idx": i,
                    "title": rec["title"],
                    "ingredients": rec["ingredients"],
                    "instructions": rec["instructions"],
                    "parse_status": status,
                    "n_output_tokens": r.get("n_output_tokens"),
                    "meta": r.get("meta") or {},
                }
            )
    return pd.DataFrame(rows)


def alias_recipes(recipes: pd.DataFrame, cdb=None) -> pd.DataFrame:
    """Aliased ingredient lines (same columns as data/generated/aliased/lines.parquet)."""
    from flavoraudit.data import load_culinarydb

    cdb = cdb or load_culinarydb()
    alias = make_aliaser(cdb, curated_entries())
    ing = cdb.ingredients.set_index("entity_id")
    names, cats = ing["name"].to_dict(), ing["category"].to_dict()
    cache: dict[str, tuple] = {}
    keep = ["mode", "model_key", "family", "region_code", "region_label", "condition", "rep", "call_id"]
    rows = []
    for r in recipes.itertuples(index=False):
        for j, line in enumerate(r.ingredients):
            key = str(line).strip().lower()
            if key not in cache:
                cache[key] = alias(key)
            ent, how = cache[key]
            rows.append({**{k: getattr(r, k) for k in keep}, "recipe_uid": r.recipe_uid, "recipe_idx": r.recipe_idx,
                         "line_idx": j, "line": line, "entity_id": ent, "entity": names.get(ent),
                         "category": cats.get(ent), "how": how})  # fmt: skip
    out = pd.DataFrame(rows)
    if len(out):
        out["entity_id"] = out["entity_id"].astype("Int64")
    return out
