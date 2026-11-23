"""Collect one steering run: parse and alias its downloaded Kaggle outputs (steering/steer_io.py).

Usage (repo root), after `generation/kaggle/push.py output --model M --mode MODE`:
    .venv/bin/python steering/collect.py --model qwen3.5-9b --mode bon
Output: data/interim/steering/{recipes,lines}_<mode>_<model>.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from steering.steer_io import alias_recipes, load_outputs  # noqa: E402

OUT = REPO / "data" / "interim" / "steering"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True)
    args = ap.parse_args()
    recipes = load_outputs(args.model, args.mode)
    if recipes.empty:
        print(f"no outputs for {args.model} / {args.mode} under data/interim/steering/raw/{args.model}/")
        return 1
    lines = alias_recipes(recipes)
    recipes.assign(meta=recipes["meta"].map(lambda m: json.dumps(m, ensure_ascii=False))).to_parquet(
        OUT / f"recipes_{args.mode}_{args.model}.parquet", index=False
    )  # meta as a JSON string (parquet cannot store free-form dicts)
    lines.to_parquet(OUT / f"lines_{args.mode}_{args.model}.parquet", index=False)
    calls = recipes["call_id"].nunique()
    print(
        f"{args.model} {args.mode}: {calls} calls, {len(recipes)} recipes, {len(lines)} lines, "
        f"{lines['entity_id'].notna().mean():.1%} aliased"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
