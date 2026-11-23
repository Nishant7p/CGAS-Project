"""Command line: audit a recipe collection against the real cuisines it claims to be.

    flavoraudit audit recipes.csv --out audit/
    flavoraudit pairing --cuisine Korea

recipes.csv is long format, one row per ingredient line, with columns recipe_id, cuisine (a CulinaryDB region
name, e.g. "Korea") and line (the raw ingredient line). Every claimed cuisine with at least 20 recipes gets a
scorecard row (flavoraudit.audit.scorecard_row) in <out>/scorecard.csv; every line's aliasing goes to
<out>/aliased_lines.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .stats import food_pairing
from .world import load_world

MIN_RECIPES = 20


def cmd_audit(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.recipes)
    missing = {"recipe_id", "cuisine", "line"} - set(df.columns)
    if missing:
        print(f"{args.recipes}: missing columns {sorted(missing)}", file=sys.stderr)
        return 2
    world = load_world(args.data_dir)
    unknown = sorted(set(df["cuisine"]) - set(world.regions))
    if unknown:
        print(f"unknown cuisines {unknown}; choose from {world.regions}", file=sys.stderr)
        return 2
    df["entity_id"] = pd.array([world.alias(str(x)) for x in df["line"]], dtype="Int64")  # ints, NA if unmatched
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "aliased_lines.csv", index=False)
    rows = []
    for region, g in df.groupby("cuisine"):
        recipes = [list(gg["line"].astype(str)) for _, gg in g.groupby("recipe_id")]
        if len(recipes) < MIN_RECIPES:
            print(f"skipping {region}: {len(recipes)} recipes (< {MIN_RECIPES})", file=sys.stderr)
            continue
        cuisine = world.cuisine_from_lines(f"input:{region}", recipes)
        rows.append(world.scorecard(cuisine, region, n_boot=args.n_boot))
        print(f"{region}: {len(cuisine)} recipes scored", file=sys.stderr)
    pd.DataFrame(rows).to_csv(out / "scorecard.csv", index=False)
    print(f"matched {df['entity_id'].notna().mean():.1%} of lines -> {out / 'scorecard.csv'}")
    return 0


def cmd_pairing(args: argparse.Namespace) -> int:
    world = load_world(args.data_dir)
    regions = [args.cuisine] if args.cuisine else world.regions
    for r in regions:
        res = food_pairing(world.cuisine(r), world.S, args.null, n_null=args.n_null, seed=args.seed)
        print(f"{r:22s} Ns={res.ns_cuisine:7.3f} null={res.ns_null:7.3f} Z={res.z_sb:8.1f} d={res.d:7.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="flavoraudit", description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", help="folder with the CulinaryDB, FlavorDB and Ahn data (default: data/raw)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="scorecard of a recipe collection")
    a.add_argument("recipes")
    a.add_argument("--out", default="flavoraudit_out")
    a.add_argument("--n-boot", type=int, default=1000)
    a.set_defaults(func=cmd_audit)
    p = sub.add_parser("pairing", help="food-pairing Z of real CulinaryDB cuisines (Singh & Bagler 2018)")
    p.add_argument("--cuisine")
    p.add_argument("--null", default="random", choices=["random", "frequency", "category", "frequency_category"])
    p.add_argument("--n-null", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_pairing)
    for sp in (a, p):  # also accept --data-dir after the subcommand; SUPPRESS keeps the top-level value otherwise
        sp.add_argument("--data-dir", default=argparse.SUPPRESS, help="same as the top-level --data-dir")
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
