"""Equal-N food pairing: own-pool random-null ΔNs and d of genuine subsamples of every real cuisine, by sample size.

Why this exists (found 2026-11-22 with analysis/audit/real_subsamples.py, before the full LLM analysis): the random
null of Singh & Bagler (2018) draws from the cuisine's OWN ingredient pool. A small sample of a cuisine has a smaller
pool, without the long tail of rare ingredients, and the remaining popular ingredients share more compounds. The
null's expected Ns therefore rises with smaller N while the real mean Ns stays the same, so ΔNs and Z shrink or flip:
USA ΔNs = 12.7 at full size but about 0 at 160 recipes (pool 595 -> 238 ingredients); against the full cuisine's pool
the same subsamples give 12.8. LLM cuisines (160 recipes per region) must be compared with genuine samples of the same
size, and cuisines of different sizes (Korea 301 vs USA 16,106 recipes) are not comparable on raw Z.

For each region and N, ``reps`` random subsamples of N pairing recipes give: the mean Ns, the exact null mean
(flavoraudit.contribution.pair_mean over the subsample's pool), the null SD (simulated), ΔNs, d, and also ΔNs against
the FULL cuisine's pool (a fixed-pool null that removes the pool effect).

Usage (repo root):
    .venv/bin/python analysis/audit/equal_n_pairing.py                       # N grid 60..160 for the LLM comparison
    .venv/bin/python analysis/audit/equal_n_pairing.py --n 290 --tag matched290 --reps 200   # Fig. 4 at equal size
Output: data/interim/audit/real/equal_n_pairing[_<tag>].parquet
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_io import REAL, REPO, load_reference, worker_count  # noqa: E402

from flavoraudit.contribution import pair_mean  # noqa: E402
from flavoraudit.nulls import size_controlled  # noqa: E402
from flavoraudit.pairing import Cuisine, ns_from_groups, ns_values  # noqa: E402

N_GRID = (60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160)
_REF = None


def _init() -> None:
    global _REF
    _REF = load_reference()[0]


def _run(task: tuple) -> list[dict]:
    region, ns_grid, reps, n_sd, seed = task
    S = _REF.S
    real = _REF.real[region].pairing_recipes()
    full_pool = real.pool()[0]
    e_full = pair_mean(full_pool, S)
    rng = np.random.default_rng([seed, zlib.crc32(region.encode())])
    rows = []
    for n in ns_grid:
        if n > len(real):
            continue
        for k in range(reps):
            pick = rng.choice(len(real), size=n, replace=False)
            sub = Cuisine(region, [real.recipes[i] for i in pick], real.categories)
            pool = sub.pool()[0]
            ns = float(ns_values(sub, S).mean())
            e_own = pair_mean(pool, S)
            sd = float(ns_from_groups(size_controlled(sub, n_sd, rng), S).std(ddof=1))
            rows.append(
                {
                    "region": region,
                    "N": n,
                    "rep": k,
                    "pool": len(pool),
                    "ns": ns,
                    "ns_null": e_own,
                    "sd_null": sd,
                    "delta_ns": ns - e_own,
                    "d": (ns - e_own) / sd,
                    "delta_ns_fullpool": ns - e_full,
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="*", default=list(N_GRID))
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--n-sd", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=worker_count())
    args = ap.parse_args()
    t0 = time.time()
    ref = load_reference()[0]
    regions = sorted(ref.real, key=lambda r: -len(ref.real[r]))
    tasks = [(r, tuple(args.n), args.reps, args.n_sd, args.seed) for r in regions]
    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init) as ex:
        for region, out in zip(regions, ex.map(_run, tasks), strict=True):
            rows.extend(out)
            print(f"{region:22s} {len(out)} subsamples | {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    path = REAL / f"equal_n_pairing{'_' + args.tag if args.tag else ''}.parquet"
    df.to_parquet(path, index=False)
    z = pd.read_csv(REPO / "data" / "interim" / "replication" / "results_v1_keep_empty.csv")
    zfull = z[z["null_model"] == "random"].set_index("region")["d"]
    summ = df.groupby(["region", "N"]).agg(
        d_median=("d", "median"), positive_share=("d", lambda s: (s > 0).mean()), pool=("pool", "mean")
    )
    summ = summ.reset_index().assign(d_full=lambda t: t["region"].map(zfull))
    print(summ[summ["N"] == max(args.n)].round(3).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s -> {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
