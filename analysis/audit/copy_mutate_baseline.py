"""Phase 7: the copy-mutate baseline cuisines, scored with the same scorecard as the LLM cuisines.

For each region and model (primary: CM-R with ranked fitness, Jain & Bagler 2018's "CM-fitness ranked"; secondary:
CM-R with random fitness, Kinouchi 2008 / Tuwani 2019), 10 instances are evolved with Tuwani et al.'s (2019)
Algorithm 1 (flavoraudit.evolution). The nature basket is the region's real ingredients, ranked fitness is their real
frequency, and the seed recipes' sizes follow the region's size distribution. Each instance is scored at full size
(the region's recipe count; novelty estimated on 2,000 recipes) and as a random equal-N subsample (n = 160, the
LLMs' per-region sample size).

Usage (repo root):
    .venv/bin/python analysis/audit/copy_mutate_baseline.py
    .venv/bin/python analysis/audit/copy_mutate_baseline.py --instances 2 --regions Korea Japan
Output: data/interim/audit/copy_mutate/{scorecard,runs}.parquet
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
from audit_io import AUDIT, LLM_N_PER_REGION, REPO, load_reference, worker_count  # noqa: E402

from flavoraudit.audit import scorecard_row  # noqa: E402
from flavoraudit.evolution import copy_mutate_with_stats  # noqa: E402
from flavoraudit.pairing import Cuisine  # noqa: E402

MODELS = (("CM-R", "ranked"), ("CM-R", "random"))  # primary, secondary (plan, Phase 7)
OUT = AUDIT / "copy_mutate"
_REF = None


def _init() -> None:
    global _REF
    _REF = load_reference()[0]


def _run(task: tuple) -> tuple[list[dict], list[dict]]:
    region, m, instances, n, seed = task
    variant, fitness = MODELS[m]
    source = f"{variant}/{fitness}"
    real = _REF.real[region]
    rows, runs = [], []
    for k in range(instances):
        s = int(np.random.SeedSequence([seed, zlib.crc32(region.encode()), m, k]).generate_state(1)[0])
        cm, stats = copy_mutate_with_stats(real, variant=variant, fitness=fitness, sizes="empirical", seed=s)
        full = scorecard_row(cm, region, _REF, max_novelty_recipes=2000, seed=k)
        pick = np.random.default_rng(s).choice(len(cm), size=min(n, len(cm)), replace=False)
        sub = Cuisine(f"{cm.name}@{n}", [cm.recipes[i] for i in pick], cm.categories)
        small = scorecard_row(sub, region, _REF, seed=k)
        rows.append({"source": source, "instance": k, "scope": "full", **full})
        rows.append({"source": source, "instance": k, "scope": f"n={n}", **small})
        runs.append({"region": region, "source": source, "instance": k, "seed": s, **stats})
    return rows, runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--n", type=int, default=LLM_N_PER_REGION)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regions", nargs="*")
    ap.add_argument("--workers", type=int, default=worker_count())
    args = ap.parse_args()
    t0 = time.time()
    ref = load_reference()[0]
    regions = args.regions or sorted(ref.real)
    # largest regions first, so the slowest tasks don't start last
    tasks = sorted(
        ((r, m, args.instances, args.n, args.seed) for r in regions for m in range(len(MODELS))),
        key=lambda t: -len(ref.real[t[0]]),
    )
    rows, runs = [], []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init) as ex:
        for task, (r_rows, r_runs) in zip(tasks, ex.map(_run, tasks), strict=True):
            rows.extend(r_rows)
            runs.extend(r_runs)
            print(f"{task[0]:22s} {'/'.join(MODELS[task[1]])}: {len(r_runs)} instances | {time.time() - t0:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.regions is None else "_partial"
    sc, rn = pd.DataFrame(rows), pd.DataFrame(runs)
    sc.to_parquet(OUT / f"scorecard{suffix}.parquet", index=False)
    rn.to_parquet(OUT / f"runs{suffix}.parquet", index=False)
    summary = sc.groupby(["source", "scope"]).agg(
        sign_match=("sign_match", "mean"),
        sign_error=("sign_error", "mean"),
        prev_jsd=("prevalence_jsd", "median"),
        sai=("sai", "median"),
        freq_rank_mae=("freq_rank_mae", "median"),
        fp_rho=("fingerprint_spearman", "median"),
        size_ks=("size_ks", "median"),
        near_copy=("near_copy_rate", "median"),
    )
    print(summary.round(3).to_string())
    print(rn.groupby("source")[["acceptance_rate", "duplicate_share"]].mean().round(3).to_string())
    print(f"done in {time.time() - t0:.0f}s -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
