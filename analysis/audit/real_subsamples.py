"""Equal-N reference distribution: scorecard metrics of random n-recipe subsamples of each REAL cuisine.

At the LLMs' per-region sample size (n = 160), even genuine recipes differ from the full cuisine: prevalence JSD
is above 0, SAI scatters around 1, fingerprints are noisy and pairing CIs are wide. Per region and metric, these
subsamples show the range a genuine 160-recipe sample of the cuisine falls in (plan §7, equal-N comparisons). They
also give the false-positive rate of the pre-registered sign-error rule and the H2c reference (fingerprint
similarity under resampling). Novelty is leave-one-out: each real recipe is compared with every other real recipe.

Usage (repo root):
    .venv/bin/python analysis/audit/real_subsamples.py                 # 1,000 subsamples, fingerprints on 200
    .venv/bin/python analysis/audit/real_subsamples.py --reps 20 --fingerprint-reps 5 --regions Korea Japan
Output: data/interim/audit/real/subsamples_n160.parquet (one row per region x subsample)
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
from audit_io import LLM_N_PER_REGION, REAL, REPO, load_reference, worker_count  # noqa: E402

from flavoraudit.audit import real_subsample, scorecard_row  # noqa: E402

_REF = None


def _init() -> None:
    global _REF
    _REF = load_reference()[0]


def _run(task: tuple) -> list[dict]:
    region, reps, n, fp_reps, seed = task
    rng = np.random.default_rng([seed, zlib.crc32(region.encode())])
    rows = []
    for k in range(reps):
        sub, self_idx = real_subsample(_REF, region, n, rng)
        row = scorecard_row(sub, region, _REF, self_index=self_idx, with_fingerprint=k < fp_reps, seed=k)
        rows.append({"source": "real", "rep": k, **row})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=LLM_N_PER_REGION)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--fingerprint-reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regions", nargs="*")
    ap.add_argument("--workers", type=int, default=worker_count())
    args = ap.parse_args()
    t0 = time.time()
    ref = load_reference()[0]
    regions = args.regions or sorted(ref.real)
    tasks = [(r, args.reps, args.n, args.fingerprint_reps, args.seed) for r in regions]
    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init) as ex:
        for region, out in zip(regions, ex.map(_run, tasks), strict=True):
            rows.extend(out)
            print(f"{region:22s} {len(out)} subsamples | {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    path = REAL / f"subsamples_n{args.n}{'' if args.regions is None else '_partial'}.parquet"
    df.to_parquet(path, index=False)
    q = df.groupby("region").agg(
        z_median=("z_sb", "median"),
        z_real=("z_real", "first"),
        sign_match=("sign_match", "mean"),
        sign_error=("sign_error", "mean"),
        prev_jsd=("prevalence_jsd", "median"),
        sai_lo=("sai", lambda s: s.quantile(0.025)),
        sai_hi=("sai", lambda s: s.quantile(0.975)),
        fp_rho=("fingerprint_spearman", "median"),
        near_copy=("near_copy_rate", "median"),
    )
    print(q.round(3).to_string())
    print(f"sign-error rule false-positive rate (all regions): {df['sign_error'].mean():.4f}")
    print(f"done in {time.time() - t0:.0f}s -> {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
