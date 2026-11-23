"""Phase 2 replication gate: reproduce Singh & Bagler (2018) Fig. 4 on CulinaryDB with our implementation.

For every calibration variant (the paper under-specifies a few choices) and every region x null model, it
computes Z against 100,000 null recipes and compares with the published values in
data/reference/singh_bagler_2018_targets.json (sign agreement, Spearman rho, MAE).

Outputs: data/interim/replication/{results_<variant>.csv, summary.csv}, figures/F2_replication_*.png and
docs/replication_report.md (tables are generated; the narrative is written by hand).

Usage (repo root):  .venv/bin/python analysis/replication/replicate_singh_bagler.py [--variants v0 v1 ...] [--n-null 100000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

from flavoraudit.data import load_ahn, load_culinarydb, load_flavordb
from flavoraudit.flavor import build_profiles
from flavoraudit.nulls import NULL_MODELS
from flavoraudit.pairing import build_cuisines
from flavoraudit.stats import food_pairing

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "interim" / "replication"
FIG = REPO / "figures"
TARGETS = json.loads((REPO / "data" / "reference" / "singh_bagler_2018_targets.json").read_text())
CODE = {v: k for k, v in TARGETS["region_codes"].items()}

VARIANTS = {
    "v0_baseline": dict(desc="drop empty-profile entities; compounds = union of constituents; Cayenne from Ahn 2011"),
    "v1_keep_empty": dict(
        desc="as v0 but keep empty-profile entities in recipes (they add zero-sharing pairs)", keep_empty=True
    ),
    "v2_cayenne_capsicum": dict(desc="as v0 but Cayenne takes the FlavorDB Capsicum profile", cayenne="capsicum"),
    "v3_drop_compounds": dict(desc="as v0 but compound ingredients are removed from recipes", drop_compounds=True),
    "v4_drop_additives": dict(
        desc="as v0 but the Additive category (salt, sugar, water, ...) is removed", drop_additives=True
    ),
}


def seed_for(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def run_variant(name: str, cfg: dict, cdb, fdb, ahn, n_null: int) -> pd.DataFrame:
    fp = build_profiles(cdb, fdb, ahn, cayenne_source=cfg.get("cayenne", "ahn"))
    S, idx = fp.shared_matrix()
    drop = set()
    if not cfg.get("keep_empty"):
        drop |= {e for e, p in fp.profiles.items() if len(p) == 0}
    if cfg.get("drop_compounds"):
        drop |= {e for e in fp.profiles if e >= 2000}
    if cfg.get("drop_additives"):
        drop |= {e for e, c in fp.categories.items() if c == "Additive"}
    cuisines = build_cuisines(cdb.recipe_entities, cdb.recipes, idx, fp.categories, drop, labels=cdb.regions)
    rows = []
    for region, cu in cuisines.items():
        for null in NULL_MODELS:
            r = food_pairing(cu, S, null, n_null=n_null, seed=seed_for(name, region, null))
            rows.append({"variant": name, "region": region, "code": CODE[region], **r.as_dict()})
    return pd.DataFrame(rows)


def agreement(df: pd.DataFrame) -> dict:
    out = {}
    for null in NULL_MODELS:
        pub = TARGETS["z_vs_null"][null]
        sub = df[df["null_model"] == null].set_index("code")
        ours = sub.loc[list(pub), "z_sb"].to_numpy()
        theirs = np.array([pub[c] for c in pub], dtype=float)
        rho = st.spearmanr(ours, theirs).statistic
        out[null] = {
            "sign_agree": int((np.sign(ours) == np.sign(theirs)).sum()),
            "spearman": round(float(rho), 3),
            "mae": round(float(np.abs(ours - theirs).mean()), 1),
        }
    return out


def plot(df: pd.DataFrame, variant: str) -> Path:
    import matplotlib.pyplot as plt

    FIG.mkdir(exist_ok=True)
    titles = {
        "random": "Random",
        "frequency": "Ingredient frequency",
        "category": "Ingredient category",
        "frequency_category": "Frequency + category",
    }
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, null in zip(axes, NULL_MODELS, strict=True):
        pub = TARGETS["z_vs_null"][null]
        sub = df[df["null_model"] == null].set_index("code")
        x = np.array([pub[c] for c in pub])
        y = sub.loc[list(pub), "z_sb"].to_numpy()
        lim = [min(x.min(), y.min()) * 1.1 - 5, max(x.max(), y.max()) * 1.1 + 5]
        ax.plot(lim, lim, color="#999", lw=1, zorder=0)
        ax.axhline(0, color="#ddd", lw=0.8, zorder=0)
        ax.axvline(0, color="#ddd", lw=0.8, zorder=0)
        neg = set(TARGETS["expected_sign"]["contrasting_negative"])
        colors = ["#c0392b" if c in neg else "#2a78d6" for c in pub]
        ax.scatter(x, y, c=colors, s=28, zorder=3)
        for c, xi, yi in zip(pub, x, y, strict=True):
            ax.annotate(c, (xi, yi), fontsize=7, xytext=(3, 2), textcoords="offset points")
        rho = st.spearmanr(x, y).statistic
        agree = int((np.sign(x) == np.sign(y)).sum())
        ax.set_title(f"{titles[null]} null\nSpearman ρ = {rho:.2f}, sign agreement {agree}/22", fontsize=10)
        ax.set_xlabel("Published Z (Singh & Bagler 2018)")
        ax.set_ylabel("Our Z (this work)")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    fig.suptitle(
        f"F2. Replication of Singh & Bagler (2018) Fig. 4 on CulinaryDB ({variant}); blue = published uniform, red = published contrasting",
        fontsize=11,
    )
    fig.tight_layout()
    path = FIG / f"F2_replication_{variant}.png"
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--n-null", type=int, default=100_000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cdb, fdb, ahn = load_culinarydb(), load_flavordb(), load_ahn()
    summary = []
    for v in args.variants:
        t = time.time()
        df = run_variant(v, VARIANTS[v], cdb, fdb, ahn, args.n_null)
        df.to_csv(OUT / f"results_{v}.csv", index=False)
        ag = agreement(df)
        summary.append(
            {
                "variant": v,
                "desc": VARIANTS[v]["desc"],
                **{f"{n}_{k}": val for n, d in ag.items() for k, val in d.items()},
            }
        )
        print(
            f"{v}: {time.time() - t:.0f}s | "
            + " | ".join(f"{n}: sign {d['sign_agree']}/22, rho {d['spearman']}, MAE {d['mae']}" for n, d in ag.items()),
            flush=True,
        )
    sm = pd.DataFrame(summary)
    sm.to_csv(OUT / "summary.csv", index=False)
    print(sm.drop(columns="desc").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
