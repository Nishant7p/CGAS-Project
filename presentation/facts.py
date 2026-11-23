"""Facts layer of the presentation pack: every number shown on a slide, the poster or a handout comes from here.

Nothing is typed by hand. Numbers are read from the pipeline outputs:

* ``results/summary.json`` (written by ``analysis/summarize.py``): the headline numbers;
* ``tables/rq_tests_full.csv``, ``tables/h1_equal_n_full.csv``, ``tables/sign_rule_calibration_n160.csv``;
* ``data/interim/steering/scorer_metrics.json``: details of the D3 fidelity scorer;
* ``figures/*.png``: figures are shown only if the file exists.

A few inputs are reference or design information rather than results, and are read from their source files too:

* ``data/reference/singh_bagler_2018_targets.json``: published values (digitised from Singh & Bagler 2018);
* ``data/interim/aliasing/metrics.json``: held-out evaluation of the aliaser against CulinaryDB's expert mapping;
* ``data/interim/steering/bon_main.parquet``: best-of-N design metadata (pool and selection sizes only);
* ``configs/models.yaml``, ``configs/regions.yaml``, ``configs/prompts/v1/*.txt``, ``steering/design.py``,
  ``steering/prompt_sets/steer_bon_v1.jsonl``, ``src/flavoraudit`` and ``tests/`` (module, constant and test counts).

Anything that is missing becomes a ``[pending: ...]`` marker, so the documents show what will still change.
"""

from __future__ import annotations

import ast
import contextlib
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "results" / "summary.json"
TABLES = REPO / "tables"
FIGURES = REPO / "figures"
SCORER_METRICS = REPO / "data" / "interim" / "steering" / "scorer_metrics.json"
ALIAS_METRICS = REPO / "data" / "interim" / "aliasing" / "metrics.json"
TARGETS = REPO / "data" / "reference" / "singh_bagler_2018_targets.json"
BON_PARQUET = REPO / "data" / "interim" / "steering" / "bon_main.parquet"
BON_PROMPTS = REPO / "steering" / "prompt_sets" / "steer_bon_v1.jsonl"
STEER_DESIGN = REPO / "steering" / "design.py"
MODELS_YAML = REPO / "configs" / "models.yaml"
REGIONS_YAML = REPO / "configs" / "regions.yaml"
PROMPTS = REPO / "configs" / "prompts" / "v1"
CURATION_OVERRIDES = REPO / "curation" / "alias_overrides.csv"
LIBRARY = REPO / "src" / "flavoraudit"
TESTS = REPO / "tests"
CI_YAML = REPO / ".github" / "workflows" / "ci.yml"
STYLE_PY = REPO / "analysis" / "figures" / "style.py"
SUMMARIZE_PY = REPO / "analysis" / "summarize.py"

ALPHA = 0.05  # BH-FDR level of the plan (section 7)

# Figure keys -> file name patterns (the replication figures carry the calibration variant in their names).
FIGURE_PATTERNS = {
    "F2": "F2_replication_*.png",
    "F2b": "F2b_fig4_style_*.png",
    "F2c": "F2c_matched_size_signs.png",
    "F2d": "F2d_pool_effect.png",
    "F3": "F3_pairing_dotplot.png",
    "F4": "F4_effect_sizes.png",
    "F5": "F5_sign_matrix.png",
    "F8": "F8_sai.png",
    "F12": "F12_scorecard.png",
    "F13": "F13_copy_mutate.png",
    "F14": "F14_persona_vs_plain.png",
    "F15": "F15_best_of_n.png",
    "F15b": "F15b_ablation.png",
    "F15c": "F15c_critique_revise.png",
    "F1": "F1_pipeline.png",
    "F4b": "F4b_ns_distributions.png",
    "F6": "F6_fingerprints.png",
    "F7": "F7_drivers.png",
    "F9": "F9_diversity.png",
    "F10": "F10_sizes.png",
    "F11": "F11_networks.png",
    "F11b": "F11b_showcase_pairs.png",
}


# --------------------------------------------------------------------------------------------------------------------
# Formatting helpers (shared by the deck, the poster and the Markdown templates)
# --------------------------------------------------------------------------------------------------------------------
def pending(what: str) -> str:
    return f"[pending: {what}]"


def is_pending(x: Any) -> bool:
    return isinstance(x, str) and x.startswith("[pending")


def _missing(x: Any) -> bool:
    return x is None or (isinstance(x, float) and not math.isfinite(x))


def f(x: Any, nd: int = 2, what: str = "value not yet in results/summary.json") -> str:
    """Fixed decimals."""
    return pending(what) if _missing(x) else f"{float(x):.{nd}f}"


def sf(x: Any, nd: int = 2, what: str = "value not yet in results/summary.json") -> str:
    """Signed fixed decimals (for shifts and differences)."""
    if _missing(x):
        return pending(what)
    s = f"{float(x):+.{nd}f}"
    return s.replace("-", "−")


def pct(x: Any, nd: int = 1, what: str = "value not yet in results/summary.json") -> str:
    """A share in [0, 1] as a percentage."""
    return pending(what) if _missing(x) else f"{100 * float(x):.{nd}f}%"


def n0(x: Any, what: str = "value not yet in results/summary.json") -> str:
    """Integer with thousands separators."""
    return pending(what) if _missing(x) else f"{int(round(float(x))):,}"


def pv(p: Any) -> str:
    """A p-value with two significant digits (or '< 0.001')."""
    if _missing(p):
        return "n/a"
    p = float(p)
    return "< 0.001" if p < 0.001 else f"= {p:.2g}"


def minus(s: str) -> str:
    """Typographic minus signs for numbers in running text."""
    return re.sub(r"(?<![\w.])-(?=\d)", "−", s)


def join_and(items: list[str]) -> str:
    items = [str(i) for i in items]
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# --------------------------------------------------------------------------------------------------------------------
# Small readers
# --------------------------------------------------------------------------------------------------------------------
def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, dtype={"effect": str})
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _module_constants(path: Path, names: tuple[str, ...]) -> dict[str, Any]:
    """Literal module-level constants of a Python file, read with ``ast`` (no import, no side effects)."""
    out: dict[str, Any] = {}
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return out
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets, value = [node.target.id], node.value
        for t in targets:
            if t in names and value is not None:
                with contextlib.suppress(ValueError):
                    out[t] = ast.literal_eval(value)
    return out


def _model_labels() -> dict[str, str]:
    """Model display names in the order the figures use (analysis/figures/style.py LABELS)."""
    for path, name in ((STYLE_PY, "LABELS"), (SUMMARIZE_PY, "MODEL_NAMES")):
        labels = _module_constants(path, (name,)).get(name)
        if isinstance(labels, dict):
            return {k: v for k, v in labels.items() if k not in ("real", "copy-mutate")}
    return {}


def figure(key: str) -> Path | None:
    pattern = FIGURE_PATTERNS.get(key)
    if pattern is None:
        return None
    hits = sorted(FIGURES.glob(pattern))
    return hits[0] if hits else None


# --------------------------------------------------------------------------------------------------------------------
# Verdict helpers
# --------------------------------------------------------------------------------------------------------------------
def direction(effect: float | None, p_adj: float | None, better: str) -> str:
    """'better', 'worse' or 'no difference' for a difference whose direction of improvement is ``better``."""
    if effect is None or p_adj is None or p_adj >= ALPHA:
        return "no difference"
    good = effect < 0 if better == "lower" else effect > 0
    return "better" if good else "worse"


def supported(p_adj: float | None) -> bool:
    return p_adj is not None and p_adj < ALPHA


# --------------------------------------------------------------------------------------------------------------------
# The facts
# --------------------------------------------------------------------------------------------------------------------
def build_facts() -> dict[str, Any]:
    S = _read_json(SUMMARY)
    if S is None:
        raise SystemExit(f"missing {SUMMARY}: run `.venv/bin/python analysis/summarize.py` first")
    F: dict[str, Any] = {"warnings": [], "pending": []}
    warn, todo = F["warnings"].append, F["pending"].append

    F["generated_utc"] = S.get("generated_utc", "")
    try:
        F["generated_date"] = dt.datetime.fromisoformat(F["generated_utc"]).strftime("%d %b %Y, %H:%M UTC")
    except ValueError:
        F["generated_date"] = F["generated_utc"]
    F["build_date"] = dt.datetime.now(dt.timezone.utc).strftime("%d %b %Y, %H:%M UTC")

    # ---------------- regions and design ----------------
    regions_cfg = yaml.safe_load(REGIONS_YAML.read_text())
    code_of = {v["label"]: k for k, v in regions_cfg.items()}
    label_of = {k: v["label"] for k, v in regions_cfg.items()}
    targets = _read_json(TARGETS) or {}
    contrasting = [label_of[c] for c in targets.get("expected_sign", {}).get("contrasting_negative", []) if c in label_of]
    rep = S.get("replication", {})
    regions = sorted(rep.get("n_recipes", {}) or label_of.values())
    uniform = [r for r in regions if r not in contrasting]
    F["regions"] = {
        "all": regions,
        "n": len(regions),
        "contrasting": contrasting,
        "n_contrasting": len(contrasting),
        "uniform": uniform,
        "n_uniform": len(uniform),
        "code_of": code_of,
        "label_of": label_of,
        "phrase_korea": regions_cfg.get("KOR", {}).get("phrase", "Korean"),
        "phrase_of": {v["label"]: v.get("phrase", v["label"]) for v in regions_cfg.values()},
    }

    prompt_text = {p.stem: p.read_text().strip() for p in PROMPTS.glob("*.txt")}
    m = re.search(r"exactly (\d+) recipes", prompt_text.get("format", ""))
    recipes_per_call = int(m.group(1)) if m else None
    gen = _module_constants(REPO / "generation" / "common.py", ("CONDITIONS", "N_REPS_FULL", "PROMPT_SET_VERSION"))
    conditions = list(gen.get("CONDITIONS", ()))
    calls_per_cell = gen.get("N_REPS_FULL")
    planned_calls = None
    if calls_per_cell and conditions:
        planned_calls = len(regions) * len(conditions) * calls_per_cell
    F["design"] = {
        "recipes_per_call": recipes_per_call,
        "conditions": conditions,
        "n_conditions": len(conditions),
        "calls_per_cell": calls_per_cell,
        "planned_calls": planned_calls,
        "recipes_per_model": planned_calls * recipes_per_call if planned_calls and recipes_per_call else None,
        "recipes_per_region": len(conditions) * calls_per_cell * recipes_per_call
        if conditions and calls_per_cell and recipes_per_call
        else None,
        "prompt_plain": prompt_text.get("plain", "").split("\n\n")[0],
        "prompt_persona": prompt_text.get("persona", "").split("\n\n")[0],
        "prompt_system": prompt_text.get("system", ""),
        "prompt_version": gen.get("PROMPT_SET_VERSION", "v1"),
        "proposal_min_recipes": None,
    }
    # The proposal's D1 minimum, as encoded by analysis/summarize.py (meets_proposal) — read, not retyped.
    msrc = SUMMARIZE_PY.read_text() if SUMMARIZE_PY.exists() else ""
    m = re.search(r">=\s*([\d_]+)\s*and len\(per\)\s*>=\s*(\d+)\s*and len\(families\)\s*>=\s*(\d+)", msrc)
    if m:
        F["design"]["proposal_min_recipes"] = int(m.group(1).replace("_", ""))
        F["design"]["proposal_min_models"] = int(m.group(2))
        F["design"]["proposal_min_open_families"] = int(m.group(3))

    lib_consts = _module_constants(LIBRARY / "audit.py", ("NEAR_COPY",))
    src_audit = (LIBRARY / "audit.py").read_text() if (LIBRARY / "audit.py").exists() else ""
    m = re.search(r"n_null:\s*int\s*=\s*([\d_]+)", src_audit)
    n_rand = int(m.group(1).replace("_", "")) if m else None
    F["method"] = {
        "n_rand": n_rand,
        "sqrt_n_rand": math.sqrt(n_rand) if n_rand else None,
        "near_copy": lib_consts.get("NEAR_COPY"),
    }

    # ---------------- models ----------------
    mcfg = yaml.safe_load(MODELS_YAML.read_text())
    palette = mcfg.get("palette", {})
    labels = _model_labels()
    ds = S.get("dataset", {})
    dmodels = ds.get("models", {})
    planned = [k for k, v in mcfg.get("models", {}).items() if v.get("tier") in ("core", "extra")]
    order = [k for k in labels if k in set(planned) | set(dmodels)]
    order += [k for k in list(planned) + list(dmodels) if k not in order]
    models = {}
    for k in order:
        cfg = mcfg.get("models", {}).get(k, {})
        d = dmodels.get(k)
        info = {
            "key": k,
            "name": (d or {}).get("name") or labels.get(k, k),
            "family": (d or {}).get("family") or cfg.get("family", ""),
            "proprietary": cfg.get("provider") in ("openai", "gemini"),
            "tier": cfg.get("tier", "?"),
            "color": palette.get(k, "#666666"),
            "data": d,
            "complete": bool(d and d.get("complete")),
            "analysed": k in S.get("llm_audit", {}),
        }
        if d is None:
            if cfg.get("tier") == "core":
                info["status"] = pending("not yet run")
            else:
                info["status"] = pending("running on Kaggle" + (" (RQ4 scale pair)" if k == "qwen3.5-4b" else "")
                                         if cfg.get("runner") == "kaggle" else "not yet run")  # fmt: skip
        elif not d.get("complete"):
            why = "; free-tier daily request cap" if cfg.get("provider") == "gemini" else ""
            info["status"] = pending(f"{d['regions']}/{len(regions)} regions so far ({d['calls']:,}/"
                                     f"{d['planned_calls']:,} calls{why})")  # fmt: skip
        else:
            info["status"] = "complete"
        models[k] = info
    F["models"] = models
    F["model_order"] = order
    F["analysed"] = [k for k in order if models[k]["analysed"]]
    F["complete_models"] = [k for k in order if models[k]["complete"]]
    F["incomplete_models"] = [k for k in order if not models[k]["complete"]]
    for k in F["incomplete_models"]:
        mi = models[k]
        todo({"item": f"{mi['name']} ({mi['tier']})", "detail": mi["status"]})
    F["dataset"] = {
        "total_recipes": ds.get("total_recipes"),
        "total_calls": ds.get("total_calls"),
        "n_models": ds.get("n_models"),
        "n_proprietary": ds.get("n_proprietary"),
        "n_open_families": ds.get("n_open_families"),
        "meets_proposal": ds.get("meets_proposal"),
        "n_planned": len(planned),
        "planned_total": len(planned) * F["design"]["recipes_per_model"] if F["design"]["recipes_per_model"] else None,
    }
    if not ds.get("meets_proposal"):
        todo({"item": "Proposal minimum for D1", "detail": pending(
            f"{n0(ds.get('total_recipes'))} recipes and {ds.get('n_open_families')} open-weight family so far; "
            f"needs ≥ {n0(F['design'].get('proposal_min_recipes'))} recipes and "
            f"≥ {F['design'].get('proposal_min_open_families', '?')} open families")})
    alias_share = [dmodels[k]["alias_full_share"] for k in dmodels if dmodels[k].get("alias_full_share") is not None]
    matched_share = [dmodels[k]["alias_matched_share"] for k in dmodels if dmodels[k].get("alias_matched_share")]
    F["dataset"]["alias_full_min"] = min(alias_share) if alias_share else None
    F["dataset"]["alias_full_max"] = max(alias_share) if alias_share else None
    F["dataset"]["alias_matched_min"] = min(matched_share) if matched_share else None
    F["dataset"]["alias_matched_max"] = max(matched_share) if matched_share else None
    F["dataset"]["curation_pending"] = not CURATION_OVERRIDES.exists()
    if F["dataset"]["curation_pending"]:
        todo({"item": "Alias curation of unmatched LLM terms", "detail": pending(
            "proposed aliases awaiting review; LLM numbers use the uncurated aliasing")})

    am = _read_json(ALIAS_METRICS) or {}
    test = am.get("test", {})
    F["aliasing"] = {
        "acc_lines": test.get("accuracy_lines"),
        "acc_rows": test.get("accuracy_rows_weighted"),
        "coverage": test.get("coverage_lines"),
        "test_lines": test.get("lines"),
        "chosen": am.get("chosen"),
    }

    # ---------------- replication ----------------
    by_null = rep.get("by_null", {})
    null_names = {
        "random": "Random",
        "frequency": "Ingredient frequency",
        "category": "Ingredient category",
        "frequency_category": "Frequency + category",
    }
    F["replication"] = {
        "nulls": [
            {"key": k, "name": null_names.get(k, k), "signs": v.get("sign_agreement"), "n": v.get("n"),
             "rho": v.get("spearman")}
            for k, v in by_null.items()
        ],  # fmt: skip
        "random": by_null.get("random", {}),
        "korea_top3": rep.get("korea_top3", []),
        "z": rep.get("z_random", {}),
        "d": rep.get("d_random", {}),
        "n_recipes": rep.get("n_recipes", {}),
    }
    pub = targets.get("top3_contribution_percent", {}).get("negative", {}).get("KOR", {})
    top1 = (rep.get("korea_top3") or [{}])[0]
    F["replication"]["korea_top1"] = top1.get("name")
    F["replication"]["korea_top1_ours"] = top1.get("chi_percent")
    F["replication"]["korea_top1_published"] = pub.get(top1.get("name")) if top1 else None
    F["replication"]["korea_top3_published"] = pub
    pubz = targets.get("z_vs_null", {}).get("random", {})
    F["replication"]["z_published"] = {label_of.get(c, c): v for c, v in pubz.items()}
    zk, dk = rep.get("z_random", {}).get("Korea"), rep.get("d_random", {}).get("Korea")
    F["replication"]["korea_z"], F["replication"]["korea_d"] = zk, dk
    F["replication"]["korea_z_published"] = pubz.get("KOR")
    F["replication"]["z_over_d_korea"] = zk / dk if zk and dk else None
    z, d = rep.get("z_random", {}), rep.get("d_random", {})
    ratios = [z[r] / d[r] for r in z if d.get(r)]
    F["replication"]["z_over_d_min"] = min(ratios) if ratios else None
    F["replication"]["z_over_d_max"] = max(ratios) if ratios else None
    sizes = rep.get("n_recipes", {})
    smallest7 = sorted(sizes, key=sizes.get)[:7]
    F["replication"]["contrasting_in_smallest7"] = sum(r in contrasting for r in smallest7)
    F["replication"]["smallest_region"] = min(sizes, key=sizes.get) if sizes else None
    F["replication"]["smallest_n"] = min(sizes.values()) if sizes else None

    # ---------------- equal-N (sample-size) finding ----------------
    eq = S.get("equal_n", {})
    matched_key = next((k for k in eq if re.fullmatch(r"matched\d+_positive_share", k)), None)
    matched_n = int(re.search(r"\d+", matched_key).group()) if matched_key else None
    n_sub_key = next((k for k in eq if re.fullmatch(r"n\d+_sign_match", k)), None)
    n_sub = int(re.search(r"\d+", n_sub_key).group()) if n_sub_key else None
    share = eq.get(matched_key, {}) if matched_key else {}
    weakened = sorted(r for r in uniform if r in share and 0.5 <= share[r] < 0.95)
    sign_match = eq.get(n_sub_key, {}) if n_sub_key else {}
    calib = _read_csv(TABLES / "sign_rule_calibration_n160.csv")
    fp_rates = {}
    if calib is not None:
        fp_rates = dict(zip(calib["region"], calib["sign_rule_false_positive"], strict=True))
    fp_top = sorted(((r, v) for r, v in fp_rates.items() if v > 0.1), key=lambda t: -t[1])
    rq_tab = _read_csv(TABLES / "rq_tests_full.csv")
    n_genuine = None
    if rq_tab is not None:
        row = rq_tab[(rq_tab["hypothesis"] == "H1") & (rq_tab["model"].astype(str).str.startswith("real@"))]
        if len(row) and len(regions):
            n_genuine = int(row["n"].iloc[0]) // len(regions)
    F["equal_n"] = {
        "matched_n": matched_n,
        "n_sub": n_sub,
        "n_genuine_per_region": n_genuine,
        "share": share,
        "contrasting_robust": eq.get("contrasting_robust"),
        "uniform_robust": eq.get("uniform_robust"),
        "uniform_flipped": eq.get("uniform_flipped", []),
        "uniform_weakened": weakened,
        "fpr": eq.get("n160_sign_rule_false_positive_rate"),
        "fpr_by_region": fp_rates,
        "fpr_top": fp_top,
        "sign_match_by_region": sign_match,
        "genuine_sign_agreement": sum(sign_match.values()) / len(sign_match) if sign_match else None,
        "ref": eq.get("n160_reference_medians", {}),
        "sai_band": eq.get("n160_sai_band", [None, None]),
        "usa": eq.get("usa_pool_effect", {}),
    }

    # ---------------- LLM audit and pre-registered tests ----------------
    la = S.get("llm_audit", {})
    h1t = _read_csv(TABLES / "h1_equal_n_full.csv")

    def rq(hyp: str, model: str | None = None, contains: str | None = None) -> dict | None:
        if rq_tab is None:
            return None
        t = rq_tab[rq_tab["hypothesis"] == hyp]
        if model is not None:
            t = t[t["model"] == model]
        if contains is not None:
            t = t[t["test"].str.contains(contains, regex=False)]
        if not len(t):
            return None
        r = t.iloc[0]
        return {"test": r["test"], "p": _num(r["p"]), "p_adj": _num(r["p_adj"]), "effect": r["effect"],
                "effect_num": _num(r["effect"]), "n": _num(r["n"]),
                "note": "" if pd.isna(r["note"]) else str(r["note"])}  # fmt: skip

    per = {}
    for k in F["analysed"]:
        a = la[k]
        p = {
            "name": models[k]["name"],
            "color": models[k]["color"],
            "proprietary": models[k]["proprietary"],
            "complete": models[k]["complete"],
            "regions": a.get("regions"),
            "recipes": a.get("recipes"),
            **{x: a.get(x) for x in a},
            "h1_fisher": rq("H1", k, "Fisher"),
            "h1_count": rq("H1", k, "size-aware"),
            "h1_explore": rq("H1-exploratory", k),
            "h1eq_wilcoxon": rq("H1-equalN", k, "Wilcoxon"),
            "h1eq_fisher": rq("H1-equalN", k, "Fisher"),
            "h2b_sai": rq("H2b", k),
            "h2b_div": rq("H2b-diversity", k),
            "h2c": rq("H2c", k),
            "persona": {
                met: rq("H4-persona", k, f": {met}") for met in ("abs_d_gap", "prevalence_jsd", "fingerprint_spearman",
                                                               "abs_log_sai")
            },  # fmt: skip
            "cm": {
                met: rq("CM", k, f": {met}")
                for met in ("abs_d_gap", "prevalence_jsd", "fingerprint_spearman", "abs_log_sai")
            },
        }
        h2c = p["h2c"]
        m2 = re.search(r"(\d+) regions? below", h2c["note"]) if h2c else None
        p["h2c_regions_below"] = int(m2.group(1)) if m2 else None
        per[k] = p
    F["llm"] = per

    better = {"abs_d_gap": "lower", "prevalence_jsd": "lower", "fingerprint_spearman": "higher", "abs_log_sai": "lower"}
    F["metric_names"] = {
        "abs_d_gap": "food-pairing gap |d_LLM − d_real|",
        "prevalence_jsd": "prevalence JSD",
        "fingerprint_spearman": "fingerprint ρ",
        "abs_log_sai": "|log SAI|",
    }
    F["metric_better"] = better
    for p in per.values():
        p["persona_verdict"] = {
            met: direction(r["effect_num"] if r else None, r["p_adj"] if r else None, better[met])
            for met, r in p["persona"].items()
        }
        p["cm_verdict"] = {
            met: direction(r["effect_num"] if r else None, r["p_adj"] if r else None, better[met])
            for met, r in p["cm"].items()
        }

    # proprietary vs open (H4) and scale
    prop = {}
    if rq_tab is not None:
        for met in better:
            r = rq("H4-proprietary", None, f": {met}")
            if r:
                prop[met] = {**r, "verdict": direction(r["effect_num"], r["p_adj"], better[met])}
    F["proprietary"] = prop
    F["proprietary_models"] = [models[k]["name"] for k in F["analysed"] if models[k]["proprietary"]]
    F["open_models"] = [models[k]["name"] for k in F["analysed"] if not models[k]["proprietary"]]
    scale_rows = rq_tab[rq_tab["hypothesis"].str.startswith("H4-scale")] if rq_tab is not None else None
    F["scale"] = None if scale_rows is None or not len(scale_rows) else scale_rows.to_dict(orient="records")
    if F["scale"] is None:
        todo({"item": "RQ4 scale contrast (Qwen3.5-9B vs Qwen3.5-4B)", "detail": pending(
            "Qwen3.5-4B running on Kaggle")})

    # Region-level consensus of the equal-N verdicts (every analysed model with data for the region agrees).
    consensus = []
    if h1t is not None and len(h1t):
        for region, g in h1t.groupby("region"):
            verdicts = set(g["verdict"])
            if len(g) >= 2 and len(verdicts) == 1 and next(iter(verdicts)) != "typical":
                consensus.append({"region": region, "verdict": next(iter(verdicts)), "k": len(g)})
        consensus.sort(key=lambda c: (-c["k"], c["verdict"], c["region"]))
    F["consensus"] = consensus
    F["n_analysed"] = len(F["analysed"])

    # Aggregates across the analysed models (for headline ranges)
    def rng(key: str) -> tuple[float | None, float | None]:
        vals = [per[k].get(key) for k in per if per[k].get(key) is not None]
        return (min(vals), max(vals)) if vals else (None, None)

    F["ranges"] = {
        key: rng(key)
        for key in (
            "median_prevalence_jsd",
            "median_category_jsd",
            "median_sai",
            "median_fingerprint_spearman",
            "median_diversity_ratio",
            "median_freq_rank_mae",
            "mean_near_copy_rate",
            "median_novelty",
            "sign_agreement",
        )
    }
    complete_per = {k: v for k, v in per.items() if v["complete"]}
    F["ranges_complete"] = {
        key: (
            min(v[key] for v in complete_per.values() if v.get(key) is not None),
            max(v[key] for v in complete_per.values() if v.get(key) is not None),
        )
        if complete_per
        else (None, None)
        for key in ("median_diversity_ratio", "median_fingerprint_spearman", "median_prevalence_jsd", "median_sai")
    }

    # Hypothesis verdicts per model (counts, used in summaries)
    def count(fn) -> int:
        return sum(1 for p in per.values() if fn(p))

    F["verdicts"] = {
        "h1_fisher_supported": count(lambda p: p["h1_fisher"] and supported(p["h1_fisher"]["p_adj"])),
        "h1eq_supported": count(lambda p: p["h1eq_wilcoxon"] and supported(p["h1eq_wilcoxon"]["p_adj"])),
        "h2b_sai_supported": count(lambda p: p["h2b_sai"] and supported(p["h2b_sai"]["p_adj"])),
        "h2b_div_supported": count(lambda p: p["h2b_div"] and supported(p["h2b_div"]["p_adj"])),
        "h2c_supported": count(lambda p: p["h2c"] and supported(p["h2c"]["p_adj"])),
        "persona_jsd_better": count(lambda p: p["persona_verdict"]["prevalence_jsd"] == "better"),
        "persona_rho_better": count(lambda p: p["persona_verdict"]["fingerprint_spearman"] == "better"),
        "persona_gap_better": count(lambda p: p["persona_verdict"]["abs_d_gap"] == "better"),
        "persona_sai_better": count(lambda p: p["persona_verdict"]["abs_log_sai"] == "better"),
        "cm_better": {met: count(lambda p, m=met: p["cm_verdict"][m] == "better") for met in better},
        "cm_worse": {met: count(lambda p, m=met: p["cm_verdict"][m] == "worse") for met in better},
        "n": len(per),
    }
    F["persona_rho_models"] = [p["name"] for p in per.values() if p["persona_verdict"]["fingerprint_spearman"] == "better"]
    F["cm_gap_models"] = [p["name"] for p in per.values() if p["cm_verdict"]["abs_d_gap"] == "better"]

    # ---------------- RQ2a flavor drivers ----------------
    dr = S.get("drivers") or {}
    real_top = dr.get("korea_real_top5") or []
    drv = {k: v for k, v in dr.items() if isinstance(v, dict) and k in models}
    F["drivers"] = {
        "models": {k: {**v, "name": models[k]["name"],
                       "korea_first": (v.get("korea_top5") or [None])[0]} for k, v in drv.items()},
        "order": [k for k in order if k in drv],
        "real": dr.get("real@160", {}),
        "korea_real_top5": real_top,
        "korea_match": [models[k]["name"] for k in order if k in drv and real_top
                        and (drv[k].get("korea_top5") or [None])[0] == real_top[0]],
        "korea_miss": [models[k]["name"] for k in order if k in drv and real_top and drv[k].get("korea_top5")
                       and drv[k]["korea_top5"][0] != real_top[0]],
    } if drv else None  # fmt: skip
    if not drv:
        todo({"item": "RQ2a flavor drivers", "detail": pending("drivers not yet in results/summary.json")})
    jac = [v["median_jaccard5"] for v in drv.values() if v.get("median_jaccard5") is not None]
    if F["drivers"]:
        F["drivers"]["jac_min"], F["drivers"]["jac_max"] = (min(jac), max(jac)) if jac else (None, None)

    F["networks"] = S.get("networks") or None
    if not F["networks"]:
        todo({"item": "Co-occurrence networks", "detail": pending("networks not yet in results/summary.json")})

    # ---------------- copy-mutate baseline ----------------
    cm = S.get("copy_mutate", {})
    cm_key = next((k for k in cm if k.startswith("CM-R/ranked|n=")), None)
    F["cm"] = {
        "key": cm_key,
        "ranked_n": cm.get(cm_key, {}) if cm_key else {},
        "ranked_full": cm.get("CM-R/ranked|full", {}),
        "random_n": cm.get(cm_key.replace("ranked", "random"), {}) if cm_key else {},
        "all": cm,
    }

    # ---------------- D3 steering ----------------
    st = S.get("steering", {})
    sm = _read_json(SCORER_METRICS) or {}
    sc = st.get("scorer", {})
    sel = sm.get("classifier", {}).get("selection_on_val", {})
    chosen = sm.get("classifier", {}).get("selected")
    per_region_f1 = sm.get("classifier", {}).get("per_region_f1", {})
    f1_sorted = sorted(per_region_f1.items(), key=lambda t: t[1])
    F["scorer"] = {
        **sc,
        "classifier_name": {"linear_svm": "linear SVM", "hist_gbdt": "gradient boosting", "logreg": "logistic regression"}
        .get(sc.get("classifier", ""), sc.get("classifier", "")),
        "val_macro_f1_selected": sel.get(chosen, {}).get("val_macro_f1"),
        "val_macro_f1_gbdt": sel.get("hist_gbdt", {}).get("val_macro_f1"),
        "val_macro_f1_logreg_best": max((v["val_macro_f1"] for k, v in sel.items() if k.startswith("logreg")),
                                        default=None),
        "fit_seconds_selected": sel.get(chosen, {}).get("fit_seconds"),
        "fit_seconds_gbdt": sel.get("hist_gbdt", {}).get("fit_seconds"),
        "temperature": sm.get("classifier", {}).get("temperature"),
        "kappa": sm.get("kappa"),
        "vocab_size": sm.get("vocab_size"),
        "weights_selection": sm.get("weights_selection", {}),
        "f1_worst": f1_sorted[:3],
        "f1_best": f1_sorted[::-1][:3],
        "chance": 1 / len(regions) if regions else None,
        "eval_model": sm.get("eval_classifier", {}).get("model"),
    }
    design_c = _module_constants(STEER_DESIGN, ("STEER_REGIONS", "STEER_MODELS", "N_VALUES", "N_SELECTED",
                                                 "NEAR_COPY"))  # fmt: skip
    pool_target = None
    if design_c.get("N_SELECTED") and design_c.get("N_VALUES"):
        pool_target = design_c["N_SELECTED"] * max(design_c["N_VALUES"])
    try:
        n_bon_prompts = sum(1 for _ in BON_PROMPTS.open())
    except OSError:
        n_bon_prompts = None
    res = S.get("resources", {})
    qwen_calls = (dmodels.get("qwen3.5-9b") or {}).get("calls")
    minutes_per_call = res.get("kaggle_qwen_full_minutes") / qwen_calls if res.get("kaggle_qwen_full_minutes") and \
        qwen_calls else None  # fmt: skip
    run_rates = {}
    for run, info in (res.get("kaggle_runs") or {}).items():
        if not run.endswith("-full") or not info.get("minutes"):
            continue
        slug = run.removeprefix("llmcuisine22-").removesuffix("-full")
        key = next((k for k in dmodels if k.replace(".", "-") == slug), None)
        if key and dmodels[key].get("calls"):
            run_rates[key] = {"minutes": info["minutes"], "engine": info.get("engine"),
                              "per_call": info["minutes"] / dmodels[key]["calls"]}  # fmt: skip
    F["steer_design"] = {
        "regions": [label_of.get(c, c) for c in design_c.get("STEER_REGIONS", ())],
        "n_regions_planned": len(design_c.get("STEER_REGIONS", ())),
        "models_planned": list(design_c.get("STEER_MODELS", ())),
        "n_values_planned": list(design_c.get("N_VALUES", ())),
        "n_selected_planned": design_c.get("N_SELECTED"),
        "pool_target": pool_target,
        "extra_calls_per_model": n_bon_prompts,
        "minutes_per_call": minutes_per_call,
        "gpu_hours_per_model": minutes_per_call * n_bon_prompts / 60 if minutes_per_call and n_bon_prompts else None,
        "run_rates": {models.get(k, {}).get("name", k): v for k, v in run_rates.items()},
        "gpu_hours_min": min(v["per_call"] for v in run_rates.values()) * n_bon_prompts / 60
        if run_rates and n_bon_prompts else None,
        "gpu_hours_max": max(v["per_call"] for v in run_rates.values()) * n_bon_prompts / 60
        if run_rates and n_bon_prompts else None,
    }

    bon = st.get("bon_main")
    F["bon"] = None
    if bon:
        bdf = pd.DataFrame(bon)
        rows = {}
        for k, g in bdf.groupby("model"):
            g = g.sort_values("N")
            first, last = g.iloc[0], g.iloc[-1]
            acc = g["indep_classifier_acc"].to_numpy()
            div = g["diversity_ratio"].to_numpy()
            rows[k] = {
                "name": models.get(k, {}).get("name", k),
                "N_min": int(first["N"]),
                "N_max": int(last["N"]),
                "Ns": [int(x) for x in g["N"]],
                "first": first.to_dict(),
                "last": last.to_dict(),
                "acc_monotonic": bool((acc[1:] > acc[:-1]).all()),
                "div_monotonic_down": bool((div[1:] <= div[:-1]).all()),
                "series": g.to_dict(orient="list"),
            }
        order_b = [k for k in order if k in rows] + [k for k in rows if k not in order]
        meta = {}
        try:
            bp = pd.read_parquet(BON_PARQUET, columns=["model", "code", "method", "N", "n_pool", "n_selected",
                                                       "indep_classifier_acc", "prevalence_jsd"])
            comp = bp[bp["method"] == "composite"]
            meta = {
                "pool_max": int(comp["n_pool"].max()),
                "pool_median": float(comp["n_pool"].median()),
                "selected_max": int(comp["n_selected"].max()),
                "selected_median": float(comp["n_selected"].median()),
                "regions_by_model": comp.groupby("model")["code"].nunique().to_dict(),
                "n_regions": int(comp["code"].nunique()),
                "ablation_N": int(bp[bp["method"] != "composite"]["N"].max()) if (bp["method"] != "composite").any()
                else None,
            }  # fmt: skip
            if meta["ablation_N"]:
                at = bp[bp["N"] == meta["ablation_N"]]
                meta["ablation"] = at.groupby("method")[["indep_classifier_acc", "prevalence_jsd"]].mean().to_dict(
                    orient="index")
        except (OSError, ValueError, KeyError):
            meta = {}
        F["bon"] = {
            "rows": rows,
            "order": order_b,
            "meta": meta,
            "all_acc_up": all(r["acc_monotonic"] for r in rows.values()),
            "all_div_down": all(r["last"]["diversity_ratio"] < r["first"]["diversity_ratio"] for r in rows.values()),
            "near_copy_max": max(float(bdf["near_copy_rate"].max()), 0.0),
            "n_models": len(rows),
        }
    else:
        todo({"item": "D3 best-of-N results", "detail": pending("steering.bon_main not yet in results/summary.json")})
    F["revise"] = None
    rev = st.get("revise_main")
    m = re.search(r'"--rounds",\s*type=int,\s*default=(\d+)', (REPO / "steering" / "run_revise.py").read_text()) \
        if (REPO / "steering" / "run_revise.py").exists() else None  # fmt: skip
    rounds_planned = int(m.group(1)) if m else None
    if rev:
        rdf = pd.DataFrame(rev)
        rrows = {}
        for k, g in rdf.groupby("model"):
            g = g.sort_values("round")
            first, last = g.iloc[0].to_dict(), g.iloc[-1].to_dict()
            bon_first = (F["bon"] or {}).get("rows", {}).get(k, {}).get("first", {})
            rrows[k] = {
                "name": models.get(k, {}).get("name", k),
                "rounds": int(last["round"]),
                "first": first,
                "last": last,
                "from_bon_n1": bool(bon_first) and abs(bon_first.get("indep_classifier_acc", -1)
                                                       - first["indep_classifier_acc"]) < 1e-3,
            }  # fmt: skip
        order_r = [k for k in order if k in rrows] + [k for k in rrows if k not in order]

        def improved(metric: str, better: str) -> int:
            n = 0
            for r in rrows.values():
                a, b = r["first"].get(metric), r["last"].get(metric)
                if a is not None and b is not None and (b < a if better == "lower" else b > a):
                    n += 1
            return n

        kept = [r["last"].get("identity_jaccard") for r in rrows.values()
                if r["last"].get("identity_jaccard") is not None and math.isfinite(r["last"]["identity_jaccard"])]
        F["revise"] = {
            "raw": rev,
            "rows": rrows,
            "order": order_r,
            "n_models": len(rrows),
            "rounds_planned": rounds_planned,
            "improved": {"indep_classifier_acc": improved("indep_classifier_acc", "higher"),
                         "prevalence_jsd": improved("prevalence_jsd", "lower"),
                         "fingerprint_spearman": improved("fingerprint_spearman", "higher"),
                         "sign_match": improved("sign_match", "higher")},
            "kept_min": min(kept) if kept else None,
            "kept_max": max(kept) if kept else None,
            "all_from_bon_n1": all(r["from_bon_n1"] for r in rrows.values()),
        }  # fmt: skip
        short = [r["name"] for r in rrows.values() if rounds_planned and r["rounds"] < rounds_planned]
        if short:
            todo({"item": "D3 critique-and-revise, remaining rounds", "detail": pending(
                f"round {rounds_planned} not run for {join_and(short)}")})
    else:
        todo({"item": "D3 critique-and-revise", "detail": pending(
            "round 1 running (GPT-6-Luna Batch, Qwen3.5-9B on Kaggle); steering.revise_main not yet in summary.json")})
    steered = set((F["bon"] or {}).get("rows", {})) | set((F["revise"] or {}).get("rows", {}))
    F["unsteered_models"] = [models[k]["name"] for k in F["analysed"] if k not in steered]
    if F["unsteered_models"]:
        todo({"item": "D3 steering coverage", "detail": pending(f"steering not run for {join_and(F['unsteered_models'])}")})

    # ---------------- resources ----------------
    F["resources"] = {
        "openai_usd": res.get("openai_usd_spent"),
        "budget_usd": res.get("budget_usd"),
        "kaggle_minutes": res.get("kaggle_qwen_full_minutes"),
        "kaggle_hours": res.get("kaggle_qwen_full_minutes") / 60 if res.get("kaggle_qwen_full_minutes") else None,
        "kaggle_engine": res.get("kaggle_engine") or next(
            (f"{v['engine']} fallback, vLLM failed" if v.get("vllm_failed") else v["engine"]
             for k, v in (res.get("kaggle_runs") or {}).items() if k.endswith("-full") and v.get("engine")), None),
    }

    # ---------------- library and app (counted, not typed) ----------------
    modules = sorted(p.stem for p in LIBRARY.glob("*.py") if p.stem != "__init__")
    n_tests = 0
    for p in TESTS.glob("test_*.py"):
        n_tests += len(re.findall(r"^\s*def test_", p.read_text(), flags=re.M))
    ci = {}
    with contextlib.suppress(OSError, yaml.YAMLError):
        ci = yaml.safe_load(CI_YAML.read_text())
    pyvers = []
    for job in (ci or {}).get("jobs", {}).values():
        pyvers += job.get("strategy", {}).get("matrix", {}).get("python", []) or []
    F["library"] = {
        "modules": modules,
        "n_modules": len(modules),
        "n_tests": n_tests,
        "n_test_files": len(list(TESTS.glob("test_*.py"))),
        "ci_python": sorted(set(pyvers), key=lambda v: [int(x) for x in v.split(".")]),
    }

    # ---------------- figures ----------------
    F["figures"] = {k: figure(k) for k in FIGURE_PATTERNS}
    if F["figures"]["F15c"] is None and F["revise"]:
        warn("steering.revise_main is present but figures/F15c_critique_revise.png is missing.")
    # F5 draws regions without data as sign errors (NaN -> -1 in analysis/figures/d2_figures.py), so it is shown only
    # when every analysed model covers every region.
    F["f5_safe"] = bool(per)  # F5 now marks regions a model does not cover with "·" (fixed in d2_figures.py)
    summary_mtime = SUMMARY.stat().st_mtime
    newer = [p.name for p in FIGURES.glob("F*.png") if p.stat().st_mtime > summary_mtime + 120]
    if newer:
        warn("figures newer than results/summary.json (re-run analysis/summarize.py?): " + ", ".join(sorted(newer)))
    for k in per:
        dreg = (dmodels.get(k) or {}).get("regions")
        if dreg is not None and dreg != per[k]["regions"]:
            warn(f"{per[k]['name']}: dataset covers {dreg} regions but the audit has {per[k]['regions']}.")

    # Statuses used for tags
    F["pending_core"] = [models[k]["name"] for k in F["incomplete_models"] if models[k]["tier"] == "core"]
    F["llm_pending"] = bool(F["pending_core"])
    F["pending_models"] = [models[k]["name"] for k in F["incomplete_models"]]
    F["pending_core"] = [models[k]["name"] for k in F["incomplete_models"] if models[k]["tier"] == "core"]
    F["complete_names"] = [models[k]["name"] for k in F["complete_models"]]
    F["analysed_names"] = [models[k]["name"] for k in F["analysed"]]
    return F


def provisional_reason(F: dict) -> str:
    """Why an LLM-based finding is provisional (empty if nothing is pending)."""
    parts = []
    for k in F["incomplete_models"]:
        mi = F["models"][k]
        if mi["tier"] == "core":
            parts.append(f"{mi['name']}: {mi['status'].removeprefix('[pending: ').removesuffix(']')}")
    return "; ".join(parts)


def tag(F: dict, depends_on_llm: bool, extra: str = "") -> str:
    """FINAL or PROVISIONAL (with the reason), for findings_summary.md."""
    reason = provisional_reason(F) if depends_on_llm else ""
    if extra:
        reason = f"{reason}; {extra}" if reason else extra
    return f"PROVISIONAL — depends on {reason}" if reason else "FINAL"


if __name__ == "__main__":  # quick look: python presentation/facts.py
    facts = build_facts()
    print("generated:", facts["generated_utc"])
    print("analysed:", facts["analysed_names"], "complete:", facts["complete_names"])
    for p in facts["pending"]:
        print("PENDING", p["item"], "->", p["detail"])
    for w in facts["warnings"]:
        print("WARNING", w)
