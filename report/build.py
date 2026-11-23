#!/usr/bin/env python
"""Build the project report (report/main.pdf) from the pipeline's outputs.

Every number in the report is a LaTeX macro that this script computes; main.tex contains no hand-typed results.
Inputs that do not exist yet become visible "[pending: ...]" marks instead of errors, so the report can be rebuilt
at any time and every number (and every model-dependent sentence) updates when new data arrive: the Llama-3.1-8B
run, the rest of Gemini 3.5 Flash-Lite, critique-and-revise, any new model in results/summary.json.

Primary sources (the agreed single sources of numbers):
  results/summary.json                       analysis/summarize.py
  tables/rq_tests_full.csv                   analysis/audit/rq_tests.py (pre-registered tests, BH-adjusted p)
  tables/h1_equal_n_full.csv                 analysis/audit/rq_tests.py (per-region equal-N verdicts)
  tables/sign_rule_calibration_n160.csv      analysis/audit/rq_tests.py
  data/interim/steering/scorer_metrics.json  steering/train_scorer.py
  figures/*.pdf                              analysis/figures/*.py and steering (included as files)
Secondary sources (pipeline inputs or outputs that summary.json does not carry yet; read-only and optional):
  data/raw/culinarydb/extracted/*.csv        CulinaryDB release files (dataset counts, Sec. III)
  data/reference/singh_bagler_2018_targets.json   published Z and contributions (read from the paper's figures)
  data/interim/replication/{results_v1_keep_empty,contributions_summary}.csv   analysis/replication/*.py
  data/interim/aliasing/metrics.json         analysis/aliasing/evaluate_aliaser.py
  data/interim/steering/bon_main.parquet     steering/run_bon.py (pool sizes, SAI and novelty by N, ablation)
  configs/*.yaml, configs/prompts/v1/*, generation/prompt_sets/v1.sha256 and constants in the code (design)

Usage (repo root):
  .venv/bin/python report/build.py                 # regenerate report/generated/* and compile report/main.pdf
  .venv/bin/python report/build.py --no-pdf        # only the LaTeX inputs
  .venv/bin/python report/build.py --include-f5    # include figures/F5 (withheld: it draws missing regions as errors)
  .venv/bin/python report/build.py --draft         # add the draft-status box listing the pending items
It never writes outside report/ (run analysis/summarize.py first to refresh results/summary.json).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd
import yaml

REPORT = Path(__file__).resolve().parent
REPO = REPORT.parent
GEN = REPORT / "generated"
FIGURES = REPO / "figures"

# Display order and macro stems of the models; models not listed here are appended and named by summary.json.
PREFERRED = ["openai-gpt-6-luna", "gemini-3.5-flash-lite", "qwen3.5-9b", "llama-3.1-8b", "qwen3.5-4b", "ministral-3-8b"]
STEM = {
    "openai-gpt-6-luna": "Luna",
    "gemini-3.5-flash-lite": "Gemini",
    "qwen3.5-9b": "Qwen",
    "llama-3.1-8b": "Llama",
    "qwen3.5-4b": "QwenSmall",
    "ministral-3-8b": "Ministral",
}
FALLBACK_NAMES = {
    "openai-gpt-6-luna": "GPT-6-Luna",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "qwen3.5-9b": "Qwen3.5-9B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "qwen3.5-4b": "Qwen3.5-4B",
    "ministral-3-8b": "Ministral-3-8B",
}
# Why a planned model has no (or partial) data. Status text only; it is shown only while the data are missing.
STATUS = {
    "llama-3.1-8b": "extension in progress",
    "gemini-3.5-flash-lite": "subset set by the free-tier daily request cap",
    "qwen3.5-4b": "extension: generation in progress",
    "ministral-3-8b": "extension: generation in progress",
}
NULL_LABELS = {
    "random": "Random",
    "frequency": "Ingredient frequency",
    "category": "Ingredient category",
    "frequency_category": "Frequency + category",
}
NULL_STEMS = {"random": "Random", "frequency": "Frequency", "category": "Category", "frequency_category": "FreqCat"}
ALIAS_STAGES = {
    "A_literal": "Protocol, all CulinaryDB entities, no partial matches",
    "B_vocab": "+ vocabulary of entities used by the curators",
    "C_partial": "+ partial matches",
    "D_prefs": "+ tie-break preferences learned from expert lines",
    "D_prefs_culinary_sw": "+ extended culinary stop-words",
    "E_mined": "+ mined synonyms (support $\\geq$5, precision $\\geq$0.9)",
    "F_mined_strict": "+ mined synonyms (support $\\geq$10, precision $\\geq$0.95)",
}
WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
Q = 0.05  # replaced by the value parsed from analysis/audit/rq_tests.py when available


# ===================================================================================================== formatting
def tex(s: object) -> str:
    """Escape LaTeX special characters in text that comes from data."""
    rep = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(rep.get(ch, ch) for ch in str(s))


def finite(x: object) -> bool:
    try:
        return x is not None and not isinstance(x, bool) and math.isfinite(float(x))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def num(x: object, nd: int = 2, signed: bool = False) -> str | None:
    """A number for LaTeX: thousands separators, a real minus sign, optional explicit plus."""
    if not finite(x):
        return None
    v = float(x)  # type: ignore[arg-type]
    s = f"{abs(v):,.{nd}f}"
    if float(s.replace(",", "")) == 0:
        v = 0.0
    s = s.replace(",", "{,}")
    if v < 0:
        return r"\ensuremath{-}" + s
    return "+" + s if signed and v > 0 else s


def integer(x: object) -> str | None:
    return None if not finite(x) else f"{int(round(float(x))):,}".replace(",", "{,}")  # type: ignore[arg-type]


def pct(x: object, nd: int = 1) -> str | None:
    return None if not finite(x) else num(100 * float(x), nd) + r"\%"  # type: ignore[arg-type,operator]


def pfmt(p: object) -> str:
    if not finite(p):
        return "--"
    v = float(p)  # type: ignore[arg-type]
    if v < 0.001:
        return r"\ensuremath{<}0.001"
    return f"{v:.3f}" if v < 0.1 else f"{v:.2f}"


def p_phrase(p: object, op: str) -> str:
    """'$p \\geq 0.49$'-style text (op is 'geq' or 'leq'); tiny values read '< 0.001'."""
    if not finite(p):
        return "--"
    v = float(p)  # type: ignore[arg-type]
    if v < 0.001:
        return r"$p<0.001$"
    return rf"$p\{op} {pfmt(v)}$"


def word(n: int) -> str:
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


def join_and(items: Iterable[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def span(values: Iterable[object], nd: int = 2, as_pct: bool = False) -> str | None:
    """'lo--hi' over finite values (a single value if they agree); 'lo to hi' when negatives are involved."""
    vals = [float(v) for v in values if finite(v)]  # type: ignore[arg-type]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    f = (lambda v: num(100 * v, nd)) if as_pct else (lambda v: num(v, nd))
    a, b = f(lo), f(hi)
    out = a if a == b else (f"{a} to {b}" if lo < 0 else f"{a}--{b}")
    return out + (r"\%" if as_pct else "")  # type: ignore[operator]


def count_phrase(k: int, n: int, names: list[str]) -> str:
    """'all three models', 'none of the three models', 'two of the three models (A and B)'."""
    if n == 0:
        return "no model (no test available yet)"
    if n == 1:
        return f"the one model tested ({names[0]})" if k == 1 else "no model (one model tested)"
    if k == n:
        return f"all {word(n)} models"
    if k == 0:
        return f"none of the {word(n)} models"
    return f"{word(k)} of the {word(n)} models ({join_and(names)})"


# ===================================================================================================== I/O helpers
def load_json(rel: str) -> dict | list | None:
    try:
        return json.loads((REPO / rel).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_csv(rel: str) -> pd.DataFrame | None:
    p = REPO / rel
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def load_yaml(rel: str) -> dict:
    try:
        return yaml.safe_load((REPO / rel).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def dig(d: object, *keys: str) -> object:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def code_text(rel: str) -> str:
    try:
        return (REPO / rel).read_text(encoding="utf-8")
    except OSError:
        return ""


def const(rel: str, pattern: str, cast: Callable = int) -> object:
    """A design constant parsed from the code that defines it (None if the pattern no longer matches)."""
    m = re.search(pattern, code_text(rel), flags=re.M)
    if not m:
        return None
    try:
        return cast(m.group(1).replace("_", ""))
    except ValueError:
        return None


def const_tuple(rel: str, pattern: str) -> list[str]:
    m = re.search(pattern, code_text(rel), flags=re.M)
    if not m:
        return []
    return [t.strip().strip("\"'") for t in m.group(1).split(",") if t.strip()]


# ===================================================================================================== registry
class Registry:
    """Every macro of numbers.tex, plus the list of pending items shown in the draft-status box."""

    def __init__(self) -> None:
        self.m: dict[str, str] = {}
        self.pending: list[str] = []

    def set(self, name: str, value: object, note: str | None = None) -> None:
        assert re.fullmatch(r"N[A-Z][A-Za-z]*", name), f"bad macro name {name}"
        if value is None or value == "":
            self.pend(name, note or rf"value of \texttt{{{name}}}")
        else:
            self.m[name] = str(value)

    def pend(self, name: str, note: str) -> None:
        self.m[name] = rf"\pending{{{note}}}"
        self.note(note)

    def note(self, note: str) -> None:
        if note not in self.pending:
            self.pending.append(note)


# ===================================================================================================== context
def order(keys: Iterable[str]) -> list[str]:
    keys = list(dict.fromkeys(keys))
    return [k for k in PREFERRED if k in keys] + sorted(k for k in keys if k not in PREFERRED)


def head(name: str) -> str:
    """A two-line table header for a (TeX-escaped) model name: break at the last space, else after the last hyphen."""
    if " " in name:
        a, b = name.rsplit(" ", 1)
    elif "-" in name:
        a, b = name.rsplit("-", 1)
        a += "-"
    else:
        return rf"\makecell{{{name}}}"
    return rf"\makecell{{{a}\\{b}}}"


def stem(key: str) -> str:
    return STEM.get(key) or "".join(w.capitalize() for w in re.findall(r"[A-Za-z]+", key))


class Ctx:
    """Everything loaded once: summary, tables, scorer metrics, configs and the derived model/region lists."""

    def __init__(self, summary_path: Path) -> None:
        try:
            self.S: dict = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.S = {}
        self.rq = load_csv("tables/rq_tests_full.csv")
        if self.rq is not None:
            self.rq["effect_num"] = pd.to_numeric(self.rq["effect"], errors="coerce")
        self.h1 = load_csv("tables/h1_equal_n_full.csv")
        self.cal = load_csv("tables/sign_rule_calibration_n160.csv")
        self.scorer = load_json("data/interim/steering/scorer_metrics.json") or {}
        self.targets = load_json("data/reference/singh_bagler_2018_targets.json") or {}
        self.models_cfg = {k: v for k, v in (load_yaml("configs/models.yaml").get("models") or {}).items() if v}
        self.regions_cfg = {k: v for k, v in load_yaml("configs/regions.yaml").items() if isinstance(v, dict)}
        self.code_of = {v["label"]: k for k, v in self.regions_cfg.items() if "label" in v}
        ds = dig(self.S, "dataset", "models") or {}
        audit = dig(self.S, "llm_audit") or {}
        audit = {} if "pending" in audit else audit
        planned = [k for k, v in self.models_cfg.items() if v.get("tier") in ("core", "extra")]
        self.roster = order(list(ds) + planned + list(audit))
        self.present = order(ds)
        self.audited = order(audit)
        self.names = {k: dig(ds, k, "name") or FALLBACK_NAMES.get(k) or k for k in self.roster}
        self.proprietary = {
            k for k, v in self.models_cfg.items() if v.get("runner") == "api" and v.get("provider") in ("openai", "gemini")
        }
        codes = dig(self.targets, "region_codes") or {}
        self.contrasting = [codes[c] for c in dig(self.targets, "expected_sign", "contrasting_negative") or [] if c in codes]
        self.uniform = [codes[c] for c in dig(self.targets, "expected_sign", "uniform_positive") or [] if c in codes]
        z = dig(self.S, "replication", "z_random") or {}
        self.region_order = sorted(z, key=lambda r: -z[r]) if z else sorted(self.code_of)
        self.ok_figures = {p.name for p in FIGURES.glob("*.pdf")} if FIGURES.exists() else set()

    def name(self, key: str) -> str:
        return tex(self.names.get(key) or FALLBACK_NAMES.get(key) or key)

    def code(self, region: str) -> str:
        return tex(self.code_of.get(region, region))

    def rq_rows(self, hyp: str, sub: str) -> pd.DataFrame:
        if self.rq is None:
            return pd.DataFrame()
        return self.rq[(self.rq["hypothesis"] == hyp) & self.rq["test"].str.contains(sub, regex=False)]

    def rq_row(self, hyp: str, sub: str, model: str) -> pd.Series | None:
        d = self.rq_rows(hyp, sub)
        d = d[d["model"] == model]
        return None if d.empty else d.iloc[0]


# ===================================================================================================== sections
def sec_meta(c: Ctx, R: Registry) -> None:
    g = c.S.get("generated_utc")
    m = re.match(r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})", str(g or ""))
    R.set("NSummaryUTC", f"{m.group(1)} {m.group(2)} UTC" if m else None, "results/summary.json")


def sec_design(c: Ctx, R: Registry) -> None:
    """Design constants, parsed from the configs and code that define them (pending if a pattern stops matching)."""
    regions = [k for k, v in c.regions_cfg.items() if "label" in v]
    R.set("NRegions", str(len(regions)) if regions else None, "region list (configs/regions.yaml)")
    places = {len(v.get("places", [])) for v in c.regions_cfg.values() if "label" in v}
    R.set("NPersonaPlaces", str(places.pop()) if len(places) == 1 else None, "persona places (configs/regions.yaml)")
    reps = const("generation/common.py", r"^N_REPS_FULL = (\d+)")
    conds = const_tuple("generation/common.py", r"^CONDITIONS = \(([^)]*)\)")
    per_call = const("analysis/audit/llm_audit.py", r"^RECIPES_PER_CALL = (\d+)")
    R.set("NReplicates", reps)
    R.set("NConditions", len(conds) or None)
    R.set("NConditionsWord", word(len(conds)) if conds else None)
    R.set("NRecipesPerCall", per_call)
    R.set("NRecipesPerCallWord", word(per_call) if isinstance(per_call, int) else None)
    if regions and conds and reps and per_call:
        calls = len(regions) * len(conds) * reps  # type: ignore[operator]
        R.set("NCallsPerModel", integer(calls))
        R.set("NRecipesPerModel", integer(calls * per_call))  # type: ignore[operator]
        R.set("NRecipesPerRegionModel", integer(len(conds) * reps * per_call))  # type: ignore[operator]
        R.set("NRecipesPerCondition", integer(reps * per_call))  # type: ignore[operator]
        R.set("NPilotCalls", integer(len(regions) * len(conds)))
    for name in ("NCallsPerModel", "NRecipesPerModel", "NRecipesPerRegionModel", "NRecipesPerCondition", "NPilotCalls"):
        if name not in R.m:
            R.set(name, None, "generation design constants")
    mc = load_yaml("configs/models.yaml")
    R.set("NTemperature", dig(mc, "open_weight_sampling", "temperature"))
    R.set("NTopP", dig(mc, "open_weight_sampling", "top_p"))
    R.set("NMaxTokens", integer(dig(mc, "defaults", "max_tokens")))
    sha = code_text("generation/prompt_sets/v1.sha256").strip()
    R.set("NPromptHash", sha[:12] if re.fullmatch(r"[0-9a-f]{64}", sha) else None, "prompt-set hash")
    R.set("NRecipesTarget", integer(const("analysis/summarize.py", r">= (\d[\d_]*) and len\(per\)")))
    # statistics and baselines
    n_rand = None
    rep = load_csv("data/interim/replication/results_v1_keep_empty.csv")
    if rep is not None and "n_null" in rep:
        n_rand = rep["n_null"].max()
    R.set("NNRand", integer(n_rand or const("analysis/audit/rq_tests.py", r"^N_RAND = ([\d_]+)")))
    R.set("NNullCandidates", integer(const("analysis/audit/llm_audit.py", r'"--n-null", type=int, default=([\d_]+)')))
    R.set("NBoot", integer(const("src/flavoraudit/audit.py", r"n_boot: int = (\d+)")))
    R.set("NNearCopy", const("src/flavoraudit/audit.py", r"^NEAR_COPY = ([\d.]+)", float))
    R.set("NSignatureK", const("src/flavoraudit/audit.py", r"k_signature: int = (\d+)"))
    R.set("NLLMN", const("analysis/audit/audit_io.py", r"^LLM_N_PER_REGION = (\d+)"))
    R.set("NRefFpSubsamples", const("analysis/audit/real_subsamples.py", r'"--fingerprint-reps", type=int, default=(\d+)'))
    R.set("NEqReps", const("analysis/audit/equal_n_pairing.py", r'"--reps", type=int, default=(\d+)'))
    grid = [int(g) for g in const_tuple("analysis/audit/equal_n_pairing.py", r"^N_GRID = \(([^)]*)\)") if g.isdigit()]
    R.set("NEqGridMin", min(grid) if grid else None)
    R.set("NEqGridMax", max(grid) if grid else None)
    R.set("NEqGridStep", grid[1] - grid[0] if len(grid) > 1 else None)
    R.set("NMatchedN", const("analysis/audit/equal_n_pairing.py", r"--n (\d+) --tag matched"))
    R.set("NMatchedReps", const("analysis/audit/equal_n_pairing.py", r"--tag matched\d+ --reps (\d+)"))
    R.set("NCMInstances", const("analysis/audit/copy_mutate_baseline.py", r'"--instances", type=int, default=(\d+)'))
    R.set("NCMSeedPool", const("src/flavoraudit/evolution.py", r"m0: int = (\d+)"))
    R.set("NCMMutations", const("src/flavoraudit/evolution.py", r'"CM-R": (\d+)'))
    R.set("NChiTopK", const("analysis/replication/replicate_contributions.py", r"top_k=(\d+)"))
    q = const("analysis/audit/rq_tests.py", r'p_adj"\] >= ([\d.]+)', float)
    global Q
    Q = float(q) if finite(q) else Q  # type: ignore[arg-type]
    R.set("NQ", q)
    R.set("NAliasTestShare", const("analysis/aliasing/evaluate_aliaser.py", r"TEST \((\d+)%\)"))
    # pre-registered replication criteria (docs/IMPLEMENTATION_PLAN.md, Phase 2)
    plan = code_text("docs/IMPLEMENTATION_PLAN.md")
    m = re.search(r"Spearman ρ ≥ ([\d.]+) with the published Z for the random and category nulls, and ≥ ([\d.]+)", plan)
    R.set("NAccRhoStrict", m.group(1) if m else None, "replication criteria (plan)")
    R.set("NAccRhoLoose", m.group(2) if m else None, "replication criteria (plan)")
    # D3 scorer and steering design
    frac = [float(f) for f in const_tuple("steering/train_scorer.py", r"^FRACTIONS = \(([^)]*)\)") if f]
    for nm, f in zip(("NSplitTrain", "NSplitVal", "NSplitTest"), frac or [None] * 3, strict=False):
        R.set(nm, integer(100 * f) if finite(f) else None, "scorer splits")
    R.set("NMaxOther", integer(const("steering/train_scorer.py", r"^MAX_OTHER = (\d+)")))
    R.set("NKappa", num(c.scorer.get("kappa"), 0))
    R.set("NTemp", num(dig(c.scorer, "classifier", "temperature"), 2))
    R.set("NVocab", integer(c.scorer.get("vocab_size")))
    regions_pre = const_tuple("steering/design.py", r"^STEER_REGIONS = \(([^)]*)\)")
    models_pre = const_tuple("steering/design.py", r"^STEER_MODELS = \(([^)]*)\)")
    n_values = [int(v) for v in const_tuple("steering/design.py", r"^N_VALUES = \(([^)]*)\)") if v.isdigit()]
    n_sel = const("steering/design.py", r"^N_SELECTED = (\d+)")
    R.set("NSteerPreRegions", len(regions_pre) or None)
    R.set("NSteerPreModels", join_and([c.name(k) for k in models_pre]) or None)
    R.set("NSteerPreN", ", ".join(str(v) for v in n_values) or None)
    R.set("NSteerPreSelected", n_sel)
    R.set("NSteerPrePool", integer(n_sel * max(n_values)) if n_sel and n_values else None)  # type: ignore[operator]
    R.set("NFeedbackItems", const("steering/design.py", r"^MAX_FEEDBACK_ITEMS = (\d+)"))
    R.set("NRoundsMax", const("steering/run_revise.py", r'"--rounds", type=int, default=(\d+)'))
    # reconstruction of the profiles missing from FlavorDB (flavoraudit.flavor)
    try:
        sys.path.insert(0, str(REPO / "src"))
        import flavoraudit.flavor as fl  # noqa: PLC0415

        coarse, ahn = len(fl.COARSE_PARENTS), len(fl.AHN_INGREDIENTS)
        add = len(fl.ADDITIVE_MOLECULES) + len(fl.NO_PROFILE)
        R.set("NMissingCoarse", coarse)
        R.set("NMissingAhn", ahn)
        R.set("NMissingAdditives", add)
        R.set("NMissingProfiles", coarse + ahn + add)
        R.set("NAhnNames", join_and([tex(v.capitalize()) for v in fl.AHN_INGREDIENTS.values()]))
    except Exception:  # noqa: BLE001 -- any import problem only makes these counts pending
        for nm in ("NMissingCoarse", "NMissingAhn", "NMissingAdditives", "NMissingProfiles", "NAhnNames"):
            R.set(nm, None, "flavor-profile reconstruction constants (flavoraudit.flavor)")


def sec_culinarydb(c: Ctx, R: Registry) -> None:
    """CulinaryDB counts from its release files, and the recipes entering the pairing analysis (summary.json)."""
    base = REPO / "data/raw/culinarydb/extracted"
    note = "CulinaryDB release files (data/raw/culinarydb)"
    try:
        rec = pd.read_csv(base / "01_Recipe_Details.csv", usecols=["Cuisine"])
        ing = pd.read_csv(base / "02_Ingredients.csv", usecols=["Category"])
        comp = pd.read_csv(base / "03_Compound_Ingredients.csv")
        ali = pd.read_csv(base / "04_Recipe-Ingredients_Aliases.csv", usecols=[1])
        misc = rec["Cuisine"].astype(str).str.startswith("Misc")
        R.set("NCdbRecipes", integer(len(rec)))
        R.set("NCdbLabels", rec["Cuisine"].nunique())
        R.set("NCdbMiscLabels", word(int(rec.loc[misc, "Cuisine"].nunique())))
        R.set("NCdbMiscRecipes", integer(int(misc.sum())))
        R.set("NCdbIngredients", integer(len(ing)))
        R.set("NCdbCompound", integer(len(comp)))
        R.set("NCdbEntities", integer(len(ing) + len(comp)))
        R.set("NCdbCategories", ing["Category"].nunique())
        R.set("NCdbRows", integer(len(ali)))
        R.set("NCdbPhrases", integer(ali.iloc[:, 0].nunique()))
    except (OSError, ValueError, KeyError):
        for nm in ("NCdbRecipes", "NCdbLabels", "NCdbMiscLabels", "NCdbMiscRecipes", "NCdbIngredients",
                   "NCdbCompound", "NCdbEntities", "NCdbCategories", "NCdbRows", "NCdbPhrases"):  # fmt: skip
            R.set(nm, None, note)
    n = dig(c.S, "replication", "n_recipes") or {}
    if n:
        lo, hi = min(n, key=n.get), max(n, key=n.get)
        R.set("NPairingRecipesTotal", integer(sum(n.values())))
        R.set("NPairingMin", integer(n[lo]))
        R.set("NPairingMinRegion", tex(lo))
        R.set("NPairingMax", integer(n[hi]))
        R.set("NPairingMaxRegion", tex(hi))
        k = 7
        smallest = sorted(n, key=n.get)[:k]
        R.set("NSmallestK", word(k))
        R.set("NContrastingAmongSmallest", word(sum(r in c.contrasting for r in smallest)))
    else:
        for nm in ("NPairingRecipesTotal", "NPairingMin", "NPairingMinRegion", "NPairingMax", "NPairingMaxRegion",
                   "NSmallestK", "NContrastingAmongSmallest"):  # fmt: skip
            R.set(nm, None, "replication outputs (summary.json)")
    R.set("NContrastingCount", word(len(c.contrasting)) if c.contrasting else None, "published signs")
    R.set("NUniformCount", len(c.uniform) or None, "published signs")


def sec_dataset(c: Ctx, R: Registry) -> dict[str, str]:
    ds = dig(c.S, "dataset") or {}
    models = ds.get("models") or {}
    R.set("NRecipesTotal", integer(ds.get("total_recipes")), "LLMCuisine-22 recipes")
    R.set("NCallsTotal", integer(ds.get("total_calls")), "LLMCuisine-22 calls")
    present = c.present
    R.set("NModelsDone", word(len(present)) if present else None, "LLM runs")
    R.set("NModelsDoneNum", len(present) or None, "LLM runs")
    R.set("NModelNamesDone", join_and([c.name(k) for k in present]) or None, "LLM runs")
    complete = [k for k in present if dig(models, k, "complete")]
    R.set("NModelsComplete", word(len(complete)))
    R.set("NModelNamesComplete", join_and([c.name(k) for k in complete]) or "none")
    R.set("NOpenFamiliesDone", word(int(ds.get("n_open_families") or 0)))
    R.set("NProprietaryDone", word(int(ds.get("n_proprietary") or 0)))
    regions_total = R.m.get("NRegions", "22")
    partial = []
    for k in present:
        if not dig(models, k, "complete"):
            v = models[k]
            partial.append(
                f"{c.name(k)} has returned {integer(v.get('calls'))} of its {integer(v.get('planned_calls'))} calls "
                f"({integer(v.get('regions'))} of {regions_total} regions; {tex(STATUS.get(k, 'run in progress'))})"
            )
    R.set("NPartialText", (join_and(partial) + ".") if partial else "All runs that have started are complete.")
    for k in present:
        if not dig(models, k, "complete"):
            R.note(f"{c.name(k)}: {tex(STATUS.get(k, 'run in progress'))}")
    missing = [k for k in c.roster if k not in models]
    core_missing = [k for k in missing if dig(c.models_cfg, k, "tier") == "core"]
    extra_missing = [k for k in missing if dig(c.models_cfg, k, "tier") != "core"]
    for k in missing:
        R.note(f"{c.name(k)}: {tex(STATUS.get(k, 'not run yet'))}")
    R.set(
        "NMissingCoreText",
        join_and([rf"\pending{{{c.name(k)}: {tex(STATUS.get(k, 'not run yet'))}}}" for k in core_missing])
        or "none",
    )
    R.set("NMissingExtraText", join_and([c.name(k) for k in extra_missing]) or "none")
    if ds.get("meets_proposal"):
        R.set("NMeetsProposal", "The D1 targets of the proposal are met.")
    else:
        R.set(
            "NMeetsProposal",
            rf"\pending{{D1 targets ({R.m.get('NRecipesTarget', '10{,}000')} recipes; two open-weight families): "
            rf"{integer(ds.get('total_recipes')) or '--'} recipes and {word(int(ds.get('n_open_families') or 0))} "
            r"open-weight family so far}",
        )
        R.note("D1 targets not yet met (waiting for the Llama-3.1-8B run)")
    # ranges over models with data
    def rng_of(key: str, nd: int = 1, as_pct: bool = True) -> str | None:
        return span([dig(models, k, key) for k in present], nd, as_pct)

    R.set("NExactFiveRange", rng_of("exactly5_share"))
    R.set("NParseOkRange", rng_of("parse_ok_share"))
    R.set("NParseAnyRange", rng_of("parse_any_share"))
    R.set("NAliasMatchedRange", rng_of("alias_matched_share"))
    R.set("NAliasFullRange", rng_of("alias_full_share"))
    R.set("NLinesPerRecipeRange", rng_of("lines_per_recipe", 1, False))
    for k in c.roster:
        s = stem(k)
        v = models.get(k)
        note = f"{c.name(k)}: {tex(STATUS.get(k, 'not run yet'))}"
        R.set(f"N{s}Recipes", integer(dig(v, "recipes")) if v else None, note)
        R.set(f"N{s}Calls", integer(dig(v, "calls")) if v else None, note)
        R.set(f"N{s}Regions", integer(dig(v, "regions")) if v else None, note)
    # resources
    res = dig(c.S, "resources") or {}
    R.set("NOpenAISpend", num(res.get("openai_usd_spent"), 2), "OpenAI spend")
    R.set("NBudget", num(res.get("budget_usd"), 0), "budget")
    R.set("NKaggleQwenMinutes", integer(res.get("kaggle_qwen_full_minutes")), "Kaggle run time")
    for key, st in (("kaggle_qwen_full_minutes", "Qwen"), ("kaggle_llama_full_minutes", "Llama")):
        v = res.get(key)
        R.set(f"NKaggle{st}Hours", num(float(v) / 60, 1) if finite(v) else None, f"Kaggle {st} run time")
    runs = res.get("kaggle_runs") or {}
    full = {k: v for k, v in runs.items() if k.endswith("-full")}
    if full and all(v.get("vllm_failed") for v in full.values()):
        R.set("NKaggleEngineText", "every Kaggle run used the Hugging Face transformers fallback after vLLM failed on the T4")
    elif full:
        R.set("NKaggleEngineText", "the engine (vLLM or transformers fallback) is logged per call")
    else:
        R.set("NKaggleEngineText", None, "Kaggle run logs")
    return {"tab_dataset.tex": table_dataset(c, R)}


def access_of(c: Ctx, key: str) -> str:
    cfg = c.models_cfg.get(key, {})
    if cfg.get("runner") == "kaggle":
        return "Kaggle GPU"
    return {"openai": "OpenAI Batch API", "gemini": "Gemini API (free)", "groq": "Groq API"}.get(
        cfg.get("provider", ""), "API"
    )


def table_dataset(c: Ctx, R: Registry) -> str:
    models = dig(c.S, "dataset", "models") or {}
    rows = []
    for k in c.roster:
        v = models.get(k)
        fam = tex(dig(v, "family") or c.models_cfg.get(k, {}).get("family", ""))
        head = f"{c.name(k)} & {fam} & {tex(access_of(c, k))}"
        if not v:
            rows.append(head + rf" & \multicolumn{{8}}{{l}}{{\pending{{{tex(STATUS.get(k, 'not run yet'))}}}}} \\")
            continue
        rows.append(
            head
            + f" & {integer(v.get('calls'))}/{integer(v.get('planned_calls'))} & {integer(v.get('recipes'))}"
            + f" & {integer(v.get('regions'))} & {pct(v.get('exactly5_share'))} & {pct(v.get('parse_ok_share'))}"
            + f" & {pct(v.get('parse_any_share'))} & {num(v.get('lines_per_recipe'), 1)}"
            + f" & {pct(v.get('alias_matched_share'))} \\\\"
        )
    ds = dig(c.S, "dataset") or {}
    total = rf"\midrule Total & & & {integer(ds.get('total_calls')) or '--'} & {integer(ds.get('total_recipes')) or '--'} & & & & & & \\"  # noqa: E501
    return "\n".join(
        [
            r"\begin{tabular}{@{}lllrrrrrrrr@{}}",
            r"\toprule",
            r"Model & Family & Access & Calls & Recipes & Regions & \makecell{Exactly\\5 recipes} & "
            r"\makecell{Parsed\\(strict)} & \makecell{Parsed after\\repair} & \makecell{Lines/\\recipe} & "
            r"\makecell{Lines\\aliased} \\",
            r"\midrule",
            *rows,
            total,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def sec_replication(c: Ctx, R: Registry) -> dict[str, str]:
    by = dig(c.S, "replication", "by_null") or {}
    for null, st in NULL_STEMS.items():
        v = by.get(null) or {}
        R.set(f"NRepl{st}Signs", v.get("sign_agreement"), "replication")
        R.set(f"NRepl{st}N", v.get("n"), "replication")
        R.set(f"NRepl{st}Rho", num(v.get("spearman"), 2), "replication")
    R.set("NReplRhoRange", span([v.get("spearman") for v in by.values()], 2), "replication")
    kor = dig(c.S, "replication", "korea_top3") or []
    pub_kor = dig(c.targets, "top3_contribution_percent", "negative", "KOR") or {}
    for i, label in enumerate(("One", "Two", "Three")):
        if i < len(kor):
            R.set(f"NKoreaTop{label}", tex(kor[i]["name"]))
            R.set(f"NKoreaTop{label}Chi", num(kor[i]["chi_percent"], 1))
            R.set(f"NKoreaTop{label}Pub", num(pub_kor.get(kor[i]["name"]), 2), "published Korea contribution")
        else:
            for suffix in ("", "Chi", "Pub"):
                R.set(f"NKoreaTop{label}{suffix}", None, "Korea contributions")
    cs = load_csv("data/interim/replication/contributions_summary.csv")
    row = None if cs is None else cs[cs["procedure"] == "ns_ingredient_noadditive"]
    ok = row is not None and not row.empty
    R.set("NContribTopOne", tex(row.iloc[0]["published_top1_in_our_top3"]) if ok else None, "contribution replication")
    R.set("NContribTopThree", tex(row.iloc[0]["top3_overlap"]) if ok else None, "contribution replication")
    z = dig(c.S, "replication", "z_random") or {}
    pub = dig(c.targets, "z_vs_null", "random") or {}
    for region in ("Korea", "Scandinavia", "Italy"):
        code = c.code_of.get(region, "")
        s = stem(region)
        R.set(f"NZ{s}Ours", num(z.get(region), 1), "replication Z")
        R.set(f"NZ{s}Pub", num(pub.get(code), 0), "published Z")
    d = dig(c.S, "replication", "d_random") or {}
    R.set("NDRandomRange", span(d.values(), 2), "replication d")
    return {"tab_replication.tex": table_replication(c, R)}


def table_replication(c: Ctx, R: Registry) -> str:
    by = dig(c.S, "replication", "by_null") or {}
    crit = {
        "random": rf"all signs, $\rho\geq{R.m.get('NAccRhoStrict', '--')}$",
        "frequency": rf"$\rho\geq{R.m.get('NAccRhoLoose', '--')}$",
        "category": rf"$\rho\geq{R.m.get('NAccRhoStrict', '--')}$",
        "frequency_category": rf"$\rho\geq{R.m.get('NAccRhoLoose', '--')}$",
    }
    rows = []
    for null, label in NULL_LABELS.items():
        v = by.get(null) or {}
        signs = f"{v.get('sign_agreement')}/{v.get('n')}" if v else r"\pending{replication}"
        rows.append(f"{label} & {signs} & {num(v.get('spearman'), 3) or '--'} & {crit[null]} \\\\")
    rows.append(
        r"\midrule \multicolumn{4}{@{}p{0.97\columnwidth}@{}}{Contributions ($\chi\%$): published top-1 in our top-3 for "
        + f"{R.m.get('NContribTopOne')} regions; top-3 overlap {R.m.get('NContribTopThree')}"
        + r"} \\"
    )
    return "\n".join(
        [
            r"\begin{tabular}{@{}lccl@{}}",
            r"\toprule",
            r"Null model & Signs & Spearman $\rho$ & Pre-registered criterion \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def sec_equal_n(c: Ctx, R: Registry) -> dict[str, str]:
    e = dig(c.S, "equal_n") or {}
    note = "equal-N reference (analysis/audit/real_subsamples.py, equal_n_pairing.py)"
    R.set("NFPRatePct", pct(e.get("n160_sign_rule_false_positive_rate")), note)
    fp = {} if c.cal is None else dict(zip(c.cal["region"], c.cal["sign_rule_false_positive"], strict=True))
    for region in ("Canada", "China", "France", "South East Asia", "USA"):
        R.set(f"NFP{stem(region)}Pct", pct(fp.get(region), 0), note)
    match = e.get("n160_sign_match") or {}
    R.set("NRefSignMatchPct", pct(sum(match.values()) / len(match), 0) if match else None, note)
    for region in ("Canada", "China", "France"):
        R.set(f"NMatch{stem(region)}Pct", pct(match.get(region), 0), note)
    rq_real = c.rq_rows("H1", "false-positive")
    n_real = int(rq_real.iloc[0]["n"]) if not rq_real.empty else None
    R.set("NRefSamplesTotal", integer(n_real), note)
    nreg = len(match) or None
    R.set("NRefSubsamples", integer(n_real / nreg) if n_real and nreg else None, note)
    usa = e.get("usa_pool_effect") or {}
    R.set("NUsaOwnPool", num(usa.get("delta_ns_own_pool_n160"), 2), note)
    R.set("NUsaFullPool", num(usa.get("delta_ns_full_pool_n160"), 2), note)
    R.set("NUsaPool", integer(usa.get("pool_n160")), note)
    rep = load_csv("data/interim/replication/results_v1_keep_empty.csv")
    usa_full = None
    if rep is not None:
        r = rep[(rep["region"] == "USA") & (rep["null_model"] == "random")]
        usa_full = None if r.empty else r.iloc[0]["delta_ns"]
    R.set("NUsaFullDeltaNs", num(usa_full, 2), "replication outputs")
    share = e.get("matched290_positive_share") or {}
    k_c = e.get("contrasting_robust")
    R.set("NContrastingRobust", word(int(k_c)) if finite(k_c) else None, note)
    R.set(
        "NContrastingRobustText",
        (f"all {word(len(c.contrasting))}" if k_c == len(c.contrasting) else f"{word(int(k_c))} of the {word(len(c.contrasting))}")  # noqa: E501
        if finite(k_c) and c.contrasting
        else None,
        note,
    )
    R.set("NUniformRobust", word(int(e["uniform_robust"])) if finite(e.get("uniform_robust")) else None, note)
    flipped = e.get("uniform_flipped") or []
    R.set("NUniformFlipped", join_and([tex(f) for f in flipped]) or "none", note)
    R.set("NUniformFlippedCount", word(len(flipped)), note)
    R.set("NUniformFlippedShares", join_and([pct(share.get(f), 0) for f in flipped]) or "--", note)
    mid = sorted(
        (r for r in c.uniform if r in share and r not in flipped and share[r] < 0.95), key=lambda r: -share[r]
    )
    R.set("NUniformMiddle", join_and([tex(r) for r in mid]) or "none", note)
    R.set("NUniformMiddleRange", span([share[r] for r in mid], 0, True) or "--", note)
    med = e.get("n160_reference_medians") or {}
    R.set("NRefPrevJSD", num(med.get("prevalence_jsd"), 3), note)
    R.set("NRefCatJSD", num(med.get("category_jsd"), 4), note)
    R.set("NRefSAI", num(med.get("sai"), 2), note)
    R.set("NRefFpRho", num(med.get("fingerprint_spearman"), 2), note)
    R.set("NRefFreqMAE", num(med.get("freq_rank_mae"), 4), note)
    R.set("NRefNearCopyPct", pct(med.get("near_copy_rate"), 1), note)
    band = e.get("n160_sai_band") or [None, None]
    R.set("NRefSAIBandLo", num(band[0], 2), note)
    R.set("NRefSAIBandHi", num(band[1], 2), note)
    return {"tab_regions.tex": table_regions(c)}


def table_regions(c: Ctx) -> str:
    z = dig(c.S, "replication", "z_random") or {}
    n = dig(c.S, "replication", "n_recipes") or {}
    codes = dig(c.targets, "region_codes") or {}
    pub = {codes.get(k, k): v for k, v in (dig(c.targets, "z_vs_null", "random") or {}).items()}
    e = dig(c.S, "equal_n") or {}
    m160 = e.get("n160_sign_match") or {}
    pos = e.get("matched290_positive_share") or {}
    fp = {} if c.cal is None else dict(zip(c.cal["region"], c.cal["sign_rule_false_positive"], strict=True))
    rows = []
    for region in c.region_order:
        mark = r"$^\dagger$" if region in c.contrasting else ""
        rows.append(
            f"{c.code(region)}{mark} & {integer(n.get(region)) or '--'} & {num(z.get(region), 1) or '--'}"
            f" & {num(pub.get(region), 0) or '--'} & {pct(m160.get(region), 0) or '--'}"
            f" & {pct(fp.get(region), 1) or '--'} & {pct(pos.get(region), 0) or '--'} \\\\"
        )
    return "\n".join(
        [
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            r"& & \multicolumn{2}{c}{$Z$ vs random null} & \multicolumn{2}{c}{160 recipes} & 290 \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-7}",
            r"Region & $N$ & ours & publ. & \makecell{sign\\match} & \makecell{rule\\fires} & \makecell{share\\$d>0$} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def sec_copy_mutate(c: Ctx, R: Registry) -> None:
    cm = dig(c.S, "copy_mutate") or {}
    note = "copy-mutate baseline (analysis/audit/copy_mutate_baseline.py)"
    n = R.m.get("NLLMN", "160")
    for key, st in ((f"CM-R/ranked|n={n}", "CM"), (f"CM-R/random|n={n}", "CMRand"), ("CM-R/ranked|full", "CMFull")):
        v = cm.get(key) or {}
        R.set(f"N{st}SignMatchPct", pct(v.get("sign_match"), 0), note)
        R.set(f"N{st}SignErrorPct", pct(v.get("sign_error"), 0), note)
        R.set(f"N{st}PrevJSD", num(v.get("prevalence_jsd"), 2), note)
        R.set(f"N{st}SAI", num(v.get("sai"), 2), note)
        R.set(f"N{st}FreqMAE", num(v.get("freq_rank_mae"), 3), note)
        R.set(f"N{st}FpRho", num(v.get("fingerprint_spearman"), 2), note)
        R.set(f"N{st}NearCopyPct", pct(v.get("near_copy_rate"), 1), note)


def sec_audit(c: Ctx, R: Registry) -> dict[str, str]:
    a = dig(c.S, "llm_audit") or {}
    a = {} if "pending" in a else a
    M = c.audited
    note = "LLM audit (analysis/audit/llm_audit.py)"
    R.set("NAuditModels", word(len(M)) if M else None, note)
    R.set("NAuditModelNames", join_and([c.name(k) for k in M]) or None, note)

    def vals(key: str) -> list:
        return [dig(a, k, key) for k in M]

    R.set("NSignAgreeRange", span(vals("sign_agreement"), 0, True), note)
    R.set("NPrevJSDRange", span(vals("median_prevalence_jsd"), 2), note)
    R.set("NCatJSDRange", span(vals("median_category_jsd"), 3), note)
    R.set("NSAIRange", span(vals("median_sai"), 2), note)
    R.set("NFpRhoRange", span(vals("median_fingerprint_spearman"), 2), note)
    R.set("NDivRange", span(vals("median_diversity_ratio"), 2), note)
    R.set("NDivRangePct", span(vals("median_diversity_ratio"), 0, True), note)
    R.set("NFreqMAERange", span(vals("median_freq_rank_mae"), 3), note)
    R.set("NNearCopyRangePct", span(vals("mean_near_copy_rate"), 0, True), note)
    R.set("NNoveltyRange", span(vals("median_novelty"), 2), note)
    errs = [int(dig(a, k, "sign_errors_preregistered") or 0) for k in M]
    regs = [int(dig(a, k, "regions") or 0) for k in M]
    R.set("NSignErrTotal", sum(errs) if M else None, note)
    R.set("NSignErrRegionsTotal", sum(regs) if M else None, note)
    R.set(
        "NSignErrText",
        join_and([f"{e} of {r} ({c.name(k)})" for e, r, k in zip(errs, regs, M, strict=True)]) or None,
        note,
    )
    # expected false positives of the rule under genuine sampling, for the regions each model covers
    if c.cal is not None and c.h1 is not None and M:
        fp = dict(zip(c.cal["region"], c.cal["sign_rule_false_positive"], strict=True))
        exp = [sum(fp.get(r, 0.0) for r in c.h1.loc[c.h1["model"] == k, "region"]) for k in M]
        R.set(
            "NSignErrExpectedText",
            join_and([f"{num(x, 1)} ({c.name(k)})" for x, k in zip(exp, M, strict=True)]),
        )
    else:
        R.set("NSignErrExpectedText", None, note)
    errs_c = [int(dig(a, k, "sign_errors_in_contrasting") or 0) for k in M]
    R.set("NSignErrContrastingTotal", sum(errs_c) if M else None, note)
    # equal-N verdicts
    tu = [int(dig(a, k, "equal_n_too_uniform") or 0) for k in M]
    tc = [int(dig(a, k, "equal_n_too_contrasting") or 0) for k in M]
    ty = [int(dig(a, k, "equal_n_typical") or 0) for k in M]
    R.set("NEqCuisinesTotal", sum(tu) + sum(tc) + sum(ty) if M else None, note)
    R.set("NEqTypicalTotal", sum(ty) if M else None, note)
    R.set("NEqUniformTotal", sum(tu) if M else None, note)
    R.set("NEqContrastingTotal", sum(tc) if M else None, note)
    R.set(
        "NEqPerModelText",
        join_and(
            [f"{u} too uniform and {t} too contrasting of {r} ({c.name(k)})" for u, t, r, k in zip(tu, tc, regs, M, strict=True)]  # noqa: E501
        )
        or None,
        note,
    )
    R.set("NEqShiftRange", span(vals("equal_n_median_shift_d"), 2), note)
    if c.h1 is not None and len(M) >= 2:
        same: dict[str, list[str]] = {"too uniform": [], "too contrasting": []}
        for region, g in c.h1.groupby("region"):
            v = set(g["verdict"])
            if len(g) >= 2 and len(v) == 1 and next(iter(v)) in same:
                same[next(iter(v))].append(tex(region))
        parts = []
        if same["too uniform"]:
            parts.append(f"{join_and(same['too uniform'])} {'was' if len(same['too uniform']) == 1 else 'were'} too uniform")  # noqa: E501
        if same["too contrasting"]:
            parts.append(f"{join_and(same['too contrasting'])} {'was' if len(same['too contrasting']) == 1 else 'were'} too contrasting")  # noqa: E501
        R.set(
            "NUnanimousText",
            (", and ".join(parts) + ", for every model that covers them")
            if parts
            else "no region received the same non-typical verdict from every model that covers it",
        )
    else:
        R.set("NUnanimousText", None, "equal-N verdicts of at least two models")
    # model-specific macros (only for audited models; pending otherwise)
    for k in c.roster:
        s = stem(k)
        v = a.get(k)
        miss = f"{c.name(k)}: {tex(STATUS.get(k, 'not run yet'))}"
        R.set(f"N{s}AudRegions", integer(dig(v, "regions")) if v else None, miss)
        R.set(f"N{s}SignAgreePct", pct(dig(v, "sign_agreement"), 0) if v else None, miss)
        R.set(f"N{s}SignErr", dig(v, "sign_errors_preregistered") if v else None, miss)
        R.set(f"N{s}EqContrasting", dig(v, "equal_n_too_contrasting") if v else None, miss)
        R.set(f"N{s}EqUniform", dig(v, "equal_n_too_uniform") if v else None, miss)
    return {"tab_scorecard.tex": table_scorecard(c, R), "tab_verdicts.tex": table_verdicts(c)}


def table_scorecard(c: Ctx, R: Registry) -> str:
    a = dig(c.S, "llm_audit") or {}
    e = dig(c.S, "equal_n") or {}
    med = e.get("n160_reference_medians") or {}
    n = R.m.get("NLLMN", "160")
    cm = dig(c.S, "copy_mutate", f"CM-R/ranked|n={n}") or {}
    match = e.get("n160_sign_match") or {}
    M = c.audited
    cols = len(M) + 2

    def model_cells(fn: Callable[[dict], str | None]) -> str:
        return " & ".join((fn(a.get(k) or {}) or "--") for k in M)

    def row(label: str, real: str | None, cmv: str | None, fn: Callable[[dict], str | None]) -> str:
        return f"{label} & {real or '--'} & {cmv or '--'} & {model_cells(fn)} \\\\"

    ref_sign = pct(sum(match.values()) / len(match), 0) if match else None

    def err_rate(v: dict) -> str | None:
        if not v:
            return None
        return f"{pct(v['sign_errors_preregistered'] / v['regions'], 0)} ({v['sign_errors_preregistered']}/{v['regions']})"

    def verdicts(v: dict) -> str | None:
        if not v or "equal_n_typical" not in v:
            return None
        return f"{v['equal_n_too_uniform']} / {v['equal_n_typical']} / {v['equal_n_too_contrasting']}"

    rows = [
        row("Regions (recipes)", None, None, lambda v: f"{v.get('regions')} ({integer(v.get('recipes'))})" if v else None),
        r"\midrule \multicolumn{" + str(cols + 1) + r"}{@{}l}{\emph{Food pairing (RQ1)}} \\",
        row("Sign agrees with real cuisine", ref_sign, pct(cm.get("sign_match"), 0), lambda v: pct(v.get("sign_agreement"), 0)),  # noqa: E501
        row("Pre-registered sign-error rate", pct(e.get("n160_sign_rule_false_positive_rate"), 1), pct(cm.get("sign_error"), 0), err_rate),  # noqa: E501
        row(r"Equal-$N$: too uniform / typical / too contrasting", None, None, verdicts),
        row(r"Median shift $d-\tilde d_{\mathrm{genuine}}$", None, None, lambda v: num(v.get("equal_n_median_shift_d"), 2)),  # noqa: E501
        r"\midrule \multicolumn{" + str(cols + 1) + r"}{@{}l}{\emph{Composition, stereotypes and fingerprints (RQ2)}} \\",  # noqa: E501
        row("Prevalence JSD", num(med.get("prevalence_jsd"), 3), num(cm.get("prevalence_jsd"), 3), lambda v: num(v.get("median_prevalence_jsd"), 3)),  # noqa: E501
        row("Category-composition JSD", num(med.get("category_jsd"), 4), None, lambda v: num(v.get("median_category_jsd"), 3)),  # noqa: E501
        row("Frequency-rank MAE", num(med.get("freq_rank_mae"), 4), num(cm.get("freq_rank_mae"), 3), lambda v: num(v.get("median_freq_rank_mae"), 3)),  # noqa: E501
        row("SAI", num(med.get("sai"), 2), num(cm.get("sai"), 2), lambda v: num(v.get("median_sai"), 2)),
        row(r"Fingerprint $\rho$", num(med.get("fingerprint_spearman"), 2), num(cm.get("fingerprint_spearman"), 2), lambda v: num(v.get("median_fingerprint_spearman"), 2)),  # noqa: E501
        row("Diversity ratio", None, None, lambda v: num(v.get("median_diversity_ratio"), 2)),
        r"\midrule \multicolumn{" + str(cols + 1) + r"}{@{}l}{\emph{Novelty}} \\",
        row(r"Near-copy rate (max.\ Jaccard $\geq$" + str(R.m.get("NNearCopy", "0.8")) + ")", pct(med.get("near_copy_rate"), 1), pct(cm.get("near_copy_rate"), 1), lambda v: pct(v.get("mean_near_copy_rate"), 1)),  # noqa: E501
        row("Mean max.\\ Jaccard to CulinaryDB", None, None, lambda v: num(v.get("median_novelty"), 2)),
    ]
    heads = " & ".join(head(c.name(k)) for k in M)
    return "\n".join(
        [
            r"\begin{tabular}{@{}l" + "r" * cols + r"@{}}",
            r"\toprule",
            rf"Metric & \makecell{{Genuine\\@{n}}} & \makecell{{Copy-mutate\\@{n}}} & {heads} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def table_verdicts(c: Ctx) -> str:
    if c.h1 is None or c.h1.empty:
        return r"\pending{equal-$N$ verdicts (tables/h1\_equal\_n\_full.csv)}"
    M = [k for k in c.audited if k in set(c.h1["model"])] or order(set(c.h1["model"]))
    rows = []
    for region in c.region_order:
        g = c.h1[c.h1["region"] == region]
        if g.empty:
            continue
        ref = g.sort_values("n_ref", ascending=False).iloc[0]
        mark = r"$^\dagger$" if region in c.contrasting else ""
        cells = []
        for k in M:
            r = g[g["model"] == k]
            if r.empty:
                cells.append(r"\textcolor{gray}{--}")
                continue
            r = r.iloc[0]
            d = num(r["d"], 2)
            if r["verdict"] == "too uniform":
                cells.append(rf"\cellcolor{{TooUniform!30}}{d}\,$\uparrow$")
            elif r["verdict"] == "too contrasting":
                cells.append(rf"\cellcolor{{TooContrasting!25}}{d}\,$\downarrow$")
            else:
                cells.append(f"{d}")
        rows.append(
            f"{c.code(region)}{mark} & {num(ref['d_ref_median'], 2)} [{num(ref['d_ref_lo'], 2)}, {num(ref['d_ref_hi'], 2)}]"
            + " & "
            + " & ".join(cells)
            + r" \\"
        )
    heads = " & ".join(head(c.name(k)) for k in M)
    return "\n".join(
        [
            r"\begin{tabular}{@{}ll" + "r" * len(M) + r"@{}}",
            r"\toprule",
            rf"Region & \makecell{{Genuine $d$: median\\{{[}}95\% range{{]}}}} & {heads} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def sec_drivers(c: Ctx, R: Registry) -> None:
    """RQ2a: top-5 chi% contributors of LLM cuisines vs the real cuisine (summary.json "drivers")."""
    d = dig(c.S, "drivers") or {}
    note = "RQ2a driver comparison (summary.json drivers)"
    M = [k for k in c.audited if isinstance(d.get(k), dict)]
    ref = d.get("real@160") or {}
    R.set("NDrvJaccardRange", span([d[k].get("median_jaccard5") for k in M], 2), note)
    R.set("NDrvRhoRange", span([d[k].get("median_spearman") for k in M], 2), note)
    R.set("NDrvPerModelText", join_and([f"{num(d[k].get('median_jaccard5'), 2)} and {num(d[k].get('median_spearman'), 2)} ({c.name(k)})" for k in M]) or None, note)  # noqa: E501
    R.set("NDrvRefJaccard", num(ref.get("median_jaccard5"), 2), note)
    R.set("NDrvRefRho", num(ref.get("median_spearman"), 2), note)
    R.set("NDrvRefReps", integer(ref.get("reps_per_region")), note)
    neg = [k for k in M if finite(d[k].get("median_spearman")) and d[k]["median_spearman"] < 0]
    caveat = r"$\chi\%$ is oriented by each cuisine's own $Z$ sign, so a model whose pairing sign differs from the real cuisine's can have a negative rank correlation"  # noqa: E501
    R.set("NDrvNegText", (f"The negative median for {join_and([c.name(k) for k in neg])} reflects that " + caveat + ".") if neg else ("Note that " + caveat + "."))  # noqa: E501
    real5 = d.get("korea_real_top5") or []
    R.set("NKoreaRealTopFive", join_and([tex(x) for x in real5]) or None, note)
    if real5:
        top = real5[0]
        first, other, absent, nodata = [], [], [], []
        for k in M:
            t5 = d[k].get("korea_top5")
            if not t5:
                nodata.append(c.name(k))
            elif t5[0] == top:
                first.append(c.name(k))
            elif top in t5:
                other.append(f"{c.name(k)} (rank {t5.index(top) + 1})")
            else:
                absent.append(c.name(k))
        bits = []
        if first:
            bits.append(f"first for {join_and(first)}")
        if other:
            bits.append(f"in the top five for {join_and(other)}")
        if absent:
            bits.append(f"absent from the top five for {join_and(absent)}")
        tail = f" ({join_and(nodata)}: no Korean recipes yet)" if nodata else ""
        R.set("NDrvKoreaText", f"Korea's top real contributor, {tex(top)}, is {join_and(bits)}{tail}")
    else:
        R.set("NDrvKoreaText", None, note)


NET_METRICS = [  # (summary key, label, digits, as percent)
    ("npmi_top100", r"\makecell[l]{NPMI top-100,\\Jaccard (primary)}", 2, False),
    ("npmi_top50", r"\makecell[l]{NPMI top-50,\\Jaccard}", 2, False),
    ("backbone", r"\makecell[l]{Backbone,\\Jaccard}", 2, False),
    ("backbone_overlap", r"\makecell[l]{Backbone overlap\\$|A\cap B|/|A|$}", 2, False),
    ("backbone_edges", r"\makecell[l]{Backbone\\edges}", 0, False),
]


def sec_networks(c: Ctx, R: Registry) -> dict[str, str]:
    """Co-occurrence networks (plan 6.5; analysis/audit/networks.py), from summary.json "networks"."""
    net = dig(c.S, "networks") or {}
    note = "co-occurrence networks (analysis/audit/networks.py)"
    M = [k for k in c.audited if isinstance(dig(net, "npmi_top100", k), dict)]
    R.set("NNetDraws", integer(net.get("n_draws")), note)
    for key, st in (("npmi_top100", "Hundred"), ("npmi_top50", "Fifty"), ("backbone", "Bb"), ("backbone_overlap", "BbOverlap"), ("backbone_edges", "BbEdges")):  # noqa: E501
        d = net.get(key) or {}
        nd = 0 if key == "backbone_edges" else 2
        pct_ = key == "backbone_overlap"
        R.set(f"NNet{st}Range", span([dig(d, k, "median") for k in M], 0 if pct_ else nd, pct_), note)
        ref = dig(d, "real@160", "median")
        R.set(f"NNet{st}Ref", (pct(ref, 0) if pct_ else num(ref, nd)), note)
        R.set(f"NNet{st}BelowText", join_and([f"{dig(d, k, 'below_reference_fdr05')} of {dig(d, k, 'regions')} ({c.name(k)})" for k in M]) or None, note)  # noqa: E501
    edges = [dig(net, "backbone_edges", k, "median") for k in M]
    ref_e = dig(net, "backbone_edges", "real@160", "median")
    R.set("NNetSparser", "sparser than" if edges and finite(ref_e) and all(finite(e) and e < ref_e for e in edges) else "of similar size to", note)  # noqa: E501
    rows = []
    for key, label, nd, _ in NET_METRICS:
        d = net.get(key) or {}
        cells = []
        for k in M:
            v = d.get(k) or {}
            val = num(v.get("median"), nd) or "--"
            cells.append(rf"\makecell{{{val}\\({v.get('below_reference_fdr05', '--')}/{v.get('regions', '--')})}}")
        rows.append(f"{label} & {num(dig(d, 'real@160', 'median'), nd) or '--'} & " + " & ".join(cells) + r" \\")
    heads = " & ".join(head(c.name(k)) for k in M)
    body = "\n".join([r"\begin{tabular}{@{}l" + "c" * (len(M) + 1) + r"@{}}", r"\toprule",
                      rf"Metric & \makecell{{Genuine\\@{R.m.get('NLLMN', '160')}}} & {heads} \\", r"\midrule",
                      *rows, r"\bottomrule", r"\end{tabular}"]) if M else r"\pending{co-occurrence networks}"  # noqa: E501
    return {"tab_networks.tex": body}


def status_table(c: Ctx, R: Registry) -> str:
    """Runs not completed at submission (neutral wording; derived from summary.json where possible)."""
    ds = dig(c.S, "dataset", "models") or {}
    rows = []
    for k in c.roster:
        v = ds.get(k)
        if v and v.get("complete"):
            continue
        if v:
            what = f"{integer(v.get('regions'))} regions analysed ({integer(v.get('recipes'))} recipes)"
        else:
            what = "planned extension"
        role = "scale contrast (RQ4)" if k == "qwen3.5-4b" else ("third open-weight family" if k == "ministral-3-8b" else "generation")  # noqa: E501
        rows.append(f"{c.name(k)}, {role} & {what} & {tex(STATUS.get(k, 'extension'))} \\\\")
    bon = dig(c.S, "steering", "bon_main") or []
    rev = dig(c.S, "steering", "revise_main") or []
    bon_models = {r.get("model") for r in bon if isinstance(r, dict)}
    rev_rounds: dict[str, int] = {}
    for r in rev:
        if isinstance(r, dict):
            rev_rounds[r["model"]] = max(rev_rounds.get(r["model"], 0), int(r.get("round", 0)))
    no_bon = [k for k in c.audited if k not in bon_models]
    if bon:
        rows.append(f"Best-of-$N$ steering & {join_and([c.name(k) for k in c.audited if k in bon_models])} & "
                    f"{'extension: ' + join_and([c.name(k) for k in no_bon]) if no_bon else '--'} \\\\")  # fmt: skip
    if rev:
        rows.append(f"Critique-and-revise & {join_and([f'{c.name(k)} ({r} round' + ('s)' if r != 1 else ')') for k, r in rev_rounds.items()])} & "  # noqa: E501
                    f"{'extension: other models' if [k for k in c.audited if k not in rev_rounds] else '--'} \\\\")  # fmt: skip
    rows.append(r"Ingredient aliasing & automatic CulinaryDB protocol (Appendix~\ref{app:alias}) & manual curation of rare terms planned \\")  # noqa: E501
    rows.append(r"Distribution of D1, D4, D5 & packaged and tested locally & public release planned \\")
    return "\n".join([r"\begin{tabular}{@{}p{0.31\columnwidth}p{0.33\columnwidth}p{0.24\columnwidth}@{}}", r"\toprule",
                      r"Item & Analysed here & Next \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])  # noqa: E501


# ------------------------------------------------------------------------------------------------- RQ tests
def support(c: Ctx, hyp: str, sub: str, good: Callable[[float], bool] = lambda e: True) -> dict:
    """Which audited models reach BH-adjusted p < q for one test (and the effect has the stated direction)."""
    rows = [(k, c.rq_row(hyp, sub, k)) for k in c.audited]
    rows = [(k, r) for k, r in rows if r is not None]
    sig = [k for k, r in rows if finite(r["p_adj"]) and r["p_adj"] < Q and good(r["effect_num"])]
    return {
        "k": len(sig),
        "n": len(rows),
        "sig": sig,
        "phrase": count_phrase(len(sig), len(rows), [c.name(k) for k in sig]),
        "p": [r["p"] for _, r in rows],
        "padj": [r["p_adj"] for _, r in rows],
        "eff": [r["effect_num"] for _, r in rows],
        "rows": rows,
    }


def frac_pair(s: str) -> tuple[int, int, int, int] | None:
    m = re.fullmatch(r"\s*(\d+)/(\d+) vs (\d+)/(\d+)\s*", str(s))
    return tuple(int(x) for x in m.groups()) if m else None  # type: ignore[return-value]


def sec_rq(c: Ctx, R: Registry) -> dict[str, str]:
    note = "pre-registered tests (tables/rq\\_tests\\_full.csv)"
    if c.rq is None or not c.audited:
        for nm in RQ_MACROS:
            R.set(nm, None, note)
        return {"tab_rq.tex": r"\pending{pre-registered tests}", "tab_rq_full.tex": r"\pending{pre-registered tests}"}
    s = support(c, "H1", "Fisher exact")
    R.set("NHOneFisherSupport", s["phrase"])
    R.set("NHOneFisherPadjMin", pfmt(min(s["padj"])) if s["padj"] else None, note)
    s = support(c, "H1-exploratory", "Wilcoxon")
    R.set("NHOneExpShiftRange", span(s["eff"], 2), note)
    R.set("NHOneExpSupport", s["phrase"])
    effs = [e for e in s["eff"] if finite(e)]
    R.set("NHOneExpDirection", "LLM cuisines were less uniform than the full cuisines" if effs and all(e < 0 for e in effs) else ("LLM cuisines were more uniform than the full cuisines" if effs and all(e > 0 for e in effs) else "the shift relative to the full cuisines varied in sign across models"))  # noqa: E501
    s = support(c, "H1-equalN", "Wilcoxon")
    R.set("NHOneEqWilcoxSupport", s["phrase"])
    R.set("NHOneEqWilcoxPmin", p_phrase(min(s["p"]), "geq") if s["p"] else None, note)
    R.set(
        "NHOneEqSummary",
        (rf"Wilcoxon {p_phrase(min(s['p']), 'geq')} for every model" if s["k"] == 0 else f"significant for {s['phrase']}")
        if s["p"]
        else None,
        note,
    )
    R.set("NFindingPairing", (rf"were not pushed toward uniform pairing (Wilcoxon {p_phrase(min(s['p']), 'geq')} for every model)" if s["k"] == 0 else f"were pushed toward uniform pairing for {s['phrase']}") if s["p"] else None, note)  # noqa: E501
    s = support(c, "H1-equalN", "Fisher")
    R.set("NHOneEqFisherSupport", s["phrase"])
    R.set("NHOneEqFisherPRange", span(s["p"], 3), note)
    R.set("NHOneEqFisherPadjRange", span(s["padj"], 3), note)
    parts, up_all = [], True
    for k, r in s["rows"]:
        f = frac_pair(r["effect"])
        if f:
            parts.append(f"{f[0]}/{f[1]} vs {f[2]}/{f[3]} ({c.name(k)})")
            up_all &= f[0] / max(f[1], 1) > f[2] / max(f[3], 1)
    R.set("NDirText", join_and(parts) or None, note)
    k_dir = s["k"]
    if parts and up_all:
        sig = "it is not significant after BH correction" if k_dir == 0 else f"it is significant after BH correction for {s['phrase']}"
        R.set("NDirInterpretation", f"This suggests that LLMs compress the pairing spectrum toward its middle; the pattern was not pre-registered, and {sig}.")  # noqa: E501
    else:
        R.set("NDirInterpretation", "No consistent direction by region type emerged.")
    R.set(
        "NDirConsistent",
        "for every model, a larger share of contrasting than of uniform regions moved toward uniform pairing"
        if parts and up_all
        else "the share of regions that moved toward uniform pairing did not differ consistently between contrasting and uniform regions",  # noqa: E501
    )
    s = support(c, "H2b", "log SAI")
    R.set("NHTwoBSupport", s["phrase"])
    R.set("NHTwoBPadjMin", pfmt(min(s["padj"])) if s["padj"] else None, note)
    sai_rng = R.m.get("NSAIRange", "--")
    R.set("NFindingSAI", (f"did not over-use signature ingredients on average (median stereotype amplification index {sai_rng})" if s["k"] == 0 else f"over-used signature ingredients for {s['phrase']} (median stereotype amplification index {sai_rng})") if s["p"] else None, note)  # noqa: E501
    s = support(c, "H2b-diversity", "diversity")
    R.set("NHTwoBDivSupport", s["phrase"])
    R.set("NHTwoBDivPadjMax", pfmt(max(s["padj"])) if s["padj"] else None, note)
    s = support(c, "H2c", "rho")
    R.set("NHTwoCSupport", s["phrase"])
    R.set("NHTwoCPadjMax", pfmt(max(s["padj"])) if s["padj"] else None, note)
    R.set(
        "NHTwoCSummary",
        (rf"Wilcoxon, BH-adjusted {p_phrase(max(s['padj']), 'leq')} for every model" if s["k"] == s["n"] else f"supported for {s['phrase']}")  # noqa: E501
        if s["padj"]
        else None,
        note,
    )
    fp_rng, fp_ref = R.m.get("NFpRhoRange", "--"), R.m.get("NRefFpRho", "--")
    if s["k"] == s["n"] and s["n"]:
        R.set("NFindingFingerprint", rf"their category fingerprints agreed less with the real cuisines than resampling allows (median $\rho$ {fp_rng} vs {fp_ref} for genuine samples)")  # noqa: E501
    elif s["k"] == 0:
        R.set("NFindingFingerprint", rf"their category fingerprints were within the range of genuine resamples (median $\rho$ {fp_rng} vs {fp_ref})")  # noqa: E501
    else:
        R.set("NFindingFingerprint", rf"their category fingerprints agreed less with the real cuisines than resampling allows for {s['phrase']} (median $\rho$ {fp_rng} vs {fp_ref})")  # noqa: E501
    below = []
    for k, r in s["rows"]:
        m = re.match(r"(\d+) regions below", str(r["note"]))
        if m:
            below.append(f"{m.group(1)} of {int(r['n'])} ({c.name(k)})")
    R.set("NHTwoCBelowText", join_and(below) or None, note)
    # H4: persona vs plain (direction: lower gap/JSD/|log SAI|, higher fingerprint rho = more faithful)
    better = {"abs_d_gap": -1, "prevalence_jsd": -1, "fingerprint_spearman": 1, "abs_log_sai": -1}
    for metric, st in (("prevalence_jsd", "JSD"), ("fingerprint_spearman", "Fp"), ("abs_d_gap", "Dgap"), ("abs_log_sai", "Sai")):  # noqa: E501
        s = support(c, "H4-persona", metric, lambda e, m=metric: finite(e) and e * better[m] > 0)
        R.set(f"NPersona{st}Support", s["phrase"])
        R.set(f"NPersona{st}Range", span(s["eff"], 3), note)
        R.set(f"NPersona{st}PadjMax", pfmt(max(s["padj"])) if s["padj"] else None, note)
        R.set(f"NPersona{st}PadjMin", pfmt(min(s["padj"])) if s["padj"] else None, note)
        s2 = support(c, "CM", metric, lambda e, m=metric: finite(e) and e * better[m] > 0)
        R.set(f"NCM{st}Support", s2["phrase"])
        R.set(f"NCM{st}Range", span(s2["eff"], 2), note)
        R.set(f"NCM{st}PadjMax", pfmt(max(s2["padj"])) if s2["padj"] else None, note)
    cm_all = [support(c, "CM", m, lambda e, m=m: finite(e) and e * better[m] > 0) for m in ("prevalence_jsd", "fingerprint_spearman", "abs_log_sai")]  # noqa: E501
    if cm_all[0]["n"] and all(x["k"] == x["n"] for x in cm_all):
        R.set("NFindingCM", "Every model was closer to the real cuisines than a copy-mutate evolution baseline on ingredient prevalence, fingerprints and signature-ingredient use.")  # noqa: E501
    else:
        R.set("NFindingCM", f"Compared with a copy-mutate evolution baseline, LLM cuisines were closer to the real cuisines in ingredient prevalence for {cm_all[0]['phrase']}.")  # noqa: E501
    # H4: proprietary vs open (one pooled row per metric)
    prop = c.rq_rows("H4-proprietary", "")
    open_models = [k for k in c.audited if k not in c.proprietary]
    prop_models = [k for k in c.audited if k in c.proprietary]
    R.set("NPropModels", join_and([c.name(k) for k in prop_models]) or None, note)
    R.set("NOpenModels", join_and([c.name(k) for k in open_models]) or None, note)
    for metric, st in (("abs_d_gap", "Dgap"), ("prevalence_jsd", "JSD"), ("fingerprint_spearman", "Fp"), ("abs_log_sai", "Sai")):  # noqa: E501
        r = prop[prop["test"].str.endswith(metric)] if not prop.empty else prop
        if r is None or r.empty:
            R.set(f"NProp{st}", None, "proprietary vs open test")
            R.set(f"NProp{st}P", None, "proprietary vs open test")
            continue
        r = r.iloc[0]
        R.set(f"NProp{st}", num(r["effect_num"], 3, signed=True))
        R.set(f"NProp{st}P", pfmt(r["p_adj"]))
    better_prop = {"abs_d_gap": -1, "prevalence_jsd": -1, "fingerprint_spearman": 1, "abs_log_sai": -1}
    sig_better = [m for m, b in better_prop.items() if not prop.empty and not prop[prop["test"].str.endswith(m)].empty
                  and finite(prop[prop["test"].str.endswith(m)].iloc[0]["p_adj"])
                  and prop[prop["test"].str.endswith(m)].iloc[0]["p_adj"] < Q
                  and prop[prop["test"].str.endswith(m)].iloc[0]["effect_num"] * b > 0]
    R.set("NPropVerdict", "no metric showed the proprietary models to be significantly more faithful" if not sig_better else "the proprietary models were significantly more faithful on " + join_and([tex(m.replace("_", " ")) for m in sig_better]))  # noqa: E501
    if len(open_models) == 1:
        R.set(
            "NPropCaveat",
            f"With a single open-weight model so far ({c.name(open_models[0])}), this contrast compares vendors more "
            "than model classes; it will be repeated when the remaining open-weight runs finish.",
        )
    else:
        R.set("NPropCaveat", "{}")
    scale = c.rq_rows("H4-scale", "")
    if scale.empty:
        R.pend("NScaleText", "this contrast is the planned extension with Qwen3.5-4B, whose generation is in progress")
    else:
        bits = [
            f"{tex(r['test'].split(': ')[-1])}: {num(r['effect_num'], 3, signed=True)} ($p_{{BH}}={pfmt(r['p_adj'])}$)"
            for _, r in scale.iterrows()
        ]
        R.set("NScaleText", "Qwen3.5-9B minus Qwen3.5-4B, median over regions: " + "; ".join(bits))
    return {"tab_rq.tex": table_rq(c), "tab_rq_full.tex": table_rq_full(c)}


RQ_MACROS = [
    "NHOneFisherSupport", "NHOneFisherPadjMin", "NHOneExpShiftRange", "NHOneExpSupport", "NHOneEqWilcoxSupport",
    "NHOneEqWilcoxPmin", "NHOneEqSummary", "NHOneEqFisherSupport", "NHOneEqFisherPRange", "NHOneEqFisherPadjRange",
    "NDirText", "NDirConsistent", "NHTwoBSupport", "NHTwoBPadjMin", "NHTwoBDivSupport", "NHTwoBDivPadjMax",
    "NHTwoCSupport", "NHTwoCPadjMax", "NHTwoCSummary", "NHTwoCBelowText", "NPropModels", "NOpenModels",
    "NPropCaveat", "NScaleText",
    *[f"NPersona{s}{x}" for s in ("JSD", "Fp", "Dgap", "Sai") for x in ("Support", "Range", "PadjMax", "PadjMin")],
    *[f"NCM{s}{x}" for s in ("JSD", "Fp", "Dgap", "Sai") for x in ("Support", "Range", "PadjMax")],
    *[f"NProp{s}{x}" for s in ("Dgap", "JSD", "Fp", "Sai") for x in ("", "P")],
]  # fmt: skip

RQ_TABLE_ROWS = [  # (hypothesis, test substring, label, kind)
    ("H1", "Fisher exact", r"H1: sign errors, contrasting vs uniform (Fisher)", "str"),
    ("H1", "size-aware", r"H1: opposite-sign regions, size-aware $p$ (BH)", "count"),
    ("H1-exploratory", "Wilcoxon", r"H1 (expl.): $d-d_{\mathrm{full}}>0$", "num"),
    ("H1-equalN", "Wilcoxon", r"H1 equal-$N$: $d-\tilde d_{\mathrm{gen}}>0$", "num"),
    ("H1-equalN", "Fisher", r"H1 equal-$N$: shift up, contrasting vs uniform", "str"),
    ("H2b", "log SAI", r"H2b: $\log\mathrm{SAI}>0$ (median SAI)", "num"),
    ("H2b-diversity", "diversity", r"H2b: $\log$ diversity ratio $<0$ (median)", "num"),
    ("H2c", "rho", r"H2c: $\rho-\tilde\rho_{\mathrm{gen}}<0$", "num"),
    ("H2c", "rho", r"H2c: regions below genuine range (BH)", "note"),
    ("H4-persona", "abs_d_gap", r"H4 persona$-$plain: $|d-d_{\mathrm{full}}|$", "num"),
    ("H4-persona", "prevalence_jsd", r"H4 persona$-$plain: prevalence JSD", "num"),
    ("H4-persona", "fingerprint_spearman", r"H4 persona$-$plain: fingerprint $\rho$", "num"),
    ("H4-persona", "abs_log_sai", r"H4 persona$-$plain: $|\log\mathrm{SAI}|$", "num"),
    ("CM", "abs_d_gap", r"LLM$-$copy-mutate: $|d-d_{\mathrm{full}}|$", "num"),
    ("CM", "prevalence_jsd", r"LLM$-$copy-mutate: prevalence JSD", "num"),
    ("CM", "fingerprint_spearman", r"LLM$-$copy-mutate: fingerprint $\rho$", "num"),
    ("CM", "abs_log_sai", r"LLM$-$copy-mutate: $|\log\mathrm{SAI}|$", "num"),
]
POOLED_ROWS = [  # rows with one value for the pooled model groups
    ("H4-proprietary", "abs_d_gap", r"H4 proprietary$-$open: $|d-d_{\mathrm{full}}|$"),
    ("H4-proprietary", "prevalence_jsd", r"H4 proprietary$-$open: prevalence JSD"),
    ("H4-proprietary", "fingerprint_spearman", r"H4 proprietary$-$open: fingerprint $\rho$"),
    ("H4-proprietary", "abs_log_sai", r"H4 proprietary$-$open: $|\log\mathrm{SAI}|$"),
    ("H4-scale", "abs_d_gap", r"H4 scale (9B$-$4B): $|d-d_{\mathrm{full}}|$"),
    ("H4-scale", "prevalence_jsd", r"H4 scale (9B$-$4B): prevalence JSD"),
    ("H4-scale", "fingerprint_spearman", r"H4 scale (9B$-$4B): fingerprint $\rho$"),
    ("H4-scale", "abs_log_sai", r"H4 scale (9B$-$4B): $|\log\mathrm{SAI}|$"),
]


def cell(r: pd.Series | None, kind: str) -> str:
    if r is None:
        return "--"
    sig = finite(r["p_adj"]) and r["p_adj"] < Q
    if kind == "count":
        return f"{int(float(r['effect']))}/{int(r['n'])}"
    if kind == "note":
        m = re.match(r"(\d+) regions below", str(r["note"]))
        return f"{m.group(1)}/{int(r['n'])}" if m else "--"
    eff = tex(r["effect"]) if kind == "str" else (num(r["effect_num"], 3) or "--")
    out = f"{eff} ({pfmt(r['p_adj'])})"
    return rf"\textbf{{{out}}}" if sig else out


def table_rq(c: Ctx) -> str:
    M = c.audited
    rows = []
    last = None
    for hyp, sub, label, kind in RQ_TABLE_ROWS:
        if c.rq_rows(hyp, sub).empty:
            continue
        fam = hyp.split("-")[0] if hyp != "CM" else "CM"
        if last is not None and fam != last:
            rows.append(r"\midrule")
        last = fam
        rows.append(label + " & " + " & ".join(cell(c.rq_row(hyp, sub, k), kind) for k in M) + r" \\")
    pooled = []
    for hyp, sub, label in POOLED_ROWS:
        d = c.rq_rows(hyp, sub)
        if d.empty:
            continue
        r = d.iloc[0]
        groups = " vs ".join(join_and([c.name(m) for m in order(str(g).split("+"))]) for g in str(r["model"]).split(" vs "))
        groups = groups.replace("qwen3.5-9b", "9B")
        pooled.append(label + rf" & \multicolumn{{{len(M)}}}{{c}}{{{cell(r, 'num')}}} \\")
    if pooled:
        rows += [r"\midrule", *pooled]
    heads = " & ".join(c.name(k) for k in M)
    return "\n".join(
        [
            r"\begin{tabular}{@{}l" + "c" * len(M) + r"@{}}",
            r"\toprule",
            rf"Test (effect) & {heads} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def table_rq_full(c: Ctx) -> str:
    rows = []
    for _, r in c.rq.iterrows():  # type: ignore[union-attr]
        model = " + ".join(c.name(m) for m in str(r["model"]).split("+"))
        stat = num(r["statistic"], 2) if finite(r["statistic"]) else ("$\\infty$" if str(r["statistic"]) == "inf" else "--")  # noqa: E501
        eff = num(r["effect_num"], 3) if finite(r["effect_num"]) else tex(r["effect"])
        rows.append(
            f"{tex(r['hypothesis'])} & {model} & {tex(r['test'])} & {integer(r['n'])} & {stat} & {eff}"
            f" & {pfmt(r['p'])} & {pfmt(r['p_adj'])} \\\\"
        )
    return "\n".join(
        [
            r"\begin{tabular}{@{}llp{0.34\textwidth}rrrrr@{}}",
            r"\toprule",
            r"Hyp. & Model & Test & $n$ & Statistic & Effect & $p$ & $p_{\mathrm{BH}}$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


# ------------------------------------------------------------------------------------------------- D3 steering
def sec_steering(c: Ctx, R: Registry) -> dict[str, str]:
    sc = dig(c.S, "steering", "scorer") or {}
    note = "D3 scorer (steering/train\\_scorer.py)"
    clf = {"linear_svm": "linear SVM", "hist_gbdt": "gradient boosting"}.get(str(sc.get("classifier")), tex(sc.get("classifier") or ""))  # noqa: E501
    R.set("NClfName", clf or None, note)
    R.set("NClfFOne", num(sc.get("test_macro_f1"), 2), note)
    R.set("NClfTopOne", num(sc.get("test_top1"), 2), note)
    R.set("NClfTopThree", num(sc.get("test_top3"), 2), note)
    R.set("NEceRaw", num(sc.get("ece_raw"), 2), note)
    R.set("NEceCal", num(sc.get("ece_calibrated"), 2), note)
    R.set("NGbdtValFOne", num(sc.get("comparator_val_macro_f1"), 2), note)
    R.set("NEvalClfFOne", num(sc.get("eval_classifier_macro_f1"), 2), note)
    R.set("NWeighting", tex(sc.get("weighting") or "") or None, note)
    sel = dig(c.scorer, "classifier", "selection_on_val") or {}
    R.set("NSvmValFOne", num(dig(sel, "linear_svm", "val_macro_f1"), 2), note)
    R.set("NLogregValFOneRange", span([v.get("val_macro_f1") for k, v in sel.items() if k.startswith("logreg")], 2), note)
    f1 = dig(c.scorer, "classifier", "per_region_f1") or {}
    if f1:
        hi, lo = max(f1, key=f1.get), min(f1, key=f1.get)
        R.set("NBestRegionF", f"{tex(hi)} ({num(f1[hi], 2)})")
        R.set("NWorstRegionF", f"{tex(lo)} ({num(f1[lo], 2)})")
    else:
        R.set("NBestRegionF", None, note)
        R.set("NWorstRegionF", None, note)
    auc = sc.get("auc_test") or {}
    negs = {"other_cuisines": "Other", "copy_mutate": "CM", "random_null": "Null"}
    comps = {"composite": "Composite", "classifier": "Clf", "cooccurrence": "Cooc", "ns_consistency": "Ns"}
    for nk, ns in negs.items():
        for ck, cs in comps.items():
            R.set(f"NAuc{ns}{cs}", num(dig(auc, nk, ck), 2), note)
    R.set("NAucNsRange", span([dig(auc, nk, "ns_consistency") for nk in negs], 2), note)
    tabs = {"tab_scorer.tex": table_scorer(c, R)}
    tabs.update(sec_bon(c, R))
    tabs.update(sec_revise(c, R))
    return tabs


def table_scorer(c: Ctx, R: Registry) -> str:
    sc = dig(c.S, "steering", "scorer") or {}
    auc = sc.get("auc_test") or {}
    rows = [
        rf"Macro-F1 / top-1 / top-3 & \multicolumn{{4}}{{l}}{{{num(sc.get('test_macro_f1'), 3)} / {num(sc.get('test_top1'), 3)} / {num(sc.get('test_top3'), 3)}}} \\",  # noqa: E501
        rf"ECE raw $\rightarrow$ scaled & \multicolumn{{4}}{{l}}{{{num(sc.get('ece_raw'), 3)} $\rightarrow$ {num(sc.get('ece_calibrated'), 3)}}} \\",  # noqa: E501
        rf"GBDT comparator, val.\ F1 & \multicolumn{{4}}{{l}}{{{num(sc.get('comparator_val_macro_f1'), 3)}}} \\",  # noqa: E501
        rf"Independent classifier, F1 & \multicolumn{{4}}{{l}}{{{num(sc.get('eval_classifier_macro_f1'), 3)}}} \\",  # noqa: E501
        r"\midrule",
        r"AUC: genuine vs & Comp. & Classif. & PMI & $N_s$ \\",
        r"\midrule",
    ]
    for nk, label in (("other_cuisines", "other cuisines"), ("copy_mutate", "copy-mutate"), ("random_null", "random null")):  # noqa: E501
        v = auc.get(nk) or {}
        rows.append(
            f"{label} & {num(v.get('composite'), 3) or '--'} & {num(v.get('classifier'), 3) or '--'} & "
            f"{num(v.get('cooccurrence'), 3) or '--'} & {num(v.get('ns_consistency'), 3) or '--'} \\\\"
        )
    return "\n".join([r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", *rows, r"\bottomrule", r"\end{tabular}"])


def sec_bon(c: Ctx, R: Registry) -> dict[str, str]:
    bon = dig(c.S, "steering", "bon_main")
    note = "best-of-$N$ results (steering/run\\_bon.py)"
    names = ["NBonModels", "NBonNValues", "NBonNMax", "NBonClfText", "NBonClfMonotone", "NBonDivText",
             "NBonDivMonotone", "NBonSignText", "NBonFpText", "NBonJSDText", "NBonNearCopy", "NBonClfLowRange",
             "NBonClfHighRange", "NBonPoolMax", "NBonSelectedMax", "NBonSaiText", "NBonNoveltyText",
             "NBonAblationText", "NBonPoolMin", "NBonSelectedMin"]  # fmt: skip
    if not isinstance(bon, list) or not bon:
        for nm in names:
            R.set(nm, None, note)
        return {"tab_bon.tex": rf"\pending{{{note}}}"}
    df = pd.DataFrame(bon)
    models = order(df["model"].unique())
    nvals = sorted(int(n) for n in df["N"].unique())
    n1, nmax = nvals[0], nvals[-1]
    R.set("NBonModels", word(len(models)))
    R.set("NBonNValues", ", ".join(str(n) for n in nvals))
    R.set("NBonNMax", nmax)

    def trend(col: str, nd: int = 2) -> tuple[str, int]:
        parts, mono = [], 0
        for k in models:
            g = df[df["model"] == k].sort_values("N")
            v = g[col].tolist()
            parts.append(f"{num(v[0], nd)} to {num(v[-1], nd)} ({c.name(k)})")
            mono += int(all(b >= a for a, b in zip(v, v[1:], strict=False)))
        return join_and(parts), mono

    def mono_phrase(k: int, n: int, up: bool) -> str:
        verb = "rose" if up else "fell"
        if k == n:
            return f"{verb} monotonically with $N$ for {'every model' if n > 1 else 'the model'}"
        return f"{verb} monotonically with $N$ for {word(k)} of the {word(n)} models"

    t, mono = trend("indep_classifier_acc")
    R.set("NBonClfText", t)
    R.set("NBonClfMonotone", mono_phrase(mono, len(models), True))
    t, _ = trend("diversity_ratio")
    df["neg_div"] = -df["diversity_ratio"]
    _, mono_d = trend("neg_div")
    R.set("NBonDivText", t)
    R.set("NBonDivMonotone", mono_phrase(mono_d, len(models), False))
    R.set("NBonSignText", trend("sign_match")[0])
    R.set("NBonFpText", trend("fingerprint_spearman")[0])
    R.set("NBonJSDText", trend("prevalence_jsd", 3)[0])
    nc = df["near_copy_rate"]
    R.set("NBonNearCopy", num(nc.max(), 0) if nc.max() == nc.min() else span(nc, 3))
    lo = df[df["N"] == n1]["indep_classifier_acc"]
    hi = df[df["N"] == nmax]["indep_classifier_acc"]
    R.set("NBonClfLowRange", span(lo, 2))
    R.set("NBonClfHighRange", span(hi, 2))
    # secondary: run_bon.py's per-region output (pool sizes; SAI and novelty by N; ablation at the largest N)
    try:
        pq = pd.read_parquet(REPO / "data/interim/steering/bon_main.parquet")
        comp = pq[pq["method"] == "composite"]
        R.set("NBonPoolMax", integer(pq["n_pool"].max()))
        R.set("NBonPoolMin", integer(pq["n_pool"].min()))
        R.set("NBonSelectedMax", integer(pq["n_selected"].max()))
        R.set("NBonSelectedMin", integer(pq["n_selected"].min()))
        agg = comp.groupby(["model", "N"])[["sai", "novelty_mean"]].mean().reset_index()
        parts_s, parts_n = [], []
        for k in models:
            g = agg[agg["model"] == k].sort_values("N")
            if not g.empty:
                parts_s.append(f"{num(g['sai'].iloc[0], 2)} to {num(g['sai'].iloc[-1], 2)} ({c.name(k)})")
                parts_n.append(f"{num(g['novelty_mean'].iloc[0], 2)} to {num(g['novelty_mean'].iloc[-1], 2)} ({c.name(k)})")  # noqa: E501
        R.set("NBonSaiText", join_and(parts_s) or None, note)
        R.set("NBonNoveltyText", join_and(parts_n) or None, note)
        abl = pq[pq["N"] == nmax].groupby("method")[["indep_classifier_acc", "prevalence_jsd"]].mean()
        if {"composite", "without_classifier", "without_cooccurrence", "without_ns_consistency"} <= set(abl.index):
            f = lambda m, col, nd=2: num(abl.loc[m, col], nd)  # noqa: E731
            R.set(
                "NBonAblationText",
                f"the independent classifier's share was {f('composite', 'indep_classifier_acc')} with all three "
                f"components, {f('without_classifier', 'indep_classifier_acc')} without the classifier, "
                f"{f('without_cooccurrence', 'indep_classifier_acc')} without co-occurrence and "
                f"{f('without_ns_consistency', 'indep_classifier_acc')} without $N_s$ consistency, while prevalence "
                f"JSD was {f('composite', 'prevalence_jsd', 3)}, {f('without_classifier', 'prevalence_jsd', 3)}, "
                f"{f('without_cooccurrence', 'prevalence_jsd', 3)} and {f('without_ns_consistency', 'prevalence_jsd', 3)}",  # noqa: E501
            )
        else:
            R.set("NBonAblationText", None, note)
    except (OSError, ValueError, KeyError, ImportError):
        for nm in ("NBonPoolMax", "NBonPoolMin", "NBonSelectedMax", "NBonSelectedMin", "NBonSaiText",
                   "NBonNoveltyText", "NBonAblationText"):  # fmt: skip
            R.set(nm, None, "best-of-$N$ per-region output (data/interim/steering/bon\\_main.parquet)")
    return {"tab_bon.tex": table_bon(c, df, models, nvals)}


def table_bon(c: Ctx, df: pd.DataFrame, models: list[str], nvals: list[int]) -> str:
    metrics = [
        ("indep_classifier_acc", "Independent classifier: share in claimed cuisine", 2),
        ("diversity_ratio", "Diversity ratio", 2),
        ("prevalence_jsd", "Prevalence JSD", 3),
        ("fingerprint_spearman", r"Fingerprint $\rho$", 2),
        ("sign_match", "Sign agreement (share of regions)", 2),
        ("near_copy_rate", "Near-copy rate", 2),
    ]
    rows = []
    for col, label, nd in metrics:
        rows.append(rf"\multicolumn{{{len(models) + 1}}}{{@{{}}l}}{{\emph{{{label}}}}} \\")
        for n in nvals:
            vals = []
            for k in models:
                g = df[(df["model"] == k) & (df["N"] == n)]
                vals.append(num(g[col].iloc[0], nd) if not g.empty else "--")
            rows.append(rf"\quad $N={n}$ & " + " & ".join(v or "--" for v in vals) + r" \\")
    heads = " & ".join(head(c.name(k)) for k in models)
    return "\n".join(
        [
            r"\begin{tabular}{@{}l" + "c" * len(models) + r"@{}}",
            r"\toprule",
            rf"Metric (mean over regions) & {heads} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def sec_revise(c: Ctx, R: Registry) -> dict[str, str]:
    rev = dig(c.S, "steering", "revise_main")
    note = "critique-and-revise (round 1 running: GPT-6-Luna via Batch, Qwen3.5-9B on Kaggle)"
    if not isinstance(rev, list) or not rev:
        R.pend("NReviseText", note)
        for nm in ("NReviseModels", "NReviseRoundsText", "NReviseVerb", "NReviseIdentityRange", "NReviseIdentityPctRange",
                   *[f"NRevise{st}{x}" for st in ("Clf", "Sign", "Fp", "JSD") for x in ("Text", "StartRange", "EndRange")]):  # fmt: skip
            R.set(nm, None, note)
        return {"tab_revise.tex": rf"\pending{{{note}}}"}
    df = pd.DataFrame(rev)
    models = order(df["model"].unique())
    cols = [
        ("indep_classifier_acc", r"\makecell{Indep.\\classif.}", 2),
        ("sign_match", r"\makecell{Sign\\agree.}", 2),
        ("fingerprint_spearman", r"\makecell{Finger-\\print $\rho$}", 2),
        ("prevalence_jsd", r"\makecell{Prev.\\JSD}", 3),
        ("identity_jaccard", r"\makecell{Identity\\(Jaccard)}", 2),
    ]
    cols = [x for x in cols if x[0] in df]
    rows = []
    for k in models:
        for _, r in df[df["model"] == k].sort_values("round").iterrows():
            rows.append(
                f"{c.name(k) if int(r['round']) == 0 else ''} & {int(r['round'])} & " + " & ".join(num(r[col], nd) or "--" for col, _, nd in cols) + r" \\"
            )
    R.set("NReviseModels", join_and([c.name(k) for k in models]))
    rounds = {k: int(df[df["model"] == k]["round"].max()) for k in models}
    R.set("NReviseRoundsText", join_and([f"{word(r)} round{'s' if r != 1 else ''} for {c.name(k)}" for k, r in rounds.items()]))  # noqa: E501

    def rtrend(col: str, nd: int = 2) -> str | None:
        if col not in df:
            return None
        parts = []
        for k in models:
            g = df[df["model"] == k].sort_values("round")
            parts.append(f"{num(g[col].iloc[0], nd)} to {num(g[col].iloc[-1], nd)} ({c.name(k)})")
        return join_and(parts)

    for col, st, nd in (("indep_classifier_acc", "Clf", 2), ("sign_match", "Sign", 2), ("fingerprint_spearman", "Fp", 2), ("prevalence_jsd", "JSD", 3)):  # noqa: E501
        R.set(f"NRevise{st}Text", rtrend(col, nd), note)
        first = [df[df["model"] == k].sort_values("round")[col].iloc[0] for k in models] if col in df else []
        last = [df[df["model"] == k].sort_values("round")[col].iloc[-1] for k in models] if col in df else []
        R.set(f"NRevise{st}StartRange", span(first, nd), note)
        R.set(f"NRevise{st}EndRange", span(last, nd), note)
    up = [df[df["model"] == k].sort_values("round")["indep_classifier_acc"].iloc[-1] > df[df["model"] == k].sort_values("round")["indep_classifier_acc"].iloc[0] for k in models] if "indep_classifier_acc" in df else []  # noqa: E501
    R.set("NReviseVerb", "raised" if up and all(up) else "changed")
    ident = df["identity_jaccard"].dropna() if "identity_jaccard" in df else pd.Series(dtype=float)
    R.set("NReviseIdentityRange", span(ident, 2), note)
    R.set("NReviseIdentityPctRange", span(ident, 0, True), note)
    R.set("NReviseText", "{}")
    return {
        "tab_revise.tex": "\n".join(
            [
                r"\begin{tabular}{@{}lc" + "c" * len(cols) + r"@{}}",
                r"\toprule",
                "Model & Round & " + " & ".join(label for _, label, _ in cols) + r" \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )
    }


# ------------------------------------------------------------------------------------------------- appendix pieces
def sec_aliasing(c: Ctx, R: Registry) -> dict[str, str]:
    m = load_json("data/interim/aliasing/metrics.json") or {}
    note = "aliaser evaluation (data/interim/aliasing/metrics.json)"
    t = m.get("test") or {}
    R.set("NAliasAccLinesPct", pct(t.get("accuracy_lines"), 1), note)
    R.set("NAliasAccRowsPct", pct(t.get("accuracy_rows_weighted"), 1), note)
    R.set("NAliasCoveragePct", pct(t.get("coverage_lines"), 2), note)
    R.set("NAliasTestLines", integer(t.get("lines")), note)
    R.set("NAliasPartialPct", pct(t.get("share_partial"), 1), note)
    chosen = m.get("chosen")
    R.set("NAliasChosen", tex(ALIAS_STAGES.get(chosen, chosen or "")).lstrip("+ ") if chosen else None, note)
    by = m.get("test_by_source") or {}
    R.set("NAliasSourceRange", span(by.values(), 1, True), note)
    if by:
        lo = min(by, key=by.get)
        R.set("NAliasWorstSource", f"{tex(lo.replace('_', ' ').title())} ({pct(by[lo], 1)})")
    else:
        R.set("NAliasWorstSource", None, note)
    rows = []
    for stage, v in (m.get("val") or {}).items():
        label = ALIAS_STAGES.get(stage, tex(stage))
        mark = r"$^\ast$" if stage == chosen else ""
        rows.append(
            f"{tex(stage.split('_')[0])}{mark} & {label} & {pct(v.get('accuracy_lines'), 1)} & "
            f"{pct(v.get('accuracy_rows_weighted'), 1)} & {pct(v.get('coverage_lines'), 1)} \\\\"
        )
    if t:
        rows += [
            r"\midrule",
            rf"Test & chosen variant, refit on all training lines & {pct(t.get('accuracy_lines'), 1)} & "
            rf"{pct(t.get('accuracy_rows_weighted'), 1)} & {pct(t.get('coverage_lines'), 1)} \\",
        ]
    src = ", ".join(f"{tex(k.replace('_', ' ').title())} {pct(v, 1)}" for k, v in by.items())
    if src:
        rows += [r"\midrule", rf"\multicolumn{{5}}{{@{{}}p{{0.97\columnwidth}}}}{{Test accuracy by source: {src}}} \\"]
    body = (
        "\n".join(
            [
                r"\begin{tabular}{@{}lp{0.44\columnwidth}rrr@{}}",
                r"\toprule",
                r"& Variant & \makecell{Unique\\lines} & \makecell{Occur-\\rences} & Coverage \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )
        if rows
        else rf"\pending{{{note}}}"
    )
    return {"tab_aliasing.tex": body}


def sec_code_counts(c: Ctx, R: Registry) -> None:
    mods = [p for p in (REPO / "src/flavoraudit").glob("*.py") if p.name != "__init__.py"]
    tests = sum(len(re.findall(r"^\s*def test_", p.read_text(encoding="utf-8"), flags=re.M)) for p in (REPO / "tests").glob("test_*.py"))  # noqa: E501
    pages = list((REPO / "docs/site").glob("*.md"))
    R.set("NLibModules", len(mods) or None, "library modules")
    R.set("NLibTests", tests or None, "test functions")
    R.set("NDocsPages", len(pages) or None, "documentation pages")
    R.set("NAppTabs", "three")


def prompts_tex() -> str:
    """The frozen prompt templates, verbatim (escaped) from configs/prompts/v1/."""
    base = REPO / "configs/prompts/v1"
    out = []
    for name, label in (("system.txt", "System"), ("plain.txt", "Plain"), ("persona.txt", "Persona"), ("format.txt", "{format}")):  # noqa: E501
        p = base / name
        if not p.exists():
            out.append(rf"\pending{{prompt file {tex(name)}}}")
            continue
        body = tex(p.read_text(encoding="utf-8").strip()).replace("\n\n", r"\par ").replace("\n", " ")
        out.append(rf"\noindent\textbf{{{tex(label)}:}} {{\ttfamily\footnotesize {body}}}\par\smallskip")
    return "\n".join(out)


def traceability(c: Ctx, R: Registry) -> str:
    """Proposal (as submitted) vs delivered; the status column is data-driven."""
    ds = dig(c.S, "dataset") or {}
    bon = dig(c.S, "steering", "bon_main")
    rev = dig(c.S, "steering", "revise_main")
    d1 = (
        f"{integer(ds.get('total_recipes'))} recipes from {word(len(c.present))} LLMs so far; "
        + ("targets met" if ds.get("meets_proposal") else "targets not yet met")  # noqa: E501
    )
    d3 = (
        "scorer trained and validated; best-of-$N$ "
        + ("done (deviated design)" if bon else "not run")
        + "; critique-and-revise "
        + ("done" if rev else "not run")
    )
    rows = [
        (r"D1 LLMCuisine-22", r"$\geq$10{,}000 recipes, 22 regions, $\geq$3 LLMs, plain + persona prompts, CulinaryDB format, datasheet",  # noqa: E501
         r"\texttt{generation/}, \texttt{data/release/}, datasheet", d1),
        (r"D2 Audit", r"$N_s$, $Z$ vs four nulls with CIs, contributions, fingerprints, SAI, diversity, novelty; copy-mutate baseline; scorecard",  # noqa: E501
         r"\texttt{analysis/}, \texttt{tables/}, \texttt{figures/}",
         r"replication, equal-$N$ reference, scorecard, pre-registered tests" + (r", $\chi\%$ drivers" if dig(c.S, "drivers") else "") + (", co-occurrence networks" if dig(c.S, "networks") else "") + " done"),  # noqa: E501
        (r"D3 Steering", r"Fidelity scorer (classifier + co-occurrence + $N_s$) steering best-of-$N$ and critique-and-revise", r"\texttt{steering/}, \texttt{flavoraudit.scoring}", d3),  # noqa: E501
        (r"D4 FlavorAudit", r"pip-installable, tested, documented library (proposed as ``CuisineKit'')",
         r"\texttt{src/flavoraudit/}, \texttt{tests/}, MkDocs site",
         rf"{R.m.get('NLibModules')} modules, {R.m.get('NLibTests')} tests, documentation built; PyPI release awaits the team's approval"),  # noqa: E501
        (r"D5 FlavorPrint", r"Streamlit app on Hugging Face Spaces: audit a recipe; dashboard", r"\texttt{app/}",
         r"Audit, Dashboard and About tabs built and tested locally; deployment awaits the team's approval"),
    ]
    if not dig(c.S, "drivers"):
        R.note(r"RQ2a: $\chi\%$ driver comparison for LLM cuisines (not in summary.json yet)")
    R.note("Publishing D1 (Kaggle/HF/Zenodo), D4 (PyPI) and D5 (HF Space): team approval")
    body = [f"{a} & {b} & {cc} & {d} \\\\" for a, b, cc, d in rows]
    return "\n".join(
        [
            r"\begin{tabular}{@{}p{0.11\textwidth}p{0.32\textwidth}p{0.2\textwidth}p{0.3\textwidth}@{}}",
            r"\toprule",
            r"Deliverable & Proposed (Oct.\ 2026) & Where & Status at this build \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


# ===================================================================================================== output
def write_numbers(R: Registry, flags: dict[str, bool]) -> None:
    lines = [
        "% Generated by report/build.py -- do not edit. Every value is computed from the pipeline outputs;",
        "% \\pending{...} marks values whose inputs do not exist yet.",
        "\\providecommand{\\pending}[1]{{\\color{PendingColor}\\sffamily\\bfseries\\footnotesize[pending:~#1]}}",
    ]
    for flag, on in flags.items():
        lines += [f"\\newif\\if{flag}", f"\\{flag}{'true' if on else 'false'}"]
    for k in sorted(R.m):
        lines.append(f"\\newcommand{{\\{k}}}{{{R.m[k]}}}")
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    items = "\n".join(rf"  \item {p}" for p in R.pending) or r"  \item none"
    (GEN / "pending_list.tex").write_text(
        "\\begin{itemize}\\setlength{\\itemsep}{0pt}\n" + items + "\n\\end{itemize}\n", encoding="utf-8"
    )


def check_macros(R: Registry) -> list[str]:
    """Macros used in main.tex but not produced here become pending marks (and are reported)."""
    main_tex = REPORT / "main.tex"
    if not main_tex.exists():
        return []
    used = set(re.findall(r"\\(N[A-Z][A-Za-z]*)", main_tex.read_text(encoding="utf-8")))
    missing = sorted(u for u in used if u not in R.m)
    for u in missing:
        R.pend(u, rf"undefined number \texttt{{{u}}}")
    return missing


def compile_pdf() -> int:
    tectonic = shutil.which("tectonic") or "/opt/homebrew/bin/tectonic"
    cmd = [tectonic, "--keep-logs", "--outdir", ".", "main.tex"]
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPORT, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(out[-4000:])
        print("tectonic failed")
        return proc.returncode
    warn = [ln for ln in out.splitlines() if "warning" in ln.lower() and "downloading" not in ln]
    log = (REPORT / "main.log").read_text(encoding="utf-8", errors="replace") if (REPORT / "main.log").exists() else ""
    over = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log)
    big = [float(x) for x in over if float(x) > 2.0]
    undef = re.findall(r"LaTeX Warning: (Reference|Citation) `([^']+)'", log)
    pages = None
    try:
        info = subprocess.run(["pdfinfo", str(REPORT / "main.pdf")], capture_output=True, text=True).stdout
        m = re.search(r"Pages:\s+(\d+)", info)
        pages = int(m.group(1)) if m else None
    except OSError:
        pass
    print(f"main.pdf: {pages} pages; overfull boxes > 2pt: {len(big)} (max {max(big) if big else 0:.1f}pt); "
          f"undefined refs/cites: {len(undef)}; tectonic warnings: {len(warn)}")  # fmt: skip
    for w in warn[:20]:
        print("  ", w)
    for kind, key in undef[:20]:
        print(f"   undefined {kind}: {key}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summary", type=Path, default=REPO / "results" / "summary.json")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--include-f5", action="store_true", help="include figures/F5 (withheld by default)")
    ap.add_argument("--draft", action="store_true", help="show the draft-status box with the pending items")
    args = ap.parse_args()
    GEN.mkdir(parents=True, exist_ok=True)
    c = Ctx(args.summary)
    R = Registry()
    tables: dict[str, str] = {}
    sec_meta(c, R)
    sec_design(c, R)
    sec_culinarydb(c, R)
    tables.update(sec_dataset(c, R))
    tables.update(sec_replication(c, R))
    tables.update(sec_equal_n(c, R))
    sec_copy_mutate(c, R)
    tables.update(sec_audit(c, R))
    sec_drivers(c, R)
    tables.update(sec_networks(c, R))
    tables.update(sec_rq(c, R))
    tables.update(sec_steering(c, R))
    tables.update(sec_aliasing(c, R))
    sec_code_counts(c, R)
    tables["tab_traceability.tex"] = traceability(c, R)
    tables["tab_status.tex"] = status_table(c, R)
    tables["prompts.tex"] = prompts_tex()
    if not args.include_f5:
        R.note("Fig.~F5 withheld: figures/F5\\_sign\\_matrix draws the regions a model has not covered yet as sign errors")
    missing = check_macros(R)
    flags = {
        "figfive": args.include_f5,
        "draftstatus": args.draft,
        "figfifteen": "F15_best_of_n.pdf" in c.ok_figures,
        "figfifteenb": "F15b_ablation.pdf" in c.ok_figures,
        "figfifteenc": "F15c_critique_revise.pdf" in c.ok_figures,
    }
    for name, body in tables.items():
        (GEN / name).write_text("% Generated by report/build.py -- do not edit.\n" + body + "\n", encoding="utf-8")
    write_numbers(R, flags)
    print(f"wrote {len(R.m)} macros and {len(tables)} tables to {GEN.relative_to(REPO)}; pending items: {len(R.pending)}")
    for p in R.pending:
        print("  pending:", re.sub(r"\\[a-z]+\{|[{}\\$]", "", p))
    if missing:
        print("WARNING: macros used in main.tex but not computed:", ", ".join(missing))
    return 0 if args.no_pdf else compile_pdf()


if __name__ == "__main__":
    raise SystemExit(main())
