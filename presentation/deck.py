"""The Week-14 slide deck (slides.pptx): 15 main slides plus backups, every number from the facts layer."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from facts import (  # noqa: E402
    direction,
    f,
    join_and,
    minus,
    n0,
    pct,
    pending,
    pv,
    sf,
    supported,
    tag,
)
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx_kit import MONO, Deck, Para, T, footer, header  # noqa: E402

FORMULAS = {
    "ns": r"$N_s(R)=\frac{2}{n(n-1)}\sum_{i<j\in R}|C_i\cap C_j|$",
    "z": r"$Z=\sqrt{N_{rand}}\;\frac{\overline{N}_s^{\,cuisine}-\overline{N}_s^{\,rand}}{\sigma_{rand}}=\sqrt{N_{rand}}\;d$",
    "sai": r"$\mathrm{SAI}_c=\sum_{i\in\Sigma_c}P_{ic}^{LLM}\;/\;\sum_{i\in\Sigma_c}P_{ic}^{real}$",
    "prev": r"$p_{ic}=P_{ic}-\langle P_{ic'}\rangle_{c'\neq c}$",
}


def render_formulas(outdir: Path) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"mathtext.fontset": "dejavusans"})
    paths = {}
    for key, tex in FORMULAS.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, tex, fontsize=30, color=T.ink)
        path = outdir / f"formula_{key}.png"
        fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        paths[key] = path
    return paths


def model_name(F, k):
    return F["models"][k]["name"]


def rng_text(lo, hi, nd=2):
    if lo is None or hi is None:
        return pending("no analysed model yet")
    return f"{lo:.{nd}f}" if f"{lo:.{nd}f}" == f"{hi:.{nd}f}" else f"{lo:.{nd}f}–{hi:.{nd}f}"


def pct_rng(lo, hi):
    if lo is None or hi is None:
        return pending("no analysed model yet")
    a, b = f"{100 * lo:.0f}", f"{100 * hi:.0f}"
    return f"{a}%" if a == b else f"{a}–{b}%"


def padj(r):
    return f"p_adj {pv(r['p_adj'])}" if r else pending("test not yet in tables/rq_tests_full.csv")


def drivers_text(F) -> str:
    """One RQ2a sentence: top-5 flavor-driver overlap with the real cuisine and Korea's top driver."""
    DR = F["drivers"]
    if not DR:
        return "**Flavor drivers (RQ2a):** " + pending("drivers not yet in results/summary.json")
    vals = "; ".join(f"{DR['models'][k]['name']} {f(DR['models'][k].get('median_jaccard5'))}" for k in DR["order"])
    txt = (
        f"**Flavor drivers (RQ2a):** overlap of the top-5 χ% contributors with the real cuisine (median "
        f"Jaccard@5): {vals}; genuine samples {f(DR['real'].get('median_jaccard5'))}."
    )
    if DR["korea_real_top5"]:
        top = DR["korea_real_top5"][0]
        if DR["korea_match"]:
            txt += f" Korea: {join_and(DR['korea_match'])} put {top} first, as the real cuisine does"
            txt += f"; {join_and(DR['korea_miss'])} do{'es' if len(DR['korea_miss']) == 1 else ''} not." \
                if DR["korea_miss"] else "."  # fmt: skip
    return txt


# =====================================================================================================================
# Main slides
# =====================================================================================================================
def s_title(deck: Deck, F) -> None:
    s = deck.new("main", "title")
    deck.rect(s, 0, 0, 0.28, 7.5, T.accent)
    deck.text(s, 0.9, 1.05, 11.6, 1.5, [Para("Do AI Chefs Respect Culinary Fingerprints?", 46, bold=True)],
              anchor="bottom")  # fmt: skip
    deck.text(s, 0.9, 2.6, 11.6, 0.95, [Para(
        f"Auditing and steering LLM-generated recipes across {F['regions']['n']} world cuisines", 26,
        color=T.muted)])  # fmt: skip
    deck.hline(s, 0.9, 3.65, 4.2, T.accent, 2.5)
    deck.text(s, 0.9, 3.85, 11.5, 1.55, [
        Para("Group 44  ·  Roll nos. 2023355 and 2023477", 22, bold=True, space_before=0),
        Para("BIO544 Computational Gastronomy  ·  IIIT-Delhi  ·  Monsoon 2026", 19, color=T.muted, space_before=6),
        Para("Instructor: Prof. Ganesh Bagler", 19, color=T.muted, space_before=4),
    ])  # fmt: skip
    chips = [("D1", "LLMCuisine-22 dataset"), ("D2", "Food-pairing audit"), ("D3", "Fingerprint-guided steering"),
             ("D4", "FlavorAudit library"), ("D5", "FlavorPrint web app")]  # fmt: skip
    x, w, gap = 0.9, 2.26, 0.13
    for i, (d, name) in enumerate(chips):
        cx = x + i * (w + gap)
        deck.rect(s, cx, 5.55, w, 0.86, T.panel, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        deck.text(s, cx + 0.08, 5.58, w - 0.16, 0.8, [Para(d, 15, bold=True, color=T.accent, align="center",
                                                           space_before=0),
                                                      Para(name, 14, align="center", space_before=1)],
                  anchor="middle")  # fmt: skip
    deck.text(s, 0.9, 6.7, 11.9, 0.4, [Para(
        f"Every number is regenerated from results/summary.json by presentation/build.py  ·  data snapshot "
        f"{F['generated_date']}", 11, color=T.muted)], fit=False)  # fmt: skip
    deck.notes(
        s,
        f"""
WHAT TO SAY (about 40 s)
Good morning. We are Group 44. Our question is simple: when a large language model writes a "traditional" recipe for
a cuisine, does that recipe carry the cuisine's measurable culinary fingerprint? We use this lab's own framework:
flavor sharing Ns, food-pairing Z against null cuisines, category fingerprints and ingredient prevalence.

We built five deliverables: a dataset of LLM recipes (D1), the audit itself (D2), a fingerprint-based steering method
(D3), the FlavorAudit library (D4) and the FlavorPrint app (D5).

HOW TO READ THE DECK
Every number on these slides is generated from results/summary.json (snapshot {F["generated_date"]}). Runs that are
still in progress are shown as a neutral status line, never as a predicted value. The deck ends on the Conclusions slide; backup slides follow it.
""",
    )


def s_motivation(deck: Deck, F) -> None:
    s = deck.new("main", "motivation")
    header(deck, s, "Why audit LLM recipes with flavor chemistry?",
           "Because this lab's framework turns “authentic” into numbers we can test, and, as far as we found, no one has "
           "applied it to LLM recipes.")  # fmt: skip
    deck.text(s, 0.55, 1.95, 5.5, 4.4, [
        Para("Motivation", 20, bold=True, space_before=0),
        Para("LLMs now write recipes for millions of people and call them “traditional”.", 17, bullet=True,
             space_before=10),
        Para("Existing audits of LLM recipes are textual (e.g. Carichon et al. 2026): they measure wording, not how "
             "ingredients combine.", 17, bullet=True, space_before=10),
        Para("Food pairing (Ahn et al. 2011), Z against null cuisines and culinary fingerprints (Jain, Rakhi & Bagler "
             "2015; Singh & Bagler 2018) are measurable and falsifiable.", 17, bullet=True, space_before=10),
        Para(f"So we score LLM cuisines with the same metrics, on the same {F['regions']['n']} CulinaryDB regions.", 17,
             bullet=True, space_before=10),
    ])  # fmt: skip
    rows = [
        ["", "Question", "Pre-registered hypothesis"],
        ["RQ1", "Same food-pairing sign and strength?",
         f"H1: pushed toward uniform pairing; sign errors cluster in the {F['regions']['n_contrasting']} contrasting "
         "regions"],
        ["RQ2", "Same drivers, stereotypes and fingerprints?",
         "H2b: signature ingredients over-used (SAI > 1), lower diversity; H2c: weaker category fingerprint"],
        ["RQ3", "Can fingerprint feedback steer generation without copying?",
         "H3: best-of-N and critique-and-revise improve held-out fidelity, novelty stays high"],
        ["RQ4", "Do prompt framing, vendor or scale matter?",
         "H4: home-cook persona helps slightly; proprietary or larger models not consistently better"],
    ]  # fmt: skip
    deck.table(s, 6.3, 1.98, [0.7, 2.45, 3.35], rows, size=13, row_h=0.84, aligns=["c", "l", "l"],
               bold_first_col=True)  # fmt: skip
    deck.text(s, 0.55, 6.42, 12.25, 0.55, [Para(
        "Hypotheses, metrics and tests were fixed in the implementation plan before any full-scale LLM data were "
        "analysed; one amendment (the equal-size comparison, slide 6) was logged before the analysis.", 14,
        color=T.muted, italic=True, space_before=0)])  # fmt: skip
    deck.notes(
        s,
        f"""
WHAT TO SAY (about 50 s)
LLMs are increasingly where people get recipes, and they confidently label them "traditional". Earlier audits of LLM
recipes look at text, for example novelty of wording. Nobody, as far as our literature search found, has checked LLM
recipes against flavor-compound pairing or culinary fingerprints. Your framework is ideal for this because it is
quantitative and has null models, so "authentic" becomes a falsifiable statement.

We have four research questions. RQ1: does each LLM cuisine keep the region's food-pairing sign and strength? Our
pre-registered hypothesis H1 was that LLMs push cuisines toward uniform (positive) pairing, so errors would cluster in
the {F["regions"]["n_contrasting"]} contrasting regions. RQ2: composition, stereotypes and the category fingerprint.
RQ3: can we steer generation with the fingerprints? RQ4: prompt framing, proprietary versus open models, and scale.

IF ASKED: "Were the hypotheses fixed in advance?" Yes, in the plan's section 1 before any full-run data were analysed.
The only change is the equal-N amendment (slide 6), recorded in the decision log with its reason before the full LLM
analysis. See qa_prep.md.
""",
    )


def s_data(deck: Deck, F) -> None:
    s = deck.new("main", "data")
    D, ds = F["design"], F["dataset"]
    header(deck, s, "What did we ask the models, and how much is in?",
           f"{n0(ds['total_recipes'])} recipes from {ds['n_models']} models so far. The design gives every model "
           f"{n0(D['recipes_per_model'])} recipes: {F['regions']['n']} regions × {D['n_conditions']} prompts × "
           f"{D['calls_per_cell']} calls × {D['recipes_per_call']} recipes.")  # fmt: skip
    deck.rect(s, 0.55, 1.95, 5.15, 5.0, T.panel)
    korea = F["regions"]["phrase_korea"]
    plain_prompt = D["prompt_plain"].replace("{phrase}", korea)
    place = "Seoul, South Korea"
    persona = D["prompt_persona"].replace("{place}", place)
    al = F["aliasing"]
    deck.text(s, 0.7, 2.02, 4.9, 4.86, [
        Para(f"Frozen prompt set {D['prompt_version']} (example: Korea)", 17, bold=True, space_before=0),
        Para("**Plain:** “" + plain_prompt + "”", 15, space_before=8),
        Para("**Home-cook persona:** “" + persona + "”", 15, space_before=8),
        Para("Same system prompt and JSON format for every model (title, ingredient lines with quantities, steps). "
             "Persona places rotate over 8 places per region.", 13, color=T.muted, space_before=8),
        Para("Open weights: T = 0.7, top-p = 0.95, thinking off. APIs: provider defaults, lowest reasoning level.",
             13, color=T.muted, space_before=6),
        Para(f"Every call is logged (prompt, model id, parameters, usage, raw output). Ingredient lines are aliased "
             f"with CulinaryDB's own protocol: **{pct(al['acc_lines'])}** of held-out lines and "
             f"**{pct(al['acc_rows'])}** of occurrences match CulinaryDB's expert mapping.", 13, space_before=8),
    ], min_scale=0.7)  # fmt: skip
    rows = [["Model", "Calls", "Recipes", "Regions", "Lines aliased", "Status"]]
    for k in F["model_order"]:
        mi = F["models"][k]
        d = mi["data"]
        kind = "API" if mi["proprietary"] else "open"
        name = f"{mi['name']} ({kind})"
        if d:
            status = "complete" if mi["complete"] else pending("in progress")
            rows.append([name, f"{d['calls']:,} / {d['planned_calls']:,}", n0(d["recipes"]),
                         f"{d['regions']} / {F['regions']['n']}", pct(d["alias_full_share"]), status])  # fmt: skip
        else:
            status = pending("not yet run") if mi["tier"] == "core" else pending("planned extra")
            rows.append([name, f"0 / {n0(D['planned_calls'])}", "—", "—", "—", status])
    rows.append(["Total so far", f"{n0(ds['total_calls'])}", n0(ds["total_recipes"]), "", "", ""])
    fills = {(len(rows) - 1, c): T.panel2 for c in range(6)}
    deck.table(s, 5.95, 1.98, [2.3, 0.95, 0.85, 0.8, 0.85, 1.2], rows, size=12, row_h=0.38,
               aligns=["l", "c", "r", "c", "c", "l"], fills=fills, header_size=12)  # fmt: skip
    note = [Para(f"Design target: {ds['n_planned']} models × {n0(D['recipes_per_model'])} = "
                 f"{n0(ds['planned_total'])} recipes ({ds['n_proprietary']} proprietary APIs so far; open-weight "
                 f"families so far: {ds['n_open_families']}).", 13, space_before=0)]  # fmt: skip
    if not ds["meets_proposal"]:
        note.append(Para(
            f"Proposal minimum (≥ {n0(D['proposal_min_recipes'])} recipes, ≥ {D['proposal_min_models']} LLMs, "
            f"≥ {D['proposal_min_open_families']} open-weight families): "
            + pending(f"{n0(ds['total_recipes'])} recipes; Llama-3.1-8B run will add the 2nd open family"),
            13, space_before=6))  # fmt: skip
    else:
        note.append(Para("Proposal minimum for D1: **met**.", 13, space_before=6))
    if ds["curation_pending"]:
        note.append(
            Para(
                f"LLM lines fully aliased: {pct_rng(ds['alias_full_min'], ds['alias_full_max'])}; unmatched terms (e.g. "
                "“gochugaru”) go to human curation " + pending("curation/alias_overrides.csv"),
                13,
                space_before=6,
            )
        )
    y_note = 1.98 + deck.last_table_h + 0.12
    deck.text(s, 5.95, y_note, 6.85, 6.95 - y_note, note, min_scale=0.75)
    pend = [F["models"][k] for k in F["incomplete_models"]]
    deck.notes(
        s,
        f"""
WHAT TO SAY (about 60 s)
We asked every model for five traditional recipes per call, in two framings: a plain request and a home-cook persona
that grew up in a specific place of the region. {F["regions"]["n"]} regions × {D["n_conditions"]} prompts ×
{D["calls_per_cell"]} calls × {D["recipes_per_call"]} recipes gives {n0(D["recipes_per_model"])} recipes per model,
i.e. {D["recipes_per_region"]} per region, the same for every model. Prompts were frozen after a pilot and are
identical across models. Nothing in the prompt mentions flavor or pairing.

The ingredient lines are mapped to CulinaryDB entities with our reconstruction of CulinaryDB's own aliasing protocol.
We validated it on CulinaryDB's raw lines, which carry the expert mapping: {pct(al["acc_lines"])} of held-out unique
lines and {pct(al["acc_rows"])} of occurrences agree. That matters, because LLM and human recipes must be mapped the
same way for the comparison to be fair.

STATUS (read the table): complete runs: {join_and(F["complete_names"])}.
Still pending: {join_and([p["name"] + " " + p["status"] for p in pend]) if pend else "nothing"}.
So far {n0(ds["total_recipes"])} recipes; the proposal's minimum is {n0(D["proposal_min_recipes"])} recipes with two
open-weight families, which the Llama run completes.

IF ASKED "why these models?": free open-weight models on Kaggle GPUs plus two cheap proprietary APIs; total OpenAI
spend ${f(F["resources"]["openai_usd"])}. See qa_prep.md.
""",
    )


def s_method(deck: Deck, F, formulas: dict[str, Path]) -> None:
    s = deck.new("main", "method")
    header(deck, s, "How do we measure a cuisine's culinary fingerprint?",
           "With the lab's own definitions, unchanged, and one scoring function for real, LLM and copy-mutate "
           "cuisines.")  # fmt: skip
    nr = F["method"]["n_rand"]
    cards = [
        ("Flavor sharing", "ns", [
            "C_i = flavor compounds of ingredient i (FlavorDB). A cuisine's score is the mean over its recipes "
            "(Ahn et al. 2011; Jain et al. 2015, eq. 2)."]),
        ("Food pairing against 4 nulls", "z", [
            "Each null keeps the cuisine's own ingredients and recipe sizes: random, frequency, category, "
            f"frequency + category (Singh & Bagler 2018). N_rand = {n0(nr)}, so Z = {f(F['method']['sqrt_n_rand'], 0)} "
            "× effect size d."]),
        ("Category fingerprint", None, [
            "Re-draw one ingredient category at a time, keep everything else (Jain et al. 2015): the vector of "
            "ΔNs per category is the cuisine's fingerprint.",
            "Real vs LLM fingerprints are compared by Spearman ρ."]),
        ("Composition and stereotypes", "sai", [
            "Signature set Σ_c = top 10 ingredients by relative prevalence (Jain & Bagler 2018). Also: prevalence "
            "JSD and the diversity ratio (distinct ingredients vs a genuine sample of equal size)."]),
        ("Two baselines", None, [
            f"**Genuine samples:** {n0(F['equal_n']['n_genuine_per_region'])} random "
            f"{F['equal_n']['n_sub']}-recipe samples per region give the noise floor of every metric.",
            "**Copy-mutate:** Tuwani et al. 2019 Algorithm 1 with ranked fitness (Jain & Bagler 2018), i.e. what a "
            "simple cultural-evolution model reproduces."]),
        ("Statistics", None, [
            "95% bootstrap CIs; region is the unit; Wilcoxon signed-rank and Fisher exact tests; "
            "Benjamini–Hochberg FDR across regions (q = 0.05).",
            "The same function (flavoraudit.audit.scorecard_row) scores every cuisine."]),
    ]  # fmt: skip
    cw, ch, gx, gy = 3.95, 2.42, 0.2, 0.2
    for i, (title, fkey, body) in enumerate(cards):
        r, c = divmod(i, 3)
        x, y = 0.55 + c * (cw + gx), 1.98 + r * (ch + gy)
        deck.rect(s, x, y, cw, ch, T.panel)
        deck.rect(s, x, y, cw, 0.06, T.accent if i < 4 else T.blue)
        deck.text(s, x + 0.12, y + 0.1, cw - 0.24, 0.4, [Para(title, 16, bold=True, space_before=0)], fit=False)
        ty = y + 0.52
        if fkey:
            h_img = 0.62 if fkey != "z" else 0.66
            deck.picture(s, formulas[fkey], x + 0.12, ty, cw - 0.24, h_img, align="left")
            ty += h_img + 0.06
        deck.text(
            s,
            x + 0.12,
            ty,
            cw - 0.24,
            y + ch - ty - 0.05,
            [Para(b, 13, space_before=0 if j == 0 else 4) for j, b in enumerate(body)],
            min_scale=0.75,
        )
    deck.notes(
        s,
        f"""
WHAT TO SAY (about 60 s)
Everything here is your lab's definition, implemented exactly and cited in the code.
1) Ns of a recipe is the mean number of shared flavor compounds over its ingredient pairs; a cuisine's Ns is the mean
   over recipes.
2) Z compares the cuisine with null cuisines that keep its own ingredient set and recipe-size distribution. Important:
   with N_rand = {n0(nr)}, Z equals √N_rand ≈ {f(F["method"]["sqrt_n_rand"], 1)} times the standardised effect d =
   ΔNs/σ_rand. Cuisine size does not enter Z, so we always report d and compare models at equal size.
3) The culinary fingerprint is ΔNs when only one category is randomised at a time (Jain 2015).
4) Prevalence, relative prevalence and the signature set follow Jain & Bagler 2018; the Stereotype Amplification
   Index asks whether signature ingredients are used more often than in real recipes (1 = as real).
5) Baselines: genuine equal-size samples of real cuisines are the noise floor; the copy-mutate model is a baseline,
   not our contribution (Kinouchi 2008; Jain & Bagler 2018; Tuwani 2019; also our Assignment 3).
6) Region is the statistical unit; BH-FDR across the {F["regions"]["n"]} regions.
""",
    )


def s_replication(deck: Deck, F) -> None:
    s = deck.new("main", "replication")
    R = F["replication"]
    rnd = R["random"]
    header(deck, s, "Does our pipeline reproduce Singh & Bagler's Fig. 4?",
           f"Yes: {rnd.get('sign_agreement')}/{rnd.get('n')} signs against the random null and Spearman ρ = "
           f"{f(rnd.get('spearman'))} with the published Z, checked before any LLM recipe was analysed.")  # fmt: skip
    deck.picture(s, F["figures"]["F2b"], 0.55, 1.92, 6.0, 5.08, align="center")
    rows = [["Null model", "Signs", "Spearman ρ"]]
    for n in R["nulls"]:
        rows.append([n["name"], f"{n['signs']}/{n['n']}", f(n["rho"], 3)])
    deck.table(s, 6.85, 1.98, [3.15, 1.25, 1.55], rows, size=15, row_h=0.45, aligns=["l", "c", "c"])
    k = R["korea_top3"]
    top = ", ".join(f"{t['name']} {f(t['chi_percent'])}%" for t in k)
    pub = R["korea_top3_published"]
    pub_txt = ", ".join(f"{n} {v}%" for n, v in pub.items()) if pub else pending("published values")
    deck.text(s, 6.85, 4.45, 5.95, 2.5, [
        Para(f"**Korea's top contributors** (ours): {top}.", 15, space_before=0),
        Para(f"Published (Fig. 5): {pub_txt}. Same ranking; our magnitudes differ (% change in mean Ns).", 14,
             color=T.muted, space_before=4),
        Para("All four pre-registered acceptance criteria of the replication gate were met (plan §5, Phase 2).", 15,
             space_before=10),
        Para(minus(f"Known difference: contrasting cuisines come out more contrasting (Korea Z = {f(R['korea_z'], 1)} "
                   f"vs {R['korea_z_published']} published), most likely because the 2018 FlavorDB is no longer "
                   "online."), 14, color=T.muted, space_before=10),
    ], min_scale=0.75)  # fmt: skip
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 50 s)
Before touching any LLM recipe we had to reproduce your Fig. 4; that was a pre-registered gate.
HOW TO READ F2b: one row per region, filled markers are ours, open markers are the published values (read from the
CulinaryDB site figures). Blue dots are Z against the random null; the grey band marks the
{F["regions"]["n_contrasting"]} contrasting cuisines.
RESULT: {rnd.get("sign_agreement")}/{rnd.get("n")} signs against the random null; Spearman ρ = {f(rnd.get("spearman"), 3)}
with the published values; the table gives the other three nulls. As in the paper, the frequency-preserving nulls
bring Z close to zero (popularity explains much of the pattern) while the category null does not.
Korea's top contributor is Soybean Sauce, as published ({f(R["korea_top1_ours"])}% in our run vs
{R["korea_top1_published"]}% published); rankings replicate, magnitudes differ because the exact removal procedure is
not specified in the paper, so we calibrated it (details in docs/replication_report.md).

IF ASKED "why is Korea at {f(R["korea_z"], 1)} and not {R["korea_z_published"]}?": the 2018 FlavorDB snapshot is no
longer online; today's FlavorDB has updated molecule lists. Signs and ranks are preserved; a molecule-dropout
robustness analysis is planned. Remember Z = √N_rand · d, so a difference of {f(abs(R["korea_z"] - R["korea_z_published"]), 1)} in Z is a difference of {f(abs(R["korea_z"] - R["korea_z_published"]) / F["method"]["sqrt_n_rand"], 3)} in d.
"""),
    )


def s_sample_size(deck: Deck, F) -> None:
    s = deck.new("main", "sample size")
    E = F["equal_n"]
    flipped = join_and(E["uniform_flipped"])
    header(deck, s, "Does the food-pairing sign depend on how many recipes you sample?",
           f"Yes. At equal size ({E['matched_n']} recipes) all {E['contrasting_robust']} contrasting cuisines stay "
           f"contrasting, but {flipped} flip. So LLM cuisines must be compared with genuine samples of their own "
           "size.")  # fmt: skip
    deck.picture(s, F["figures"]["F2c"], 0.55, 1.92, 7.55, 5.08, align="left")
    x, w = 8.3, 4.5
    deck.tile(s, x, 1.98, w, 1.15, f"{E['contrasting_robust']} / {F['regions']['n_contrasting']}",
              f"contrasting cuisines are negative in every matched-size sample ({E['matched_n']} recipes)",
              color=T.accent, big_size=30, cap_size=14)  # fmt: skip
    deck.tile(s, x, 3.25, w, 1.15, f"{E['uniform_robust']} / {F['regions']['n_uniform']}",
              f"uniform cuisines stay positive in ≥ 95% of samples; flipped: {flipped}", color=T.blue, big_size=30,
              cap_size=14)  # fmt: skip
    top = ", ".join(f"{r} {100 * v:.0f}%" for r, v in E["fpr_top"][:2])
    deck.tile(s, x, 4.52, w, 1.15, pct(E["fpr"]),
              f"of genuine {E['n_sub']}-recipe samples trigger the pre-registered sign-error rule ({top})",
              color=T.ink, big_size=30, cap_size=14)  # fmt: skip
    u = E["usa"]
    deck.text(s, x, 5.78, w, 1.18, [Para(minus(
        f"**Why:** a small sample loses the rare, low-sharing ingredients, so its own-pool null scores higher. USA at "
        f"{E['n_sub']} recipes: ΔNs = {f(u.get('delta_ns_own_pool_n160'))} against its own "
        f"{n0(u.get('pool_n160'))}-ingredient pool, but {f(u.get('delta_ns_full_pool_n160'))} against the full pool "
        "(Backup B1)."), 13, space_before=0)], min_scale=0.75)  # fmt: skip
    weak = join_and(E["uniform_weakened"])
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 75 s) — this is our main methodological finding, take your time.
While calibrating the LLM comparison on genuine data we found that the own-pool random null is sample-size
dependent. The null draws from the sample's own ingredient pool. A small sample loses the cuisine's long tail of
rare ingredients, which share few compounds; the pool becomes richer in common, high-sharing ingredients, so the
null's mean Ns rises while the real recipes' mean Ns does not. ΔNs therefore falls as N shrinks.

HOW TO READ F2c: each row is a cuisine cut to {E["matched_n"]} recipes (just below the smallest cuisine,
{F["replication"]["smallest_region"]}). Dot = median effect size d over 200 random samples, line = 95% range,
diamond = full cuisine; vermillion = published contrasting, blue = published uniform. The text on the right is the
share of samples with d > 0.

RESULT: all {E["contrasting_robust"]} published contrasting cuisines are negative in every matched-size sample, so
the contrasting pattern of Fig. 4 is robust. {E["uniform_robust"]} of {F["regions"]["n_uniform"]} uniform cuisines stay
positive in at least 95% of samples, {weak} weaken, and {flipped} flip: their published positive sign relies on
their larger ingredient pool.

CONSEQUENCE FOR LLMs: our pre-registered sign-error rule compared a 160-recipe LLM cuisine with the full real cuisine.
On genuine 160-recipe samples that rule fires {pct(E["fpr"])} of the time overall, and
{", ".join(f"{r} {100 * v:.0f}%" for r, v in E["fpr_top"])} for the worst regions. So we added an equal-N comparison
(logged in the decision log before the full LLM analysis): an LLM cuisine is compared with
{n0(E["n_genuine_per_region"])} genuine samples of the same region at the same N. The old rule is still reported.

IF ASKED "are the contrasting cuisines just small?": {F["replication"]["contrasting_in_smallest7"]} of the
{F["regions"]["n_contrasting"]} contrasting cuisines are among CulinaryDB's 7 smallest, which would bias toward
"contrasting" — but at equal size they all stay contrasting, so the pattern is real.
"""),
    )


def s_pairing(deck: Deck, F) -> None:
    s = deck.new("main", "pairing")
    L, V = F["llm"], F["verdicts"]
    E = F["equal_n"]
    keys = list(L)
    if keys:

        def top(col):
            return max(keys, key=lambda k: (L[k][col] or 0) / max(L[k]["regions"], 1))

        typ, uni, con = top("equal_n_typical"), top("equal_n_too_uniform"), top("equal_n_too_contrasting")
        h1 = (
            "not supported for any model"
            if V["h1eq_supported"] == 0
            else f"supported for {V['h1eq_supported']}/{V['n']}"
        )
        parts = [f"{L[typ]['name']} is typical in {L[typ]['equal_n_typical']}/{L[typ]['regions']} regions"]
        if uni not in (typ,) and L[uni]["equal_n_too_uniform"]:
            parts.append(f"{L[uni]['name']} too uniform in {L[uni]['equal_n_too_uniform']}/{L[uni]['regions']}")
        if con not in (typ, uni) and L[con]["equal_n_too_contrasting"]:
            parts.append(f"{L[con]['name']} too contrasting in {L[con]['equal_n_too_contrasting']}/"
                         f"{L[con]['regions']}")  # fmt: skip
        answer = (
            f"No common drift to uniform pairing (H1 {h1}); the errors are model-specific. At equal size, "
            + join_and(parts)
            + "."
        )
    else:
        answer = pending("LLM audit not yet in results/summary.json")
    header(deck, s, "Do LLM cuisines reproduce each region's food-pairing sign?", answer)
    deck.picture(s, F["figures"]["F3"], 0.55, 1.92, 5.75, 5.08, align="left")
    rows = [["Model (regions)", "Typical", "Too uniform", "Too contr.", "Sign agr."]]
    for k in keys:
        p = L[k]
        rows.append([f"{p['name']} ({p['regions']})", str(p["equal_n_typical"]), str(p["equal_n_too_uniform"]),
                     str(p["equal_n_too_contrasting"]), f(p["sign_agreement"])])  # fmt: skip
    rows.append([f"Genuine {E['n_sub']}-recipe samples", "", "", "", f(E["genuine_sign_agreement"])])
    fills = {(len(rows) - 1, c): T.panel2 for c in range(5)}
    deck.table(s, 6.5, 1.98, [2.6, 0.85, 0.95, 0.95, 0.95], rows, size=13, row_h=0.42,
               aligns=["l", "c", "c", "c", "c"], fills=fills)  # fmt: skip
    y = 1.98 + deck.last_table_h + 0.12
    cons = F["consensus"]
    cons_txt = "; ".join(f"{c['region']} {c['verdict']} ({c['k']}/{c['k']})" for c in cons) or "none"
    paras = [
        Para(f"Equal-N verdicts: each LLM's effect size d against {n0(E['n_genuine_per_region'])} genuine samples of "
             "the same region and size (two-sided, BH). Sign agreement = same sign as the full cuisine.", 13,
             color=T.muted, space_before=0),
        Para(f"**Same error in every model with data:** {cons_txt}.", 14, space_before=8),
        Para("**Pre-registered rule** (sign errors cluster in contrasting regions, Fisher): "
             + "; ".join(f"{L[k]['name']} {L[k]['h1_fisher']['effect']}" for k in keys if L[k]["h1_fisher"])
             + f" (supported for {V['h1_fisher_supported']}/{V['n']}).", 13, space_before=8),
        Para("F3 compares each LLM cuisine with the full real cuisine; small samples look more contrasting "
             "(slide 6), so the equal-N verdicts are the primary test.", 13, color=T.muted, space_before=8),
    ]  # fmt: skip
    deck.text(s, 6.5, y, 6.3, 6.97 - y, paras, min_scale=0.7)
    shifts = "; ".join(
        f"{L[k]['name']} {sf(L[k]['equal_n_median_shift_d'])} ({padj(L[k]['h1eq_wilcoxon'])})" for k in keys
    )
    fishers = "; ".join(
        f"{L[k]['name']} {L[k]['h1eq_fisher']['effect']} ({padj(L[k]['h1eq_fisher'])})"
        for k in keys
        if L[k]["h1eq_fisher"]
    )
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 75 s)
HOW TO READ F3: one row per region, black diamond = real cuisine's Z against the random null, coloured markers = each
LLM cuisine. The x-axis is symmetric-log. Grey band = the {F["regions"]["n_contrasting"]} contrasting regions. Note
that F3 compares a {E["n_sub"]}-recipe LLM cuisine with the full real cuisine, which slide 6 showed is biased toward
"contrasting", so read it as descriptive.

PRIMARY TEST (equal N): we place each LLM cuisine's effect size d in the distribution of d over genuine samples of
the same region and size. Median shift of d (LLM minus genuine median) and Wilcoxon test of a shift toward uniform
(H1): {shifts}. H1 is {("not supported for any model" if V["h1eq_supported"] == 0 else "supported for " + str(V["h1eq_supported"]))}.
Direction × region type (Fisher, shift toward uniform in contrasting vs uniform regions): {fishers}.
The baseline for sign agreement is not 1: genuine {E["n_sub"]}-recipe samples agree with the full-cuisine sign
{f(E["genuine_sign_agreement"])} of the time on average.

THE STORY: there is no general drift toward uniform pairing. The errors are model-specific and region-specific;
the consistent ones are listed on the slide ({cons_txt}).

CAVEAT: {("Gemini covers " + str(L["gemini-3.5-flash-lite"]["regions"]) + " regions so far; ") if "gemini-3.5-flash-lite" in L else ""}
pending runs: {join_and(F["pending_models"]) or "none"}.
"""),
    )


def s_composition(deck: Deck, F) -> None:
    s = deck.new("main", "composition")
    L, V, E = F["llm"], F["verdicts"], F["equal_n"]
    rs = F["ranges"]
    if L:
        amp = "no model" if V["h2b_sai_supported"] == 0 else f"{V['h2b_sai_supported']}/{V['n']} models"
        answer = (
            f"No systematic amplification (median SAI {rng_text(*rs['median_sai'])}; H2b supported for {amp}), "
            f"but they narrow the cuisine: {pct_rng(*rs['median_diversity_ratio'])} of the distinct ingredients "
            "of a genuine sample of the same size."
        )
    else:
        answer = pending("LLM audit not yet in results/summary.json")
    header(deck, s, "Do LLMs amplify the stereotypes of each cuisine?", answer)
    deck.picture(s, F["figures"]["F8"], 0.55, 1.92, 6.05, 5.08, align="left")
    rows = [["Model", "Median SAI", "Diversity ratio", "Prevalence JSD"]]
    for p in L.values():
        rows.append([p["name"], f"{f(p['median_sai'])} ({pv(p['h2b_sai']['p_adj']) if p['h2b_sai'] else 'n/a'})",
                     f"{f(p['median_diversity_ratio'])} ({pv(p['h2b_div']['p_adj']) if p['h2b_div'] else 'n/a'})",
                     f(p["median_prevalence_jsd"], 3)])  # fmt: skip
    band = E["sai_band"]
    rows.append([f"Genuine {E['n_sub']}-recipe samples", f"{f(E['ref'].get('sai'))} ({f(band[0], 3)}–{f(band[1], 3)})",
                 "1 (by definition)", f(E["ref"].get("prevalence_jsd"), 3)])  # fmt: skip
    fills = {(len(rows) - 1, c): T.panel2 for c in range(4)}
    deck.table(s, 6.8, 1.98, [2.15, 1.35, 1.35, 1.15], rows, size=13, row_h=0.46, aligns=["l", "c", "c", "c"],
               fills=fills)  # fmt: skip
    y = 1.98 + deck.last_table_h + 0.12
    lo, hi = rs["median_prevalence_jsd"]
    ref = E["ref"].get("prevalence_jsd")
    ratio = f"{lo / ref:.0f}–{hi / ref:.0f}×" if lo and ref else pending("ratio")
    deck.text(s, 6.8, y, 6.0, 6.97 - y, [
        Para("In brackets: BH-adjusted p of the one-sided Wilcoxon test over regions (SAI > 1; diversity < 1); "
             "genuine SAI with its 95% range.", 12, color=T.muted, space_before=0),
        Para(f"**Prevalence** differs {ratio} more from the real cuisine than a genuine sample of the same size does.",
             15, space_before=10),
        Para(f"**Diversity** is lower in {V['h2b_div_supported']}/{V['n']} models: LLMs reuse a narrower set of "
             "ingredients.", 15, space_before=8),
        Para(drivers_text(F), 14, space_before=8),
    ], min_scale=0.7)  # fmt: skip
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 60 s)
HOW TO READ F8: one row per region; the grey bar is the 95% range of SAI over genuine {E["n_sub"]}-recipe samples
({f(band[0], 3)} to {f(band[1], 3)}); markers are the LLMs (log scale; 1 = signature ingredients used exactly as
often as in real recipes).
RESULT: H2b predicted SAI > 1 (stereotype amplification). The median SAI is {rng_text(*rs["median_sai"])} across
models and the one-sided Wilcoxon test is not significant for {"any" if V["h2b_sai_supported"] == 0 else "all"} model(s).
Per region there are large deviations in both directions, so the models are not uniformly "over-signatured".
What they do is narrow the cuisine: the diversity ratio (distinct ingredients relative to a genuine sample of the same
size) is {rng_text(*rs["median_diversity_ratio"])}, significant in {V["h2b_div_supported"]}/{V["n"]} models; and
the ingredient-prevalence JSD is {rng_text(*rs["median_prevalence_jsd"], 3)} against
{f(ref, 3)} for genuine samples, i.e. {ratio} the sampling noise.
ONE-LINER: LLMs don't caricature a cuisine with more signature ingredients; they cook a smaller, more repetitive
version of it.
"""),
    )


def s_fingerprint(deck: Deck, F) -> None:
    s = deck.new("main", "fingerprint")
    L, V, E, C = F["llm"], F["verdicts"], F["equal_n"], F["cm"]
    rs = F["ranges"]
    cmr = C["ranked_n"]
    if L:
        answer = (
            f"The category fingerprint is weakened in {V['h2c_supported']}/{V['n']} models (ρ "
            f"{rng_text(*rs['median_fingerprint_spearman'])} vs {f(E['ref'].get('fingerprint_spearman'))} for "
            "genuine samples), yet LLMs beat copy-mutate on composition and fingerprints, not on food-pairing "
            "strength."
        )
    else:
        answer = pending("LLM audit not yet in results/summary.json")
    header(deck, s, "Do LLMs keep the category fingerprint, and beat a copy-mutate model?", answer)
    deck.picture(s, F["figures"]["F13"], 0.55, 1.92, 8.3, 3.72, align="left")
    below = [f"{L[k]['name']} {L[k]['h2c_regions_below']}/{L[k]['regions']}" for k in L]
    cmb, n = V["cm_better"], V["n"]
    deck.text(s, 0.55, 5.72, 8.3, 1.25, [
        Para(f"**LLM vs copy-mutate** (Wilcoxon over regions, BH; models where the LLM is closer to real): prevalence "
             f"JSD {cmb['prevalence_jsd']}/{n} · fingerprint ρ {cmb['fingerprint_spearman']}/{n} · |log SAI| "
             f"{cmb['abs_log_sai']}/{n} · food-pairing gap {cmb['abs_d_gap']}/{n}"
             + (f" ({join_and(F['cm_gap_models'])})" if F["cm_gap_models"] else "") + ".", 14, space_before=0),
        Para(f"Regions whose LLM fingerprint falls below the genuine resampling range (BH): {'; '.join(below)}.", 13,
             color=T.muted, space_before=6),
    ], min_scale=0.7)  # fmt: skip
    x, w = 9.1, 3.7
    deck.tile(
        s,
        x,
        1.98,
        w,
        1.45,
        rng_text(*rs["median_fingerprint_spearman"]),
        f"median fingerprint ρ of the LLMs (genuine samples {f(E['ref'].get('fingerprint_spearman'))}; "
        f"copy-mutate {f(cmr.get('fingerprint_spearman'))})",
        color=T.accent,
        big_size=28,
        cap_size=13,
    )
    deck.tile(s, x, 3.55, w, 1.45, f(cmr.get("prevalence_jsd")),
              f"copy-mutate prevalence JSD (LLMs {rng_text(*rs['median_prevalence_jsd'])}; genuine "
              f"{f(E['ref'].get('prevalence_jsd'), 3)}); its SAI is {f(cmr.get('sai'))}", color=T.ink, big_size=28,
              cap_size=13)  # fmt: skip
    deck.tile(
        s,
        x,
        5.12,
        w,
        1.45,
        pct(cmr.get("sign_match"), 0),
        f"of regions where copy-mutate gets the pairing sign right; it does reproduce the popularity curve "
        f"(rank-frequency MAE {f(cmr.get('freq_rank_mae'), 3)})",
        color=T.blue,
        big_size=28,
        cap_size=13,
    )
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 70 s)
HOW TO READ F13: four panels, one box per source, one point per region: genuine {E["n_sub"]}-recipe samples (dark
grey), the LLMs, and the copy-mutate model (light grey, {C["key"]}). Lower is better for JSD and MAE, higher is better
for fingerprint ρ, 1 is ideal for the diversity ratio.
H2c — the category fingerprint: genuine resamples of the same region correlate with the full cuisine's fingerprint at
ρ ≈ {f(E["ref"].get("fingerprint_spearman"))} (not 1, because of sampling noise). LLM fingerprints correlate at
{rng_text(*rs["median_fingerprint_spearman"])}; the one-sided Wilcoxon test is significant in
{V["h2c_supported"]}/{V["n"]} models. Regions below the resampling range: {"; ".join(below)}.
COPY-MUTATE is our baseline, not a contribution: it is Tuwani et al. 2019 Algorithm 1 with Jain & Bagler's ranked
fitness, i.e. the lab's own model and our Assignment 3. It reproduces the rank-frequency curve (MAE
{f(cmr.get("freq_rank_mae"), 3)}) but not composition (JSD {f(cmr.get("prevalence_jsd"))}), signatures (SAI
{f(cmr.get("sai"))}) or fingerprints (ρ {f(cmr.get("fingerprint_spearman"))}), and its pairing sign is a coin flip
({pct(cmr.get("sign_match"), 0)} of regions).
LLMs beat it on prevalence, fingerprint and SAI in every model, but on the food-pairing gap only
{join_and(F["cm_gap_models"]) or "no model"} is significantly better: knowing which ingredients belong to a cuisine
does not automatically give its pairing structure.
"""),
    )


def s_persona(deck: Deck, F) -> None:
    s = deck.new("main", "persona")
    L, V = F["llm"], F["verdicts"]
    n = V["n"]
    if L:
        rho = join_and(F["persona_rho_models"]) if F["persona_rho_models"] else "no model"
        answer = (
            f"Partly. The persona brings ingredient prevalence closer to real in {V['persona_jsd_better']}/{n} "
            f"models and improves the fingerprint only for {rho}; food pairing is unchanged "
            f"({V['persona_gap_better']}/{n})."
        )
    else:
        answer = pending("LLM audit not yet in results/summary.json")
    header(deck, s, "Does a home-cook persona make the recipes more faithful?", answer)
    deck.picture(s, F["figures"]["F14"], 0.55, 1.92, 12.25, 4.25, align="center")

    def cell(k, met):
        r = L[k]["persona"][met]
        return f"{sf(r['effect_num'], 3)} ({padj(r)})" if r else pending("test")

    P = F["proprietary"]
    prop_txt = "; ".join(
        f"{F['metric_names'][m]} {sf(P[m]['effect_num'], 3)} ({padj(P[m])})"
        for m in ("abs_d_gap", "prevalence_jsd")
        if m in P
    )
    scale = F["scale"]
    deck.text(s, 0.55, 6.22, 12.25, 0.78, [
        Para(f"**Persona − plain** (Wilcoxon over regions, BH): prevalence JSD lower in {V['persona_jsd_better']}/{n} · "
             f"fingerprint ρ higher in {V['persona_rho_better']}/{n} · pairing gap smaller in "
             f"{V['persona_gap_better']}/{n} · |log SAI| smaller in {V['persona_sai_better']}/{n}.", 14,
             space_before=0),
        Para(f"**Proprietary − open** (open = {join_and(F['open_models'])} so far): {prop_txt}. Scale (Qwen 9B vs 4B): "
             + ("see backup" if scale else pending("Qwen3.5-4B run")) + ".", 13, color=T.muted, space_before=4),
    ], min_scale=0.7)  # fmt: skip
    lines = "\n".join(
        f"  {L[k]['name']}: JSD {cell(k, 'prevalence_jsd')}; ρ {cell(k, 'fingerprint_spearman')}; "
        f"gap {cell(k, 'abs_d_gap')}; |log SAI| {cell(k, 'abs_log_sai')}"
        for k in L
    )
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 50 s)
HOW TO READ F14: each point is a region; the value is persona minus plain for that model (each condition has
{F["design"]["recipes_per_region"] // 2 if F["design"]["recipes_per_region"] else "?"} recipes per region); the bar is the median.
Below zero is better for JSD and SAI deviation, above zero is better for fingerprint ρ and diversity.
RESULT (persona − plain, Wilcoxon over regions, BH-adjusted):
{lines}
In words: asking the model to answer as a home cook from a specific place makes the ingredient mix more realistic and
more varied (right panel), but it does not fix food pairing. H4 predicted a slight improvement: that is what we see,
for composition only.
PROPRIETARY VS OPEN (H4): proprietary models (Gemini, GPT-6-Luna) vs open ({join_and(F["open_models"])}): {prop_txt}.
Positive = proprietary further from real. Provisional: only one open-weight model is analysed so far.
SCALE: {("available" if scale else "pending — needs the Qwen3.5-4B run")}.
"""),
    )


def s_bon(deck: Deck, F) -> None:
    s = deck.new("main", "steering")
    B, SC, SD = F["bon"], F["scorer"], F["steer_design"]
    if B:
        rows = B["rows"]
        cand = [r for k, r in rows.items() if F["models"].get(k, {}).get("complete")] or list(rows.values())
        best = max(cand, key=lambda r: r["last"]["indep_classifier_acc"] - r["first"]["indep_classifier_acc"])
        up = "for every model" if B["all_acc_up"] else "for some models"
        answer = (
            f"Best-of-N makes recipes more recognisable ({best['name']}: an independent classifier assigns "
            f"{pct(best['first']['indep_classifier_acc'], 0)} → {pct(best['last']['indep_classifier_acc'], 0)} "
            f"to the claimed cuisine, N = {best['N_min']} → {best['N_max']}; rising {up}) but less diverse; "
            "cuisine-level pairing and fingerprints do not improve."
        )
    else:
        answer = pending("best-of-N results not yet in results/summary.json")
    header(deck, s, "Can the fidelity scorer steer an LLM by picking its best recipes?", answer)
    if F["figures"]["F15"]:
        deck.picture(s, F["figures"]["F15"], 0.55, 1.92, 12.25, 3.68, align="center")
    else:
        deck.text(s, 0.55, 1.92, 12.25, 3.68, [Para(pending("F15 steering figure"), 18, align="center")],
                  anchor="middle", fill=T.panel)  # fmt: skip
    auc = SC.get("auc_test", {})
    comp = " / ".join(f(auc.get(k, {}).get("composite")) for k in ("other_cuisines", "copy_mutate", "random_null"))
    deck.rect(s, 0.55, 5.72, 6.0, 1.25, T.panel)
    deck.text(s, 0.65, 5.75, 5.85, 1.2, [
        Para("**Fidelity scorer:** " + f"{SC.get('classifier_name')} cuisine classifier + co-occurrence PMI + Ns "
             f"consistency, {SC.get('weighting')} weights (chosen on validation).", 13, space_before=0),
        Para(f"Test macro-F1 {f(SC.get('test_macro_f1'))} · top-3 {f(SC.get('test_top3'))} · ECE "
             f"{f(SC.get('ece_raw'))} → {f(SC.get('ece_calibrated'))} after temperature scaling · AUC vs other "
             f"cuisines / copy-mutate / null recipes: {comp}.", 13, space_before=4),
    ], min_scale=0.7)  # fmt: skip
    meta = (B or {}).get("meta", {})
    deck.rect(s, 6.8, 5.72, 6.0, 1.25, T.panel)
    deck.text(s, 6.9, 5.75, 5.85, 1.2, [
        Para(f"**Guards:** near-copies (Jaccard ≥ {F['method']['near_copy']} to any CulinaryDB recipe) and duplicates "
             f"are never selected (near-copy rate {f((B or {}).get('near_copy_max'))}); evaluation uses held-out "
             "metrics only, never the scorer.", 13, space_before=0),
        Para(f"**Deviation:** reranked the existing {meta.get('pool_max', '?')}-recipe pool per region, "
             f"{meta.get('selected_max', '?')} picks per N (pre-registered: {n0(SD['pool_target'])} candidates, "
             f"{SD['n_selected_planned']} picks); see Q&A.", 13, space_before=4),
    ], min_scale=0.7)  # fmt: skip
    bon_lines = ""
    if B:
        for k in B["order"]:
            r = B["rows"][k]
            a, b = r["first"], r["last"]
            bon_lines += minus(
                f"\n  {r['name']}: classifier share {f(a['indep_classifier_acc'])} → {f(b['indep_classifier_acc'])}; "
                f"diversity {f(a['diversity_ratio'])} → {f(b['diversity_ratio'])}; JSD {f(a['prevalence_jsd'], 3)} → "
                f"{f(b['prevalence_jsd'], 3)}; fingerprint ρ {f(a['fingerprint_spearman'])} → "
                f"{f(b['fingerprint_spearman'])}; sign agreement {f(a['sign_match'])} → {f(b['sign_match'])}"
            )
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 70 s)
THE SCORER (D3): three components, each standardised against genuine recipes of the cuisine: a {SC.get("classifier_name")}
cuisine classifier (linear so that per-ingredient contributions can drive feedback and the app's suggestions), the
mean shrinkage-smoothed PMI of the recipe's ingredient pairs within the cuisine, and how typical its Ns is. Test
macro-F1 {f(SC.get("test_macro_f1"))} over {F["regions"]["n"]} classes (chance {f(SC.get("chance"), 3)}), top-3
{f(SC.get("test_top3"))}; composite AUC against other cuisines / copy-mutate / null recipes: {comp}.

BEST-OF-N: from each model's existing recipes per region we repeatedly take N candidates and keep the best-scoring
admissible one (not a near-copy of a CulinaryDB recipe, not a duplicate). N = 1 is the unsteered control under the
same guards. We evaluate only with held-out metrics, never the scorer.
HOW TO READ F15: x = N, one line per model, mean over regions with 95% bootstrap CIs.
RESULTS (N = {(B or {}).get("rows", {}) and min(r["N_min"] for r in B["rows"].values())} → {(B or {}).get("rows", {}) and max(r["N_max"] for r in B["rows"].values())}):{bon_lines}
Honest reading: reranking makes recipes more recognisable to an independent classifier, but it narrows them (lower
diversity) and, per F15, pushes SAI up and recipes closer to existing CulinaryDB recipes (higher max-Jaccard), while
cuisine-level pairing sign, fingerprint and prevalence JSD do not improve at {meta.get("selected_max", "?")} recipes per
region. That is a trade-off, not a free win, and it is why critique-and-revise (which edits a recipe instead of
selecting one) is the next slide.

DEVIATION TO DISCLOSE: the plan pre-registered a {n0(SD["pool_target"])}-candidate pool per region. Generating it needs
{n0(SD["extra_calls_per_model"])} extra calls per model; at the measured Kaggle rate ({f(SD["minutes_per_call"], 2)} min
per call on the T4 transformers fallback; Kaggle full runs give {f(SD["gpu_hours_min"], 0)}–{f(SD["gpu_hours_max"], 0)}
GPU-hours per model) that is too much for the free GPU quota, so we reranked
the existing samples instead and documented the deviation.
"""),
    )


def s_revise(deck: Deck, F) -> None:
    s = deck.new("main", "revise")
    RV = F["revise"]
    if RV:
        rows = RV["rows"]
        parts = [
            f"{r['name']} {f(r['first']['indep_classifier_acc'])} → {f(r['last']['indep_classifier_acc'])} in "
            f"{r['rounds']} round{'s' if r['rounds'] > 1 else ''}"
            for r in rows.values()
        ]
        n = RV["n_models"]
        jsd = RV["improved"]["prevalence_jsd"]
        answer = (
            f"Yes, modestly: the independent classifier share rises ({'; '.join(parts)}) and prevalence JSD falls "
            f"in {jsd}/{n} models, while revised recipes keep a Jaccard overlap of "
            f"{rng_text(RV['kept_min'], RV['kept_max'])} with the original dish."
        )
    else:
        answer = pending("critique-and-revise results not yet in results/summary.json")
    header(deck, s, "Does fingerprint feedback improve a recipe while keeping the dish?", answer)
    deck.picture(s, F["figures"]["F15c"], 0.55, 1.92, 12.25, 3.95, align="center",
                 missing=pending("F15c critique-and-revise figure"))  # fmt: skip
    src = "the unsteered best-of-N picks (N = 1)" if RV and RV["all_from_bon_n1"] else "each model's recipes"
    imp = RV["improved"] if RV else {}
    pend_rounds = [p for p in F["pending"] if p["item"].startswith("D3 critique")]
    deck.text(s, 0.55, 5.95, 12.25, 1.02, [
        Para("**Feedback** (Self-Refine; Madaan et al. 2023): up to 3 ingredients unusual for the cuisine, up to 3 "
             "missing signature ingredients that pair well, and whether Ns lies outside the cuisine's interquartile "
             f"range. Applied to {src}; scored on held-out metrics only.", 13, space_before=0),
        Para(f"Models improving from round 0 to the last round: classifier share {imp.get('indep_classifier_acc', '?')}/"
             f"{RV['n_models'] if RV else '?'} · prevalence JSD {imp.get('prevalence_jsd', '?')}/{RV['n_models'] if RV else '?'} "
             f"· fingerprint ρ {imp.get('fingerprint_spearman', '?')}/{RV['n_models'] if RV else '?'} · sign agreement "
             f"{imp.get('sign_match', '?')}/{RV['n_models'] if RV else '?'}. No CIs yet. "
             + (" ".join(p["detail"] for p in pend_rounds)), 13, color=T.muted, space_before=4),
    ], min_scale=0.7)  # fmt: skip
    lines = ""
    if RV:
        for k in RV["order"]:
            r = RV["rows"][k]
            a, b = r["first"], r["last"]
            lines += minus(
                f"\n  {r['name']} (round 0 → {r['rounds']}): classifier {f(a['indep_classifier_acc'])} → "
                f"{f(b['indep_classifier_acc'])}; JSD {f(a['prevalence_jsd'], 3)} → {f(b['prevalence_jsd'], 3)}; "
                f"fingerprint ρ {f(a['fingerprint_spearman'])} → {f(b['fingerprint_spearman'])}; sign agreement "
                f"{f(a['sign_match'])} → {f(b['sign_match'])}; ingredient Jaccard with the original "
                f"{f(b.get('identity_jaccard'))}"
            )
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 60 s)
Best-of-N can only choose among recipes the model already wrote. Critique-and-revise instead tells the model what is
off, using the scorer's interpretable parts: the linear classifier's most atypical ingredients, the signature
ingredients that are missing but pair well (positive PMI with the rest of the recipe), and whether the recipe's flavor
sharing is outside the cuisine's interquartile range. The model is asked to revise the recipe but keep the dish.
HOW TO READ F15c: x = revision round (0 = original); panels: independent classifier share, fingerprint ρ, prevalence
JSD, and the Jaccard overlap between the revised and the original ingredient sets (how much of the dish is kept).
RESULTS:{lines}
Honest reading: feedback moves recipes toward the cuisine on the held-out metrics while keeping most of each dish,
so it does better than reranking, which gained recognisability but lost diversity. These are means over regions
without confidence intervals yet, at small cuisine sizes, so treat them as promising, not conclusive.
{"Pending: " + " ".join(p["detail"] for p in pend_rounds) if pend_rounds else ""}
"""),
    )


def s_tools(deck: Deck, F) -> None:
    s = deck.new("main", "tools")
    LB = F["library"]
    header(
        deck,
        s,
        "How can anyone audit their own recipes?",
        "With FlavorAudit, a tested Python library (D4), and FlavorPrint, a web app built on it (D5).",
    )
    for x in (0.55, 6.8):
        deck.rect(s, x, 1.98, 6.0, 4.97, T.panel)
    mods = " · ".join(LB["modules"])
    deck.text(s, 0.7, 2.05, 5.7, 1.9, [
        Para("D4  FlavorAudit (pip-installable, MIT)", 18, bold=True, color=T.accent, space_before=0),
        Para(f"{LB['n_modules']} modules: {mods}.", 13, space_before=6),
        Para(f"{LB['n_tests']} tests in {LB['n_test_files']} files, incl. property tests and a slow regression that "
             f"reproduces the Fig. 4 signs; CI on Python {join_and(LB['ci_python'])}; MkDocs site with "
             "“Replicate Singh & Bagler in 10 lines”.", 13, space_before=6),
    ], min_scale=0.7)  # fmt: skip
    code = [
        "from flavoraudit.data import load_culinarydb, load_flavordb, load_ahn",
        "from flavoraudit.flavor import build_profiles",
        "from flavoraudit.pairing import build_cuisines",
        "from flavoraudit.stats import food_pairing",
        "cdb, fdb, ahn = load_culinarydb(), load_flavordb(), load_ahn()",
        "S, index = build_profiles(cdb, fdb, ahn).shared_matrix()",
        "cuisines = build_cuisines(...)   # see README quick start",
        'food_pairing(cuisines["Korea"], S, null_model="random").z_sb',
        "",
        "$ flavoraudit audit recipes.csv --cuisine-col cuisine --out out/",
    ]
    deck.rect(s, 0.7, 4.0, 5.7, 2.35, "#FFFFFF", line=T.rule)
    deck.text(s, 0.78, 4.04, 5.55, 2.28, [Para(c or " ", 10.5, font=MONO, space_before=0) for c in code],
              min_scale=0.8)  # fmt: skip
    deck.text(s, 0.7, 6.4, 5.7, 0.5, [Para("PyPI release and public docs " + pending("need the team's go-ahead"), 12,
                                           color=T.muted, space_before=0)], fit=False)  # fmt: skip
    steps = [
        "Paste ingredient lines and pick the claimed cuisine.",
        "Lines are mapped to CulinaryDB entities with the same aliaser as the study.",
        "The recipe's Ns is placed in the cuisine's real distribution; its flavor network is drawn.",
        "Fidelity percentile from the three scorer components; top-5 predicted cuisines.",
        "Suggestions: atypical ingredients, missing signature ingredients that pair well, and a stereotype flag.",
    ]
    paras = [Para("D5  FlavorPrint (Streamlit)", 18, bold=True, color=T.accent, space_before=0)]
    paras += [Para(f"{i + 1}.  {t}", 14, space_before=7) for i, t in enumerate(steps)]
    paras += [
        Para("Plus a Dashboard tab (model × region results) and an About tab (methods, data, licences).", 13,
             color=T.muted, space_before=10),
        Para("Runs locally today; Hugging Face Space (Docker) is packaged " + pending("deployment needs the team's "
                                                                                     "go-ahead") + ".", 13,
             color=T.muted, space_before=6),
        Para("→ Live demo: genuine Korean recipe vs an Italian dish claimed as Korean.", 15, bold=True,
             space_before=12),
    ]  # fmt: skip
    deck.text(s, 6.95, 2.05, 5.7, 4.85, paras, min_scale=0.7)
    deck.notes(
        s,
        f"""
WHAT TO SAY (about 40 s, then switch to the demo — see demo_script.md)
D4 is FlavorAudit: the whole framework as a library with {LB["n_modules"]} modules and {LB["n_tests"]} tests, including a
slow regression test that reproduces the Fig. 4 signs from scratch. It has no dependency on our research code and
never bundles CulinaryDB/FlavorDB data inside the MIT package: the data are downloaded on first use under their
CC BY-NC-SA licence. The snippet replicates Korea's Z in a few lines; the CLI audits any CSV of recipes.

D5 is FlavorPrint, a Streamlit app built on the library: paste a recipe, pick the claimed cuisine, and it shows the
aliased ingredients, where the recipe's flavor sharing lies among real recipes of that cuisine, the fidelity score
with its three components, the most likely cuisines and concrete suggestions.

STATUS: runs locally; the Hugging Face Space and the PyPI release are prepared but publishing waits for the team's
go-ahead.
DEMO: switch to the browser now (3 minutes). If the demo fails, use Backup "Demo fallback".
""",
    )


def s_conclusions(deck: Deck, F) -> None:
    s = deck.new("main", "conclusions")
    L, V, E, R = F["llm"], F["verdicts"], F["equal_n"], F["replication"]
    rs = F["ranges"]
    header(deck, s, "So, do AI chefs respect culinary fingerprints?",
           "Partly. They use a cuisine's signature ingredients about as real cooks do, but lose its breadth and its "
           "category fingerprint; food pairing depends on the model.")  # fmt: skip
    rnd = R["random"]
    cons = "; ".join(f"{c['region']} {c['verdict']}" for c in F["consensus"] if c["k"] == F["n_analysed"]) or "none"
    typ = "; ".join(f"{p['name']} typical in {p['equal_n_typical']}/{p['regions']}" for p in L.values())
    items = [
        (False, f"**Replication:** the lab's framework is reproduced ({rnd.get('sign_agreement')}/{rnd.get('n')} signs, "
                f"ρ = {f(rnd.get('spearman'))})."),
        (False, f"**Sample size matters:** the own-pool null drifts with N; at equal size the "
                f"{E['contrasting_robust']} contrasting cuisines hold, {join_and(E['uniform_flipped'])} flip; the "
                f"naive sign rule misfires on {pct(E['fpr'])} of genuine samples, hence an equal-N reference."),
        (True, f"**Pairing:** no drift toward uniform (H1 {V['h1eq_supported']}/{V['n']}); {typ}; same error in all "
               f"models: {cons}."),
        (True, f"**Composition:** no stereotype amplification (SAI {rng_text(*rs['median_sai'])}), but narrower "
               f"cuisines (diversity {rng_text(*rs['median_diversity_ratio'])})."),
        (True, f"**Fingerprints** weakened in {V['h2c_supported']}/{V['n']} models"
               + (f"; top flavor drivers overlap less than in genuine samples (Jaccard@5 "
                  f"{rng_text(F['drivers']['jac_min'], F['drivers']['jac_max'])} vs "
                  f"{f(F['drivers']['real'].get('median_jaccard5'))})" if F["drivers"] else "")
               + "; LLMs still beat copy-mutate on composition and fingerprints. A persona improves composition, not "
                 "pairing."),
        (True, "**Steering:** best-of-N raises recognisability but lowers diversity; " + (
            f"critique-and-revise improves classifier share and prevalence JSD "
            f"({F['revise']['improved']['indep_classifier_acc']}/{F['revise']['n_models']} and "
            f"{F['revise']['improved']['prevalence_jsd']}/{F['revise']['n_models']} models) while keeping the dish."
            if F["revise"] else "critique-and-revise " + pending("running") + ".")),
    ]  # fmt: skip
    y = 1.95
    avail = 6.4 - y
    per = avail / len(items)
    for prov, text in items:
        t = "PROVISIONAL" if (prov and F["llm_pending"]) else "FINAL"
        color = T.amber if t == "PROVISIONAL" else T.green
        deck.rect(s, 0.55, y + 0.1, 1.55, 0.36, color, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        deck.text(s, 0.55, y + 0.1, 1.55, 0.36, [Para(t, 10.5, bold=True, color="#FFFFFF", align="center",
                                                      space_before=0)], anchor="middle", fit=False)  # fmt: skip
        deck.text(s, 2.25, y, 10.55, per - 0.04, [Para(text, 15, space_before=0)], anchor="top", min_scale=0.72)
        y += per
    deck.rect(s, 0.55, 6.45, 12.25, 0.52, T.ink)
    deck.text(s, 0.55, 6.45, 12.25, 0.52, [Para("Thank you. Questions?   (backup slides and qa_prep.md cover the "
                                                "likely ones)", 18, bold=True, color="#FFFFFF", align="center",
                                                space_before=0)], anchor="middle", fit=False)  # fmt: skip
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 60 s)
Answer to the title question: partly.
FINAL (complete data): the replication, and the sample-size finding with its consequence that LLM cuisines must be
compared with genuine samples of their own size.
PROVISIONAL (more LLM data pending: {join_and(F["pending_models"]) or "none"}): no drift toward uniform pairing; errors
are model- and region-specific; no stereotype amplification but narrower cuisines; fingerprints weakened; LLMs beat
copy-mutate except on pairing strength; persona helps composition; best-of-N trades diversity for recognisability.
WHAT IS NEW vs BASELINE: new = the LLMCuisine-22 dataset, the first flavor-compound audit of LLM recipes, the
equal-N reference that exposes the sample-size dependence, fingerprint-guided steering, the library and the app.
Reused (not contributions) = the nulls, Z and χ (Ahn 2011; Jain 2015; Singh & Bagler 2018) and the copy-mutate model
(Kinouchi 2008; Jain & Bagler 2018; Tuwani 2019; our A2/A3).
Keep this slide up during questions. Backups: B1 sample-size mechanism, B2 all pre-registered tests, B3 scorecard,
B4 effect sizes, B5 steering ablation, B6 demo fallback, B7 credits and licences. qa_prep.md has prepared answers.
"""),
    )


def s_limitations(deck: Deck, F) -> None:
    s = deck.new("main", "limitations")
    E, ds, SD = F["equal_n"], F["dataset"], F["steer_design"]
    meta = (F["bon"] or {}).get("meta", {})
    header(
        deck,
        s,
        "What could change these conclusions?",
        "Data versions, an incomplete model roster and small LLM cuisines. Each is flagged, tested or pending.",
    )
    pend = "; ".join(f"{F['models'][k]['name']} {F['models'][k]['status']}" for k in F["incomplete_models"]
                     if F["models"][k]["tier"] == "core")  # fmt: skip
    left = [
        Para("Data and method", 19, bold=True, color=T.accent, space_before=0),
        Para(minus(f"The 2018 FlavorDB is offline; today's version makes contrasting cuisines more contrasting "
                   f"(Korea Z {f(F['replication']['korea_z'], 1)} vs {F['replication']['korea_z_published']}). "
                   "Signs and ranks replicate; molecule-dropout robustness ") + pending("not yet run") + ".", 15,
             bullet=True, space_before=8),
        Para(f"LLM cuisines have {F['design']['recipes_per_region']} recipes per region; genuine samples of that size "
             f"share the full cuisine's sign only {pct(E['genuine_sign_agreement'], 0)} of the time on average, so we "
             "test at region level against equal-size samples.", 15, bullet=True, space_before=8),
        Para(f"LLM lines are aliased before curation ({pct_rng(ds['alias_full_min'], ds['alias_full_max'])} fully "
             "matched); unmatched terms await human review " + pending("curation") + ".", 15, bullet=True,
             space_before=8),
        Para("“Real” means CulinaryDB: web recipes (AllRecipes, Food Network, Epicurious, TarlaDalal), not home "
             "cooking.", 15, bullet=True, space_before=8),
    ]  # fmt: skip
    right = [
        Para("Roster and steering", 19, bold=True, color=T.accent, space_before=0),
        Para(f"Core runs still pending: {pend or 'none'}. Open-weight families so far: {ds['n_open_families']} "
             f"({join_and(F['open_models'])}), so the proprietary-vs-open contrast is provisional; scale contrast "
             + pending("Qwen3.5-4B") + ".", 15,
             bullet=True, space_before=8),
        Para(f"Best-of-N reranked {meta.get('pool_max', '?')} existing recipes per region instead of the pre-registered "
             f"{n0(SD['pool_target'])}; with {meta.get('selected_max', '?')} picks per N, cuisine-level metrics are "
             "noisy.", 15, bullet=True, space_before=8),
        Para("The independent evaluation classifier is still an ingredient-based cuisine classifier, so part of its "
             "gain is shared with the scorer (see F15b).", 15, bullet=True, space_before=8),
        Para("No human or chef rating of taste or authenticity yet"
             + (f"; steering not yet run for {join_and(F['unsteered_models'])}" if F["unsteered_models"] else "")
             + ".", 15, bullet=True, space_before=8),
    ]  # fmt: skip
    deck.text(s, 0.55, 1.98, 6.0, 4.97, left, min_scale=0.72)
    deck.text(s, 6.8, 1.98, 6.0, 4.97, right, min_scale=0.72)
    deck.notes(
        s,
        f"""
WHAT TO SAY (about 45 s)
Be upfront: these are the points a careful reviewer would raise, and each one is either measured, tested or explicitly
pending. The most important: (1) FlavorDB drift, which we handle by replicating signs and ranks and by planned
perturbation analyses; (2) small LLM cuisines, which is exactly why we built the equal-N reference; (3) the incomplete
roster: {pend or "none"}; (4) steering was run on the existing recipes, a documented, compute-driven deviation.
""",
    )


def b_credits(deck: Deck, F) -> None:
    s = deck.new("backup", "credits")
    header(
        deck,
        s,
        "Data, licences and references",
        "Everything we used is credited; derived data stay share-alike, code is MIT.",
    )
    chips = [("D1", "LLMCuisine-22"), ("D2", "Audit"), ("D3", "Steering"), ("D4", "FlavorAudit"), ("D5", "FlavorPrint")]
    for i, (d, name) in enumerate(chips):
        cx = 0.9 + i * 2.39
        deck.rect(s, cx, 2.0, 2.26, 0.62, T.panel, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        deck.text(s, cx, 2.0, 2.26, 0.62, [Para(f"**{d}** {name}", 15, align="center", space_before=0)],
                  anchor="middle", fit=False)  # fmt: skip
    deck.text(s, 0.9, 2.85, 11.9, 4.1, [
        Para("Data and licences", 17, bold=True, space_before=0),
        Para("CulinaryDB and FlavorDB (CoSyLab, IIIT-Delhi; CC BY-NC-SA 3.0); Ahn et al. 2011 data (CC BY 4.0). Our "
             "derived data (LLMCuisine-22, flavor matrices, app artifacts): CC BY-NC-SA 4.0 with attribution. Code: "
             "MIT. LLM output terms (Llama 3.1, Qwen, OpenAI, Gemini) are recorded in the datasheet.", 14,
             space_before=4),
        Para("Key references", 17, bold=True, space_before=12),
        Para("Ahn et al., Sci. Rep. 2011 · Jain, Rakhi & Bagler, PLoS ONE 2015 · Singh & Bagler, ICDE Workshops 2018 · "
             "Jain & Bagler, Physica A 2018 · Tuwani et al., ICDE Workshops 2019 · Kinouchi et al., New J. Phys. 2008 · "
             "Madaan et al. 2023 (Self-Refine) · Carichon et al. 2026.", 14, space_before=4),
        Para("Built with the help of an AI coding assistant under the team's direction and review; every number on "
             "these slides is regenerated from results/summary.json by presentation/build.py.", 13, color=T.muted,
             space_before=12),
    ], min_scale=0.75)  # fmt: skip
    deck.notes(s, "Credits, licences and references. Show if asked about data use or licensing.")


# =====================================================================================================================
# Backup slides
# =====================================================================================================================
def b_mechanism(deck: Deck, F) -> None:
    s = deck.new("backup", "mechanism")
    E = F["equal_n"]
    u = E["usa"]
    header(deck, s, "Why does food pairing depend on the number of recipes?",
           minus(f"A small sample loses rare, low-sharing ingredients, so its own-pool null scores higher: USA at "
                 f"{E['n_sub']} recipes has ΔNs = {f(u.get('delta_ns_own_pool_n160'))} against its own pool but "
                 f"{f(u.get('delta_ns_full_pool_n160'))} against the full pool."))  # fmt: skip
    deck.picture(s, F["figures"]["F2d"], 0.55, 2.0, 12.25, 3.7, align="center")
    deck.text(s, 0.55, 5.85, 12.25, 1.1, [
        Para("Orange: ΔNs against the sample's own ingredient pool (the published procedure). Blue: the same sample "
             "against the full cuisine's pool. Dashed: full cuisine. The blue line is flat, so the drift comes from "
             "the null's pool, not from the recipes.", 15, space_before=0),
    ], min_scale=0.75)  # fmt: skip
    deck.notes(
        s,
        minus(f"""
Use when asked "why does N matter?". Mechanism: the random null draws ingredients uniformly from the pool that the
sample itself contains. In a small sample, rare ingredients (which typically have small flavor profiles and share few
compounds) are missing, so the pool is dominated by frequent, high-sharing ingredients; the null's mean Ns rises,
real mean Ns does not, and ΔNs falls. Against the fixed full-cuisine pool the drift disappears (blue).
USA at {E["n_sub"]} recipes: own pool of {n0(u.get("pool_n160"))} ingredients, ΔNs {f(u.get("delta_ns_own_pool_n160"))};
full pool: {f(u.get("delta_ns_full_pool_n160"))}.
"""),
    )


def b_tests(deck: Deck, F) -> None:
    s = deck.new("backup", "tests")
    L = F["llm"]
    keys = list(L)
    header(deck, s, "Which pre-registered tests passed?",
           "Green: significant in the hypothesised or better direction (BH-adjusted p < 0.05); orange: significant the "
           "other way; blank: not significant.")  # fmt: skip
    spec = [
        ("H1  sign errors cluster in contrasting (Fisher)", "h1_fisher", None, "effect"),
        ("H1  equal-N shift toward uniform (Wilcoxon)", "h1eq_wilcoxon", None, "num"),
        ("H1  shift: contrasting vs uniform (Fisher)", "h1eq_fisher", None, "effect"),
        ("H2b SAI > 1 (median, Wilcoxon)", "h2b_sai", None, "level"),
        ("H2b diversity < 1 (median, Wilcoxon)", "h2b_div", None, "level"),
        ("H2c fingerprint below genuine (Wilcoxon)", "h2c", None, "num"),
        ("H4  persona: prevalence JSD", "persona", "prevalence_jsd", "num"),
        ("H4  persona: fingerprint ρ", "persona", "fingerprint_spearman", "num"),
        ("H4  persona: pairing gap", "persona", "abs_d_gap", "num"),
        ("LLM vs CM: prevalence JSD", "cm", "prevalence_jsd", "num"),
        ("LLM vs CM: fingerprint ρ", "cm", "fingerprint_spearman", "num"),
        ("LLM vs CM: pairing gap", "cm", "abs_d_gap", "num"),
    ]  # fmt: skip
    rows = [["Test (effect, BH-adjusted p)"] + [L[k]["name"] for k in keys]]
    fills = {}
    for i, (label, key, met, kind) in enumerate(spec, start=1):
        row = [label]
        for j, k in enumerate(keys, start=1):
            r = L[k][key][met] if met else L[k][key]
            if not r:
                row.append(pending("n/a"))
                continue
            eff = (
                r["effect"]
                if kind == "effect"
                else (f(r["effect_num"], 2) if kind == "level" else sf(r["effect_num"], 2))
            )
            row.append(minus(f"{eff}  ({pv(r['p_adj'])})"))
            if met:
                v = direction(r["effect_num"], r["p_adj"], F["metric_better"][met])
                if v == "better":
                    fills[(i, j)] = T.good
                elif v == "worse":
                    fills[(i, j)] = T.bad
            elif supported(r["p_adj"]):
                fills[(i, j)] = T.good
        rows.append(row)
    widths = [4.45] + [(12.25 - 4.45) / max(len(keys), 1)] * len(keys)
    deck.table(s, 0.55, 1.98, widths, rows, size=12, row_h=0.37, aligns=["l"] + ["c"] * len(keys), fills=fills,
               zebra=False)  # fmt: skip
    deck.notes(
        s,
        """
All pre-registered tests from tables/rq_tests_full.csv. Region is the unit; Wilcoxon signed-rank over regions; Fisher
exact on region type × outcome; Benjamini–Hochberg across tests within each family. Effects: medians of differences
(LLM minus genuine or minus copy-mutate, persona minus plain), or counts for Fisher tests.
""",
    )


def b_scorecard(deck: Deck, F) -> None:
    s = deck.new("backup", "scorecard")
    header(
        deck,
        s,
        "How does every source score on every metric?",
        "Genuine 160-recipe samples set the ceiling; copy-mutate the floor; the LLMs sit in between.",
    )
    deck.picture(s, F["figures"]["F12"], 0.55, 2.0, 12.25, 4.95, align="center")
    deck.notes(
        s,
        f"""
F12: median over the regions each source covers (Gemini: {F["llm"].get("gemini-3.5-flash-lite", {}).get("regions", "?")}
regions so far); colour = better → brighter within each column. Real@160 = genuine samples of the
same size as the LLM cuisines, the realistic ceiling. Copy-mutate = Tuwani 2019 with ranked fitness.
""",
    )


def b_effects(deck: Deck, F) -> None:
    s = deck.new("backup", "effects")
    header(
        deck,
        s,
        "How large is each model's food-pairing effect?",
        "Effect sizes d = ΔNs / SD(null) with 95% bootstrap CIs; the black tick is the full real cuisine.",
    )
    deck.picture(s, F["figures"]["F4"], 0.55, 2.0, 12.25, 4.95, align="center")
    deck.notes(s, "F4: per model and region, the LLM cuisine's d with its bootstrap CI; black tick = full real "
                  "cuisine. Remember small samples are biased toward negative d (slide 6), so the equal-N verdicts are "
                  "primary.")  # fmt: skip


def b_ablation(deck: Deck, F) -> None:
    s = deck.new("backup", "ablation")
    SC = F["scorer"]
    meta = (F["bon"] or {}).get("meta", {})
    ab = meta.get("ablation", {})
    if ab.get("composite") and len(ab) > 1:
        drops = {
            m: ab["composite"]["indep_classifier_acc"] - v["indep_classifier_acc"]
            for m, v in ab.items()
            if m != "composite"
        }
        worst = max(drops, key=drops.get)

        def nice(m):
            return (
                m.replace("without_", "")
                .replace("_", " ")
                .replace("ns ", "Ns ")
                .replace("cooccurrence", "co-occurrence")
            )  # noqa: E731

        answer = (
            f"Ablation at N = {meta.get('ablation_N', '?')} (mean over models and regions): classifier share "
            f"{f(ab['composite']['indep_classifier_acc'])} with all three components, "
            + "; ".join(
                f"{f(v['indep_classifier_acc'])} without {nice(m)} (JSD {f(v['prevalence_jsd'], 3)})"
                for m, v in ab.items()
                if m != "composite"
            )
            + f". Dropping the {nice(worst)} costs most."
        )
    else:
        answer = pending("ablation rows not available")
    header(deck, s, "Which scorer component drives the best-of-N gain?", answer)
    deck.picture(s, F["figures"]["F15b"], 0.55, 2.0, 6.6, 4.95, align="left",
                 missing=pending("F15b ablation figure"))  # fmt: skip
    auc = SC.get("auc_test", {})
    rows = [["Test AUC", "Composite", "Classifier", "Co-occurrence", "Ns cons."]]
    for neg, label in (("other_cuisines", "vs other cuisines"), ("copy_mutate", "vs copy-mutate"),
                       ("random_null", "vs random null")):  # fmt: skip
        a = auc.get(neg, {})
        rows.append([label, f(a.get("composite")), f(a.get("classifier")), f(a.get("cooccurrence")),
                     f(a.get("ns_consistency"))])  # fmt: skip
    deck.table(s, 7.35, 2.05, [1.7, 1.05, 0.95, 1.1, 0.65], rows, size=12, row_h=0.42,
               aligns=["l", "c", "c", "c", "c"])  # fmt: skip
    deck.text(s, 7.35, 3.95, 5.45, 3.0, [
        Para(f"Classifier: {SC.get('classifier_name')} (validation macro-F1 {f(SC.get('val_macro_f1_selected'), 3)}; "
             f"gradient-boosting comparator {f(SC.get('val_macro_f1_gbdt'), 3)}, used as the independent evaluator). "
             f"Weights: {SC.get('weighting')}, chosen by mean validation AUC over the three negative types.", 13,
             space_before=0),
        Para("No single component separates all three negative types: the classifier separates other cuisines, "
             "co-occurrence separates copy-mutate and null recipes, so the composite is the robust choice.", 13,
             space_before=8),
    ], min_scale=0.75)  # fmt: skip
    deck.notes(s, "Use when asked whether the classifier gain is circular. The independent evaluator is a different "
                  "model class on a disjoint split, but it is still an ingredient classifier; F15b shows the gain "
                  "shrinks when the scorer's classifier is dropped, which is why non-classifier held-out metrics "
                  "(JSD, fingerprint, sign) are reported and did not improve.")  # fmt: skip


def b_demo(deck: Deck, F, demo: dict | None) -> None:
    s = deck.new("backup", "demo fallback")
    header(
        deck,
        s,
        "Demo fallback: what FlavorPrint shows",
        "Computed by running the app's own audit function on its built-in examples (presentation/demo_probe.py).",
    )
    if not demo:
        deck.text(s, 0.55, 2.0, 12.25, 4.9, [Para(pending("app artifacts not available at build time"), 18)])
        return
    sc = demo["scenarios"]
    rows = [["", "Genuine", "Off-cuisine", "Stereotype check"]]
    order = ["genuine", "off_cuisine", "stereotype"]

    def top(d):
        return f"{d['top5'][0][0]} ({100 * d['top5'][0][1]:.0f}%)" if d["top5"] else "—"

    spec = [
        ("Recipe (claimed cuisine)", lambda d: f"{d['title']} ({d['claimed']})"),
        ("Fidelity percentile", lambda d: f"{d['fidelity_pct']:.0f}"),
        ("Flavor sharing Ns (percentile)", lambda d: f"{d['ns']:.1f} ({d['ns_pct']:.0f})"),
        ("Most likely cuisine", top),
        ("Ingredients matched", lambda d: f"{d['matched']}/{d['n_lines']}"),
        ("Unusual for the cuisine", lambda d: ", ".join(d["atypical"]) or "none"),
        ("Signature ingredients used", lambda d: f"{len(d['signature_used'])} of top 10"),
        ("Stereotype flag", lambda d: "yes" if d["stereotype_flag"] else "no"),
    ]
    for label, fn in spec:
        rows.append([label] + [fn(sc[k]) for k in order])
    deck.table(s, 0.55, 2.0, [3.1, 3.05, 3.05, 3.05], rows, size=13, row_h=0.5, aligns=["l", "c", "c", "c"],
               bold_first_col=True)  # fmt: skip
    deck.notes(s, "If the live demo fails, walk through this table: the same three recipes as in demo_script.md. "
                  "Values are recomputed at build time from app/artifacts/core.pkl.")  # fmt: skip


def b_signs(deck: Deck, F) -> None:
    s = deck.new("backup", "sign matrix")
    header(deck, s, "Where do models get the food-pairing sign wrong?",
           "Pre-registered sign-error rule (CI of ΔNs vs the full cuisine's sign); see slide 6 for its false-positive "
           "rate.")  # fmt: skip
    deck.picture(s, F["figures"]["F5"], 0.55, 2.0, 12.25, 4.95, align="center")
    deck.notes(s, "F5: orange = CI-significant sign error under the pre-registered rule, green = significant and "
                  "correct, grey = not significant, '·' = region not covered yet (Gemini). Remember the rule's "
                  "false-positive rate on genuine samples (slide 6).")  # fmt: skip


# =====================================================================================================================
def build_deck(F: dict, out: Path, assets: Path, demo: dict | None) -> dict:
    formulas = render_formulas(assets)
    deck = Deck()
    s_title(deck, F)
    s_motivation(deck, F)
    s_data(deck, F)
    s_method(deck, F, formulas)
    s_replication(deck, F)
    s_sample_size(deck, F)
    s_pairing(deck, F)
    s_composition(deck, F)
    s_fingerprint(deck, F)
    s_persona(deck, F)
    s_bon(deck, F)
    s_revise(deck, F)
    s_tools(deck, F)
    s_limitations(deck, F)
    s_conclusions(deck, F)
    b_mechanism(deck, F)
    b_tests(deck, F)
    b_scorecard(deck, F)
    b_effects(deck, F)
    if F["figures"]["F15b"] or F["scorer"]:
        b_ablation(deck, F)
    b_demo(deck, F, demo)
    b_credits(deck, F)
    if F["figures"]["F5"] and F["f5_safe"]:
        b_signs(deck, F)
    footer(deck, F["generated_date"])
    deck.save(out)
    n_main = sum(1 for m in deck.meta if m["kind"] == "main")
    return {"slides": len(deck.meta), "main": n_main, "backup": len(deck.meta) - n_main,
            "labels": [m["label"] for m in deck.meta]}  # fmt: skip


__all__ = ["build_deck", "tag"]
