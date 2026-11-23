"""Deck v2 (Week-14 talk): a visual, one-message-per-slide research talk built from the facts layer.

Main slides follow the user's outline (at most 20); earlier analysis slides are kept as backups. New figures
(F1, F6, F7, F9, F10, F11, F11b), app screenshots (app/screenshots/*.png) and results/steering_example.json are
picked up by name when present; otherwise a native fallback (diagram, chart or table) is drawn.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import deck as d1
from facts import FIGURES, REPO, f, join_and, minus, n0, pct, pending, pv, sf
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.util import Inches, Pt
from pptx_kit import FONT, MONO, Deck, Para, T, footer, header, rgb

SHOTS = REPO / "app" / "screenshots"
STEER_EXAMPLE = REPO / "results" / "steering_example.json"
NEW_FIGS = {"F1": "F1_pipeline.png", "F4b": "F4b_ns_distributions.png", "F6": "F6_fingerprints.png", "F7": "F7_drivers.png", "F9": "F9_diversity.png",
            "F10": "F10_sizes.png", "F11": "F11_networks.png", "F11b": "F11b_showcase_pairs.png"}  # fmt: skip
UNIFORM_C, CONTRAST_C = "#0072B2", "#D55E00"


def fig(F, key):
    if key == "F15c":
        return F["figures"].get("F15c")
    if key in NEW_FIGS:
        p = FIGURES / NEW_FIGS[key]
        return p if p.exists() else None
    return F["figures"].get(key)


def short(name: str) -> str:
    return (name.replace("Gemini 3.5 Flash-Lite", "Gemini FL").replace("Qwen3.5-", "Qwen-")
            .replace("Llama-3.1-", "Llama-").replace("Ministral-3-", "Ministral-"))  # fmt: skip


def caption(deck, s, x, y, w, text, size=11):
    deck.text(s, x, y, w, 0.32, [Para(text, size, color=T.muted, space_before=0)], fit=True, min_scale=0.75)


def card(deck, s, x, y, w, h, title, paras, accent=T.accent, title_size=16, body_size=13, fill=T.panel):
    deck.rect(s, x, y, w, h, fill)
    deck.rect(s, x, y, w, 0.06, accent)
    deck.text(s, x + 0.12, y + 0.1, w - 0.24, 0.42, [Para(title, title_size, bold=True, space_before=0)],
              min_scale=0.75)  # fmt: skip
    body = [p if isinstance(p, Para) else Para(p, body_size, space_before=3) for p in paras]
    deck.text(s, x + 0.12, y + 0.52, w - 0.24, h - 0.58, body, min_scale=0.7)


def bar_chart(deck, s, x, y, w, h, title, cats, vals, colors, fmt="0.00", ref=None, ymax=None):
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series(title, [None if v is None or (isinstance(v, float) and math.isnan(v)) else v for v in vals])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.font.name = FONT
    ch.font.size = Pt(10)
    ch.has_legend = False
    ch.has_title = True
    ch.chart_title.text_frame.text = title
    for r in ch.chart_title.text_frame.paragraphs[0].runs:
        r.font.size, r.font.bold, r.font.name = Pt(13), True, FONT
    plot = ch.plots[0]
    plot.gap_width = 45
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = fmt, False
    dl.font.size, dl.font.name = Pt(10), FONT
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    ser = plot.series[0]
    for i, c in enumerate(colors):
        pt = ser.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = rgb(c)
    va = ch.value_axis
    va.has_major_gridlines = False
    va.visible = False
    va.minimum_scale = 0
    finite = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    va.maximum_scale = ymax or (max(finite) * 1.3 if finite else 1)
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(9)
    ca.tick_labels.font.name = FONT
    ca.format.line.color.rgb = rgb(T.rule)
    return ch


def network(deck, s, cx, cy, radius, nodes, sizes, edges, node_w=1.35):
    """Draw a small ingredient network: nodes on a circle, edge width ∝ shared compounds, labels on edges."""
    n = len(nodes)
    pos = {}
    for i, name in enumerate(nodes):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        pos[name] = (cx + radius * math.cos(ang), cy + radius * math.sin(ang))
    wmax = max((e[2] for e in edges), default=1) or 1
    for a, b, w in edges:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        c.line.color.rgb = rgb("#9AA5B1" if w >= 0.3 * wmax else CONTRAST_C)
        c.line.width = Pt(1 + 7 * w / wmax)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        deck.text(
            s,
            mx - 0.25,
            my - 0.15,
            0.5,
            0.3,
            [Para(str(w), 11, bold=True, align="center", space_before=0, color=T.ink)],
            anchor="middle",
            fit=False,
            fill="#FFFFFF",
        )
    for name in nodes:
        x, y = pos[name]
        shp = deck.rect(s, x - node_w / 2, y - 0.3, node_w, 0.6, UNIFORM_C if sizes.get(name, 0) > 50 else CONTRAST_C,
                        shape=MSO_SHAPE.ROUNDED_RECTANGLE)  # fmt: skip
        shp.shadow.inherit = False
        deck.text(s, x - node_w / 2, y - 0.3, node_w, 0.6, [
            Para(name, 12, bold=True, color="#FFFFFF", align="center", space_before=0),
            Para(f"{sizes.get(name, '?')} compounds", 9.5, color="#FFFFFF", align="center", space_before=0)],
            anchor="middle", fit=True, min_scale=0.7)  # fmt: skip


# =====================================================================================================================
def s_problem(deck: Deck, F) -> None:
    s = deck.new("main", "problem")
    header(deck, s, "Why ask whether AI chefs respect culinary fingerprints?",
           "LLMs now write “traditional” recipes for millions of people. This lab's flavor-network framework lets us "
           "test that claim with numbers instead of taste.")  # fmt: skip
    why = [
        ("LLMs are recipe authors now", "People ask chatbots for “a traditional Korean recipe” and trust the answer."),
        ("“Authentic” can be measured", "Food pairing Ns and Z, flavor drivers and category fingerprints (Ahn 2011; "
                                       "Jain, Rakhi & Bagler 2015; Singh & Bagler 2018) are quantitative and have "
                                       "null models."),
        ("Nobody has tested LLM recipes this way", "Earlier audits of LLM recipes are textual (e.g. Carichon et al. "
                                                  "2026); none uses flavor compounds or fingerprints."),
    ]  # fmt: skip
    for i, (t, b) in enumerate(why):
        card(deck, s, 0.55, 1.98 + i * 1.66, 5.3, 1.54, t, [b], accent=T.accent, body_size=14)
    rqs = [
        ("RQ1", "Pairing", "Do LLM cuisines keep each region's food-pairing sign and strength?"),
        ("RQ2", "Fingerprints", "Do they keep the flavor drivers, signature ingredients, category fingerprints and "
                                "co-occurrence patterns?"),
        ("RQ3", "Steering", "Can the fingerprints steer generation toward a cuisine without copying recipes?"),
        ("RQ4", "Framing and models", "Do a home-cook persona, the vendor or the model size matter?"),
    ]  # fmt: skip
    deck.text(s, 6.15, 1.95, 6.6, 0.4, [Para("What we test", 17, bold=True, space_before=0)], fit=False)
    for i, (rq, t, b) in enumerate(rqs):
        y = 2.4 + i * 1.03
        deck.rect(s, 6.15, y, 0.9, 0.9, T.ink, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        deck.text(s, 6.15, y, 0.9, 0.9, [Para(rq, 17, bold=True, color="#FFFFFF", align="center", space_before=0)],
                  anchor="middle", fit=False)  # fmt: skip
        deck.text(
            s,
            7.15,
            y - 0.02,
            5.65,
            0.94,
            [Para(t, 15, bold=True, color=T.accent, space_before=0), Para(b, 14, space_before=2)],
            anchor="middle",
            min_scale=0.75,
        )
    deck.text(s, 6.15, 6.58, 6.65, 0.36, [Para(
        "Hypotheses and tests were pre-registered before the full LLM data were analysed.", 12, color=T.muted,
        italic=True, space_before=0)], fit=True)  # fmt: skip
    deck.notes(
        s,
        """
WHAT TO SAY (about 45 s)
Large language models are now recipe authors: people ask for "a traditional Korean recipe" and trust what comes back.
Is that recipe Korean in any measurable sense? Your lab's framework gives us exactly that measure: food pairing Ns
and Z against null models, the ingredients and categories that drive it, and the ingredients a cuisine relies on.
As far as our literature search found, nobody has audited LLM recipes with flavor compounds: earlier work is textual.
We ask four questions: pairing (RQ1), the finer fingerprints (RQ2), steering (RQ3) and framing/vendor/scale (RQ4).
All hypotheses were written into the plan before we analysed the full LLM data.
""",
    )


def s_fingerprint_concept(deck: Deck, F) -> None:
    s = deck.new("main", "fingerprint concept")
    R = F["replication"]
    z = R["z"]
    header(deck, s, "What is a culinary fingerprint?",
           "The measurable way a cuisine combines ingredients: how much flavor its ingredient pairs share, which "
           "ingredients and categories drive that, and which ingredients it relies on.")  # fmt: skip
    k1 = R["korea_top3"][0] if R["korea_top3"] else {}
    items = [
        ("1  Food-pairing sign", UNIFORM_C,
         [minus(f"**Uniform** (Z > 0): pairs share more flavor compounds than chance, e.g. Italy Z = {f(z.get('Italy'), 1)}."),
          minus(f"**Contrasting** (Z < 0): they share fewer, e.g. Korea Z = {f(z.get('Korea'), 1)}."),
          f"{F['regions']['n_uniform']} of the {F['regions']['n']} regions are uniform, {F['regions']['n_contrasting']} "
          "contrasting (Singh & Bagler 2018)."]),
        ("2  Flavor drivers", CONTRAST_C,
         ["Ingredients whose removal changes the cuisine's Ns the most (χ%).",
          f"Korea: **{k1.get('name', '?')}** is the top driver ({f(k1.get('chi_percent'))}% in our replication)."]),
        ("3  Category fingerprint", T.green,
         ["Randomise one ingredient category at a time and see how ΔNs moves (Jain, Rakhi & Bagler 2015).",
          "The vector over categories is the cuisine's fingerprint."]),
        ("4  Ingredient use", T.ink,
         ["Signature ingredients: top-10 by relative prevalence (Jain & Bagler 2018).",
          "Co-occurrence: which ingredient pairs appear together more than chance (NPMI networks)."]),
    ]  # fmt: skip
    w = (12.25 - 3 * 0.2) / 4
    for i, (t, col, body) in enumerate(items):
        card(deck, s, 0.55 + i * (w + 0.2), 2.0, w, 3.55, t, body, accent=col, body_size=14)
    deck.rect(s, 0.55, 5.75, 12.25, 1.17, T.panel2)
    deck.text(s, 0.75, 5.8, 11.9, 1.08, [
        Para("**Our test:** compute all four for every LLM cuisine and compare them with the real cuisine and with "
             "genuine samples of the same size.", 16, space_before=0),
        Para("Layers 1–3 use FlavorDB's flavor compounds; layer 4 uses only which ingredients appear together.", 13,
             color=T.muted, space_before=4),
    ], anchor="middle", min_scale=0.8)  # fmt: skip
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 45 s)
A culinary fingerprint is not one number. Layer 1 is the food-pairing sign: in a uniform cuisine like Italy
(Z = {f(z.get("Italy"), 1)}) ingredient pairs share more flavor compounds than random recipes built from the same
ingredients; in a contrasting cuisine like Korea (Z = {f(z.get("Korea"), 1)}) they share fewer. Layer 2: which
ingredients drive that, measured by χ% (Korea's Soybean Sauce). Layer 3: the category fingerprint of Jain, Rakhi &
Bagler 2015. Layer 4: signature ingredients and co-occurrence. We test every layer for every LLM cuisine.
"""),
    )


def s_data(deck: Deck, F, X) -> None:
    s = deck.new("main", "data")
    ds, D = F["dataset"], F["design"]
    C = X.get("counts") or {}
    header(deck, s, "Which data and which models?",
           f"CulinaryDB and FlavorDB, both from this lab, are the ground truth; {ds['n_models']} LLMs wrote "
           f"{n0(ds['total_recipes'])} recipes for the same {F['regions']['n']} regions.")  # fmt: skip
    src = ", ".join(k.replace("_", " ").title() for k in C.get("sources", {}))
    card(deck, s, 0.55, 1.98, 4.1, 2.45, "CulinaryDB (Singh & Bagler 2018)", [
        f"**{n0(C.get('recipes'))}** recipes from {len(C.get('sources', {}))} sources ({src})",
        f"**{n0(C.get('basic_ingredients'))}** + {n0(C.get('compound_ingredients'))} compound ingredients in "
        f"{C.get('categories', '?')} categories",
        f"**{n0(C.get('rows'))}** recipe lines with the expert mapping ({n0(C.get('phrases'))} distinct phrases): "
        "the gold standard for our aliaser"], body_size=13)  # fmt: skip
    card(deck, s, 0.55, 4.55, 4.1, 1.62, "FlavorDB (Garg et al. 2018)", [
        f"**{n0(C.get('fdb_entities'))}** ingredients with flavor profiles, **{n0(C.get('fdb_molecules'))}** "
        f"molecules, {n0(C.get('fdb_links'))} ingredient–molecule links"], accent=T.blue, body_size=13)  # fmt: skip
    deck.text(s, 0.55, 6.25, 4.1, 0.7, [Para(
        "Same lab, same definitions: our numbers are directly comparable with Singh & Bagler's Fig. 4.", 12,
        color=T.muted, italic=True, space_before=0)], min_scale=0.8)  # fmt: skip
    # 22-region grid
    z, n_rec = F["replication"]["z"], F["replication"]["n_recipes"]
    order = sorted(z, key=lambda r: -z[r])
    cols, tw, th, gx, gy, x0, y0 = 6, 1.25, 0.6, 0.08, 0.07, 4.9, 1.98
    for i, r in enumerate(order):
        rr, cc = divmod(i, cols)
        x, y = x0 + cc * (tw + gx), y0 + rr * (th + gy)
        col = UNIFORM_C if z[r] > 0 else CONTRAST_C
        deck.rect(s, x, y, tw, th, col)
        deck.text(s, x, y, tw, th, [Para(r, 10.5, bold=True, color="#FFFFFF", align="center", space_before=0),
                                    Para(f"{n0(n_rec.get(r))} recipes", 9, color="#FFFFFF", align="center",
                                         space_before=0)], anchor="middle", min_scale=0.7)  # fmt: skip
    yl = y0 + 4 * (th + gy) + 0.02
    deck.text(
        s,
        x0,
        yl,
        7.9,
        0.3,
        [
            Para(
                "Blue: uniform pairing (Z > 0); vermillion: contrasting (Z < 0); ordered by Z (our replication).",
                11,
                color=T.muted,
                space_before=0,
            )
        ],
        fit=True,
    )
    # models
    ym = yl + 0.38
    keys = F["model_order"]
    mw = (7.9 - 0.1 * (len(keys) - 1)) / max(len(keys), 1)
    for i, k in enumerate(keys):
        mi = F["models"][k]
        x = x0 + i * (mw + 0.1)
        deck.rect(s, x, ym, mw, 1.3, T.panel)
        deck.rect(s, x, ym, mw, 0.06, mi["color"])
        dd = mi["data"] or {}
        lines = [Para(short(mi["name"]), 12, bold=True, align="center", space_before=0),
                 Para(("API" if mi["proprietary"] else "open weights"), 10, color=T.muted, align="center",
                      space_before=0)]  # fmt: skip
        if dd:
            lines.append(Para(f"{n0(dd.get('recipes'))} recipes", 11, align="center", space_before=3))
        lines.append(Para("complete" if mi["complete"] else mi["status"], 9.5, align="center", space_before=2,
                          color=T.green if mi["complete"] else T.muted))  # fmt: skip
        deck.text(s, x + 0.04, ym + 0.07, mw - 0.08, 1.2, lines, min_scale=0.6)
    deck.text(s, x0, ym + 1.36, 7.9, 0.42, [Para(
        f"Design: {F['regions']['n']} regions × {D['n_conditions']} prompts (plain, home-cook persona) × "
        f"{D['calls_per_cell']} calls × {D['recipes_per_call']} recipes = {n0(D['recipes_per_model'])} per model; "
        f"aliaser agrees with CulinaryDB's experts on {pct(F['aliasing']['acc_lines'])} of held-out lines.", 11.5,
        color=T.muted, space_before=0)], min_scale=0.75)  # fmt: skip
    deck.notes(
        s,
        f"""
WHAT TO SAY (about 50 s)
Both databases come from this lab, so every definition matches Singh & Bagler 2018. CulinaryDB gives
{n0(C.get("recipes"))} recipes with their raw ingredient lines and the expert mapping to ingredient entities; FlavorDB
gives each ingredient's flavor molecules. The grid shows the {F["regions"]["n"]} regions ordered by Z, blue uniform,
vermillion contrasting, with their recipe counts.
We generated recipes with {ds["n_models"]} LLMs ({join_and(F["analysed_names"])}): the same frozen prompts, plain and
home-cook persona, {D["recipes_per_region"]} recipes per region and model. LLM ingredient lines are mapped with our
reconstruction of CulinaryDB's own aliasing protocol, validated against CulinaryDB's expert mapping
({pct(F["aliasing"]["acc_lines"])} of held-out lines, {pct(F["aliasing"]["acc_rows"])} of occurrences).
STATUS: {"; ".join(F["models"][k]["name"] + " " + F["models"][k]["status"].removeprefix("[pending: ").removesuffix("]") for k in F["incomplete_models"])}.
""",
    )


def s_pipeline(deck: Deck, F) -> None:
    s = deck.new("main", "pipeline")
    header(deck, s, "How does the analysis pipeline fit together?",
           "Real and LLM recipes go through the same aliaser and the same scoring function; the results feed the "
           "tests, the steering scorer, the library and the app.")  # fmt: skip
    f1 = fig(F, "F1")
    if f1:
        deck.picture(s, f1, 0.55, 1.95, 12.25, 4.95, align="center")
    else:
        boxes = [
            (0.55, 2.1, "CulinaryDB + FlavorDB", "recipes, flavor profiles", T.blue),
            (0.55, 3.55, "LLM generation", f"{F['dataset']['n_models']} models · 2 prompts · logged", T.accent),
            (0.55, 5.0, "Copy-mutate model", "Tuwani 2019 baseline", "#7A7A7A"),
            (3.75, 2.1, "Shared-compound matrix S", "|C_i ∩ C_j| for every pair", T.blue),
            (3.75, 3.55, "Aliaser", "CulinaryDB protocol", T.accent),
            (6.95, 3.55, "Scorecard", "Ns · Z · fingerprints · SAI · networks", T.ink),
            (10.0, 2.1, "Equal-size tests", "genuine samples · BH-FDR", T.green),
            (10.0, 3.55, "D3 steering", "scorer · best-of-N · revise", T.green),
            (10.0, 5.0, "D4 FlavorAudit · D5 FlavorPrint", "library and app", T.green),
        ]
        for x, y, t, sub, col in boxes:
            deck.rect(s, x, y, 2.8, 1.15, T.panel)
            deck.rect(s, x, y, 0.07, 1.15, col)
            deck.text(s, x + 0.15, y + 0.05, 2.6, 1.05, [Para(t, 14, bold=True, space_before=0),
                                                       Para(sub, 11, color=T.muted, space_before=2)],
                      anchor="middle", min_scale=0.7)  # fmt: skip
        arrows = [
            (3.35, 2.67, 3.75, 2.67),
            (3.35, 4.12, 3.75, 4.12),
            (6.55, 2.67, 6.95, 3.9),
            (6.55, 4.12, 6.95, 4.12),
            (3.35, 5.57, 6.95, 4.4),
            (9.75, 4.12, 10.0, 2.67),
            (9.75, 4.12, 10.0, 4.12),
            (9.75, 4.12, 10.0, 5.57),
        ]
        for x1, y1, x2, y2 in arrows:
            c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
            c.line.color.rgb = rgb(T.muted)
            c.line.width = Pt(2)
            c.line._get_or_add_ln().append(_arrow_tail())
        caption(deck, s, 0.55, 6.45, 12.25, "Diagram drawn from the repository structure (F1 pipeline figure not "
                                            "available at build time).")  # fmt: skip
    deck.notes(
        s,
        """
WHAT TO SAY (about 40 s)
Left: the inputs. CulinaryDB and FlavorDB give the real cuisines and the shared-compound matrix S; the LLMs give
generated recipes; the copy-mutate model gives a baseline cuisine. Every recipe, real or generated, goes through the
same aliaser and the same scoring function (flavoraudit.audit.scorecard_row), so differences come from the recipes,
not from the code. The scorecard feeds the equal-size tests, the D3 steering scorer, the library and the app.
""",
    )


def _arrow_tail():
    from lxml import etree
    from pptx.oxml.ns import qn

    tail = etree.Element(qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    return tail


def s_ns_intuition(deck: Deck, F, X, formulas) -> None:
    s = deck.new("main", "ns intuition")
    M = F["method"]
    header(deck, s, "How do Ns, Z and the four null models work?",
           "Ns counts the flavor compounds an average ingredient pair shares; Z asks whether real recipes share more "
           "(uniform) or fewer (contrasting) than random recipes built from the same ingredients.")  # fmt: skip
    net = X.get("network")
    deck.rect(s, 0.55, 1.98, 5.6, 4.97, T.panel)
    if net:
        deck.text(s, 0.7, 2.03, 5.3, 0.4, [Para(f"{net['region']}'s top-5 signature ingredients", 15, bold=True,
                                                 space_before=0)], fit=False)  # fmt: skip
        network(deck, s, 3.35, 4.2, 1.55, net["nodes"], net["sizes"], net["edges"])
        sb = [e for e in net["edges"] if "Soybean Sauce" in e[:2]]
        sb_rng = f"{min(e[2] for e in sb)}–{max(e[2] for e in sb)}" if sb else "?"
        deck.text(
            s,
            0.7,
            5.98,
            5.3,
            0.95,
            [
                Para(
                    f"Edge label = shared compounds. Ns of these five = mean over the 10 pairs = **{f(net['ns'], 1)}**. "
                    f"Soybean Sauce has only {net['sizes'].get('Soybean Sauce', '?')} compounds, so it shares {sb_rng} with "
                    "each partner: this is why it drives Korea's contrasting pairing.",
                    12,
                    space_before=0,
                )
            ],
            min_scale=0.75,
        )
    deck.picture(s, formulas["ns"], 6.4, 2.0, 3.0, 0.62, align="left")
    deck.picture(s, formulas["z"], 9.55, 2.0, 3.25, 0.62, align="left")
    deck.text(s, 6.4, 2.68, 6.4, 0.55, [Para(minus(
        f"N_rand = {n0(M['n_rand'])}, so Z = {f(M['sqrt_n_rand'], 0)} × d (an effect size, not a p-value). Z > 0: "
        "uniform; Z < 0: contrasting."), 12.5, color=T.muted, space_before=0)], min_scale=0.8)  # fmt: skip
    f4b = fig(F, "F4b")
    if f4b:
        deck.picture(s, f4b, 6.4, 3.22, 6.4, 2.62, align="center")
        nulls = [("Random", "uniform draws from the cuisine's ingredients"),
                 ("Frequency", "draws in proportion to use"),
                 ("Category", "keeps each recipe's category mix"),
                 ("Freq. + category", "both constraints")]  # fmt: skip
        w = (6.4 - 3 * 0.1) / 4
        for i, (t, b) in enumerate(nulls):
            x = 6.4 + i * (w + 0.1)
            deck.rect(s, x, 5.92, w, 1.03, "#FFFFFF", line=T.rule)
            deck.rect(s, x, 5.92, w, 0.05, T.blue)
            deck.text(s, x + 0.06, 5.97, w - 0.12, 0.96, [Para(t + " null", 11.5, bold=True, space_before=0),
                                                         Para(b, 10.5, space_before=1)], min_scale=0.7)  # fmt: skip
    else:
        nulls = [
            ("Random", "pick each recipe's ingredients uniformly from the cuisine's own ingredient set"),
            ("Frequency", "pick them in proportion to how often the cuisine uses them"),
            ("Category", "keep each recipe's category mix (2 spices, 1 meat, …), fill each slot at random"),
            ("Frequency + category", "both: category mix kept, popular ingredients picked more often"),
        ]
        w, h = 3.1, 1.6
        for i, (t, b) in enumerate(nulls):
            r, c = divmod(i, 2)
            x, y = 6.4 + c * (w + 0.2), 3.35 + r * (h + 0.15)
            card(deck, s, x, y, w, h, t + " null", [b], accent=T.blue, title_size=14, body_size=12.5, fill="#FFFFFF")
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 70 s)
Left, a real example from FlavorDB: Korea's five signature ingredients. Each edge is the number of flavor compounds
the two ingredients share. Ns of a recipe is simply the average edge weight over its ingredient pairs; for these
five it is {f((net or {}).get("ns"), 1)}. Soybean Sauce has a tiny flavor profile, so every pair with it shares very
few compounds; that pulls Korean recipes' Ns down, which is why Soybean Sauce is Korea's top contributor.
Right: Z compares the real cuisine's mean Ns with that of 100,000 random recipes built from the same ingredients and
recipe sizes. F4b (if shown) puts it in one picture: grey = random recipes from the cuisine's own ingredients, colour
= real recipes; Korea's real recipes sit to the left of their null (contrasting), Italy's to the right (uniform); the
LLM recipes are the density lines. Every null keeps the cuisine's own ingredients and recipe sizes. The four nulls differ in how they draw: uniformly, by popularity, by category mix, or both. If even
the frequency null removes the effect, popularity explains it — your Fig. 4 shows that.
Key subtlety: Z = √N_rand × d, so it measures effect size times {f(M["sqrt_n_rand"], 0)}, not significance; we
always report d and compare at equal size.
"""),
    )


def s_recipes(deck: Deck, F, X) -> None:
    s = deck.new("main", "recipes")
    P = X.get("pair")
    if not P:
        header(deck, s, "What does a real recipe look like next to an LLM recipe?", pending("recipe pair unavailable"))
        return
    rl, ll = P["real"], P["llm"]
    adj = F["regions"].get("phrase_of", {}).get(P["region"], P["region"])
    header(deck, s, f"What does a typical {adj} recipe look like, real vs LLM?",
           minus(f"Both are the median-Ns recipe of their source. The LLM recipe shares more flavor compounds per pair "
                 f"(Ns {f(ll['ns'], 1)} vs {f(rl['ns'], 1)}) and lands at the {ll['ns_pct']:.0f}th percentile of real "
                 f"{adj} recipes."))  # fmt: skip
    for x, side, lab, col in ((0.55, rl, f"CulinaryDB · {rl['source']} (real)", T.ink),
                              (6.75, ll, f"LLM · {ll['model']} · plain prompt", T.accent)):  # fmt: skip
        deck.rect(s, x, 1.98, 6.05, 4.42, T.panel)
        deck.rect(s, x, 1.98, 6.05, 0.42, col)
        deck.text(s, x + 0.12, 1.98, 5.8, 0.42, [Para(lab, 13, bold=True, color="#FFFFFF", space_before=0)],
                  anchor="middle", fit=True)  # fmt: skip
        deck.text(s, x + 0.12, 2.45, 5.8, 0.5, [Para(side["title"], 19, bold=True, space_before=0)], fit=True,
                  min_scale=0.7)  # fmt: skip
        lines = side["lines"][:11]
        more = len(side["lines"]) - len(lines)
        paras = [Para(ln[:78], 12, bullet=True, space_before=1) for ln in lines]
        if more > 0:
            paras.append(Para(f"… and {more} more lines", 11, color=T.muted, space_before=1))
        deck.text(s, x + 0.12, 2.98, 5.8, 3.4, paras, min_scale=0.7)
        sig = side["signature"]
        stats = (
            f"**{side['n']}** ingredients mapped · Ns **{f(side['ns'], 1)}** ({side['ns_pct']:.0f}th percentile of "
            f"real {adj}) · signature ingredients: {len(sig)} of 10"
        )
        deck.text(s, x, 6.45, 6.05, 0.5, [Para(stats, 12, space_before=0)], min_scale=0.75)
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 50 s)
How we picked these, so nobody can say we cherry-picked: in {P["region"]}, the lab's showcase contrasting cuisine,
we took the real CulinaryDB recipe whose Ns is closest to the median Ns of all {rl["pool"]} real {P["region"]}
recipes ({f(rl["median_ns"], 1)}), and the LLM recipe whose Ns is closest to the median of all {ll["pool"]} plain-prompt
LLM recipes from the complete models ({f(ll["median_ns"], 1)}). So both are the typical recipe of their source.
Left: {rl["title"]} ({rl["source"]}). Right: {ll["title"]} by {ll["model"]}. Both look plausibly Korean and share
signature ingredients ({join_and(rl["signature"][:4])} vs {join_and(ll["signature"][:4])}). The difference is
quantitative: the LLM recipe's pairs share more compounds (Ns {f(ll["ns"], 1)} vs {f(rl["ns"], 1)}). Note also
aliasing: LLM lines such as "gochujang" have no CulinaryDB entity yet; those go to curation.
"""),
    )


def s_models(deck: Deck, F) -> None:
    s = deck.new("main", "models")
    L, E, C, N = F["llm"], F["equal_n"], F["cm"]["ranked_n"], F.get("networks") or {}
    header(deck, s, "How do the models compare with real cuisines?",
           "No model reaches genuine samples of the same size on any layer; copy-mutate copies popularity but not "
           "composition. Models differ most on pairing sign and diversity.")  # fmt: skip
    keys = list(L)
    names = [short(L[k]["name"]) for k in keys]
    cols = [L[k]["color"] for k in keys]
    nt = N.get("npmi_top100") or {}
    specs = [
        ("Sign agreement ↑", [L[k]["sign_agreement"] for k in keys], E["genuine_sign_agreement"], C.get("sign_match")),
        ("Fingerprint ρ ↑", [L[k]["median_fingerprint_spearman"] for k in keys], E["ref"].get("fingerprint_spearman"),
         C.get("fingerprint_spearman")),
        ("Diversity ratio ↑", [L[k]["median_diversity_ratio"] for k in keys], 1.0, None),
        ("Prevalence JSD ↓", [L[k]["median_prevalence_jsd"] for k in keys], E["ref"].get("prevalence_jsd"),
         C.get("prevalence_jsd")),
        ("Pair overlap ↑", [(nt.get(k) or {}).get("median") for k in keys], (nt.get("real@160") or {}).get(
            "median"), None),
    ]  # fmt: skip
    w = (12.25 - 4 * 0.12) / 5
    for i, (t, vals, gen, cmv) in enumerate(specs):
        cats, v, c = list(names), list(vals), list(cols)
        cats.append("Genuine")
        v.append(gen)
        c.append("#000000")
        if cmv is not None:
            cats.append("Copy-mut.")
            v.append(cmv)
            c.append("#999999")
        bar_chart(deck, s, 0.55 + i * (w + 0.12), 1.98, w, 3.75, t, cats, v, c, fmt="0.00")
    V = F["verdicts"]
    deck.text(s, 0.55, 5.85, 12.25, 1.1, [
        Para("Medians over regions; genuine = random real samples of the same size; ↑/↓ = better. Overlap = Jaccard of "
             "the top-100 NPMI ingredient pairs with the real cuisine.", 11.5, color=T.muted, space_before=0),
        Para(f"**Home-cook persona** (RQ4): brings prevalence closer to real in {V['persona_jsd_better']}/{V['n']} "
             f"models, the fingerprint only for {join_and(F['persona_rho_models']) or 'none'}, pairing in "
             f"{V['persona_gap_better']}/{V['n']} (Backup). Scale pair Qwen 9B vs 4B: "
             + (F["models"].get("qwen3.5-4b", {}).get("status", "") or "n/a") + ".", 13, space_before=5),
    ], min_scale=0.75)  # fmt: skip
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 60 s)
Five layers, one panel each; bars are the models, black is genuine real samples of the same size (the realistic
ceiling), grey is the copy-mutate model (a baseline from Tuwani 2019 with ranked fitness, also our Assignment 3).
Sign agreement: genuine samples agree with the full cuisine {f(E["genuine_sign_agreement"])} of the time; models range
{d1.rng_text(*F["ranges"]["sign_agreement"])}. Fingerprint ρ: genuine {f(E["ref"].get("fingerprint_spearman"))}, models
{d1.rng_text(*F["ranges"]["median_fingerprint_spearman"])}, copy-mutate {f(C.get("fingerprint_spearman"))}.
Diversity: models use {d1.pct_rng(*F["ranges"]["median_diversity_ratio"])} of the distinct ingredients of a genuine
sample. Prevalence JSD: models {d1.rng_text(*F["ranges"]["median_prevalence_jsd"], 3)} vs genuine
{f(E["ref"].get("prevalence_jsd"), 3)}. Networks: top-100 pair overlap {", ".join(short(L[k]["name"]) + " " + f((nt.get(k) or {}).get("median")) for k in keys)} vs
genuine {f((nt.get("real@160") or {}).get("median"))}.
Persona (RQ4) details are in the backup slide.
"""),
    )


def s_cuisines(deck: Deck, F) -> None:
    s = deck.new("main", "cuisines")
    import pandas as pd

    h1 = pd.read_csv(REPO / "tables" / "h1_equal_n_full.csv")
    L, V = F["llm"], F["verdicts"]
    z = F["replication"]["z"]
    contr = sorted([r for r in z if z[r] < 0], key=lambda r: z[r])
    unif = sorted([r for r in z if z[r] > 0], key=lambda r: -z[r])
    regions = contr + unif
    code = F["regions"]["code_of"]
    cons = F["consensus"]
    header(deck, s, "Which cuisines do the models get wrong, uniform or contrasting?",
           f"Errors are not concentrated in the contrasting cuisines (H1 supported for {V['h1eq_supported']}/{V['n']} "
           "models); they are model- and region-specific, with a few shared by every model.")  # fmt: skip
    keys = list(L)
    rows = [["", *[code.get(r, r) for r in regions]]]
    fills = {}
    look = {(m, r): v for m, r, v in zip(h1["model"], h1["region"], h1["verdict"], strict=True)}
    for i, k in enumerate(keys, start=1):
        row = [short(L[k]["name"])]
        for j, r in enumerate(regions, start=1):
            v = look.get((k, r))
            if v is None:
                row.append("·")
                fills[(i, j)] = "#FFFFFF"
            elif v == "too uniform":
                row.append("U")
                fills[(i, j)] = UNIFORM_C
            elif v == "too contrasting":
                row.append("C")
                fills[(i, j)] = CONTRAST_C
            else:
                row.append("")
                fills[(i, j)] = "#E3E7EC"
        rows.append(row)
    cw = (12.25 - 1.45) / len(regions)
    deck.table(s, 0.55, 2.42, [1.45] + [cw] * len(regions), rows, size=11, row_h=0.5, aligns=["l"] + ["c"] * len(regions),
               fills=fills, zebra=False, header_size=9)  # fmt: skip
    tbl = s.shapes[-1].table
    for i in range(1, len(rows)):
        for j in range(1, len(regions) + 1):
            for p in tbl.cell(i, j).text_frame.paragraphs:
                for r in p.runs:
                    r.font.bold = True
                    if fills.get((i, j)) in (UNIFORM_C, CONTRAST_C):
                        r.font.color.rgb = rgb("#FFFFFF")
    # group labels above the table
    deck.rect(s, 0.55 + 1.45, 1.98, cw * len(contr), 0.38, CONTRAST_C)
    deck.text(s, 0.55 + 1.45, 1.98, cw * len(contr), 0.38, [Para("Contrasting cuisines (Z < 0)", 11, bold=True,
              color="#FFFFFF", align="center", space_before=0)], anchor="middle", fit=True)  # fmt: skip
    deck.rect(s, 0.55 + 1.45 + cw * len(contr), 1.98, cw * len(unif), 0.38, UNIFORM_C)
    deck.text(s, 0.55 + 1.45 + cw * len(contr), 1.98, cw * len(unif), 0.38, [Para(
        "Uniform cuisines (Z > 0), most uniform first", 11, bold=True, color="#FFFFFF", align="center",
        space_before=0)], anchor="middle", fit=True)  # fmt: skip
    y = 2.42 + deck.last_table_h + 0.15
    deck.text(s, 0.55, y, 12.25, 0.35, [Para(
        "Equal-size verdict against genuine samples of the same region (BH): U = too uniform, C = too contrasting, "
        "grey = typical, · = region not covered yet.", 12, color=T.muted, space_before=0)], fit=True)  # fmt: skip
    y += 0.45
    lines = [
        Para(
            "**Shared by every model with data:** "
            + "; ".join(f"{c['region']} {c['verdict']} ({c['k']}/{c['k']})" for c in cons)
            + ".",
            15,
            space_before=0,
        )
    ]
    lines.append(Para("**Per model:** " + "; ".join(
        f"{short(L[k]['name'])} typical {L[k]['equal_n_typical']}/{L[k]['regions']}, U {L[k]['equal_n_too_uniform']}, "
        f"C {L[k]['equal_n_too_contrasting']}" for k in keys) + ".", 13, space_before=6))  # fmt: skip
    lines.append(Para("**Pre-registered H1** (errors cluster in contrasting regions, Fisher): supported for "
                      f"{V['h1_fisher_supported']}/{V['n']} models; equal-N shift toward uniform: "
                      f"{V['h1eq_supported']}/{V['n']}.", 13, space_before=6))  # fmt: skip
    deck.text(s, 0.55, y, 12.25, 6.95 - y, lines, min_scale=0.75)
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 60 s)
HOW TO READ: columns are the {F["regions"]["n"]} regions, contrasting cuisines on the left (most contrasting first),
uniform on the right; rows are models. Each cell compares the model's effect size d with the distribution of d over
genuine samples of the same region and size: U = above the 97.5% quantile (too uniform), C = below the 2.5% quantile
(too contrasting), grey = typical, dot = region not covered yet (Gemini, free-tier daily cap).
H1 said LLMs push contrasting cuisines toward uniform. That is not what we see: the pre-registered Fisher test is
supported for {V["h1_fisher_supported"]}/{V["n"]} models and the equal-size shift for {V["h1eq_supported"]}/{V["n"]}.
Instead each model has its own pattern ({"; ".join(short(L[k]["name"]) + ": " + str(L[k]["equal_n_typical"]) + " typical" for k in keys)}),
and a few regions are wrong in every model: {", ".join(c["region"] + " (" + c["verdict"] + ")" for c in cons)}.
"""),
    )


def s_drivers(deck: Deck, F) -> None:
    s = deck.new("main", "drivers")
    DR, V, E, rs = F["drivers"], F["verdicts"], F["equal_n"], F["ranges"]
    header(deck, s, "Do LLMs keep the ingredients and categories that drive pairing?",
           minus(f"Only partly: top-5 flavor drivers overlap {d1.rng_text(DR['jac_min'], DR['jac_max']) if DR else '?'} "
                 f"(genuine {f((DR or {}).get('real', {}).get('median_jaccard5'))}), and category fingerprints "
                 f"correlate {d1.rng_text(*rs['median_fingerprint_spearman'])} (genuine "
                 f"{f(E['ref'].get('fingerprint_spearman'))}).") if DR else pending("drivers"))  # fmt: skip
    if DR:
        real = DR["korea_real_top5"]
        rows = [
            [
                "Rank",
                "Real Korea",
                *[short(DR["models"][k]["name"]) for k in DR["order"] if DR["models"][k].get("korea_top5")],
            ]
        ]
        ks = [k for k in DR["order"] if DR["models"][k].get("korea_top5")]
        fills = {}
        for i in range(5):
            row = [str(i + 1), real[i] if i < len(real) else ""]
            for j, k in enumerate(ks, start=2):
                top = DR["models"][k]["korea_top5"]
                item = top[i] if i < len(top) else ""
                row.append(item)
                if item and item in real:
                    fills[(i + 1, j)] = "#D7F0E5"
            rows.append(row)
        widths = [0.6, 1.55] + [1.55] * len(ks)
        deck.text(s, 0.55, 1.98, 6.4, 0.4, [Para("Korea's top-5 flavor drivers (χ%)", 15, bold=True, space_before=0)],
                  fit=False)  # fmt: skip
        deck.table(s, 0.55, 2.42, widths, rows, size=12, row_h=0.36, aligns=["c"] + ["l"] * (len(widths) - 1),
                   fills=fills, zebra=False)  # fmt: skip
        y = 2.42 + deck.last_table_h + 0.1
        deck.text(s, 0.55, y, 6.4, 0.9, [Para(
            f"Green = also in the real top-5. {join_and(DR['korea_match'])} put {real[0] if real else '?'} first, as the "
            f"real cuisine does" + (f"; {join_and(DR['korea_miss'])} does not." if DR["korea_miss"] else "."), 12.5,
            space_before=0)], min_scale=0.75)  # fmt: skip
        cats = [short(DR["models"][k]["name"]) for k in DR["order"]] + ["Genuine"]
        vals = [DR["models"][k].get("median_jaccard5") for k in DR["order"]] + [DR["real"].get("median_jaccard5")]
        colors = [F["models"][k]["color"] for k in DR["order"]] + ["#000000"]
        bar_chart(
            deck, s, 0.55, (y + 0.62), 6.4, 6.95 - (y + 0.62), "Top-5 driver overlap (Jaccard@5)", cats, vals, colors
        )
    right = None  # F6/F7 are multi-panel: shown full-page in the backups, a native chart is readable here
    if right:
        deck.picture(s, right, 7.2, 1.98, 5.6, 4.6, align="center")
        caption(deck, s, 7.2, 6.62, 5.6, "F6/F7: real vs LLM.")
    else:
        L = F["llm"]
        cats = [short(L[k]["name"]) for k in L] + ["Genuine", "Copy-mut."]
        vals = [L[k]["median_fingerprint_spearman"] for k in L] + [
            E["ref"].get("fingerprint_spearman"),
            F["cm"]["ranked_n"].get("fingerprint_spearman"),
        ]
        colors = [L[k]["color"] for k in L] + ["#000000", "#999999"]
        bar_chart(deck, s, 7.2, 1.98, 5.6, 3.6, "Category fingerprint ρ with the real cuisine", cats, vals, colors)
        below = "; ".join(f"{short(L[k]['name'])} {L[k]['h2c_regions_below']}/{L[k]['regions']}" for k in L)
        deck.text(
            s,
            7.2,
            5.7,
            5.6,
            1.2,
            [
                Para(
                    f"H2c (weaker fingerprint) supported for {V['h2c_supported']}/{V['n']} models. Regions below the genuine "
                    f"resampling range: {below}. Full figures F6 and F7: backup.",
                    13,
                    space_before=0,
                )
            ],
            min_scale=0.75,
        )
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 60 s)
Two finer layers of the fingerprint. Left, ingredient-level drivers (χ%: % change in mean Ns when the ingredient is
removed, the definition we calibrated in the replication). For Korea the real top driver is Soybean Sauce;
{join_and((DR or {}).get("korea_match", []))} put it first too{("; " + join_and(DR["korea_miss"]) + " leads with " + ", ".join(DR["models"][k]["korea_first"] for k in DR["order"] if DR["models"][k]["name"] in DR["korea_miss"])) if DR and DR["korea_miss"] else ""}.
Across all regions the top-5 overlap with the real drivers is well below what genuine samples achieve.
Right, category fingerprints (Jain 2015): LLM fingerprints correlate {d1.rng_text(*rs["median_fingerprint_spearman"])}
with the real ones, genuine resamples {f(E["ref"].get("fingerprint_spearman"))}; H2c is supported in
{V["h2c_supported"]}/{V["n"]} models.
"""),
    )


def s_stereotypes(deck: Deck, F) -> None:
    s = deck.new("main", "stereotypes")
    L, V, E, rs = F["llm"], F["verdicts"], F["equal_n"], F["ranges"]
    header(deck, s, "Do LLMs amplify each cuisine's stereotypes?",
           f"No: signature ingredients are used about as often as in real recipes (median SAI "
           f"{d1.rng_text(*rs['median_sai'])}). Instead the models narrow cuisines to "
           f"{d1.pct_rng(*rs['median_diversity_ratio'])} of their ingredient variety.")  # fmt: skip
    deck.picture(s, F["figures"]["F8"], 0.55, 1.95, 6.0, 5.0, align="left")
    f9 = fig(F, "F9")
    if f9:
        deck.picture(s, f9, 6.75, 1.95, 6.05, 3.2, align="center")
        y = 5.25
    else:
        cats = [short(L[k]["name"]) for k in L] + ["Genuine"]
        bar_chart(deck, s, 6.75, 1.95, 6.05, 3.1, "Diversity ratio (distinct ingredients vs genuine)", cats,
                  [L[k]["median_diversity_ratio"] for k in L] + [1.0], [L[k]["color"] for k in L] + ["#000000"],
                  ymax=1.3)  # fmt: skip
        y = 5.15
    band = E["sai_band"]
    deck.text(s, 6.75, y, 6.05, 6.95 - y, [
        Para(f"**SAI** = how often the top-10 signature ingredients are used, LLM / real. Genuine samples: 95% range "
             f"{f(band[0], 3)}–{f(band[1], 3)}. H2b (SAI > 1) supported for {V['h2b_sai_supported']}/{V['n']} "
             "models.", 13, space_before=0),
        Para(f"**Diversity** lower in {V['h2b_div_supported']}/{V['n']} models; ingredient prevalence differs by JSD "
             f"{d1.rng_text(*rs['median_prevalence_jsd'], 3)} vs {f(E['ref'].get('prevalence_jsd'), 3)} for genuine "
             "samples.", 13, space_before=6),
    ], min_scale=0.72)  # fmt: skip
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 50 s)
HOW TO READ F8: one row per region; the grey band is the 95% range of SAI in genuine samples of the same size; markers
are the models (log scale, 1 = as in real recipes). Per region the models scatter on both sides, but the median SAI is
{d1.rng_text(*rs["median_sai"])}, and the one-sided test for SAI > 1 is supported for {V["h2b_sai_supported"]}/{V["n"]}
models: no systematic stereotype amplification. What the models do is narrow the cuisine: they use
{d1.pct_rng(*rs["median_diversity_ratio"])} of the distinct ingredients a genuine sample of the same size uses. One-liner:
the AI chef does not caricature Korean food with extra soy sauce; it cooks a smaller, more repetitive Korean menu.
"""),
    )


def s_networks(deck: Deck, F) -> None:
    s = deck.new("main", "networks")
    N = F.get("networks") or {}
    nt, be, bo = N.get("npmi_top100") or {}, N.get("backbone_edges") or {}, N.get("backbone_overlap") or {}
    keys = [k for k in F["model_order"] if k in nt]
    if not keys:
        header(deck, s, "Do LLMs reproduce which ingredients go together?", pending("network results"))
        return
    llm_vals = [nt[k]["median"] for k in keys]
    gen = (nt.get("real@160") or {}).get("median")
    header(deck, s, "Do LLMs reproduce which ingredients go together?",
           minus(f"They keep a generic skeleton of strong pairings but miss each cuisine's distinctive ones: their top-100 "
                 f"ingredient pairs overlap the real ones {d1.rng_text(min(llm_vals), max(llm_vals))}, genuine samples "
                 f"{f(gen)}."))  # fmt: skip
    f11 = fig(F, "F11")
    if f11:
        deck.picture(s, f11, 0.55, 1.95, 7.75, 5.0, align="left")
    else:
        names = [short(F["models"][k]["name"]) for k in keys] + ["Genuine"]
        cols = [F["models"][k]["color"] for k in keys] + ["#000000"]
        bar_chart(deck, s, 0.55, 1.95, 7.75, 5.0, "Top-100 NPMI pair overlap with the real cuisine", names,
                  llm_vals + [gen], cols)  # fmt: skip
    full = [k for k in keys if nt[k]["regions"] == F["regions"]["n"]]
    part = [k for k in keys if k not in full]
    below = [nt[k]["below_reference_fdr05"] for k in full]
    be_v = [be.get(k, {}).get("median") for k in keys if be.get(k, {}).get("median") is not None]
    bo_v = [bo.get(k, {}).get("median") for k in keys if bo.get(k, {}).get("median") is not None]
    x, w = 8.5, 4.3
    deck.tile(s, x, 1.98, w, 1.5, f"{d1.rng_text(min(llm_vals), max(llm_vals))} vs {f(gen)}",
              "Jaccard overlap of the top-100 NPMI ingredient pairs with the real cuisine (LLMs vs genuine "
              "samples of the same size)", color=T.accent, big_size=24, cap_size=12.5)  # fmt: skip
    btxt = f"{min(below)}–{max(below)} / {F['regions']['n']}" if below else "?"
    ptxt = "; ".join(
        f"{short(F['models'][k]['name'])} {nt[k]['below_reference_fdr05']}/{nt[k]['regions']}" for k in part
    )
    deck.tile(s, x, 3.6, w, 1.5, btxt, "regions per model below the genuine range (BH)" + (f"; {ptxt}" if ptxt else ""),
              color=T.ink, big_size=24, cap_size=12.5)  # fmt: skip
    deck.tile(
        s,
        x,
        5.22,
        w,
        1.73,
        f"{f(min(be_v), 0)}–{f(max(be_v), 0)} vs {f((be.get('real@160') or {}).get('median'), 0)}",
        f"edges in the disparity-filter backbone; but {d1.pct_rng(min(bo_v), max(bo_v))} of the LLM backbone edges "
        "are real backbone edges: a sparser, generic skeleton",
        color=T.blue,
        big_size=24,
        cap_size=12.5,
    )
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 50 s)
This layer ignores chemistry: which ingredient pairs co-occur more than chance (normalised PMI, NPMI) in each
cuisine. HOW TO READ F11: one row per region; the grey bar is the range of genuine samples of the same size, the
markers are the models. Our pre-registered metric is the Jaccard overlap between the top-100 NPMI pairs of the LLM
cuisine and of the real cuisine: genuine samples reach {f(gen)}, the models {d1.rng_text(min(llm_vals), max(llm_vals))},
below the genuine range in most regions ({", ".join(short(F["models"][k]["name"]) + " " + str(nt[k]["below_reference_fdr05"]) + "/" + str(nt[k]["regions"]) for k in keys)}).
The disparity-filter backbones are sparser ({", ".join(f(be.get(k, {}).get("median"), 0) for k in keys)} edges vs about
{f((be.get("real@160") or {}).get("median"), 0)}), yet most of their edges are real backbone edges
({", ".join(pct(bo.get(k, {}).get("median"), 0) for k in keys)}). So the models know the strong, generic pairings but
miss the pairings that make each cuisine distinctive (examples in backup F11b).
"""),
    )


def s_steer_method(deck: Deck, F) -> None:
    s = deck.new("main", "steering method")
    SC, SD = F["scorer"], F["steer_design"]
    auc = SC.get("auc_test", {})
    header(deck, s, "How can a fingerprint steer generation?",
           "A fidelity scorer built from three fingerprint layers either picks the best of N recipes or tells the "
           "model what to fix (critique-and-revise).")  # fmt: skip
    comps = [("Cuisine classifier", f"{SC.get('classifier_name')} on ingredients + categories; calibrated; its "
              "weights say which ingredients are atypical"),
             ("Co-occurrence", "mean shrinkage-smoothed PMI of the recipe's ingredient pairs in the cuisine"),
             ("Ns consistency", "how typical the recipe's flavor sharing is for the cuisine")]  # fmt: skip
    deck.text(s, 0.55, 1.95, 4.3, 0.4, [Para("Fidelity scorer (D3)", 16, bold=True, space_before=0)], fit=False)
    for i, (t, b) in enumerate(comps):
        card(deck, s, 0.55, 2.4 + i * 1.18, 4.3, 1.08, t, [b], accent=T.green, title_size=14, body_size=12)
    deck.text(s, 0.55, 5.98, 4.3, 0.97, [Para(
        f"Equal weights (chosen on validation). Test AUC, real held-out recipes vs other cuisines / copy-mutate / null "
        f"recipes: **{f(auc.get('other_cuisines', {}).get('composite'))} / {f(auc.get('copy_mutate', {}).get('composite'))}"
        f" / {f(auc.get('random_null', {}).get('composite'))}**.", 12, space_before=0)], min_scale=0.75)  # fmt: skip
    # two methods
    for x, t, steps, col in (
        (5.2, "Best-of-N (selection)", ["Take N recipes the model wrote for the region",
                                         "Score each; drop near-copies and duplicates",
                                         "Keep the best; repeat for N = 1 (control), 2, 4, 8"], T.blue),
        (9.05, "Critique-and-revise (feedback)", ["Score the recipe and build feedback: atypical ingredients, missing "
                                                  "signature ingredients that pair well, Ns too high/low",
                                                  "Ask the model to revise but keep the dish (Self-Refine)",
                                                  "Re-score; up to 2 rounds"], T.accent),
    ):  # fmt: skip
        deck.rect(s, x, 1.98, 3.75, 3.85, T.panel)
        deck.rect(s, x, 1.98, 3.75, 0.5, col)
        deck.text(s, x + 0.12, 1.98, 3.5, 0.5, [Para(t, 14, bold=True, color="#FFFFFF", space_before=0)],
                  anchor="middle", fit=True)  # fmt: skip
        deck.text(s, x + 0.12, 2.58, 3.5, 3.2, [Para(f"{i + 1}.  {st}", 13, space_before=8) for i, st in enumerate(steps)],
                  min_scale=0.75)  # fmt: skip
    deck.rect(s, 5.2, 5.98, 7.6, 0.97, T.panel2)
    deck.text(s, 5.32, 5.98, 7.4, 0.97, [Para(
        f"**Guards:** a recipe with Jaccard ≥ {F['method']['near_copy']} to any CulinaryDB recipe is never selected; "
        "evaluation uses only held-out metrics (an independent gradient-boosting classifier on a disjoint split, "
        "JSD, SAI, fingerprint, diversity, novelty), never the scorer itself.", 12, space_before=0)], anchor="middle",
        min_scale=0.75)  # fmt: skip
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 50 s)
The scorer turns the fingerprint into a number per recipe: a linear cuisine classifier (linear so its weights can
name the atypical ingredients), the co-occurrence of the recipe's ingredient pairs in the cuisine, and how typical
its Ns is. It separates real held-out recipes from other cuisines, copy-mutate recipes and null recipes with AUC
{f(auc.get("other_cuisines", {}).get("composite"))}, {f(auc.get("copy_mutate", {}).get("composite"))} and
{f(auc.get("random_null", {}).get("composite"))}.
Two ways to use it. Best-of-N only selects among recipes the model already wrote. Critique-and-revise uses the
scorer's feedback to ask for a better version of the same dish. Both are guarded against copying, and both are
judged by metrics the scorer never saw.
DEVIATION (if asked): best-of-N reranked the existing {(F["bon"] or {}).get("meta", {}).get("pool_max", "?")} recipes per
region instead of a {n0(SD["pool_target"])}-candidate pool, because that pool costs about {f(SD["gpu_hours_min"], 0)}–{f(SD["gpu_hours_max"], 0)}
GPU-hours per model on the Kaggle T4 fallback.
"""),
    )


def _load_example():
    try:
        return json.loads(STEER_EXAMPLE.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def line_chart(deck, s, x, y, w, h, title, cats, series, fmt="0.00"):
    """series: list of (name, values, color)."""
    from pptx.enum.chart import XL_LEGEND_POSITION

    cd = CategoryChartData()
    cd.categories = cats
    for name, vals, _ in series:
        cd.add_series(name, vals)
    ch = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    ch.font.name, ch.font.size = FONT, Pt(10)
    ch.has_title = True
    ch.chart_title.text_frame.text = title
    for r in ch.chart_title.text_frame.paragraphs[0].runs:
        r.font.size, r.font.bold, r.font.name = Pt(12.5), True, FONT
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(10)
    for ser, (_, _, col) in zip(ch.plots[0].series, series, strict=True):
        ser.format.line.color.rgb = rgb(col)
        ser.format.line.width = Pt(2.5)
        ser.smooth = False
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = rgb(col)
        ser.marker.format.line.color.rgb = rgb(col)
    va = ch.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb("#E3E7EC")
    va.tick_labels.number_format, va.tick_labels.number_format_is_linked = fmt, False
    va.tick_labels.font.size = Pt(9)
    vals = [v for _, vs, _ in series for v in vs if v is not None]
    if vals:
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.25 or 0.05
        va.minimum_scale, va.maximum_scale = max(0, lo - pad), hi + pad
    ch.category_axis.tick_labels.font.size = Pt(10)
    return ch


def s_steer_example(deck: Deck, F) -> None:
    s = deck.new("main", "steering example")
    RV, B = F["revise"], F["bon"]
    ex = _load_example()
    names = {k: m["name"] for k, m in F["models"].items()}
    if ex:
        b, a = ex.get("before", {}), ex.get("after", {})
        answer = (
            f"A typical revision: the model kept the dish and added the missing signature ingredients the "
            f"feedback named; fidelity rose from the {b.get('fidelity_percentile', 0):.0f}th to the "
            f"{a.get('fidelity_percentile', 0):.0f}th percentile, but the independent classifier still says "
            f"{a.get('independent_classifier', '?')}."
        )
    else:
        answer = pending("steering example not available")
    header(deck, s, "What does steering change in practice?", answer)
    if ex:
        added = set(ex.get("entities_added") or [])
        reg = ex.get("region", "?")
        for x, obj, lab, col in ((0.55, b, "Before (original)", T.ink), (3.85, a, "After 1 revision", T.green)):
            deck.rect(s, x, 1.98, 3.15, 3.42, T.panel)
            deck.rect(s, x, 1.98, 3.15, 0.4, col)
            deck.text(s, x + 0.1, 1.98, 2.95, 0.4, [Para(f"{lab} · {reg}", 12.5, bold=True, color="#FFFFFF",
                                                         space_before=0)], anchor="middle", fit=True)  # fmt: skip
            paras = [Para(obj.get("title", ""), 13.5, bold=True, space_before=0)]
            for ln in (obj.get("ingredients") or [])[:10]:
                new = obj is a and any(e.lower() in ln.lower() for e in added)
                paras.append(Para(("+ " if new else "") + ln[:52], 11.5, bold=new, color=T.green if new else T.ink,
                                  bullet=not new, space_before=1))  # fmt: skip
            deck.text(s, x + 0.1, 2.45, 2.95, 2.45, paras, min_scale=0.65)
            cz = obj.get("components_z") or {}
            deck.text(s, x + 0.1, 4.9, 2.95, 0.48, [Para(
                f"Fidelity percentile **{obj.get('fidelity_percentile', 0):.1f}** · classifier "
                f"{sf(cz.get('classifier'))} · co-occurrence {sf(cz.get('cooccurrence'))}", 10.5, space_before=0)],
                min_scale=0.7)  # fmt: skip
        fb = ex.get("feedback_given") or {}
        ctx = ex.get("context") or {}
        deck.rect(s, 0.55, 5.5, 6.45, 1.45, T.panel2)
        deck.text(s, 0.67, 5.52, 6.25, 1.42, [
            Para("**Feedback sent:** " + (f"unusual: {', '.join(fb['atypical'])}; " if fb.get("atypical") else "")
                 + (f"missing signature ingredients that pair well: {', '.join(fb.get('missing') or [])}; " if fb.get("missing") else "")
                 + f"Ns {fb.get('ns_verdict', '?')} ({100 * (fb.get('ns_percentile') or 0):.0f}th percentile).", 11.5,
                 space_before=0),
            Para(f"**Rule:** {ex.get('rule', '?')} ({names.get(ex.get('model'), ex.get('model'))}). Across all "
                 f"{ctx.get('revisions_scored', '?')} round-1 revisions, {pct(ctx.get('share_score_raised'), 0)} raised "
                 f"the score; median percentile {f(ctx.get('median_percentile_before'), 1)} → "
                 f"{f(ctx.get('median_percentile_after'), 1)}. Ingredient overlap with the original: "
                 f"{f(ex.get('identity_jaccard'))}.", 10.5, color=T.muted, space_before=4),
        ], min_scale=0.7)  # fmt: skip
    # right: readable native curves (full figures F15 and F15c in the backups)
    if RV:
        rmax = max(r["rounds"] for r in RV["rows"].values())
        cats = [str(i) for i in range(rmax + 1)]
        series = []
        for k in RV["order"]:
            r = RV["rows"][k]
            vals = [None] * (rmax + 1)
            vals[0] = r["first"]["indep_classifier_acc"]
            vals[r["rounds"]] = r["last"]["indep_classifier_acc"]
            series.append((short(r["name"]), vals, F["models"].get(k, {}).get("color", "#666666")))
        _fill_rounds(series, RV)
        line_chart(deck, s, 7.2, 1.98, 5.6, 2.45, "Critique-and-revise: share assigned to the claimed cuisine",
                   [f"round {c}" for c in cats], series)  # fmt: skip
    if B:
        ns = sorted({n for r in B["rows"].values() for n in r["Ns"]})
        series = []
        for k in B["order"]:
            r = B["rows"][k]
            look = dict(zip(r["series"]["N"], r["series"]["indep_classifier_acc"], strict=True))
            series.append((short(r["name"]), [look.get(n) for n in ns], F["models"].get(k, {}).get("color", "#666666")))
        line_chart(deck, s, 7.2, 4.5, 5.6, 2.45, "Best-of-N: share assigned to the claimed cuisine",
                   [f"N = {n}" for n in ns], series)  # fmt: skip
    bon_div = "; ".join(f"{short(r['name'])} {f(r['first']['diversity_ratio'])} → {f(r['last']['diversity_ratio'])}"
                        for r in (B or {}).get("rows", {}).values())  # fmt: skip
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 70 s)
LEFT: one real example, chosen by a fixed rule ({(ex or {}).get("rule", "n/a")}), so it is typical, not the best case.
The feedback said the recipe lacked signature ingredients that pair well ({", ".join(((ex or {}).get("feedback_given") or {}).get("missing") or [])}) and that its flavor sharing was low.
The model kept the dish and added exactly those; the fidelity percentile rose from
{f((ex or {}).get("before", {}).get("fidelity_percentile"), 1)} to {f((ex or {}).get("after", {}).get("fidelity_percentile"), 1)}.
Honest detail: the independent classifier still assigns it to {(ex or {}).get("after", {}).get("independent_classifier", "?")}, so one revision is not
enough to make it recognisably Spanish.
RIGHT, TOP: across regions, critique-and-revise raises the independent classifier's share of recipes assigned to the
claimed cuisine ({"; ".join(short(r["name"]) + " " + f(r["first"]["indep_classifier_acc"]) + " → " + f(r["last"]["indep_classifier_acc"]) for r in (RV or {}).get("rows", {}).values())}).
RIGHT, BOTTOM: best-of-N selection raises it too but costs diversity ({bon_div}). Full figures F15 and F15c are in
the backups. No CIs yet for revise; Qwen's second round and steering for Llama were not run.
"""),
    )


def _fill_rounds(series, RV) -> None:
    """Fill intermediate rounds from the raw revise rows (summary.json keeps every round)."""
    rows = RV.get("raw") or []
    by = {}
    for r in rows:
        by.setdefault(r.get("model"), {})[int(r.get("round", 0))] = r.get("indep_classifier_acc")
    for k, (_name, vals, _col) in zip(RV["order"], list(series), strict=True):
        for rd, v in (by.get(k) or {}).items():
            if rd < len(vals):
                vals[rd] = v


def s_library(deck: Deck, F) -> None:
    s = deck.new("main", "library")
    LB, R = F["library"], F["replication"]
    L = F["llm"]
    header(deck, s, "How do you run the audit yourself with FlavorAudit?",
           f"`flavoraudit` is a pip-installable, MIT-licensed library: {LB['n_modules']} modules, {LB['n_tests']} tests, "
           "and the whole audit in a few lines or one command.")  # fmt: skip
    code = [
        "from flavoraudit import load_world",
        "from flavoraudit.stats import food_pairing",
        "",
        "world = load_world()    # data downloaded on first use",
        'res = food_pairing(world.cuisines["Korea"], world.S,',
        '                   null_model="random")',
        "res.z_sb, res.d",
    ]
    deck.rect(s, 0.55, 1.98, 6.3, 2.35, "#1E2633")
    deck.text(s, 0.7, 2.05, 6.0, 2.2, [Para(c or " ", 12.5, font=MONO, color="#E8EDF3", space_before=0) for c in code],
              min_scale=0.8)  # fmt: skip
    deck.rect(s, 0.55, 4.43, 6.3, 0.62, "#F7F2EA")
    deck.text(s, 0.7, 4.43, 6.0, 0.62, [Para(minus(f">>> ({f(R['korea_z'], 1)}, {f(R['korea_d'], 3)})"), 14,
                                              font=MONO, bold=True, color=T.accent, space_before=0)],
              anchor="middle", fit=False)  # fmt: skip
    deck.text(s, 0.55, 5.15, 6.3, 1.0, [Para(
        "Korea's Z against the random null and its effect size d, as in our replication (Fig. 4: contrasting). "
        f"Tests: property tests plus a slow regression that reproduces all {R['random'].get('n')} Fig. 4 signs; CI on "
        f"Python {join_and(LB['ci_python'])}.", 12, color=T.muted, space_before=0)], min_scale=0.75)  # fmt: skip
    deck.rect(s, 7.1, 1.98, 5.7, 1.0, "#1E2633")
    deck.text(
        s,
        7.25,
        1.98,
        5.5,
        1.0,
        [
            Para("$ flavoraudit audit my_recipes.csv \\", 12.5, font=MONO, color="#E8EDF3", space_before=0),
            Para("      --cuisine-col cuisine --out out/", 12.5, font=MONO, color="#E8EDF3", space_before=0),
        ],
        anchor="middle",
        fit=False,
    )
    rows = [["Scorecard (median over regions)", "Sign agr.", "Fingerprint ρ", "SAI", "Diversity"]]
    for _k, p in L.items():
        rows.append([p["name"], f(p["sign_agreement"]), f(p["median_fingerprint_spearman"]), f(p["median_sai"]),
                     f(p["median_diversity_ratio"])])  # fmt: skip
    deck.table(s, 7.1, 3.1, [2.25, 0.85, 1.05, 0.7, 0.85], rows, size=11.5, row_h=0.36,
               aligns=["l", "c", "c", "c", "c"])  # fmt: skip
    y = 3.1 + deck.last_table_h + 0.1
    deck.text(s, 7.1, y, 5.7, 6.95 - y, [Para(
        "One row per model × region in out/scorecard.csv (shown: medians from our LLMCuisine-22 run). Modules: "
        + " · ".join(LB["modules"]) + ".", 11.5, color=T.muted, space_before=0)], min_scale=0.75)  # fmt: skip
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 40 s)
D4 is FlavorAudit (pip name flavoraudit). Left: three lines replicate Korea's food pairing; the output is Korea's Z
against the random null and its effect size d, exactly the values of our replication ({f(R["korea_z"], 1)},
{f(R["korea_d"], 3)}). Data are downloaded on first use under CulinaryDB/FlavorDB's CC BY-NC-SA licence and never
bundled in the MIT package. Right: the command line audits any CSV of recipes against the real cuisines and writes a
scorecard; the table shows the medians our LLM run produces. {LB["n_tests"]} tests, including a slow regression that
reproduces the Fig. 4 signs. Publishing to PyPI waits for the team's go-ahead.
"""),
    )


def s_app(deck: Deck, F, demo) -> None:
    s = deck.new("main", "app")
    header(deck, s, "How does FlavorPrint audit a single recipe?",
           "Paste a recipe and the cuisine it claims; FlavorPrint maps the ingredients, places the recipe among real "
           "recipes of that cuisine, scores its fidelity and suggests fixes.")  # fmt: skip
    shots = [
        SHOTS / n
        for n in (
            "01_audit_overview.png",
            "04_fidelity.png",
            "05_suggestions.png",
            "02_ingredient_mapping.png",
            "03_flavor_network.png",
            "06_dashboard.png",
        )
    ]
    have = [p for p in shots if p.exists()]
    if have:
        main = have[0]
        deck.picture(s, main, 0.55, 1.95, 7.6, 4.6, align="left")
        caption(deck, s, 0.55, 6.6, 7.6, f"FlavorPrint · {main.stem.split('_', 1)[1].replace('_', ' ')}")
        for i, p in enumerate(have[1:3]):
            deck.picture(s, p, 8.35, 1.95 + i * 2.5, 4.45, 2.2, align="center")
            caption(deck, s, 8.35, 4.17 + i * 2.5, 4.45, p.stem.split("_", 1)[1].replace("_", " "))
    else:
        demo_table(deck, s, demo, 1.98)
    deck.notes(
        s,
        """
WHAT TO SAY (about 30 s, then the live demo — see prep/03_demo_walkthrough.md)
FlavorPrint is D5. It uses the same aliaser, flavor network and scorer as the study. The live demo: a genuine Kimchi
jjigae scores in the upper half of real Korean recipes; an Italian spaghetti claimed as Korean drops to the bottom
percentiles with "hard wheat" and "tomato" flagged; the chana masala example triggers the stereotype flag.
If the live demo fails, use these screenshots and the "Demo fallback" backup slide.
""",
    )


def demo_table(deck: Deck, s, demo, y) -> None:
    """The three demo cases as a table (values from the app's own audit function, presentation/demo_probe.py)."""
    if not demo:
        deck.text(s, 0.55, y, 12.25, 1.0, [Para(pending("app artifacts not available at build time"), 16)])
        return
    sc = demo["scenarios"]
    order = ["genuine", "off_cuisine", "stereotype"]
    spec = [
        ("Recipe (claimed cuisine)", lambda d: f"{d['title']} ({d['claimed']})"),
        ("Fidelity percentile", lambda d: f"{d['fidelity_pct']:.0f}"),
        ("Flavor sharing Ns (percentile)", lambda d: f"{d['ns']:.1f} ({d['ns_pct']:.0f})"),
        ("Most likely cuisine", lambda d: f"{d['top5'][0][0]} ({100 * d['top5'][0][1]:.0f}%)" if d["top5"] else "—"),
        ("Unusual for the cuisine", lambda d: ", ".join(d["atypical"]) or "none"),
        ("Stereotype flag", lambda d: "yes" if d["stereotype_flag"] else "no"),
    ]
    rows = [["", "Genuine", "Off-cuisine", "Stereotype check"]] + [
        [lab] + [fn(sc[k]) for k in order] for lab, fn in spec
    ]
    deck.table(s, 0.55, y, [3.1, 3.05, 3.05, 3.05], rows, size=13, row_h=0.5, aligns=["l", "c", "c", "c"],
               bold_first_col=True)  # fmt: skip
    deck.text(s, 0.55, y + deck.last_table_h + 0.1, 12.25, 0.4, [Para(
        "App screenshots appear here when app/screenshots/ is populated; values above come from the app's own audit "
        "function on its built-in examples.", 11, color=T.muted, space_before=0)], fit=True)  # fmt: skip


def s_limitations(deck: Deck, F) -> None:
    s = deck.new("main", "limitations")
    ds, E, SD = F["dataset"], F["equal_n"], F["steer_design"]
    meta = (F["bon"] or {}).get("meta", {})
    header(deck, s, "What are the limits of these results?",
           "Data versions, small per-region LLM cuisines and an unfinished model roster. Each one is measured, "
           "corrected for, or stated.")  # fmt: skip
    st = {k: F["models"][k]["status"] for k in F["incomplete_models"]}
    items = [
        ("FlavorDB version", "The 2018 snapshot is offline. With today's FlavorDB, signs and ranks replicate (22/22, "
                             f"ρ {f(F['replication']['random'].get('spearman'))}); contrasting cuisines come out more "
                             "contrasting."),
        ("Small LLM cuisines", f"{F['design']['recipes_per_region']} recipes per region: genuine samples of that size "
                               f"share the full cuisine's sign only {pct(E['genuine_sign_agreement'], 0)} of the time, "
                               "so every test is against equal-size genuine samples."),
        ("Model roster", "; ".join(f"{F['models'][k]['name']}: {v}" for k, v in st.items()) + "."),
        ("Aliasing", f"LLM lines are aliased with CulinaryDB's protocol ({d1.pct_rng(ds['alias_full_min'], ds['alias_full_max'])} "
                     "fully matched); proposed aliases for unmatched terms (e.g. gochujang) await review."),
        ("Steering scale", f"Best-of-N reranked {meta.get('pool_max', '?')} existing recipes per region "
                           f"(pre-registered {n0(SD['pool_target'])}); critique-and-revise has no CIs yet; steering was "
                           f"not run for {join_and(F['unsteered_models']) or 'none'}."),
        ("No taste test", "Fidelity is measured against CulinaryDB web recipes, not rated by cooks or chefs."),
    ]  # fmt: skip
    w, h = (12.25 - 0.2) / 2, (4.97 - 2 * 0.15) / 3
    for i, (t, b) in enumerate(items):
        r, c = divmod(i, 2)
        card(deck, s, 0.55 + c * (w + 0.2), 1.98 + r * (h + 0.15), w, h, t, [b], accent=T.muted, title_size=15,
             body_size=13)  # fmt: skip
    deck.notes(
        s,
        """
WHAT TO SAY (about 40 s)
Say these before the professor does. The FlavorDB version is handled by replicating signs and ranks first. Small LLM
cuisines are exactly why we built the equal-size reference. The roster status is stated as status, not predicted:
Gemini is limited by the free-tier daily cap, the extra Qwen and Ministral runs are on Kaggle, alias curation awaits
review. Steering ran at a smaller scale than pre-registered for compute reasons, and there is no human taste test.
""",
    )


def s_conclusion(deck: Deck, F) -> None:
    s = deck.new("main", "conclusion")
    V, R, E, rs = F["verdicts"], F["replication"], F["equal_n"], F["ranges"]
    ds = F["dataset"]
    header(deck, s, "So, do AI chefs respect culinary fingerprints?",
           "Partly. They use a cuisine's signature ingredients about as real cooks do, but lose its variety, its "
           "distinctive pairings and its category fingerprint; feedback can steer them back toward it.")  # fmt: skip
    contrib = [
        f"**LLMCuisine-22:** {n0(ds['total_recipes'])} recipes, {F['regions']['n']} regions, {ds['n_models']} LLMs, "
        "aliased exactly as CulinaryDB.",
        f"**First flavor-compound audit of LLM recipes**, after replicating Fig. 4 ({R['random'].get('sign_agreement')}/"
        f"{R['random'].get('n')} signs, ρ {f(R['random'].get('spearman'))}).",
        f"**Equal-size calibration:** the own-pool null drifts with N; the sign rule misfires on {pct(E['fpr'])} of "
        "genuine samples; the contrasting pattern is robust.",
        "**Fingerprint-guided steering** (scorer, best-of-N, critique-and-revise), **FlavorAudit** and **FlavorPrint**.",
    ]  # fmt: skip
    future = [
        "Finish the roster (Gemini, Qwen3.5-4B scale pair, Ministral) and the alias curation.",
        "Steering at the pre-registered scale, with CIs, for every model.",
        "Blind ratings by cooks and chefs (a “Turing test for chefs”).",
        "Robustness to flavor profiles (molecule dropout, Ahn 2011 profiles).",
    ]
    deck.rect(s, 0.55, 1.98, 7.0, 4.32, T.panel)
    deck.text(
        s,
        0.7,
        2.03,
        6.75,
        4.22,
        [Para("What is new", 17, bold=True, color=T.accent, space_before=0)]
        + [Para(c, 14, bullet=True, space_before=7) for c in contrib]
        + [
            Para(
                "Baselines, not contributions: the nulls, Z and χ (Ahn; Jain; Singh & Bagler) and the copy-mutate "
                "model (Tuwani 2019; our Assignment 3).",
                12,
                color=T.muted,
                space_before=8,
            )
        ],
        min_scale=0.75,
    )
    deck.rect(s, 7.75, 1.98, 5.05, 4.32, T.panel)
    deck.text(
        s,
        7.9,
        2.03,
        4.8,
        4.22,
        [Para("What next", 17, bold=True, color=T.blue, space_before=0)]
        + [Para(c, 14, bullet=True, space_before=7) for c in future],
        min_scale=0.75,
    )
    deck.rect(s, 0.55, 6.42, 12.25, 0.53, T.ink)
    deck.text(
        s,
        0.55,
        6.42,
        12.25,
        0.53,
        [Para("Thank you. Questions?", 20, bold=True, color="#FFFFFF", align="center", space_before=0)],
        anchor="middle",
        fit=False,
    )
    deck.notes(
        s,
        minus(f"""
WHAT TO SAY (about 45 s)
The answer to the title question is: partly. The models know a cuisine's signature ingredients (median SAI
{d1.rng_text(*rs["median_sai"])}), but they cook a narrower version of it ({d1.pct_rng(*rs["median_diversity_ratio"])}
of the variety), miss its distinctive ingredient pairings, and weaken its category fingerprint (H2c in
{V["h2c_supported"]}/{V["n"]} models). Food pairing errors are model-specific, not a general drift to uniform.
Feedback from the fingerprints moves recipes back toward the cuisine while keeping the dish.
What is new: the dataset, the first flavor-compound audit of LLM recipes, the equal-size calibration, the steering,
and the library and app. What is reused: your lab's methods and the copy-mutate baseline.
Keep this slide up for questions.
"""),
    )


# =====================================================================================================================
def _as_backup(deck: Deck, fn, *args) -> None:
    n = len(deck.meta)
    fn(deck, *args)
    for m in deck.meta[n:]:
        m["kind"] = "backup"


def build_deck2(F: dict, out: Path, assets: Path, demo: dict | None, X: dict) -> dict:
    formulas = d1.render_formulas(assets)
    deck = Deck()
    d1.s_title(deck, F)
    s_problem(deck, F)
    s_fingerprint_concept(deck, F)
    s_data(deck, F, X)
    s_pipeline(deck, F)
    s_ns_intuition(deck, F, X, formulas)
    d1.s_replication(deck, F)
    d1.s_sample_size(deck, F)
    s_recipes(deck, F, X)
    s_models(deck, F)
    s_cuisines(deck, F)
    s_drivers(deck, F)
    s_stereotypes(deck, F)
    s_networks(deck, F)
    s_steer_method(deck, F)
    s_steer_example(deck, F)
    s_library(deck, F)
    s_app(deck, F, demo)
    s_limitations(deck, F)
    s_conclusion(deck, F)
    # backups
    _as_backup(deck, d1.b_mechanism, F)
    _as_backup(deck, d1.b_tests, F)
    _as_backup(deck, d1.s_pairing, F)
    _as_backup(deck, d1.s_persona, F)
    _as_backup(deck, d1.s_fingerprint, F)
    _as_backup(deck, d1.s_bon, F)
    _as_backup(deck, d1.b_ablation, F)
    _as_backup(deck, d1.b_scorecard, F)
    _as_backup(deck, d1.b_effects, F)
    if F["figures"].get("F5"):
        _as_backup(deck, d1.b_signs, F)
    for key, title in (
        ("F10", "How long are LLM recipes compared with real ones?"),
        ("F9", "How diverse are LLM cuisines?"),
        ("F7", "Which ingredients drive pairing, real vs LLM?"),
        ("F6", "How do category fingerprints compare?"),
        ("F11b", "Which pairings do LLMs miss?"),
        ("F15c", "How do the steering metrics change over revision rounds?"),
    ):
        p = fig(F, key)
        if p:
            s = deck.new("backup", key)
            header(deck, s, title, None)
            deck.picture(s, p, 0.55, 1.4, 12.25, 5.5, align="center")
            deck.notes(s, f"{key}: generated by analysis/figures; shown in full here for questions.")
    _as_backup(deck, d1.b_demo, F, demo)
    _as_backup(deck, d1.b_credits, F)
    footer(deck, F["generated_date"])
    deck.save(out)
    n_main = sum(1 for m in deck.meta if m["kind"] == "main")
    return {"slides": len(deck.meta), "main": n_main, "backup": len(deck.meta) - n_main,
            "labels": [m["label"] for m in deck.meta],
            "outline": [{"kind": m["kind"], "label": m["label"], "title": m.get("title", ""),
                         "answer": m.get("answer") or "", "notes": m.get("notes", "")} for m in deck.meta]}  # fmt: skip


__all__ = ["build_deck2", "pv", "sf"]
