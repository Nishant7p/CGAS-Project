"""The talk deck: 14 story slides (one idea each, large readable charts) plus backup slides for questions.
Speaker notes carry what to say and the [DEMO]/[SHOW] cues; SPEAKING_SCRIPT.md is generated from them.

Every number is read from results/summary.json, results/steering_example.json and the audit tables; charts are the
repository's own: slide versions from analysis/figures/slide_figures.py (large text) and the report figures for backups.

Usage (repo root): .venv/bin/python presentation/story_deck.py
Output: presentation/story/Group44_BIO544_presentation.pptx, .pdf and SPEAKING_SCRIPT.md
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import talk_script as TS  # noqa: E402

GITHUB = TS.GITHUB

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "presentation" / "story"
IMG = OUT / "_img"
FIG = REPO / "figures"
SLIDE_FIG = FIG / "slides"
SHOTS = REPO / "app" / "screenshots"

NAVY, DARK, TEAL, TEAL_LIGHT, ORANGE = "1B2631", "13202C", "0E7C86", "7FD1D8", "D55E00"
GREY, BG, CARD, WHITE = "5D6D7E", "FAFAF7", "F0F3F4", "FFFFFF"
SERIF, SANS = "Georgia", "Calibri"
W, H = 13.333, 7.5

S = json.loads((REPO / "results" / "summary.json").read_text())
EX = json.loads((REPO / "results" / "steering_example.json").read_text())
ALIAS = json.loads((REPO / "data/interim/aliasing/metrics.json").read_text())["test"]["accuracy_lines"]
MODELS = {"openai-gpt-6-luna": "GPT-6-Luna", "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
          "qwen3.5-9b": "Qwen3.5-9B", "llama-3.1-8b": "Llama-3.1-8B"}  # fmt: skip
D, AUD = S["dataset"], S["llm_audit"]
REPL = S["replication"]["by_null"]["random"]
KOREA_TOP = S["replication"]["korea_top3"][0]
GEN = S["equal_n"]["n160_reference_medians"]
GEN_SIGN = float(pd.read_parquet(REPO / "data/interim/audit/real/subsamples_n160.parquet")["sign_match"].mean())
FP = {m: AUD[m]["median_fingerprint_spearman"] for m in AUD}
NET = S["networks"]["npmi_top100"]
NET_LLM = {m: NET[m]["median"] for m in NET if m != "real@160"}
AUC = S["steering"]["scorer"]["auc_test"]
REV = {(r["model"], r["round"]): r for r in S["steering"]["revise_main"]}
BON = {(r["model"], r["N"]): r for r in S["steering"]["bon_main"]}
CTX = EX["context"]
N_REAL = 45_772
with contextlib.suppress(FileNotFoundError):
    N_REAL = len(pd.read_csv(REPO / "data/raw/culinarydb/extracted/01_Recipe_Details.csv"))
LUNA, QWEN, LLAMA, GEM = "openai-gpt-6-luna", "qwen3.5-9b", "llama-3.1-8b", "gemini-3.5-flash-lite"


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def span(vals) -> str:
    lo, hi = f"{min(vals):.2f}", f"{max(vals):.2f}"
    return lo if lo == hi else f"{lo}–{hi}"


def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


def trimmed(name: str) -> Path:
    """A report figure without its printed title (first text block at the top), for use under a slide title."""
    IMG.mkdir(parents=True, exist_ok=True)
    im = Image.open(FIG / f"{name}.png").convert("RGB")
    a = np.asarray(im.convert("L"))
    ink = (a < 150).any(axis=1)
    h = len(ink)
    r0 = int(np.argmax(ink))
    r1, gap = r0, max(4, h // 200)
    while r1 < h and ink[r1 : r1 + gap].any():
        r1 += 1
    top = r1 + 2 if r1 < 0.16 * h else 0
    dst = IMG / f"{name}.png"
    im.crop((0, top, im.width, h)).save(dst)
    return dst


class Deck:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(W), Inches(H)
        self.n = 0

    # ---- primitives
    @staticmethod
    def fill(shape, color: str, line: str | None = None, width: float = 1.25) -> None:
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(color)
        if line:
            shape.line.color.rgb = rgb(line)
            shape.line.width = Pt(width)
        else:
            shape.line.fill.background()
        shape.shadow.inherit = False

    @staticmethod
    def text(s, txt, x, y, w, h, size, color=NAVY, bold=False, align=PP_ALIGN.LEFT, font=SANS,
             anchor=MSO_ANCHOR.TOP, spacing=1.0):  # fmt: skip
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, m, Emu(0))
        for k, line in enumerate(str(txt).split("\n")):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            p.alignment, p.line_spacing = align, spacing
            r = p.add_run()
            r.text = line
            r.font.size, r.font.bold, r.font.name = Pt(size), bold, font
            r.font.color.rgb = rgb(color)
        return tb

    def rich(self, s, parts: list[list[tuple[str, bool, str]]], x, y, w, h, size, gap=8, font=SANS):
        """Paragraphs of (text, bold, colour) runs."""
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, m, Emu(0))
        for k, runs in enumerate(parts):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            p.space_after = Pt(gap)
            for t, b, c in runs:
                r = p.add_run()
                r.text = t
                r.font.size, r.font.bold, r.font.name = Pt(size), b, font
                r.font.color.rgb = rgb(c)
        return tb

    def box(self, s, x, y, w, h, color=CARD, line=None, radius=0.06, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
        b = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
        if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
            b.adjustments[0] = radius
        self.fill(b, color, line)
        return b

    def picture(self, s, path: Path, x, y, w, h, align="center") -> tuple[float, float, float, float]:
        with Image.open(path) as im:
            iw, ih = im.size
        k = min(w / iw, h / ih)
        pw, ph = iw * k, ih * k
        px = x + (w - pw) / 2 if align == "center" else x
        s.shapes.add_picture(str(path), Inches(px), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))
        return px, y + (h - ph) / 2, pw, ph

    def kpi(self, s, x, y, w, value, label, color=ORANGE, vsize=40, lsize=14):
        self.text(s, value, x, y, w, 0.8, vsize, color, bold=True, font=SERIF)
        self.text(s, label, x, y + vsize / 72 * 1.3, w, 0.9, lsize, GREY)

    def number_badge(self, s, x, y, n, color=TEAL, d=0.55):
        c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
        self.fill(c, color)
        tf = c.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Emu(0)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = str(n)
        r.font.size, r.font.bold, r.font.name = Pt(18), True, SANS
        r.font.color.rgb = rgb(WHITE)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE

    def arrow(self, s, x1, y1, x2, y2, color=GREY):
        ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        ln.line.color.rgb = rgb(color)
        ln.line.width = Pt(2.25)
        ln.line._get_or_add_ln().append(_arrowhead())

    def tag(self, s, txt, x, y, w=2.0, color=ORANGE):
        t = self.box(s, x, y, w, 0.44, color, radius=0.5)
        tf = t.text_frame
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = tf.paragraphs[0].add_run()
        r.text = txt
        r.font.size, r.font.bold, r.font.name = Pt(13), True, SANS
        r.font.color.rgb = rgb(WHITE)

    # ---- slide types
    def background(self, s, color: str) -> None:
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = rgb(color)

    def content(self, title: str, answer: str, notes: str, section: str) -> object:
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.n += 1
        self.background(s, BG)
        self.box(s, 0, 0, 0.16, H, TEAL, shape=MSO_SHAPE.RECTANGLE)
        self.text(s, section.upper(), 0.6, 0.32, 8, 0.3, 11.5, TEAL, bold=True)
        self.text(s, title, 0.6, 0.6, 12.2, 0.75, 30, NAVY, bold=True, font=SERIF)
        if answer:
            self.text(s, answer, 0.6, 1.36, 12.2, 0.6, 18.5, TEAL)
        self.text(s, "Group 44 · BIO544 Computational Gastronomy", 0.6, 7.08, 6, 0.25, 9.5, GREY)
        self.text(s, str(self.n), 12.35, 7.08, 0.4, 0.25, 9.5, GREY, align=PP_ALIGN.RIGHT)
        s.notes_slide.notes_text_frame.text = notes
        return s

    def dark(self, notes: str) -> object:
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.n += 1
        self.background(s, DARK)
        self.box(s, 0, 0, 0.16, H, TEAL, shape=MSO_SHAPE.RECTANGLE)
        s.notes_slide.notes_text_frame.text = notes
        return s


def _arrowhead():
    from lxml import etree

    return etree.fromstring('<a:tailEnd xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" type="triangle" '
                            'w="med" len="med"/>')  # fmt: skip


# ------------------------------------------------------------------------------------------------------------ deck
def build() -> Path:
    d = Deck()

    # 1 title -----------------------------------------------------------------------------------------------------
    s = d.dark(
        "Good morning. We are Group 44. Our question is simple: when an AI writes a 'traditional' recipe, does it "
        "cook like the cuisine it claims? We answer it with the flavor-network methods of this course, and built five "
        "deliverables on the way: a dataset, an audit, a steering method, a Python library and a web app."
    )
    d.text(s, "BIO544 COMPUTATIONAL GASTRONOMY  ·  GROUP 44", 0.9, 0.85, 11, 0.4, 13, TEAL_LIGHT, bold=True)
    d.text(s, "Do AI Chefs Respect\nCulinary Fingerprints?", 0.9, 1.45, 11.5, 2.3, 50, WHITE, bold=True, font=SERIF,
           spacing=0.95)  # fmt: skip
    d.text(s, "Auditing and steering LLM-generated recipes across 22 world cuisines", 0.9, 3.85, 11.5, 0.6, 22,
           TEAL_LIGHT)  # fmt: skip
    d.text(s, "Roll nos. 2023355 and 2023477  ·  IIIT-Delhi  ·  Monsoon 2026  ·  Instructor: Prof. Ganesh Bagler",
           0.9, 4.6, 11.5, 0.5, 15, "C9D3DA")  # fmt: skip
    link(d.text(s, GITHUB.removeprefix("https://"), 0.9, 5.08, 11.5, 0.4, 14, TEAL_LIGHT))
    chips = ["D1  LLMCuisine-22", "D2  Fingerprint audit", "D3  Steering", "D4  FlavorAudit", "D5  FlavorPrint"]
    for k, c in enumerate(chips):
        b = d.box(s, 0.9 + k * 2.36, 5.75, 2.2, 0.62, DARK, line=TEAL_LIGHT, radius=0.25)
        tf = b.text_frame
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = tf.paragraphs[0].add_run()
        r.text = c
        r.font.size, r.font.bold, r.font.name = Pt(14), True, SANS
        r.font.color.rgb = rgb(WHITE)

    # 2 the question + fingerprint -------------------------------------------------------------------------------
    s = d.content("Do AI chefs cook like the cuisine they claim?",
                  "A cuisine has a measurable fingerprint, so we can check AI recipes against it",
                  "Analogy: a cuisine is like handwriting. We don't check whether the AI writes the right dish names; we "
                  "check whether its handwriting matches: which ingredients it combines and how their flavors relate. "
                  "Prof. Bagler's lab gave us the measuring tools (CulinaryDB, FlavorDB, the food-pairing framework); "
                  "nobody had applied them to AI-written recipes.", "The question")  # fmt: skip
    d.box(s, 0.6, 2.15, 5.0, 4.55, CARD)
    d.text(s, "Why it matters", 0.95, 2.4, 4.4, 0.5, 20, TEAL, bold=True, font=SERIF)
    d.rich(s, [[("People ask chatbots for ", False, NAVY), ("“authentic”", True, NAVY), (" recipes every day.", False, NAVY)],
               [("An AI learns from ", False, NAVY), ("text, not taste.", True, NAVY)],
               [("So we measure AI recipes with the ", False, NAVY), ("same flavor science", True, ORANGE),
                (" used for real cuisines, and then use it to make them more authentic.", False, NAVY)]],
           0.95, 3.05, 4.4, 3.5, 18, 14)  # fmt: skip
    layers = [("Flavor pairing", "Do a recipe's ingredients share flavor molecules more (Italy) or less (Korea) "
               "than chance?"),
              ("Key ingredients", f"Which ingredients drive the flavor? Korea: {KOREA_TOP['name'].lower()}."),
              ("Category pattern", "How each ingredient category shapes flavor: the culinary fingerprint "
               "(Jain, Rakhi & Bagler 2015).")]  # fmt: skip
    d.text(s, "Three layers of a culinary fingerprint", 6.1, 2.2, 6.6, 0.5, 20, NAVY, bold=True, font=SERIF)
    for k, (h_, b_) in enumerate(layers):
        y = 2.95 + k * 1.25
        d.number_badge(s, 6.1, y, k + 1, [TEAL, ORANGE, NAVY][k])
        d.text(s, h_, 6.85, y - 0.02, 5.9, 0.4, 18, NAVY, bold=True)
        d.text(s, b_, 6.85, y + 0.38, 5.9, 0.8, 15, GREY)

    # 3 what we built: the five deliverables ---------------------------------------------------------------------
    s = d.content("What we built: five deliverables", "As part of the project we built five things, all on GitHub",
                  "", "Overview")  # fmt: skip
    for k, (code, name, blurb) in enumerate(TS.D_BLURBS):
        y = 2.1 + k * 0.97
        url = TS.D_LINKS[code]
        link(d.box(s, 0.6, y, 12.15, 0.85, CARD if k % 2 == 0 else WHITE), url)
        d.number_badge(s, 0.8, y + 0.15, code, [TEAL, ORANGE, NAVY, "7D3C98", "B9770E"][k], 0.56)
        link(d.text(s, name, 1.6, y + 0.24, 3.3, 0.45, 18, NAVY, bold=True), url)
        link(d.text(s, blurb, 5.0, y + 0.08, 6.25, 0.7, 14, GREY, anchor=MSO_ANCHOR.MIDDLE), url)
        pill = d.box(s, 11.45, y + 0.21, 1.1, 0.43, TEAL, radius=0.5)
        pill.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        r_ = pill.text_frame.paragraphs[0].add_run()
        r_.text = "Open ›"
        r_.font.size, r_.font.bold, r_.font.name = Pt(13), True, SANS
        r_.font.color.rgb = rgb(WHITE)
        link(pill, url)

    # 3 data --------------------------------------------------------------------------------------------------------
    s = d.content("What did we compare?", f"{N_REAL:,} real recipes against {D['total_recipes']:,} recipes written by "
                  f"{D['n_models']} AI models", "This is deliverable D1, LLMCuisine-22. Every model got the same frozen "
                  "prompts for every region (plain and home-cook persona), asking for five traditional recipes per call; "
                  "every call is logged. Each ingredient line is mapped to CulinaryDB ingredients with CulinaryDB's own "
                  f"protocol ({ALIAS:.1%} accurate on held-out lines), so AI and real recipes are measured identically. "
                  "[SHOW, optional] 05_Dataset_LLMCuisine-22/LLMCuisine-22/01_Recipe_Details.csv.",
                  "Data · D1")  # fmt: skip
    tiles = [(f"{N_REAL:,}", "real recipes (CulinaryDB)"), ("22", "world cuisines"),
             (str(D["n_models"]), "AI models, 2 open + 2 proprietary"), (f"{D['total_recipes']:,}", "AI recipes (our dataset)")]  # fmt: skip
    for k, (v, lab) in enumerate(tiles):
        d.box(s, 0.6 + k * 3.08, 2.15, 2.9, 1.45, CARD)
        d.kpi(s, 0.85 + k * 3.08, 2.25, 2.6, v, lab, [NAVY, NAVY, TEAL, ORANGE][k], 34, 13.5)
    rows = [(MODELS[m], D["models"][m]["family"].split(" (")[1].rstrip(")"), f"{D['models'][m]['recipes']:,}",
             f"{D['models'][m]['regions']}") for m in (LUNA, GEM, QWEN, LLAMA)]  # fmt: skip
    tbl = s.shapes.add_table(len(rows) + 1, 4, Inches(0.6), Inches(3.85), Inches(7.6), Inches(2.85)).table
    for c, (t, wdt) in enumerate(
        zip(("AI model", "Developer", "Recipes", "Cuisines"), (2.9, 1.9, 1.5, 1.3), strict=True)
    ):
        tbl.columns[c].width = Inches(wdt)
        tbl.cell(0, c).text = t
    for r, row in enumerate(rows, start=1):
        for c, v in enumerate(row):
            tbl.cell(r, c).text = v
    for r in range(len(rows) + 1):
        for c in range(4):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(NAVY if r == 0 else (WHITE if r % 2 else CARD))
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.RIGHT if c >= 2 else PP_ALIGN.LEFT
                for run in p.runs:
                    run.font.size, run.font.name, run.font.bold = Pt(16), SANS, r == 0
                    run.font.color.rgb = rgb(WHITE if r == 0 else NAVY)
    d.rich(s, [[("Same prompts ", True, NAVY), ("for every cuisine: plain and home-cook persona", False, NAVY)],
               [("Every call logged: ", True, NAVY), ("prompt, model version, settings, raw output", False, NAVY)],
               [("Ingredients mapped ", True, NAVY), (f"to CulinaryDB with its own protocol ({ALIAS:.0%} accurate)", False, NAVY)]],
           8.6, 3.95, 4.2, 2.8, 16, 12)  # fmt: skip

    # 4 pipeline ---------------------------------------------------------------------------------------------------
    s = d.content("How did we do it?", "One pipeline from real cuisines to audited and improved AI recipes",
                  "Left to right: real cuisines; first we reproduce the published result; then generate AI recipes, map "
                  "their ingredients, audit them with exactly the same functions as the real cuisines, and finally "
                  "steer the models with a fidelity scorer. The bottom row shows where each deliverable comes from.",
                  "Method")  # fmt: skip
    d.picture(s, trimmed("F1_pipeline"), 0.5, 2.05, 12.4, 4.85)

    # 5 food pairing ----------------------------------------------------------------------------------------------
    s = d.content("How do we measure food pairing?",
                  "Compare real recipes with a ‘blindfolded chef’ who picks from the same pantry",
                  "Each recipe gets a score Ns: on average, how many flavor compounds its ingredient pairs share. Grey: "
                  "recipes built at random from the same ingredients and recipe sizes, our blindfolded chef. Black: real "
                  "recipes. Italian recipes share more compounds than chance (uniform, positive); Korean recipes share "
                  "fewer (contrasting, negative). Z standardises this gap; the other three null models also keep "
                  "ingredient popularity or categories (Singh & Bagler 2018).", "Method")  # fmt: skip
    d.picture(s, SLIDE_FIG / "S_pairing.png", 0.6, 2.05, 12.1, 4.3)
    d.rich(s, [[("Ns ", True, NAVY), ("= flavor compounds shared per ingredient pair.   ", False, GREY),
                ("ΔNs and Z ", True, NAVY), ("= how far real recipes sit from the blindfolded chef.", False, GREY)]],
           0.6, 6.5, 12.1, 0.5, 15.5)  # fmt: skip

    # 6 replication ----------------------------------------------------------------------------------------------
    s = d.content("Step 1: can we reproduce the published result?",
                  f"Yes: all {REPL['sign_agreement']} cuisines match Singh & Bagler (2018), rank correlation ρ = "
                  f"{REPL['spearman']:.2f}", "Before touching any AI recipe we checked our code against the published "
                  "Figure 4. Open circles are the paper's values, filled dots are ours: every cuisine has the same sign, "
                  "the ranking matches at 0.98, and the top flavor driver for Korea is soybean sauce, as published. So "
                  "everything after this uses a verified pipeline.", "Replication")  # fmt: skip
    d.picture(s, SLIDE_FIG / "S_replication.png", 0.5, 2.0, 7.4, 5.0)
    d.kpi(s, 8.5, 2.3, 4.3, f"{REPL['sign_agreement']}/{REPL['n']}", "cuisines with the published sign")
    d.kpi(s, 8.5, 3.75, 4.3, f"{REPL['spearman']:.2f}", "rank correlation with the paper", TEAL)
    d.kpi(s, 8.5, 5.2, 4.3, KOREA_TOP["name"], f"Korea's #1 flavor driver ({KOREA_TOP['chi_percent']:.0f}% of the "
          "effect), as published", NAVY, 30)  # fmt: skip

    # 7 results at a glance -------------------------------------------------------------------------------------
    s = d.content("Results at a glance: AI vs real cuisines",
                  "Each AI cuisine (160 recipes) is compared with genuine samples of 160 real recipes",
                  f"This is the whole audit in one picture. Grey is what genuine 160-recipe samples of the real cuisine "
                  f"achieve, our fair yardstick. Pairing direction: Llama matches the real cuisine in {pct(AUD[LLAMA]['sign_agreement'])} "
                  f"of cuisines, as often as genuine samples ({pct(GEN_SIGN)}). Fingerprint similarity and signature "
                  "pairs: the AI reaches about a third to a half of the real level. Variety: AI cuisines use about half "
                  "as many different ingredients. The next two slides zoom into the fingerprint and the pairs.",
                  "Audit · D2")  # fmt: skip
    d.picture(s, SLIDE_FIG / "S_glance.png", 0.5, 2.05, 12.4, 4.85)

    # 8 fingerprint -------------------------------------------------------------------------------------------------
    s = d.content("Do AI cuisines keep the culinary fingerprint?",
                  "Partly: they agree on some categories and differ on others", "Each row is an ingredient category in "
                  "Korean food; orange means the category increases flavor sharing, purple that it decreases it. The real "
                  "column and the AI columns agree on several rows (fungus, legumes, plants for Qwen) and disagree on "
                  "others (meat, additives). Across all 22 cuisines, AI fingerprints correlate with the real ones at "
                  "about 0.3, against 0.75 for genuine samples.", "Audit · D2")  # fmt: skip
    d.picture(s, SLIDE_FIG / "S_fingerprint.png", 0.5, 2.0, 7.6, 5.0)
    d.text(s, "Korea, category by category", 8.6, 2.2, 4.2, 0.4, 16, GREY, bold=True)
    d.kpi(s, 8.6, 2.75, 4.2, span(FP.values()), "AI fingerprint similarity to the real one, all cuisines (ρ)")
    d.kpi(s, 8.6, 4.35, 4.2, f"{GEN['fingerprint_spearman']:.2f}", "genuine 160-recipe samples", GREY)
    d.text(s, "AI gets the broad strokes, not the fine detail.", 8.6, 5.95, 4.2, 0.8, 17, NAVY, bold=True)

    # 9 signature pairs -------------------------------------------------------------------------------------------
    s = d.content("Do AI chefs combine the right ingredients?",
                  "They keep the classic pairs and miss many regional signatures",
                  "We built each cuisine's ingredient co-occurrence network and took its most distinctive pairs. For "
                  "Korea, soybean sauce with sesame and egg with flour appear in every AI cuisine, but shrimp with fish "
                  "or carrot with potato do not. Across all cuisines, AI shares about 0.12 of the real top-100 pairs, "
                  "genuine samples 0.30. And the key ingredient is right: GPT-6-Luna and Qwen make soybean sauce Korea's "
                  "number-one flavor driver, as in real recipes.", "Audit · D2")  # fmt: skip
    d.picture(s, SLIDE_FIG / "S_pairs.png", 0.5, 2.0, 7.6, 5.0)
    d.text(s, "Korea's 10 most distinctive pairs", 8.6, 2.2, 4.2, 0.4, 16, GREY, bold=True)
    d.kpi(s, 8.6, 2.75, 4.2, span(NET_LLM.values()), "share of the real top-100 pairs, AI (all cuisines)")
    d.kpi(s, 8.6, 4.35, 4.2, f"{NET['real@160']['median']:.2f}", "genuine 160-recipe samples", GREY)
    d.text(s, f"Key ingredient right: {KOREA_TOP['name'].lower()} is #1 for real Korean, GPT-6-Luna and Qwen recipes.",
           8.6, 5.95, 4.2, 0.9, 16, NAVY, bold=True)  # fmt: skip

    # 10 scorer -------------------------------------------------------------------------------------------------------
    s = d.content("From audit to fix: a fidelity scorer",
                  "One score for “how much does this recipe look like the cuisine it claims?”",
                  "The scorer combines three signals with equal weights: a cuisine classifier, ingredient co-occurrence "
                  "statistics and flavor-sharing consistency. It also explains itself: it names atypical ingredients and "
                  "missing signature ingredients. On held-out data it separates real recipes from other cuisines, from "
                  "copy-mutate recipes and from random recipes with AUC 0.87 to 0.94.", "Steering · D3")  # fmt: skip
    comps = [("Cuisine classifier", "which of 22 cuisines\nit looks like", TEAL),
             ("Ingredient co-occurrence", "do its pairs appear together\nin the real cuisine?", ORANGE),
             ("Flavor-sharing consistency", "is its Ns typical\nfor that cuisine?", NAVY)]  # fmt: skip
    for k, (h_, b_, c) in enumerate(comps):
        y = 2.15 + k * 1.3
        d.box(s, 0.6, y, 4.6, 1.1, CARD)
        d.box(s, 0.6, y, 0.12, 1.1, c, shape=MSO_SHAPE.RECTANGLE)
        d.text(s, h_, 0.95, y + 0.13, 4.1, 0.4, 17, c, bold=True)
        d.text(s, b_, 0.95, y + 0.5, 4.1, 0.6, 13.5, GREY)
        d.arrow(s, 5.3, y + 0.55, 6.05, 3.6)
    core = d.box(s, 6.15, 2.95, 2.7, 1.3, NAVY)
    d.text(s, "Fidelity\nscore", 6.15, 3.05, 2.7, 1.1, 22, WHITE, bold=True, align=PP_ALIGN.CENTER, font=SERIF,
           anchor=MSO_ANCHOR.MIDDLE)  # fmt: skip
    _ = core
    uses = [("Best-of-N", "generate several recipes,\nkeep the best one"),
            ("Critique-and-revise", "send the scorer's feedback\nback to the model")]  # fmt: skip
    for k, (h_, b_) in enumerate(uses):
        y = 2.45 + k * 1.45
        d.arrow(s, 8.95, 3.6, 9.55, y + 0.55)
        d.box(s, 9.6, y, 3.2, 1.15, CARD)
        d.text(s, h_, 9.85, y + 0.13, 2.8, 0.4, 17, ORANGE, bold=True)
        d.text(s, b_, 9.85, y + 0.5, 2.8, 0.6, 13.5, GREY)
    d.text(s, "How well the score tells real recipes from impostors (AUC, 1 = perfect):", 0.6, 6.12, 12, 0.35, 14, GREY)
    for k, (key, lab) in enumerate((("other_cuisines", "recipes of other cuisines"), ("copy_mutate", "copy-mutate recipes"),
                                    ("random_null", "random recipes"))):  # fmt: skip
        d.rich(s, [[(f"{AUC[key]['composite']:.2f} ", True, ORANGE), (f"vs {lab}", False, NAVY)]],
               0.6 + k * 4.1, 6.5, 4.0, 0.45, 18)  # fmt: skip

    # 11 steering results ---------------------------------------------------------------------------------------
    r0, r2 = REV[(LUNA, 0)], REV[(LUNA, max(r for (m, r) in REV if m == LUNA))]
    b, a = EX["before"], EX["after"]
    s = d.content("Does fingerprint feedback make AI recipes more authentic?",
                  f"Yes: recipes recognised as their claimed cuisine rise from {pct(r0['indep_classifier_acc'])} to "
                  f"{pct(r2['indep_classifier_acc'])} after two rounds of feedback", "Left: we show the model the "
                  "scorer's feedback and ask it to improve the recipe while keeping the dish. Recognisability, judged by "
                  "an independent classifier that the steering never saw, rises every round. Right: simply generating 8 "
                  "candidates and keeping the best one helps every model. Example, picked by a fixed rule (the median "
                  f"successful revision): {b['title']} moved from the {b['fidelity_percentile']:.0f}th to the "
                  f"{a['fidelity_percentile']:.0f}th fidelity percentile after adding "
                  f"{', '.join(EX['entities_added']).lower()}. [DEMO cue] The app's Suggestions tab shows this feedback.",
                  "Steering · D3")  # fmt: skip
    d.picture(s, SLIDE_FIG / "S_steering.png", 0.5, 2.0, 12.4, 4.35)
    added = ", ".join(EX["entities_added"]).lower()
    d.rich(s, [[("Example: ", True, ORANGE), (f"{b['title']}: ", True, NAVY),
                (f"fidelity percentile {b['fidelity_percentile']:.0f} → {a['fidelity_percentile']:.0f} after adding "
                 f"{added}.   Across all {CTX['revisions_scored']} revisions: median {CTX['median_percentile_before']:.0f} → "
                 f"{CTX['median_percentile_after']:.0f}.", False, NAVY)]], 0.6, 6.5, 12.2, 0.5, 15)  # fmt: skip

    # 12 app -------------------------------------------------------------------------------------------------------
    s = d.content("FlavorPrint: audit any recipe in the browser", "Paste a recipe, pick its cuisine, see its fingerprint",
                  "[LIVE DEMO, about 2 minutes] Switch to the browser tab with FlavorPrint (launch: double-click "
                  "04_Demo/run_demo.command). 1) Kimchi jjigae, cuisine Korea, Audit recipe: point at 'reads as Korea', "
                  "the fidelity percentile and the ingredient mapping. 2) Flavor network tab. 3) Example 'Spaghetti, "
                  "claimed Korean': it reads as Italy and the fidelity drops to 2. 4) Suggestions tab: the same feedback "
                  "our steering gives the AI. 5) Chana masala: the stereotype flag. 6) 22-cuisine dashboard. Then back to "
                  "the slides. If the app does not load, talk over this screenshot.", "Web app · D5")  # fmt: skip
    d.tag(s, "▶  LIVE DEMO", 10.75, 0.3, 2.0)
    link(s.shapes[-1], TS.APP_LOCAL)
    shot = next((p for p in (SHOTS / "00_hero_audit.png", SHOTS / "01_audit_overview.png")
                 if p.exists()), None)  # fmt: skip
    if shot and shot.name.startswith("00_hero"):  # drop the half-visible caption line at the bottom edge
        IMG.mkdir(parents=True, exist_ok=True)
        with Image.open(shot) as im:
            im = im.convert("RGB").crop((0, int(im.height * 0.02), im.width, int(im.height * 0.962)))
            a = np.asarray(im.convert("L")).astype(int)
            cols = np.where((np.abs(a - int(np.median(a[:, :5]))) > 12).any(axis=0))[0]  # columns with page content
            left, right = max(0, cols.min() - 14), min(im.width, cols.max() + 14)
            im.crop((left, 0, right, im.height)).save(IMG / "hero.png")
        shot = IMG / "hero.png"
    if shot:
        x, y, w, h = d.picture(s, shot, 0.6, 2.05, 6.75, 4.85, align="left")
        link(s.shapes[-1], TS.APP_LOCAL)  # click the screenshot to open the running app
        frame = d.box(s, x - 0.03, y - 0.03, w + 0.06, h + 0.06, BG, line="C9D3DA", radius=0.01)
        link(frame, TS.APP_LOCAL)
        frame.fill.background()
    feats = [("Ingredient mapping", "each line linked to CulinaryDB"),
             ("Pairing & flavor network", "how the ingredients share flavor"),
             ("Fidelity & stereotypes", "does it read as its cuisine?"),
             ("Suggestions", "what the fingerprint would change"),
             ("22-cuisine dashboard", "every model on every cuisine")]  # fmt: skip
    for k, (h_, b_) in enumerate(feats):
        y = 2.15 + k * 0.97
        d.number_badge(s, 7.7, y, k + 1, TEAL, 0.5)
        d.text(s, h_, 8.4, y - 0.03, 4.4, 0.4, 18, NAVY, bold=True)
        d.text(s, b_, 8.4, y + 0.38, 4.4, 0.4, 14.5, GREY)

    # 14 conclusion ---------------------------------------------------------------------------------------------------
    s = d.dark("One-line summary: AI chefs know the skeleton of a cuisine but not its full fingerprint, and the "
               "fingerprint itself can guide them closer. Thank you, we are happy to take questions. Backup slides "
               "follow for detailed questions.")  # fmt: skip
    d.text(s, "CONCLUSION", 0.9, 0.7, 6, 0.4, 13, TEAL_LIGHT, bold=True)
    d.text(s, "So, do AI chefs respect culinary fingerprints?", 0.9, 1.1, 11.6, 0.8, 34, WHITE, bold=True, font=SERIF)
    d.text(s, "Partly, and the fingerprint itself can guide them closer.", 0.9, 1.95, 11.6, 0.6, 22, TEAL_LIGHT)
    cols3 = [("They get right", "recipe size, signature ingredients, often the pairing direction and the key flavor "
              f"driver ({KOREA_TOP['name'].lower()} for Korea)", TEAL_LIGHT),
             ("They miss", "part of the category fingerprint, many regional ingredient pairs, and variety", "F5B041"),
             ("We built", "a verified audit, a 11,656-recipe dataset, a fidelity scorer that steers models, a library "
              "and an app", WHITE)]  # fmt: skip
    for k, (h_, b_, c) in enumerate(cols3):
        x = 0.9 + k * 4.0
        d.box(s, x, 3.0, 3.7, 2.35, "1E2F3D", radius=0.05)
        d.text(s, h_, x + 0.3, 3.2, 3.2, 0.45, 20, c, bold=True, font=SERIF)
        d.text(s, b_, x + 0.3, 3.75, 3.15, 1.5, 15, "D5DBDF")
    d.text(s, "Next: a scale study (Qwen3.5-4B vs 9B), more model families, and human taste tests.", 0.9, 5.65, 11.6,
           0.5, 15, "AEB6BF")  # fmt: skip
    band = d.box(s, 4.15, 6.3, 5.0, 0.7, TEAL, radius=0.5)
    tf = band.text_frame
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    r = tf.paragraphs[0].add_run()
    r.text = "Thank you  ·  Questions?"
    r.font.size, r.font.bold, r.font.name = Pt(22), True, SERIF
    r.font.color.rgb = rgb(WHITE)
    link(d.text(s, f"Code, data and app: {GITHUB.removeprefix('https://')}", 0.9, 7.05, 11.6, 0.35, 12.5, TEAL_LIGHT,
                align=PP_ALIGN.CENTER))  # fmt: skip

    # links: everything we can open ------------------------------------------------------------------------------
    s = d.content("Links: everything we can show", f"Click any row  ·  {GITHUB.removeprefix('https://')}",
                  "Jump here to open anything the examiners ask about. GitHub links need the repository pushed; the app "
                  "link needs the app running on this laptop.", "Links")  # fmt: skip
    half = (len(TS.LINKS) + 1) // 2
    for k, (label_, url) in enumerate(TS.LINKS):
        col, row_i = divmod(k, half)
        x, y = 0.6 + col * 6.15, 1.98 + row_i * 0.47
        link(d.box(s, x, y, 5.95, 0.41, CARD if row_i % 2 == 0 else WHITE, radius=0.08), url)
        link(d.text(s, label_, x + 0.18, y + 0.08, 3.4, 0.28, 13, NAVY, bold=True), url)
        tail = ("localhost:8501" if url.startswith("http://localhost") else "github.com" if url == GITHUB
                else url.rstrip("/").split("/")[-1] + ("/" if "/tree/" in url else ""))  # fmt: skip
        link(d.text(s, tail + "  ›", x + 3.4, y + 0.1, 2.42, 0.26, 10, TEAL, align=PP_ALIGN.RIGHT), url)

    # backups ---------------------------------------------------------------------------------------------------
    s = d.content("Backup: FlavorAudit in code", "Everything above as a pip-installable package",
                  "All methods are packaged as FlavorAudit: typed, tested, documented, with a command-line tool. The data "
                  "are downloaded separately (CulinaryDB and FlavorDB are CC BY-NC-SA).", "Backup")  # fmt: skip
    code = ("pip install flavoraudit-0.1.0.dev0-py3-none-any.whl\n\nfrom flavoraudit.world import load_world\n"
            "from flavoraudit.stats import food_pairing\n\nworld = load_world()           # CulinaryDB + FlavorDB\n"
            "korea = world.cuisines['Korea']\nres = food_pairing(korea, world.S, 'random')\n"
            "print(res.z_sb, res.delta_ns)\n\n# command line\nflavoraudit audit my_recipes.csv --out report/")  # fmt: skip
    d.box(s, 0.6, 2.1, 7.6, 4.7, "1E2A36", radius=0.03)
    d.text(s, code, 0.95, 2.4, 7.0, 4.2, 15, "E8EEF2", font="Courier New")
    d.rich(s, [[("Food pairing: ", True, NAVY), ("Ns, Z against 4 null models, bootstrap CIs", False, GREY)],
               [("Fingerprints: ", True, NAVY), ("χ% drivers, category fingerprint, prevalence", False, GREY)],
               [("Networks and evolution: ", True, NAVY), ("co-occurrence, copy-mutate", False, GREY)],
               [("Fidelity scorer: ", True, NAVY), ("score and explain any recipe", False, GREY)],
               [("Quality: ", True, NAVY), ("tests, docs, 3 example notebooks", False, GREY)]],
           8.6, 2.3, 4.2, 4.4, 16, 12)  # fmt: skip
    backups = [
        ("Backup: where do models get the pairing direction right?", "Green: same direction as the real cuisine",
         "F5_sign_matrix", "Each square is a model and a cuisine; compared at equal size with genuine samples."),
        ("Backup: which ingredients drive flavor pairing?", "Korea: soybean sauce for real, GPT-6-Luna and Qwen",
         "F7_drivers", "χ% = change in the cuisine's flavor sharing when one ingredient is removed (Singh & Bagler)."),
        ("Backup: how varied are AI cuisines?", "They follow the popular ingredients but lose the long tail",
         "F9_diversity", "Diversity ratio = unique ingredients relative to genuine samples of the same size."),
        ("Backup: why compare at equal size?", "Small samples lose rare ingredients, so we compare 160 vs 160",
         "F2c_matched_size_signs", "Food pairing depends on sample size, so every AI cuisine is compared with "
         "genuine samples of the same size."),
        ("Backup: AI vs a copy-mutate model", "AI cuisines beat the classic evolution model on composition and "
         "fingerprint", "F13_copy_mutate", "Copy-mutate (Jain & Bagler; Tuwani et al.) is our baseline, not a "
         "contribution."),
        ("Backup: does a home-cook persona help?", "The persona prompt changes some measures, by model",
         "F14_persona_vs_plain", "Each point is one cuisine: persona minus plain prompt."),
        ("Backup: the full scorecard", "Every source on every measure (medians over cuisines)", "F12_scorecard",
         "Genuine samples set the ceiling; AI models sit between them and copy-mutate."),
        ("Backup: best-of-N in detail", "More candidates, more recognisable recipes", "F15_best_of_n",
         "Best-of-N reranks each model's own recipes; the judge is an independent classifier."),
        ("Backup: ingredient networks by cuisine", "Top-100 pair overlap with the real cuisine, every region",
         "F11_networks", "Grey bars: the range of genuine 160-recipe samples; markers: AI models."),
    ]  # fmt: skip
    for title, answer, fig, note in backups:
        s = d.content(title, answer, note, "Backup")
        d.picture(s, trimmed(fig), 0.5, 2.05, 12.4, 4.85)

    # speaker notes for the talk slides come from the shared script (who speaks, what to say, demo steps)
    for sd in TS.SLIDES:
        txt = f"[{sd['speaker']} · {sd['time']}]\n{sd['say']}"
        if sd["n"] == TS.DEMO_SLIDE:
            txt += "\n\nDEMO STEPS:\n" + "\n".join(f"{k}. {c}: {say}" for k, (c, _pt, say) in enumerate(TS.DEMO, 1))
        if sd.get("point"):
            txt += f"\n\nTip: {sd['point']}"
        if sd.get("handover"):
            txt += f"\n\nHand over: {sd['handover']}"
        d.prs.slides[sd["n"] - 1].notes_slide.notes_text_frame.text = txt
        if sd["title"] in TS.FOOTER_LINKS:
            label_, url = TS.FOOTER_LINKS[sd["title"]]
            sl = d.prs.slides[sd["n"] - 1]
            link(d.text(sl, label_, 7.9, 7.05, 4.3, 0.28, 11, TEAL, bold=True, align=PP_ALIGN.RIGHT), url)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "Group44_BIO544_presentation.pptx"
    d.prs.save(path)
    return path


def link(tb, url: str = "") -> None:
    """Make the whole text box clickable (default: the GitHub repository) while the text keeps its own colour."""
    tb.click_action.hyperlink.address = url or GITHUB


def speaking_script(path: Path) -> Path:
    """SPEAKING_SCRIPT.md from the deck itself: each slide's title, its one-line answer and its speaker notes."""
    prs = Presentation(str(path))
    lines = ["# Speaking script: Do AI Chefs Respect Culinary Fingerprints?", "",
             "From the deck's speaker notes. [DEMO] and [SHOW] are cues for you, not for the slide. Slides 1-14 are the "
             "talk (about 40-50 s each, plus a 2-minute demo on slide 12: about 12 minutes); slides 15 onward are backups "
             "for questions.", ""]  # fmt: skip
    for i, s in enumerate(prs.slides, 1):
        texts = []
        for sh in s.shapes:
            if sh.has_text_frame and sh.text_frame.text.strip():
                sizes = [r.font.size.pt for p in sh.text_frame.paragraphs for r in p.runs if r.font.size]
                texts.append((max(sizes) if sizes else 0, sh.top, sh.text_frame.text.strip()))
        title = max(texts, key=lambda t: (t[0], -t[1]))[2].replace("\n", " ") if texts else ""
        answer = next((t[2] for t in sorted(texts, key=lambda t: t[1]) if t[0] == 18.5), "")
        lines += [f"## Slide {i}: {title}", ""]
        if answer:
            lines += [f"*On the slide:* {answer}", ""]
        lines += [s.notes_slide.notes_text_frame.text.strip() or "(no notes)", ""]
    out = OUT / "SPEAKING_SCRIPT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    path = build()
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(OUT), str(path)],
                   check=True, capture_output=True)  # fmt: skip
    script = speaking_script(path)
    print(f"{path.relative_to(REPO)} and .pdf; {script.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
