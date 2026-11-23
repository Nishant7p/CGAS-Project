"""FlavorPrint (D5): audit any recipe against the culinary fingerprints of 22 world cuisines.

Run locally (repo root):  .venv/bin/streamlit run app/streamlit_app.py
Artifacts: app/artifacts/ (built by app/build_artifacts.py; CC BY-NC-SA 4.0, derived from CulinaryDB and FlavorDB).
Theme: app/.streamlit/config.toml (read next to this script, so it applies from any working directory).

presentation/demo_probe.py executes this file without Streamlit (it drops the trailing ``main()`` call), calls
EXAMPLES, clean_lines and audit, and parses the stereotype cut from the literal in stereotype_flag (its regex takes
the first "if share" comparison in this file). Keep them stable.
"""

from __future__ import annotations

import html
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:  # installed (pip install -e .) or on PYTHONPATH (the Hugging Face Space image)
    import flavoraudit  # noqa: F401
except ImportError:  # a plain clone, e.g. Streamlit Community Cloud: use the repository's src/ layout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flavoraudit.pairing import ns_recipe  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
EXAMPLES = {
    "Korea": (
        "Kimchi jjigae",
        "2 cups napa cabbage kimchi, chopped\n200 g pork belly, sliced\n1 block firm tofu, cubed\n"
        "1 tbsp gochugaru (Korean chili flakes)\n1 tbsp soy sauce\n2 cloves garlic, minced\n"
        "2 scallions, sliced\n1 tsp sesame oil\n2 cups water",
    ),
    "Italy": (
        "Spaghetti al pomodoro",
        "400 g spaghetti\n800 g canned San Marzano tomatoes\n3 tbsp extra-virgin olive oil\n2 cloves garlic, sliced\n"
        "1 handful fresh basil leaves\n40 g Parmesan cheese, grated\nSalt\nBlack pepper",
    ),
    "Indian Subcontinent": (
        "Chana masala",
        "2 cups chickpeas, cooked\n1 onion, finely chopped\n2 tomatoes, pureed\n1 tbsp ginger-garlic paste\n"
        "1 tsp cumin seeds\n1 tsp ground coriander\n1/2 tsp turmeric\n1 tsp garam masala\n2 tbsp ghee\n"
        "Fresh coriander leaves",
    ),
}
# Built-in demo cases: key -> (button label, EXAMPLES key, claimed cuisine, what the case shows).
PRESETS = {
    "kimchi": ("Kimchi jjigae", "Korea", "Korea", "A genuine Korean stew, claimed as Korea."),
    "spaghetti": ("Spaghetti as Korean", "Italy", "Korea", "An Italian pasta dish claimed as Korean."),
    "chana": (
        "Chana masala",
        "Indian Subcontinent",
        "Indian Subcontinent",
        "A classic dish that packs in its cuisine's signature spices.",
    ),
}
COMPONENT_NAMES = {
    "classifier": "Cuisine classifier",
    "cooccurrence": "Ingredient co-occurrence",
    "ns_consistency": "Flavor-sharing consistency",
}
MATCH_LABELS = {"full": "exact", "partial": "partial", "override": "curated", "none": "no match"}

# One meaning per color in every view.
# The talk deck's identity: navy text, teal primary (this recipe, typical, agrees), orange only for "unusual".
INK, BODY, MUTED, LINE, SOFT, PAPER = "#1B2631", "#2C3E50", "#5D6D7E", "#DCE2E6", "#F0F3F4", "#FAFAF7"
TEAL, ORANGE = "#0E7C86", "#D55E00"
ACCENT = TEAL  # this recipe
REAL = "#C3CCD3"  # real CulinaryDB recipes
TYPICAL, ATYPICAL, MILD = TEAL, ORANGE, "#AEBBC6"  # mild = below 0 but above the "unusual" cut
GRID = "#E6EBEE"
OTHER = "#B8C1C8"
CATEGORY_COLORS = {  # CulinaryDB ingredient categories; related categories share a hue family
    "Vegetable": "#3A8D3F",
    "Herb": "#8DBA45",
    "Plant": "#5E7D2B",
    "Fruit": "#D6589A",
    "Flower": "#E9A0C5",
    "Spice": "#B7791F",
    "Essential Oil": "#D4A72C",
    "Legume": "#009E73",
    "Nuts & Seed": "#8A5A2B",
    "Cereal": "#C9A227",
    "Maize": "#E3C04D",
    "Bakery": "#B07845",
    "Dairy": "#56B4E9",
    "Meat": "#9E2B25",
    "Fish": "#0072B2",
    "Seafood": "#2E9BB8",
    "Fungus": "#7D6B5D",
    "Additive": "#9A9A9A",
    "Dish": "#5C6B7A",
    "Beverage": "#7E57C2",
    "Beverage Alcoholic": "#4B2C82",
}
# Sign-agreement heatmap (dashboard).
AGREE, DIFFER, SIGN_ERROR, NOT_RUN = "#BFE0E3", "#F8D9C4", "#EFA06B", "#EEF1F3"
FONT = '"Source Sans", -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'
PLOTLY_CONFIG = {"displaylogo": False, "displayModeBar": False}

CSS = """
<style>
:root{--fp-ink:#1B2631;--fp-body:#2C3E50;--fp-muted:#5D6D7E;--fp-line:#DCE2E6;--fp-soft:#F0F3F4;--fp-card:#FFFFFF;
--fp-teal:#0E7C86;--fp-teal-soft:#E4F2F3;--fp-orange:#D55E00;--fp-orange-soft:#FDF0E6;
--fp-serif:Georgia,"Source Serif","Times New Roman",serif;
--fp-shadow:0 1px 2px rgba(27,38,49,.06),0 12px 28px -18px rgba(27,38,49,.30);--fp-radius:.8rem;}
[data-testid="stMainBlockContainer"]{padding-top:1.5rem;padding-bottom:3rem;max-width:1680px;}
[data-testid="stMain"] h1,[data-testid="stMain"] h2,[data-testid="stMain"] h3,[data-testid="stMain"] h4{
font-family:var(--fp-serif)!important;color:var(--fp-ink);letter-spacing:-.005em;}
[data-testid="stCaptionContainer"]{color:#56677A;}
.st-key-recipe_box,[class*="st-key-dash_"]{background:var(--fp-card);border-radius:var(--fp-radius);
box-shadow:var(--fp-shadow);}
.st-key-preset button{padding:.15rem .75rem;min-height:2rem;}
.st-key-preset button p{font-size:.86rem;}
[data-testid="stTab"] p{font-size:1.02rem;font-weight:600;}
[data-testid="stTabPanel"] [role="tablist"]{gap:.2rem;background:var(--fp-soft);border:1px solid var(--fp-line);
border-radius:.75rem;padding:.22rem;width:fit-content;max-width:100%;margin-bottom:.35rem;gap:.1rem;}
[data-testid="stTabPanel"] [data-testid="stTab"]{border-radius:.55rem;padding:.3rem .52rem;margin:0;height:auto;}
[data-testid="stTabPanel"] [data-testid="stTab"] p{font-size:.9rem;font-weight:500;color:var(--fp-body);}
[data-testid="stTabPanel"] [data-testid="stTab"][aria-selected="true"]{background:var(--fp-card);
box-shadow:0 1px 3px rgba(27,38,49,.16);}
[data-testid="stTabPanel"] [data-testid="stTab"][aria-selected="true"] p{color:var(--fp-teal);font-weight:650;}
[data-testid="stTabPanel"] .react-aria-SelectionIndicator{display:none;}
.fp-hero{display:grid;grid-template-columns:minmax(17rem,1fr) auto;align-items:center;gap:1rem 2rem;
background:var(--fp-card);border:1px solid var(--fp-line);border-radius:var(--fp-radius);box-shadow:var(--fp-shadow);
padding:1.1rem 1.35rem;margin:.1rem 0 1rem;}
.fp-title{font-family:var(--fp-serif);font-weight:700;font-size:2.55rem;line-height:1;color:var(--fp-ink);
letter-spacing:-.015em;}
.fp-title span{color:var(--fp-teal);}
.fp-lede{margin:.45rem 0 0;color:var(--fp-body);font-size:1.06rem;line-height:1.4;}
.fp-chips{display:grid;grid-template-columns:repeat(4,8.6rem);gap:.6rem;}
.fp-chip{display:flex;flex-direction:column;justify-content:flex-start;gap:.15rem;background:var(--fp-soft);
border:1px solid var(--fp-line);border-radius:.65rem;padding:.55rem .8rem;min-height:4.3rem;}
.fp-chip b{font-size:1.4rem;font-weight:700;color:var(--fp-ink);font-variant-numeric:tabular-nums;line-height:1.05;}
.fp-chip span{font-size:.8rem;color:var(--fp-muted);line-height:1.2;}
.fp-verdict{display:grid;grid-template-columns:minmax(0,1fr) 13rem;gap:1rem 1.4rem;align-items:center;
background:var(--fp-card);border:1px solid var(--fp-line);border-left:7px solid var(--fp-teal);
border-radius:var(--fp-radius);padding:1.1rem 1.3rem;margin:0 0 .85rem;box-shadow:var(--fp-shadow);}
.fp-verdict.warn{border-left-color:var(--fp-orange);}
.fp-vhead{font-family:var(--fp-serif);font-size:1.72rem;font-weight:700;line-height:1.2;color:var(--fp-ink);
letter-spacing:-.01em;}
.fp-vsub{color:var(--fp-body);font-size:1rem;margin-top:.45rem;line-height:1.45;}
.fp-badges{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.7rem;}
.fp-badge{font-size:.82rem;font-weight:600;border-radius:999px;padding:.16rem .7rem;border:1px solid;}
.fp-badge.flag{color:#9C4300;background:var(--fp-orange-soft);border-color:#F2C29E;}
.fp-badge.note{color:var(--fp-body);background:var(--fp-soft);border-color:var(--fp-line);}
.fp-vscore{border-left:1px solid var(--fp-line);padding-left:1.3rem;}
.fp-val.big{font-size:3.6rem;min-height:0;color:var(--fp-teal);line-height:1;margin-top:.25rem;}
.fp-verdict.warn .fp-val.big{color:var(--fp-orange);}
.fp-gauge{position:relative;height:10px;border-radius:5px;margin:.75rem 0 .35rem;
background:linear-gradient(90deg,#E9EDEF 0 25%,#CFD8DE 25% 75%,#E9EDEF 75% 100%);}
.fp-gmark{position:absolute;top:50%;width:19px;height:19px;border-radius:50%;background:var(--fp-teal);
border:3px solid #fff;box-shadow:0 0 0 1px var(--fp-teal);transform:translate(-50%,-50%);}
.fp-verdict.warn .fp-gmark{background:var(--fp-orange);box-shadow:0 0 0 1px var(--fp-orange);}
.fp-gscale{display:flex;justify-content:space-between;font-size:.74rem;color:var(--fp-muted);}
.fp-cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.75rem;margin:0 0 .9rem;}
.fp-cards.two{grid-template-columns:repeat(2,minmax(0,1fr));}
.fp-cards.two .fp-lab{white-space:normal;}
.fp-card{display:flex;flex-direction:column;background:var(--fp-card);border:1px solid var(--fp-line);
border-radius:var(--fp-radius);padding:.85rem 1rem .8rem;min-width:0;box-shadow:var(--fp-shadow);}
.fp-card.warn{border-color:#F2C6A4;background:#FFFAF6;}
.fp-card.warn .fp-val{color:var(--fp-orange);}
.fp-lab{font-size:.88rem;font-weight:600;color:var(--fp-muted);line-height:1.25;white-space:nowrap;overflow:hidden;
text-overflow:ellipsis;}
.fp-val{display:flex;align-items:flex-end;min-height:3.15rem;font-size:2.35rem;font-weight:700;line-height:1.05;
color:var(--fp-ink);margin-top:.25rem;letter-spacing:-.01em;font-variant-numeric:tabular-nums;}
.fp-val.long{font-size:1.3rem;line-height:1.2;overflow-wrap:normal;word-break:normal;}
.fp-track{position:relative;height:7px;border-radius:4px;margin:.65rem 0 .1rem;
background:linear-gradient(90deg,#E9EDEF 0 25%,#CFD8DE 25% 75%,#E9EDEF 75% 100%);}
.fp-track.blank{visibility:hidden;}
.fp-dot{position:absolute;top:50%;width:15px;height:15px;border-radius:50%;background:var(--fp-teal);
border:2px solid #fff;box-shadow:0 0 0 1px var(--fp-teal);transform:translate(-50%,-50%);}
.fp-card.warn .fp-dot{background:var(--fp-orange);box-shadow:0 0 0 1px var(--fp-orange);}
.fp-ctx{font-size:.84rem;color:var(--fp-muted);line-height:1.35;margin-top:.4rem;}
.fp-callout{border:1px solid var(--fp-line);border-left:5px solid var(--fp-teal);background:var(--fp-card);
border-radius:.7rem;padding:.75rem 1rem;color:var(--fp-body);font-size:.98rem;line-height:1.5;margin:.5rem 0 .9rem;
box-shadow:var(--fp-shadow);}
.fp-callout b{color:var(--fp-ink);}
.fp-callout.warn{border-left-color:var(--fp-orange);background:#FFFAF6;}
.fp-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid var(--fp-line);margin-bottom:.7rem;
border-radius:var(--fp-radius);background:var(--fp-card);box-shadow:var(--fp-shadow);}
table.fp-table{width:100%;border-collapse:collapse;font-size:.98rem;}
.fp-table th{font-weight:600;color:var(--fp-muted);text-align:left;padding:.55rem .8rem;font-size:.86rem;
border-bottom:1px solid var(--fp-line);background:var(--fp-soft);white-space:nowrap;}
.fp-table td{padding:.5rem .8rem;border-bottom:1px solid #E9EDF0;vertical-align:top;color:var(--fp-ink);}
.fp-table tr:last-child td{border-bottom:none;}
.fp-table .num{text-align:right;font-variant-numeric:tabular-nums;}
.fp-table td.idx{color:var(--fp-muted);width:2rem;}
.fp-table tr.miss td{background:#FFF7F1;}
.fp-table tr.miss td.ent{color:#A54800;font-weight:600;}
.fp-note{display:block;font-size:.82rem;color:var(--fp-muted);font-weight:400;}
.fp-cat{display:inline-flex;align-items:center;gap:.45rem;white-space:nowrap;}
.fp-cat i{width:.7rem;height:.7rem;border-radius:50%;display:inline-block;flex:none;}
.fp-tag{font-size:.8rem;font-weight:600;border-radius:999px;padding:.12rem .55rem;white-space:nowrap;}
.fp-tag.exact{background:var(--fp-teal-soft);color:#0A5C63;}
.fp-tag.partial,.fp-tag.curated{background:var(--fp-soft);color:var(--fp-body);border:1px solid var(--fp-line);}
.fp-tag.none{background:var(--fp-orange-soft);color:#9C4300;}
.fp-q{font-weight:650;font-size:1.1rem;color:var(--fp-ink);margin:.55rem 0 .2rem;line-height:1.3;}
.fp-lead{font-size:1.08rem;color:var(--fp-body);line-height:1.5;margin:.35rem 0 .45rem;}
.fp-lead b{color:var(--fp-ink);}
.fp-sig{display:flex;flex-wrap:wrap;gap:.45rem;margin:.4rem 0 .7rem;}
.fp-sig span{font-size:.92rem;padding:.22rem .75rem;border-radius:999px;border:1px dashed #BCC7CF;color:var(--fp-muted);}
.fp-sig span.on{border:1px solid var(--fp-teal);background:var(--fp-teal-soft);color:#0A5C63;font-weight:600;}
.fp-panel{background:var(--fp-card);border:1px solid var(--fp-line);border-radius:var(--fp-radius);padding:.9rem 1.1rem;
height:100%;box-shadow:var(--fp-shadow);}
.fp-panel h4{font-family:var(--fp-serif);font-size:1.18rem;font-weight:700;margin:0 0 .2rem;padding:0;
color:var(--fp-ink);}
.fp-panel p{font-size:.9rem;color:var(--fp-muted);margin:0 0 .35rem;line-height:1.4;}
.fp-list{list-style:none;margin:.2rem 0 0;padding:0;}
.fp-list li{display:flex;justify-content:space-between;align-items:center;gap:.7rem;padding:.48rem 0;margin:0;
border-bottom:1px solid #E9EDF0;font-size:1rem;}
.fp-list li:last-child{border-bottom:none;}
.fp-list .v{font-size:.88rem;color:var(--fp-muted);font-variant-numeric:tabular-nums;white-space:nowrap;}
.fp-empty{font-size:.95rem;color:var(--fp-muted);padding:.4rem 0;}
.fp-legend{display:flex;flex-wrap:wrap;gap:.35rem 1.1rem;font-size:.9rem;color:var(--fp-body);margin:.15rem 0 .25rem;}
.fp-legend span{display:inline-flex;align-items:center;gap:.4rem;}
.fp-legend i{width:.9rem;height:.9rem;border-radius:.22rem;display:inline-block;}
.fp-legend i.line{height:0;border-top:2px dashed var(--fp-ink);border-radius:0;width:1.15rem;}
.fp-legend i.dot{height:0;border-top:2px dotted #8A97A3;border-radius:0;width:1.15rem;}
.fp-foot{font-size:.84rem;color:var(--fp-muted);border-top:1px solid var(--fp-line);margin-top:1.8rem;padding-top:.8rem;}
@media (max-width:1000px){.fp-hero{grid-template-columns:1fr;}.fp-chips{grid-template-columns:repeat(4,minmax(0,1fr));}
.fp-verdict{grid-template-columns:1fr;}.fp-vscore{border-left:none;padding-left:0;border-top:1px solid var(--fp-line);
padding-top:.8rem;}.fp-cards{grid-template-columns:repeat(2,minmax(0,1fr));}}
@media (max-width:640px){
[data-testid="stMainBlockContainer"]{padding:1.2rem 1rem 2rem;}
.fp-title{font-size:2.1rem;}
.fp-lede{font-size:1rem;}
.fp-chips{grid-template-columns:repeat(2,minmax(0,1fr));}
.fp-cards{gap:.55rem;}
.fp-val{font-size:1.7rem;min-height:2.5rem;}
.fp-val.long{font-size:1.15rem;}
.fp-val.big{font-size:2.8rem;}
.fp-vhead{font-size:1.4rem;}
.fp-table{font-size:.88rem;}
}
</style>
"""


# ---------------------------------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading the flavor network, ingredient aliaser and fidelity scorer...")
def load_core() -> dict:
    with open(ART / "core.pkl", "rb") as fh:
        core = pickle.load(fh)  # built by app/build_artifacts.py in this repo
    _, text = EXAMPLES["Korea"]
    audit(clean_lines(text), "Korea", core)  # warm-up, so a visitor's first audit is as fast as later ones
    return core


@st.cache_data(show_spinner=False)
def load_dashboard(name: str) -> pd.DataFrame | None:
    path = ART / "dashboard" / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_summary() -> dict:
    """Headline numbers copied from results/summary.json by build_artifacts.py (empty if absent)."""
    path = ART / "dashboard" / "summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


# ---------------------------------------------------------------------------------------------------- analysis
def clean_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"^\s*([-*•]|\d+[.)])\s+", "", raw).strip()
        if line:
            lines.append(line)
    return lines


def percentile(sorted_values: np.ndarray, x: float) -> float:
    """Mid-rank percentile of x among sorted values (ties count half)."""
    if not np.isfinite(x) or len(sorted_values) == 0:
        return float("nan")
    lo, hi = np.searchsorted(sorted_values, x, "left"), np.searchsorted(sorted_values, x, "right")
    return 100.0 * (lo + hi) / (2 * len(sorted_values))


def stereotype_flag(share: float) -> bool:
    """True if at least 97.5% of real recipes of the cuisine use no more of its 10 signature ingredients.

    ``share`` is in percent. presentation/demo_probe.py parses the cut from the line below; keep it a literal.
    """
    if share >= 97.5:  # noqa: SIM103 (the literal is parsed by presentation/demo_probe.py)
        return True
    return False


def audit(lines: list[str], region: str, core: dict) -> dict:
    """Audit one recipe claimed as ``region``: ingredient mapping, Ns, fidelity, stereotype check and feedback."""
    al, sc, S = core["aliaser"], core["scorer"], core["S"]
    row_of = core["row_of_entity"]
    table, rows, first_line = [], [], {}
    for k, line in enumerate(lines, 1):
        m = al.alias(line)
        row = row_of.get(m.entity_id) if m.entity_id is not None else None
        note = ""
        if row is not None and row in first_line:
            note = f"same ingredient as line {first_line[row]}; counted once"
        elif row is not None:
            first_line[row] = k
            rows.append(row)
        elif m.entity_id is not None:
            note = "CulinaryDB entity without a flavor profile; not scored"
        table.append(
            {
                "Ingredient line": line,
                "Matched ingredient": core["names"][row] if row is not None else "",
                "Category": core["categories"][row] if row is not None else "",
                "Match": MATCH_LABELS.get(m.how, m.how) if row is not None else "no match",
                "Flavor compounds": int(S[row, row]) if row is not None else None,
                "Note": note,
            }
        )
    recipe = np.unique(np.array(rows, dtype=np.int64))
    reg = core["regions"][region]
    out = {"table": pd.DataFrame(table), "recipe": recipe}
    if len(recipe) == 0:
        return out
    proba = pd.Series(sc.predict_proba([recipe])[0], index=sc.regions).sort_values(ascending=False)
    out["proba"] = proba
    out["top5"] = list(proba.head(5).items())
    out["p_claimed"] = float(proba[region])
    out["components"] = sc.zscores([recipe], region).iloc[0]
    out["composite"] = float(sc.score([recipe], region)[0])
    out["fidelity_pct"] = percentile(reg["composite_sorted"], out["composite"])
    out["ns"] = ns_recipe(recipe, S)
    out["ns_pct"] = percentile(reg["ns_sorted"], out["ns"])
    out["feedback"] = sc.feedback(recipe, region, reg["signature"])
    out["contributions"] = sc.contributions(recipe, region)
    out["atypical_cut"] = float(sc.atypical_cut.get(region, 0.0))
    sig = set(reg["signature"])
    out["signature_used"] = [core["names"][r] for r in recipe if r in sig]
    counts = reg["signature_counts"]
    cdf = np.cumsum(counts) / counts.sum()
    out["signature_share_at_most"] = float(cdf[min(len(out["signature_used"]), len(cdf) - 1)])
    out["stereotype_flag"] = stereotype_flag(100 * out["signature_share_at_most"])
    return out


def what_if(res: dict, region: str, core: dict) -> dict | None:
    """Re-score the recipe with every suggestion applied (atypical ingredients out, missing signature ones in)."""
    fb = res["feedback"]
    if not (fb["atypical"] or fb["missing"]):
        return None
    keep = np.setdiff1d(res["recipe"], np.asarray(fb["atypical"], dtype=np.int64))
    revised = np.union1d(keep, np.asarray(fb["missing"], dtype=np.int64))
    if len(revised) < 2:
        return None
    reg = core["regions"][region]
    composite = float(core["scorer"].score([revised], region)[0])
    ns = ns_recipe(revised, core["S"])
    return {
        "fidelity_pct": percentile(reg["composite_sorted"], composite),
        "ns": ns,
        "ns_pct": percentile(reg["ns_sorted"], ns),
    }


def pair_matrix(recipe: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Shared-compound counts between the recipe's ingredients (zero diagonal)."""
    W = S[np.ix_(recipe, recipe)].astype(float)
    np.fill_diagonal(W, 0.0)
    return W


def network_layout(keys: list[tuple[str, str]]) -> np.ndarray:
    """Deterministic circle positions, clockwise from the top, with nodes sorted by (category, name) so each
    category sits together and labels can be placed radially outward without overlapping."""
    order = sorted(range(len(keys)), key=lambda k: keys[k])
    ang = np.pi / 2 - 2 * np.pi * np.arange(len(keys)) / max(len(keys), 1)
    P = np.zeros((len(keys), 2))
    P[order] = np.column_stack([np.cos(ang), np.sin(ang)])
    return P


# ---------------------------------------------------------------------------------------------------- HTML bits
def esc(text: object) -> str:
    """Escape text for st.markdown(unsafe_allow_html=True); '$' too, so Markdown never starts math mode."""
    return html.escape(str(text)).replace("$", "&#36;")


def share_text(p: float) -> str:
    return "<1%" if p < 0.005 else f"{100 * p:.0f}%"


def cat_html(category: str) -> str:
    color = CATEGORY_COLORS.get(category, OTHER)
    return f'<span class="fp-cat"><i style="background:{color}"></i>{esc(category)}</span>'


def card_html(key: str, label: str, value: str, context: str, pct: float | None = None, warn: bool = False) -> str:
    """A metric card. The 0-100 track stays as an invisible spacer when there is no percentile, so that the captions
    of a row of cards line up."""
    has = pct is not None and np.isfinite(pct)
    dot = f'<div class="fp-dot" style="left:{min(max(pct, 1.5), 98.5):.1f}%"></div>' if has else ""
    long = " long" if len(value) > 11 else ""
    return (
        f'<div class="fp-card{" warn" if warn else ""}" data-metric="{key}"><div class="fp-lab">{label}</div>'
        f'<div class="fp-val{long}">{esc(value)}</div><div class="fp-track{"" if has else " blank"}">{dot}</div>'
        f'<div class="fp-ctx">{context}</div></div>'
    )


def question(text: str) -> None:
    st.markdown(f'<div class="fp-q">{text}</div>', unsafe_allow_html=True)


def explain(text: str) -> None:
    st.caption(f":material/lightbulb: **What does this mean?** {text}")


def hero_html(core: dict, summary: dict) -> str:
    n_real = sum(r["n_recipes"] for r in core["regions"].values())
    stats = [
        (f"{len(core['regions'])}", "world cuisines"),
        (f"{n_real:,}", "CulinaryDB recipes"),
        (f"{core['S'].shape[0]:,}", "flavor-network ingredients"),
    ]
    if summary.get("total_recipes"):
        stats.append((f"{summary['total_recipes']:,}", "LLM recipes audited"))
    tiles = "".join(f'<div class="fp-chip"><b>{v}</b><span>{label}</span></div>' for v, label in stats)
    return (
        '<div class="fp-hero"><div><div class="fp-title" role="heading" aria-level="1">Flavor<span>Print</span></div>'
        '<p class="fp-lede">Does a recipe respect the culinary fingerprint of the cuisine it claims?</p></div>'
        f'<div class="fp-chips">{tiles}</div></div>'
    )


def verdict_html(res: dict, lines: list[str], region: str, title: str, n_test: int) -> str:
    """The verdict band: the headline (teal bar when the classifier agrees with the claim, orange when it does not),
    the classifier's view and flags, and the fidelity percentile on a 0-100 gauge (25-75: middle half of genuine
    recipes)."""
    top, p_top = res["top5"][0]
    name = esc(title.strip()) if title.strip() else "This recipe"
    fid, R = res["fidelity_pct"], esc(region)
    if top == region and fid >= 25:
        tone, head = "good", f"{name} reads as {R}, the cuisine it claims."
    elif top == region:
        tone, head = "good", f"{name} reads as {R}, but scores below most genuine {R} recipes."
    else:
        tone, head = "warn", f"{name} is claimed as {R} but reads as {esc(top)}."
    if top == region:
        sub = f"The cuisine classifier gives {R} {share_text(p_top)}."
    else:
        sub = f"The cuisine classifier gives {R} {share_text(res['p_claimed'])} and {esc(top)} {share_text(p_top)}."
    fb = res["feedback"]
    badges = []
    if res["stereotype_flag"]:
        badges.append('<span class="fp-badge flag">Stereotype flag: signature ingredients over-used</span>')
    if fb["atypical"]:
        n = len(fb["atypical"])
        badges.append(f'<span class="fp-badge flag">{n} ingredient{"s" if n > 1 else ""} unusual for {R}</span>')
    verdict = {"high": "higher than most", "low": "lower than most", "typical": "typical"}[fb["ns_verdict"]]
    badges.append(f'<span class="fp-badge note">Flavor sharing {verdict} for {R}</span>')
    n_miss = int((res["table"]["Match"] == "no match").sum())
    if n_miss:
        badges.append(f'<span class="fp-badge note">{n_miss} of {len(lines)} lines not in CulinaryDB</span>')
    score = (
        f'<div class="fp-vscore" data-metric="fidelity"><div class="fp-lab">Fidelity percentile</div>'
        f'<div class="fp-val big">{fid:.0f}</div>'
        f'<div class="fp-gauge"><div class="fp-gmark" style="left:{min(max(fid, 1.5), 98.5):.1f}%"></div></div>'
        '<div class="fp-gscale"><span>0</span><span>typical 25–75</span><span>100</span></div>'
        f'<div class="fp-ctx">vs {n_test} genuine {R} recipes held out from training</div></div>'
    )
    return (
        f'<div class="fp-verdict {tone}"><div><div class="fp-vhead">{head}</div><div class="fp-vsub">{sub}</div>'
        f'<div class="fp-badges">{"".join(badges)}</div></div>{score}</div>'
    )


def cards_html(res: dict, lines: list[str], region: str) -> str:
    top, p_top = res["top5"][0]
    R = esc(region)
    n_miss = int((res["table"]["Match"] == "no match").sum())
    matched = len(lines) - n_miss
    claim = "matches the claim" if top == region else f"claimed {R}: {share_text(res['p_claimed'])}"
    unmatched = f"{n_miss} unmatched line{'s' if n_miss > 1 else ''} excluded from all scores"
    used, share, flag = len(res["signature_used"]), 100 * res["signature_share_at_most"], res["stereotype_flag"]
    cards = [
        card_html("top", "Most likely cuisine", top, f"{share_text(p_top)} probability; {claim}", warn=top != region),
        card_html(
            "ns",
            "Flavor sharing Ns",
            f"{res['ns']:.1f}",
            f"compounds per ingredient pair; percentile {res['ns_pct']:.0f} in {R}",
            res["ns_pct"],
        ),
        card_html(
            "signature",
            "Signature ingredients",
            f"{used}/10",
            f"{share:.0f}% of real {R} recipes use as many or fewer" + ("; stereotype flag" if flag else ""),
            share,
            warn=flag,
        ),
        card_html(
            "matched",
            "Ingredients matched",
            f"{matched}/{len(lines)}",
            unmatched if n_miss else "every line mapped to CulinaryDB",
        ),
    ]
    return f'<div class="fp-cards">{"".join(cards)}</div>'


def callout(kind: str, title: str, body: str) -> str:
    """A note in the app's own style: kind 'info' (teal bar), 'ok' (teal) or 'warn' (orange)."""
    return f'<div class="fp-callout {kind}"><b>{title}</b> {body}</div>'


def mapping_html(table: pd.DataFrame) -> str:
    rows = []
    for k, r in enumerate(table.to_dict("records"), 1):
        miss = r["Match"] == "no match"
        tag = {"exact": "exact", "partial": "partial", "curated": "curated"}.get(r["Match"], "none")
        entity = "not found in CulinaryDB" if miss else esc(r["Matched ingredient"])
        if r["Note"]:
            entity += f'<span class="fp-note">{esc(r["Note"])}</span>'
        comp = "" if miss or pd.isna(r["Flavor compounds"]) else f"{int(r['Flavor compounds'])}"
        rows.append(
            f'<tr class="{"miss" if miss else ""}"><td class="idx">{k}</td><td>{esc(r["Ingredient line"])}</td>'
            f'<td class="ent">{entity}</td><td>{"" if miss else cat_html(r["Category"])}</td>'
            f'<td><span class="fp-tag {tag}">{esc(r["Match"])}</span></td><td class="num">{comp}</td></tr>'
        )
    head = (
        "<tr><th>#</th><th>Your line</th><th>CulinaryDB ingredient</th><th>Category</th><th>Match</th>"
        '<th class="num" title="flavor compounds in the FlavorDB profile">Compounds</th></tr>'
    )
    body = "".join(rows)
    return f'<div class="fp-scroll"><table class="fp-table"><thead>{head}</thead><tbody>{body}</tbody></table></div>'


def list_html(items: list[tuple[str, str, str]], empty: str) -> str:
    """Rows of (ingredient name, category, value text)."""
    if not items:
        return f'<div class="fp-empty">{empty}</div>'
    lis = "".join(f'<li><span>{esc(n)}&nbsp; {cat_html(c)}</span><span class="v">{v}</span></li>' for n, c, v in items)
    return f'<ul class="fp-list">{lis}</ul>'


def legend_html(items: list[tuple[str, str]]) -> str:
    """Legend rows of (swatch color, label); a swatch of 'line' or 'dot' draws a dashed or dotted line."""
    spans = []
    for swatch, label in items:
        i = f'<i class="{swatch}"></i>' if swatch in ("line", "dot") else f'<i style="background:{swatch}"></i>'
        spans.append(f"<span>{i}{esc(label)}</span>")
    return f'<div class="fp-legend">{"".join(spans)}</div>'


# ---------------------------------------------------------------------------------------------------- figures
# Layouts are written in one update_layout call with no template: plotly's validation of templates and repeated
# axis updates cost ~40 ms per figure, which made every click visibly slower.
def axis(title: str = "", **kw) -> dict:
    out = {
        "gridcolor": GRID,
        "zeroline": False,
        "linecolor": LINE,
        "automargin": True,
        "tickfont": {"size": 14, "color": BODY},
        "title": {"text": title, "font": {"size": 14, "color": MUTED}},
    }
    out.update(kw)
    return out


def styled(fig: go.Figure, height: int, x: dict | None = None, y: dict | None = None, **layout) -> go.Figure:
    base = {
        "template": "none",
        "height": height,
        "margin": {"l": 8, "r": 16, "t": 12, "b": 8},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": FONT, "size": 14, "color": INK},
        "hoverlabel": {"font": {"family": FONT, "size": 14}, "bgcolor": "#FFFFFF", "bordercolor": LINE},
        "showlegend": False,
        "xaxis": x if x is not None else axis(),
        "yaxis": y if y is not None else axis(),
    }
    base.update(layout)
    fig.update_layout(**base)
    return fig


def vline(
    x: float, color: str, label: str = "", width: float = 2.0, dash: str = "solid", right: bool = True,
    inside: bool = False,
):  # fmt: skip
    """A vertical line across the plot and its optional label, as (shape, [annotation]). The label sits above the
    plot, or just inside its top edge (``inside``) so that two labels never share a row."""
    shape = {
        "type": "line", "xref": "x", "yref": "paper", "x0": x, "x1": x, "y0": 0, "y1": 1, "layer": "above",
        "line": {"color": color, "width": width, "dash": dash},
    }  # fmt: skip
    ann = {
        "x": x, "y": 0.99 if inside else 1, "xref": "x", "yref": "paper", "text": label, "showarrow": False,
        "font": {"color": color, "size": 14}, "xanchor": "left" if right else "right",
        "yanchor": "top" if inside else "bottom", "xshift": 4 if right else -4,
        "bgcolor": "rgba(251,250,247,0.85)" if inside else "rgba(0,0,0,0)",
    }  # fmt: skip
    return shape, ([ann] if label else [])


def show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONFIG, key=key)


def distribution_figure(values: np.ndarray, x: float, x_title: str, y_title: str, fmt: str) -> go.Figure:
    """Histogram of real values with this recipe's value and the real median marked."""
    med = float(np.median(values))
    lo, hi = float(min(values.min(), x)), float(max(values.max(), x))
    pad = 0.04 * (hi - lo or 1.0)
    fig = go.Figure(
        go.Histogram(
            x=values,
            xbins={"start": lo - pad, "end": hi + pad, "size": (hi - lo + 2 * pad) / 40},
            marker={"color": REAL, "line": {"color": PAPER, "width": 0.6}},
            hovertemplate=f"%{{x}}<br>%{{y}} {y_title.lower()}<extra></extra>",
        )
    )
    span = hi - lo + 2 * pad
    frac, mfrac = (x - lo + pad) / span, (med - lo + pad) / span
    right = frac < 0.62  # label on the side with room, so it is never clipped at the plot edge
    m_right = mfrac < 0.55  # the median label has its own row (inside the plot), so only fit matters
    s1, a1 = vline(med, MUTED, f"real median {med:{fmt}}", 1.6, "dot", right=m_right, inside=True)
    s2, a2 = vline(x, ACCENT, f"<b>this recipe {x:{fmt}}</b>", 3.0, right=right)
    return styled(
        fig, 255, x=axis(x_title, range=[lo - pad, hi + pad]), y=axis(y_title), bargap=0.05,
        margin={"l": 8, "r": 16, "t": 30, "b": 8}, shapes=[s1, s2], annotations=a1 + a2,
    )  # fmt: skip


def hbar_figure(
    labels: list[str],
    values: list[float],
    colors: list[str],
    texts: list[str],
    hover: list[str],
    x: dict | None = None,
    shapes: tuple = (),
    annotations: tuple = (),
    diverging: bool = False,
) -> go.Figure:
    """Horizontal bars, first item on top. With ``diverging``, a negative bar's value label sits just right of the
    zero line (it can never run into the axis labels on the left) and a positive bar's label at the bar's end."""
    fig = go.Figure(
        go.Bar(
            x=values[::-1], y=labels[::-1], orientation="h", marker={"color": colors[::-1]},
            text=None if diverging else texts[::-1], textposition="outside", cliponaxis=False,
            hovertext=hover[::-1], hoverinfo="text",
        )
    )  # fmt: skip
    if diverging:
        fig.add_trace(
            go.Scatter(
                x=[0.0 if v < 0 else v for v in values[::-1]], y=labels[::-1], mode="text",
                text=texts[::-1], hoverinfo="skip", textposition="middle right",
                textfont={"size": 14, "color": BODY}, cliponaxis=False,
            )
        )  # fmt: skip
    return styled(
        fig, 46 + 34 * len(labels), x=x, y=axis(showgrid=False), bargap=0.28,
        margin={"l": 8, "r": 58, "t": 28, "b": 8}, shapes=list(shapes), annotations=list(annotations),
    )  # fmt: skip


def sharing_figure(recipe: np.ndarray, core: dict) -> go.Figure:
    names, cats = core["names"], core["categories"]
    W = pair_matrix(recipe, core["S"])
    mean = W.sum(axis=1) / (len(recipe) - 1)
    order = np.argsort(-mean)
    ns = float(mean.mean())
    s, a = vline(ns, ACCENT, f"<b>Ns {ns:.1f}</b>", 2.0, "dash")
    return hbar_figure(
        [names[recipe[i]] for i in order],
        [float(mean[i]) for i in order],
        [CATEGORY_COLORS.get(cats[recipe[i]], OTHER) for i in order],
        [f"{mean[i]:.0f}" for i in order],
        [
            f"{names[recipe[i]]} ({cats[recipe[i]]}): shares {mean[i]:.1f} compounds on average with the others"
            for i in order
        ],
        x=axis("Mean shared compounds"),
        shapes=(s,),
        annotations=tuple(a),
    )


def network_figure(recipe: np.ndarray, core: dict) -> go.Figure:
    S, names, cats = core["S"], core["names"], core["categories"]
    W = pair_matrix(recipe, S)
    n = len(recipe)
    P = network_layout([(cats[r], names[r]) for r in recipe])
    iu, ju = np.triu_indices(n, 1)
    w = W[iu, ju]
    wmax = float(w.max()) or 1.0
    level = np.minimum((w / wmax * 4).astype(int), 3)
    traces = []
    for b in range(4):  # one trace per width band (a plotly line has a single width)
        sel = (w > 0) & (level == b)
        xs, ys = [], []
        for i, j in zip(iu[sel], ju[sel], strict=True):
            xs += [P[i, 0], P[j, 0], None]
            ys += [P[i, 1], P[j, 1], None]
        if xs:
            traces.append(
                go.Scatter(
                    x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                    line={"width": 0.8 + 2.2 * b, "color": f"rgba(52,73,94,{0.14 + 0.16 * b:.2f})"},
                )
            )  # fmt: skip
    traces.append(
        go.Scatter(
            x=(P[iu, 0] + P[ju, 0]) / 2, y=(P[iu, 1] + P[ju, 1]) / 2, mode="markers", showlegend=False,
            marker={"size": 12, "color": "rgba(0,0,0,0)"}, hoverinfo="text",
            hovertext=[f"{names[recipe[i]]} + {names[recipe[j]]}: <b>{int(W[i, j])}</b> shared compounds" for i, j in zip(iu, ju, strict=True)],
        )
    )  # fmt: skip
    comp = np.diag(S)[recipe].astype(float)
    size = 16 + 24 * np.sqrt(comp / max(comp.max(), 1.0))
    mean = W.sum(axis=1) / max(n - 1, 1)
    for cat in sorted({cats[r] for r in recipe}):
        idx = [k for k in range(n) if cats[recipe[k]] == cat]
        traces.append(
            go.Scatter(
                x=P[idx, 0], y=P[idx, 1], mode="markers+text", name=cat, text=[names[recipe[k]] for k in idx],
                textposition=[
                    "middle right" if P[k, 0] > 0.3 else "middle left" if P[k, 0] < -0.3
                    else "top center" if P[k, 1] >= 0 else "bottom center" for k in idx
                ],
                textfont={"size": 14.5, "color": INK},
                marker={"size": size[idx], "color": CATEGORY_COLORS.get(cat, OTHER), "line": {"color": "#FFFFFF", "width": 2}},
                hoverinfo="text",
                hovertext=[f"<b>{names[recipe[k]]}</b> ({cat})<br>{int(comp[k])} flavor compounds in FlavorDB<br>"
                           f"shares {mean[k]:.1f} on average with the others" for k in idx],
            )
        )  # fmt: skip
    return styled(
        go.Figure(traces), 450, x=axis(visible=False, range=[-1.9, 1.9]),
        y=axis(visible=False, range=[-1.25, 1.25], scaleanchor="x", scaleratio=1),
        showlegend=True, margin={"l": 4, "r": 4, "t": 8, "b": 4},
        legend={"orientation": "h", "y": -0.02, "x": 0.5, "xanchor": "center", "yanchor": "top", "font": {"size": 14}},
    )  # fmt: skip


def components_figure(z: pd.Series) -> go.Figure:
    keys = list(COMPONENT_NAMES)
    vals = [float(z.get(k, np.nan)) for k in keys]
    shown = [v if np.isfinite(v) else 0.0 for v in vals]
    span = max(2.5, max(abs(v) for v in shown) * 1.2)
    return hbar_figure(
        [COMPONENT_NAMES[k] for k in keys],
        shown,
        [TYPICAL if v >= 0 else ATYPICAL for v in shown],
        [f"{v:+.2f}" if np.isfinite(v) else "n/a" for v in vals],
        [f"{COMPONENT_NAMES[k]}: z = {v:+.2f}" if np.isfinite(v) else f"{COMPONENT_NAMES[k]}: not defined" for k, v in zip(keys, vals, strict=True)],
        x=axis("z-score (0 = typical)", range=[-span, span]),
        shapes=(vline(0.0, MUTED, width=1.5)[0],),
        diverging=True,
    )  # fmt: skip


def proba_figure(proba: pd.Series, region: str) -> go.Figure:
    top = list(proba.head(5).index)
    if region not in top:
        top.append(region)
    vals = [float(proba[r]) for r in top]
    return hbar_figure(
        top,
        vals,
        [ACCENT if r == region else REAL for r in top],
        [share_text(v) for v in vals],
        [f"{r}: {100 * v:.1f}%" + (" (claimed)" if r == region else "") for r, v in zip(top, vals, strict=True)],
        x=axis("Calibrated probability", range=[0, 1.12], tickformat=".0%"),
    )


def contributions_figure(contrib: pd.Series, cut: float, core: dict) -> go.Figure:
    names = core["names"]
    items = list(contrib.sort_values().items())  # most atypical first (top)
    vals = [float(v) for _, v in items]
    lo, hi = min(min(vals), cut), max(max(vals), 0.0)
    pad = 0.4 * (hi - lo or 1.0)  # room for the value labels at the bar ends
    zero = vline(0.0, MUTED, width=1.2)[0]
    s, a = vline(cut, ORANGE, "unusual below", 1.6, "dash", right=False)
    s["layer"] = "below"  # under the value labels that sit beside the zero line
    return hbar_figure(
        [names[r] for r, _ in items],
        vals,
        [ATYPICAL if v < cut else (TYPICAL if v >= 0 else MILD) for v in vals],
        [f"{v:+.2f}" for v in vals],
        [f"{names[r]}: {v:+.3f}" + (" (flagged as unusual)" if v < cut else "") for (r, _), v in zip(items, vals, strict=True)],
        x=axis("Contribution to the cuisine's log-odds", range=[lo - pad, hi + pad]),
        shapes=(zero, s),
        annotations=tuple(a),
        diverging=True,
    )  # fmt: skip


def signature_figure(counts: np.ndarray, used: int) -> go.Figure:
    share = counts / counts.sum()
    ks = np.arange(len(counts))
    k = min(used, len(share) - 1)
    fig = go.Figure(
        go.Bar(
            x=ks, y=share, marker={"color": [ACCENT if i == k else REAL for i in ks]},
            text=[share_text(s) if s > 0 else "" for s in share], textposition="outside", cliponaxis=False,
            constraintext="none", textfont={"size": 13.5},
            hovertemplate="%{x} signature ingredients: %{y:.1%} of real recipes<extra></extra>",
        )
    )  # fmt: skip
    ann = {
        "x": k, "y": 1, "yref": "paper", "yanchor": "bottom", "text": "<b>this recipe ▾</b>", "showarrow": False,
        "font": {"color": ACCENT, "size": 14},
    }  # fmt: skip
    return styled(
        fig, 280, x=axis("Signature ingredients used (of 10)", dtick=1, showgrid=False),
        y=axis("Real recipes", tickformat=".0%", range=[0, float(share.max()) * 1.4]), bargap=0.25,
        margin={"l": 8, "r": 16, "t": 34, "b": 8}, annotations=[ann],
    )  # fmt: skip


@st.cache_resource(show_spinner=False, max_entries=64)
def audit_figures(recipe_key: tuple[int, ...], region: str, _res: dict, _core: dict) -> dict[str, go.Figure | None]:
    """Every chart of one audit, built once per (recipe, claimed cuisine); the figures are never mutated."""
    res, core, reg = _res, _core, _core["regions"][region]
    recipe = res["recipe"]
    return {
        "ns": distribution_figure(reg["ns_sorted"], res["ns"], "Ns (shared compounds per pair)", "Real recipes", ".1f"),
        "sharing": sharing_figure(recipe, core),
        "network": network_figure(recipe, core),
        "fidelity": distribution_figure(
            reg["composite_sorted"], res["composite"], "Composite fidelity score", "Genuine recipes", "+.2f"
        ),
        "components": components_figure(res["components"]),
        "proba": proba_figure(res["proba"], region),
        "contrib": contributions_figure(res["contributions"], res["atypical_cut"], core)
        if len(res["contributions"])
        else None,
        "signature": signature_figure(reg["signature_counts"], len(res["signature_used"])),
    }


# ---------------------------------------------------------------------------------------------------- audit page
def _apply_preset() -> None:
    key = st.session_state.get("preset")
    if key in PRESETS:
        _, example, claimed, _ = PRESETS[key]
        title, text = EXAMPLES[example]
        st.session_state.update(recipe_title=title, recipe_text=text, claimed=claimed)


def _edited() -> None:
    st.session_state.preset = None


def _clear() -> None:
    st.session_state.update(preset=None, recipe_title="", recipe_text="")


def recipe_input(regions: list[str]) -> None:
    if "recipe_text" not in st.session_state:
        st.session_state.preset = "kimchi"
        _apply_preset()
    with st.container(border=True, key="recipe_box"):
        st.markdown("#### Your recipe")
        st.pills(
            "Built-in examples", list(PRESETS), key="preset", format_func=lambda k: PRESETS[k][0],
            on_change=_apply_preset,
        )  # fmt: skip
        # The note always describes the inputs on screen: an example's description only while it is untouched
        # (any edit below clears the example), otherwise the current claim.
        if st.session_state.get("preset") in PRESETS:
            note = PRESETS[st.session_state.preset][3]
        elif st.session_state.get("recipe_text", "").strip():
            note = f"Your recipe, claimed as {st.session_state.get('claimed', 'Korea')}."
        else:
            note = "Paste a recipe (one ingredient per line) or pick an example."
        st.caption(note)
        st.text_input("Recipe title (optional)", key="recipe_title", on_change=_edited)
        st.text_area(
            "Ingredients, one per line", key="recipe_text", height=262, on_change=_edited,
            placeholder="2 cups napa cabbage kimchi\n200 g pork belly\n1 tbsp soy sauce\n...",
        )  # fmt: skip
        st.selectbox(
            "Claimed cuisine", regions, key="claimed", help="One of the 22 CulinaryDB regions.", on_change=_edited
        )
        b1, b2 = st.columns([3, 1])
        b1.button("Audit recipe", type="primary", icon=":material/play_arrow:", width="stretch")
        b2.button("Clear", type="tertiary", on_click=_clear, width="stretch")
    st.caption(
        "Quantities and preparation words are ignored. Your recipe is processed in memory only and is not stored."
    )


def view_ingredients(res: dict) -> None:
    question("How was each line mapped to CulinaryDB and FlavorDB?")
    st.markdown(mapping_html(res["table"]), unsafe_allow_html=True)
    explain(
        "Each line is mapped to one CulinaryDB ingredient with the CulinaryDB aliasing protocol (Singh & Bagler 2018): "
        "quantities and preparation words are dropped, then the longest matching name or synonym wins. *partial* "
        "means part of a multi-word synonym matched. Unmatched lines are left out of every score, and an ingredient "
        "listed twice counts once. *Compounds* is the size of the ingredient's FlavorDB flavor profile."
    )


def view_sharing(res: dict, region: str, reg: dict, figs: dict) -> None:
    ns, pct, R = res["ns"], res["ns_pct"], esc(region)
    st.markdown(
        f'<div class="fp-lead">On average, two ingredients of this recipe share <b>{ns:.1f} flavor compounds</b>. '
        f"That is higher than about <b>{pct:.0f}%</b> of the {len(reg['ns_sorted']):,} real {R} recipes in "
        "CulinaryDB.</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        question(f"Where does this recipe's flavor sharing fall among real {R} recipes?")
        show(figs["ns"], "ns_hist")
        explain(
            "Ns counts the FlavorDB compounds two ingredients have in common, averaged over every ingredient pair "
            "of the recipe (Ahn et al. 2011). Grey bars are real recipes of the claimed cuisine."
        )
    with c2:
        question("Which ingredients carry the flavor sharing?")
        show(figs["sharing"], "sharing_bars")
        explain(
            "Each bar is one ingredient's average overlap with the others; the recipe's Ns is the mean of the bars. "
            "Colors are CulinaryDB categories, as in the network."
        )
    z = reg["z_real"]
    body = (
        f"Real {R} recipes share {'more' if z > 0 else 'fewer'} flavor compounds than random recipes of the same "
        f"sizes built from the same ingredients: Z = {z:+.0f} against the random null (N<sub>rand</sub> = 100,000), "
        f"i.e. <b>{'uniform' if z > 0 else 'contrasting'} food pairing</b>, the same sign as Singh &amp; Bagler "
        "(2018). A single recipe has no Z, so FlavorPrint places its Ns within the cuisine's recipe-level distribution."
    )
    st.markdown(callout("info", "Cuisine level.", body), unsafe_allow_html=True)


def view_network(res: dict, core: dict, figs: dict) -> None:
    recipe, names = res["recipe"], core["names"]
    W = pair_matrix(recipe, core["S"])
    question("Which ingredients share flavor compounds, and how many?")
    c1, c2 = st.columns([2, 1], gap="medium")
    with c1:
        show(figs["network"], "network")
    with c2:
        iu, ju = np.triu_indices(len(recipe), 1)
        order = np.argsort(-W[iu, ju])[:6]
        items = [(f"{names[recipe[iu[k]]]} + {names[recipe[ju[k]]]}", int(W[iu[k], ju[k]])) for k in order]
        lis = "".join(f'<li><span>{esc(p)}</span><span class="v">{v}</span></li>' for p, v in items)
        weak = [names[recipe[i]] for i in np.argsort(W.sum(axis=1))[:2]]
        st.markdown(
            f'<div class="fp-panel"><h4>Strongest pairs</h4><p>shared flavor compounds</p><ul class="fp-list">{lis}'
            f'</ul><p style="margin-top:.6rem">Least connected: {esc(", ".join(weak))}.</p></div>',
            unsafe_allow_html=True,
        )
    explain(
        "Nodes are the matched ingredients (size: number of compounds in the FlavorDB profile; color: CulinaryDB "
        "category; nodes of one category sit together on the circle). An edge joins two ingredients that share "
        "compounds: thicker and darker means more shared. Ns is the average edge weight over all pairs. Hover over a "
        "node or an edge's midpoint for the counts."
    )


def view_fidelity(res: dict, region: str, reg: dict, summary: dict, figs: dict) -> None:
    R, n_test = esc(region), len(reg["composite_sorted"])
    acc = summary.get("scorer", {})
    note = (
        f" Held-out accuracy: top-1 {100 * acc['test_top1']:.0f}%, top-3 {100 * acc['test_top3']:.0f}%."
        if acc.get("test_top1") and acc.get("test_top3")
        else ""
    )
    c1, c2 = st.columns(2, gap="large")  # row 1: the score and the classifier's view
    with c1:
        question(f"How typical is this recipe of genuine {R} recipes?")
        show(figs["fidelity"], "fidelity_hist")
        explain(
            f"Grey: {n_test} genuine {R} recipes the scorer never saw in training. The score is the mean of the three "
            "component z-scores below."
        )
    with c2:
        question("Which cuisine does the classifier see?")
        show(figs["proba"], "proba")
        explain(f"Calibrated probabilities over the 22 cuisines; the claimed one is highlighted.{note}")
    c3, c4 = st.columns(2, gap="large")  # row 2: why
    with c3:
        question("Which component pulls the score up or down?")
        show(figs["components"], "components")
        explain(
            f"z-scores against genuine {R} recipes (0 = typical): classifier probability, co-occurrence of the "
            f"ingredient pairs in real {R} recipes (PMI), and closeness of Ns to the {R} median."
        )
    with c4:
        question(f"Which ingredients make it more or less {R}?")
        if figs["contrib"] is not None:
            show(figs["contrib"], "contrib")
        explain(
            f"Classifier weight for {R} minus its mean over all 22 cuisines. Orange bars fall below the dashed cut "
            f"(5th percentile in genuine {R} recipes) and are flagged as unusual."
        )


def view_stereotype(res: dict, region: str, reg: dict, core: dict, summary: dict, figs: dict) -> None:
    R = esc(region)
    used = len(res["signature_used"])
    share = 100 * res["signature_share_at_most"]
    have = {int(r) for r in res["recipe"]}
    chips = "".join(
        f'<span class="{"on" if r in have else ""}">{esc(core["names"][r])}</span>' for r in reg["signature"]
    )
    question(f"How many of {R}'s signature ingredients does the recipe use?")
    st.markdown(f'<div class="fp-sig">{chips}</div>', unsafe_allow_html=True)
    if res["stereotype_flag"]:
        body = (
            f"The recipe uses {used} of {R}'s 10 signature ingredients; {share:.1f}% of real {R} recipes use this many "
            "or fewer. A recipe can read as authentic and still be a caricature that piles on the cuisine's most "
            "distinctive ingredients."
        )
        st.markdown(callout("warn", "Stereotype flag.", body), unsafe_allow_html=True)
    else:
        body = (
            f"The recipe uses {used} of 10; {share:.1f}% of real {R} recipes use this many or fewer (the flag needs "
            "97.5% or more)."
        )
        st.markdown(callout("ok", "No stereotype flag.", body), unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        show(figs["signature"], "signature")
    with c2:
        band = summary.get("real_n160", {}).get("sai_band")
        band_txt = f" Random 160-recipe samples of real cuisines score {band[0]:.2f} to {band[1]:.2f}." if band else ""
        st.markdown(
            '<div class="fp-panel"><h4>Signature set and SAI</h4><p>The 10 signature ingredients (highlighted above '
            "when used) have the highest relative prevalence in the cuisine: prevalence in it minus the mean "
            "prevalence in the other 21 (Jain & Bagler 2018).</p><p>For a whole collection of recipes, the "
            "Stereotype Amplification Index (SAI) divides the summed prevalence of these 10 ingredients by their "
            f"summed prevalence in real recipes; above 1 means they are over-used.{band_txt}</p><p>For one recipe, "
            "FlavorPrint asks how unusual its count is among real recipes and flags it at the 97.5th percentile."
            "</p></div>",
            unsafe_allow_html=True,
        )
    explain(
        f"Bars show how many signature ingredients real {R} recipes use. The highlighted bar is this recipe's count; "
        "the further right of the bulk it sits, the more stereotyped the recipe."
    )


def view_suggestions(res: dict, region: str, core: dict, figs: dict) -> None:
    fb, names, cats, S, R = res["feedback"], core["names"], core["categories"], core["S"], esc(region)
    contrib = res["contributions"]
    atypical = [(names[r], cats[r], f"{float(contrib.get(r, np.nan)):+.2f}") for r in fb["atypical"]]
    keep = [r for r in res["recipe"] if r not in set(fb["atypical"])]
    missing = [(names[r], cats[r], f"shares {S[r, keep].mean():.0f} compounds on avg.") for r in fb["missing"]]
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(
            f'<div class="fp-panel"><h4>Unusual for this cuisine</h4><p>Ingredients that lower the classifier\'s '
            f"odds for {R} the most (below the cuisine's 5th-percentile cut).</p>"
            f"{list_html(atypical, f'Nothing unusual for {R}: every ingredient is above the cut.')}</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="fp-panel"><h4>Typical {R} ingredients that would pair well</h4><p>Signature ingredients '
            f"the recipe lacks that co-occur with its other ingredients in real {R} recipes (positive PMI).</p>"
            f"{list_html(missing, 'No missing signature ingredient co-occurs with this recipe.')}</div>",
            unsafe_allow_html=True,
        )
    word = {"high": "more", "low": "fewer"}.get(fb["ns_verdict"])
    if word:
        sharing = f"Its ingredients share **{word}** flavor compounds than most real {region} recipes"
    else:
        sharing = f"Its flavor sharing is within the middle half of real {region} recipes"
    st.space("small")
    st.markdown(f"**Flavor sharing.** {sharing} (Ns {res['ns']:.1f}, percentile {res['ns_pct']:.0f}).")
    c3, c4 = st.columns([3, 2], gap="large")
    with c3:
        question("Why: the score components")
        show(figs["components"], "components_sugg")
    with c4:
        wi = what_if(res, region, core)
        if wi is not None:
            question("What if every suggestion were applied?")
            d_fid, d_ns = wi["fidelity_pct"] - res["fidelity_pct"], wi["ns"] - res["ns"]
            whatif_cards = card_html(
                "whatif_fidelity",
                "Fidelity percentile",
                f"{wi['fidelity_pct']:.0f}",
                f"{d_fid:+.0f} vs now ({res['fidelity_pct']:.0f})",
                wi["fidelity_pct"],
            ) + card_html("whatif_ns", "Flavor sharing Ns", f"{wi['ns']:.1f}", f"{d_ns:+.1f} vs now ({res['ns']:.1f})")
            st.markdown(f'<div class="fp-cards two">{whatif_cards}</div>', unsafe_allow_html=True)
    explain(
        "These are the critiques our critique-and-revise steering sends back to an LLM (FidelityScorer.feedback). "
        "The what-if re-scores the ingredient set with the unusual ingredients removed and the suggested ones added; "
        "it is an estimate from the same scorer, not a tested recipe."
    )


def page_audit(core: dict, summary: dict) -> None:
    regions = sorted(core["regions"])
    left, right = st.columns([2, 3], gap="medium")
    with left:
        recipe_input(regions)
    lines = clean_lines(st.session_state.get("recipe_text", ""))
    region = st.session_state.get("claimed", "Korea")
    title = st.session_state.get("recipe_title", "")
    with right:
        if not lines:
            st.info("Paste ingredient lines on the left, or pick a built-in example.", icon=":material/edit_note:")
            return
        res = audit(lines, region, core)
        if len(res["recipe"]) < 2:
            st.warning(
                "Fewer than two lines matched a CulinaryDB ingredient, so flavor sharing and fidelity are undefined. "
                "Check the mapping below or add more ingredients.",
                icon=":material/warning:",
            )
            st.markdown(mapping_html(res["table"]), unsafe_allow_html=True)
            return
        reg = core["regions"][region]
        n_test = len(reg["composite_sorted"])
        figs = audit_figures(tuple(int(r) for r in res["recipe"]), region, res, core)
        st.markdown(verdict_html(res, lines, region, title, n_test), unsafe_allow_html=True)
        st.markdown(cards_html(res, lines, region), unsafe_allow_html=True)
        tabs = st.tabs(
            ["Ingredients", "Flavor sharing", "Flavor network", "Cuisine fidelity", "Stereotype", "Suggestions"]
        )
        with tabs[0]:
            view_ingredients(res)
        with tabs[1]:
            view_sharing(res, region, reg, figs)
        with tabs[2]:
            view_network(res, core, figs)
        with tabs[3]:
            view_fidelity(res, region, reg, summary, figs)
        with tabs[4]:
            view_stereotype(res, region, reg, core, summary, figs)
        with tabs[5]:
            view_suggestions(res, region, core, figs)


# ---------------------------------------------------------------------------------------------------- dashboard
def sign_heatmap(pooled: pd.DataFrame, models: list[str], labels: dict[str, str]) -> go.Figure:
    real = pooled.groupby("region")["z_real"].first().sort_values(ascending=False)
    regions = real.index.tolist()
    code = np.zeros((len(regions), len(models)))
    text = np.full(code.shape, "", dtype=object)
    hover = np.full(code.shape, "", dtype=object)
    for j, m in enumerate(models):
        g = pooled[pooled["model"] == m].set_index("region")
        for i, r in enumerate(regions):
            if r not in g.index:
                hover[i, j] = f"<b>{labels[m]} · {r}</b><br>no recipes for this region"
                continue
            row = g.loc[r]
            code[i, j] = 3 if row["sign_error"] else (1 if row["sign_match"] else 2)
            text[i, j] = f"<b>{row['z_sb']:+.0f}</b>" if row["sign_error"] else f"{row['z_sb']:+.0f}"
            status = "same sign as the real cuisine" if row["sign_match"] else "opposite sign; ΔNs CI includes 0"
            if row["sign_error"]:
                status = "opposite sign; ΔNs CI excludes 0"
            hover[i, j] = (
                f"<b>{labels[m]} · {r}</b><br>Z = {row['z_sb']:+.0f} (real {row['z_real']:+.0f})<br>"
                f"ΔNs = {row['delta_ns']:+.2f} [{row['delta_ns_lo']:+.2f}, {row['delta_ns_hi']:+.2f}]<br>"
                f"{int(row['n_recipes'])} recipes · {status}"
            )
    cs = [[0, NOT_RUN], [0.25, NOT_RUN], [0.25, AGREE], [0.5, AGREE], [0.5, DIFFER], [0.75, DIFFER], [0.75, SIGN_ERROR], [1, SIGN_ERROR]]  # fmt: skip
    fig = go.Figure(
        go.Heatmap(
            z=code, x=["<br>".join(labels[m].rsplit(" ", 1)) if len(labels[m]) > 12 else labels[m] for m in models],
            y=[f"{r}  ({real[r]:+.0f})" for r in regions], zmin=-0.5,
            zmax=3.5, colorscale=cs, showscale=False, text=text, texttemplate="%{text}",
            textfont={"size": 13, "color": INK}, customdata=hover, hovertemplate="%{customdata}<extra></extra>",
            xgap=3, ygap=3,
        )
    )  # fmt: skip
    n_pos = int((real > 0).sum())
    sep = {
        "type": "line", "xref": "paper", "x0": -0.4, "x1": 1, "yref": "y", "y0": n_pos - 0.5, "y1": n_pos - 0.5,
        "line": {"color": INK, "width": 1.5, "dash": "dot"},
    }  # fmt: skip
    return styled(
        fig, 34 + 22 * len(regions), x=axis(side="top", showgrid=False, tickangle=0, tickfont={"size": 14, "color": INK}),
        y=axis(autorange="reversed", showgrid=False, tickfont={"size": 13.5, "color": INK}),
        margin={"l": 4, "r": 4, "t": 34, "b": 4}, shapes=[sep],
    )  # fmt: skip


def comparison_figure(summary: dict, models: list[str], labels: dict[str, str], palette: dict) -> go.Figure:
    la, ref = summary.get("llm_audit", {}), summary.get("real_n160", {})
    cm = summary.get("copy_mutate", {}).get("CM-R/ranked|n=160", {})
    sm, med, band = ref.get("sign_match") or {}, ref.get("medians") or {}, ref.get("sai_band")
    panels = [
        ("Sign agreement", "sign_agreement", float(np.mean(list(sm.values()))) if sm else None, cm.get("sign_match"), (0, 1.05)),
        ("Prevalence JSD", "median_prevalence_jsd", med.get("prevalence_jsd"), cm.get("prevalence_jsd"), (0, 0.75)),
        ("SAI (1 = real use)", "median_sai", 1.0 if band else None, cm.get("sai"), (0, 1.25)),
        ("Fingerprint ρ", "median_fingerprint_spearman", med.get("fingerprint_spearman"), cm.get("fingerprint_spearman"), (-0.1, 0.95)),
    ]  # fmt: skip
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=[p[0] for p in panels], horizontal_spacing=0.06, vertical_spacing=0.2,
        shared_yaxes=True,
    )  # fmt: skip
    ylab = [labels[m] for m in models]
    for k, (_, key, real_v, cm_v, rng) in enumerate(panels):
        r, c = k // 2 + 1, k % 2 + 1
        xs = [la.get(m, {}).get(key) for m in models]
        if key == "median_sai" and band:
            fig.add_vrect(x0=band[0], x1=band[1], fillcolor="#E1E7EA", opacity=0.8, line_width=0, row=r, col=c)
        fig.add_trace(
            go.Bar(
                x=xs, y=ylab, orientation="h", marker={"color": [palette.get(m, MUTED) for m in models]},
                text=[f"{x:.2f}" if x is not None else "" for x in xs], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            ),
            row=r, col=c,
        )  # fmt: skip
        if real_v is not None:
            fig.add_vline(x=real_v, line={"color": INK, "width": 1.6, "dash": "dash"}, row=r, col=c)
        if cm_v is not None:
            fig.add_vline(x=cm_v, line={"color": "#8A97A3", "width": 1.8, "dash": "dot"}, row=r, col=c)
        fig.update_xaxes(range=list(rng), row=r, col=c)
    styled(fig, 440, bargap=0.28, margin={"l": 4, "r": 30, "t": 30, "b": 8})
    fig.update_xaxes(gridcolor=GRID, zeroline=False, automargin=True, tickfont={"size": 13.5})
    fig.update_yaxes(autorange="reversed", showgrid=False, automargin=True, tickfont={"size": 14, "color": INK})
    fig.update_annotations(font={"size": 14, "color": INK})
    return fig


def scorecard_table(summary: dict, pooled: pd.DataFrame, models: list[str], labels: dict[str, str]) -> pd.DataFrame:
    la = summary.get("llm_audit", {})
    rows = []
    for m in models:
        a = la.get(m)
        if a is None:  # no summary: the same aggregates from the scorecard
            g = pooled[pooled["model"] == m]
            a = {
                "regions": len(g), "recipes": int(g["n_recipes"].sum()), "sign_agreement": g["sign_match"].mean(),
                "sign_errors_preregistered": int(g["sign_error"].sum()), "mean_near_copy_rate": g["near_copy_rate"].mean(),
                **{f"median_{c}": g[c].median() for c in ("prevalence_jsd", "category_jsd", "sai", "fingerprint_spearman", "diversity_ratio")},
            }  # fmt: skip
        rows.append(
            {
                "Source": labels[m], "Regions": a.get("regions"), "Recipes": a.get("recipes"),
                "Sign agreement": a.get("sign_agreement"), "Sign errors": a.get("sign_errors_preregistered"),
                "Prevalence JSD": a.get("median_prevalence_jsd"), "Category JSD": a.get("median_category_jsd"),
                "SAI": a.get("median_sai"), "Fingerprint ρ": a.get("median_fingerprint_spearman"),
                "Diversity ratio": a.get("median_diversity_ratio"), "Near-copy rate": a.get("mean_near_copy_rate"),
            }
        )  # fmt: skip
    ref = summary.get("real_n160", {})
    if ref.get("medians"):
        md, sm = ref["medians"], ref.get("sign_match") or {}
        rows.append(
            {
                "Source": "Real cuisine, 160-recipe samples", "Regions": len(sm) or None, "Recipes": 160,
                "Sign agreement": float(np.mean(list(sm.values()))) if sm else None,
                "Prevalence JSD": md.get("prevalence_jsd"), "Category JSD": md.get("category_jsd"),
                "SAI": md.get("sai"), "Fingerprint ρ": md.get("fingerprint_spearman"),
                "Near-copy rate": md.get("near_copy_rate"),
            }
        )  # fmt: skip
    for key, name in (("CM-R/ranked|n=160", "Copy-mutate (ranked), n = 160"), ("CM-R/random|n=160", "Copy-mutate (random), n = 160")):  # fmt: skip
        c = summary.get("copy_mutate", {}).get(key)
        if c:
            rows.append(
                {
                    "Source": name, "Recipes": 160, "Sign agreement": c.get("sign_match"),
                    "Prevalence JSD": c.get("prevalence_jsd"), "SAI": c.get("sai"),
                    "Fingerprint ρ": c.get("fingerprint_spearman"), "Near-copy rate": c.get("near_copy_rate"),
                }
            )  # fmt: skip
    df = pd.DataFrame(rows)
    df["Sign agreement"] = 100 * df["Sign agreement"].astype(float)
    return df


@st.cache_resource(show_spinner=False)
def dashboard_view(_core: dict) -> dict | None:
    """Everything the dashboard draws, built once per server process (the artifacts are static)."""
    sc = load_dashboard("llm_scorecard")
    if sc is None:
        return None
    summary = load_summary()
    pooled = sc[sc["scope"] == "pooled"]
    palette = _core.get("palette", {})
    present = set(pooled["model"])
    models = [m for m in palette if m in present] + sorted(present - set(palette))
    labels = {m: summary.get("models", {}).get(m, {}).get("name") or m for m in models}
    per_model = pooled.groupby("model").agg(regions=("region", "nunique"), recipes=("n_recipes", "sum"))
    return {
        "pooled": pooled,
        "models": models,
        "labels": labels,
        "palette": palette,
        "per_model": per_model,
        "agree": pooled.groupby("model")["sign_match"].mean(),
        "heatmap": sign_heatmap(pooled, models, labels),
        "comparison": comparison_figure(summary, models, labels, palette),
        "table": scorecard_table(summary, pooled, models, labels),
    }


@st.cache_resource(show_spinner=False, max_entries=32)
def region_figure(region: str, _view: dict) -> go.Figure:
    pooled, models, labels, palette = _view["pooled"], _view["models"], _view["labels"], _view["palette"]
    g = pooled[pooled["region"] == region].set_index("model")
    have = [m for m in models if m in g.index]
    names = ["Real (CulinaryDB)"] + [labels[m] for m in have]
    vals = [float(g["z_real"].iloc[0])] + [float(g.loc[m, "z_sb"]) for m in have]
    colors = [palette.get("real", INK)] + [palette.get(m, MUTED) for m in have]
    span = max(abs(v) for v in vals) * 1.3
    return hbar_figure(
        names, vals, colors, [f"{v:+.0f}" for v in vals],
        [f"{n}: Z = {v:+.1f}" for n, v in zip(names, vals, strict=True)],
        x=axis("Z against the random null (N_rand = 100,000)", range=[-span, span]),
        shapes=(vline(0.0, MUTED, width=1.2)[0],),
        diverging=True,
    )  # fmt: skip


def page_dashboard(core: dict, summary: dict) -> None:
    view = dashboard_view(core)
    if view is None:
        st.info("The dashboard reads app/artifacts/dashboard/, built by `app/build_artifacts.py --dashboard-only`.")
        return
    models, labels, palette, per_model = view["models"], view["labels"], view["palette"], view["per_model"]
    rep = summary.get("replication", {}).get("random", {})
    sm = summary.get("real_n160", {}).get("sign_match") or {}
    agree = view["agree"]
    cards = []
    if rep:
        cards.append(
            card_html(
                "replication", "Replication of Singh & Bagler (2018)", f"{rep['sign_agreement']}/{rep['n']}",
                f"food-pairing signs reproduced; Spearman ρ = {rep['spearman']:.2f} with the published Z",
            )
        )  # fmt: skip
    cards.append(
        card_html(
            "llm_recipes", "LLM recipes audited", f"{int(per_model['recipes'].sum()):,}",
            f"{len(models)} models, up to 22 regions each, plain and persona prompts",
        )
    )  # fmt: skip
    cards.append(
        card_html(
            "sign_agreement", "Sign agreement, by model", f"{100 * agree.min():.0f}–{100 * agree.max():.0f}%",
            "share of regions where the LLM cuisine's Z has the real sign",
        )
    )  # fmt: skip
    if sm:
        cards.append(
            card_html(
                "real_reference", "Real-sample reference", f"{100 * float(np.mean(list(sm.values()))):.0f}%",
                "mean sign agreement of random 160-recipe samples of each real cuisine",
            )
        )  # fmt: skip
    st.markdown(f'<div class="fp-cards">{"".join(cards)}</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True, key="dash_heat"):
        question("Does each LLM reproduce each cuisine's food-pairing sign?")
        st.markdown(
            legend_html(
                [(AGREE, "same sign"), (DIFFER, "opposite, CI incl. 0"), (SIGN_ERROR, "opposite, CI excl. 0"), (NOT_RUN, "no recipes")]
            ),
            unsafe_allow_html=True,
        )  # fmt: skip
        show(view["heatmap"], "sign_heatmap")
        coverage = "".join(
            f" {labels[m]} covers {int(per_model.loc[m, 'regions'])} of the 22 regions."
            for m in models
            if per_model.loc[m, "regions"] < 22
        )
        explain(
            "Regions ordered by the real cuisine's Z (in brackets): uniform above the dotted line, contrasting "
            f"below. Cells show each LLM cuisine's Z; the darkest are pre-registered sign errors.{coverage}"
        )
    with c2, st.container(border=True, key="dash_compare"):
        question("How do the models compare with real recipes and a copy-mutate baseline?")
        st.markdown(
            legend_html(
                [*((palette.get(m, MUTED), labels[m]) for m in models), ("line", "real, 160-recipe samples"), ("dot", "copy-mutate baseline")]
            ),
            unsafe_allow_html=True,
        )  # fmt: skip
        show(view["comparison"], "comparison")
        explain(
            "Medians over regions for each LLM; closer to real means higher sign agreement and fingerprint ρ, lower "
            "JSD and SAI near 1. Dashed: the same metric for random 160-recipe samples of the real "
            "cuisines (what sampling alone gives); dotted: the copy-mutate evolution model (Assignment 3 baseline). "
            "The shaded SAI band is the range of real samples."
        )
    with st.container(border=True, key="dash_table"):
        question("Scorecard: every metric, per model and reference")
        fmt2 = st.column_config.NumberColumn(format="%.2f")
        st.dataframe(
            view["table"],
            width="stretch",
            hide_index=True,
            column_config={
                "Source": st.column_config.TextColumn("Source", width="medium"),
                "Regions": st.column_config.NumberColumn(format="%d"),
                "Recipes": st.column_config.NumberColumn(format="localized"),
                "Sign agreement": st.column_config.ProgressColumn("Sign agreement", format="%.0f%%", min_value=0, max_value=100),
                "Sign errors": st.column_config.NumberColumn(format="%d", help="Pre-registered: regions whose ΔNs CI excludes 0 with the opposite sign to the real cuisine"),
                "Prevalence JSD": fmt2, "Category JSD": fmt2, "SAI": fmt2, "Fingerprint ρ": fmt2,
                "Diversity ratio": fmt2, "Near-copy rate": fmt2,
            },
        )  # fmt: skip
        explain(
            "JSD: Jensen-Shannon divergence of ingredient or category usage from the real cuisine (0 = identical). SAI: "
            "over-use of the cuisine's 10 signature ingredients (1 = as in real recipes). Fingerprint ρ: Spearman "
            "correlation of the per-category ΔNs fingerprint with the real one (Jain, Rakhi & Bagler 2015). Diversity "
            "ratio: distinct ingredients relative to real recipes at the same N. Near-copy rate: share of recipes with "
            "Jaccard ≥ 0.8 to a real recipe. Values are medians over regions (sign agreement and near-copy: means)."
        )
    with st.container(border=True, key="dash_regionbox"):
        question("Region detail")
        regions = sorted(view["pooled"]["region"].unique())
        region = st.selectbox(
            "Region", regions, index=regions.index("Korea") if "Korea" in regions else 0, key="dash_region"
        )
        d1, d2 = st.columns([1, 1], gap="large")
        with d1:
            show(region_figure(region, view), "region_z")
        with d2:
            drv = load_dashboard("llm_drivers")
            if drv is not None and (drv["region"] == region).any():
                g = drv[drv["region"] == region].set_index("model")
                rows = [{"Source": "Real (CulinaryDB)", "Top-5 flavor drivers (χ%)": ", ".join(g["real_top5_names"].iloc[0]), "Overlap with real": None}]  # fmt: skip
                rows += [
                    {"Source": labels[m], "Top-5 flavor drivers (χ%)": ", ".join(g.loc[m, "top5_names"]), "Overlap with real": float(g.loc[m, "jaccard5"])}
                    for m in models if m in g.index
                ]  # fmt: skip
                st.dataframe(
                    pd.DataFrame(rows), width="stretch", hide_index=True,
                    column_config={"Overlap with real": st.column_config.NumberColumn(format="%.2f", help="Jaccard overlap of the top-5 sets")},
                )  # fmt: skip
            explain(
                f"Left: {region}'s food-pairing Z for the real cuisine and each LLM cuisine. Right: the ingredients that "
                "drive food pairing most (χ%: % change in mean Ns when the ingredient is removed, Singh & Bagler 2018)."
            )


# ---------------------------------------------------------------------------------------------------- about
def page_about() -> None:
    c1, c2 = st.columns([6, 5], gap="large")
    with c1:
        st.markdown("#### What FlavorPrint measures")
        st.markdown(
            "**Flavor sharing** (Ahn et al. 2011): the mean number of FlavorDB compounds shared by a recipe's "
            "ingredient pairs, where $C_i$ is the compound set of ingredient $i$ and $n$ the number of ingredients."
        )
        st.latex(r"N_s(R) = \frac{2}{n(n-1)} \sum_{i<j} \lvert C_i \cap C_j \rvert")
        st.markdown(
            "**Food pairing of a cuisine** (Jain, Rakhi & Bagler 2015; Singh & Bagler 2018): the cuisine's mean Ns "
            "against a random null that keeps its ingredient set and recipe sizes, with $N_{rand} = 100{,}000$ random "
            "recipes and $\\sigma_{rand}$ their standard deviation of Ns."
        )
        st.latex(r"Z = \sqrt{N_{rand}}\;\frac{\bar N_s^{\,cuisine} - \bar N_s^{\,rand}}{\sigma_{rand}}")
        st.markdown(
            "**Fidelity** (this project's scorer): the mean of three z-scores, each standardized on genuine recipes of "
            "the claimed cuisine: (1) a calibrated multinomial classifier on CulinaryDB ingredients and categories; "
            "(2) the within-cuisine co-occurrence (shrunk PMI) of the recipe's ingredient pairs; (3) how close its "
            "Ns is to the cuisine's median. The fidelity percentile ranks the recipe among genuine held-out recipes."
        )
        st.markdown(
            "**Signature ingredients and SAI** (relative prevalence, Jain & Bagler 2018): the 10 ingredients most "
            "over-represented in a cuisine relative to the other 21. SAI compares their prevalence in a recipe "
            "collection with real recipes (> 1 = over-used)."
        )
    with c2:
        st.markdown("#### Project")
        st.markdown(
            "*Do AI Chefs Respect Culinary Fingerprints? Auditing and Steering LLM-Generated Recipes across 22 World "
            "Cuisines.* Group 44, BIO544 Computational Gastronomy, IIIT-Delhi, 2026. FlavorPrint runs the same "
            "aliaser, flavor network and fidelity scorer as the study; the methods are in the `flavoraudit` library."
        )
        st.markdown("#### Data and licences")
        st.markdown(
            "- CulinaryDB and FlavorDB (CoSyLab, IIIT-Delhi) are CC BY-NC-SA 3.0; this app's precomputed artifacts "
            "are CC BY-NC-SA 4.0. Please cite Singh & Bagler (2018) and Garg et al. (2018).\n"
            "- Four flavor profiles come from Ahn et al. (2011), CC BY 4.0.\n"
            "- Code: MIT (`flavoraudit`).\n"
            "- Your recipe is processed in memory only and is not stored."
        )
        st.markdown("#### References")
        st.markdown(
            "- Ahn, Ahnert, Bagrow & Barabási (2011). Flavor network and the principles of food pairing. *Sci. Rep.*\n"
            "- Jain, Rakhi & Bagler (2015). Analysis of food pairing in regional cuisines of India. *PLOS ONE*.\n"
            "- Jain & Bagler (2018). Culinary evolution models for Indian cuisines. *Physica A*.\n"
            "- Singh & Bagler (2018). Data-driven investigations of culinary patterns in traditional recipes across "
            "the world. *IEEE ICDE Workshops*.\n"
            "- Garg et al. (2018). FlavorDB: a database of flavor molecules. *Nucleic Acids Res.*"
        )


# ---------------------------------------------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(
        page_title="FlavorPrint: audit a recipe's culinary fingerprint",
        page_icon=":material/restaurant:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.html(CSS)
    if not (ART / "core.pkl").exists():
        st.info("FlavorPrint reads app/artifacts/core.pkl, built by `python app/build_artifacts.py`.")
        return
    hero = st.empty()
    core = load_core()
    summary = load_summary()
    hero.markdown(hero_html(core, summary), unsafe_allow_html=True)
    t_audit, t_dash, t_about = st.tabs(["Audit a recipe", "22-cuisine dashboard", "About & methods"])
    with t_audit:
        page_audit(core, summary)
    with t_dash:
        page_dashboard(core, summary)
    with t_about:
        page_about()
    st.markdown(
        '<div class="fp-foot">FlavorPrint · Group 44, BIO544 Computational Gastronomy, IIIT-Delhi · data: CulinaryDB '
        "and FlavorDB (CoSyLab, IIIT-Delhi), artifacts CC BY-NC-SA 4.0 · code: MIT (flavoraudit)</div>",
        unsafe_allow_html=True,
    )


main()
