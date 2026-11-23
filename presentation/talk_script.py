"""The talk in plain words: who says what on each slide, the demo, how to show each deliverable, simple explanations,
Q&A and a numbers cheat sheet. Shared by presentation/guide.py (the presenter-guide PDF) and presentation/story_deck.py
(speaker notes), so both always say the same thing. Every number is read from the project's result files.

Edit SPEAKER_A / SPEAKER_B to change the names shown in the guide and the speaker notes.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/Nishant7p/CGAS-Project"
SPEAKER_A = "Nishant (2023355)"
SPEAKER_B = "Partner (2023477)"

S = json.loads((REPO / "results/summary.json").read_text())
EX = json.loads((REPO / "results/steering_example.json").read_text())
ALIAS = json.loads((REPO / "data/interim/aliasing/metrics.json").read_text())["test"]["accuracy_lines"]
D, AUD = S["dataset"], S["llm_audit"]
REPL = S["replication"]["by_null"]["random"]
KOREA_TOP = S["replication"]["korea_top3"][0]
GEN = S["equal_n"]["n160_reference_medians"]
GEN_SIGN = float(pd.read_parquet(REPO / "data/interim/audit/real/subsamples_n160.parquet")["sign_match"].mean())
NET = S["networks"]["npmi_top100"]
SC = S["steering"]["scorer"]
AUC = SC["auc_test"]
REV = {(r["model"], r["round"]): r for r in S["steering"]["revise_main"]}
BON = {(r["model"], r["N"]): r for r in S["steering"]["bon_main"]}
CTX = EX["context"]
N_REAL = 45_772
with contextlib.suppress(FileNotFoundError):
    N_REAL = len(pd.read_csv(REPO / "data/raw/culinarydb/extracted/01_Recipe_Details.csv"))
_rep = pd.read_csv(REPO / "data/interim/replication/results_v1_keep_empty.csv")
Z_KOREA_OURS = float(_rep[(_rep["null_model"] == "random") & (_rep["code"] == "KOR")]["z_sb"].iloc[0])
Z_KOREA_PUB = json.loads((REPO / "data/reference/singh_bagler_2018_targets.json").read_text())["z_vs_null"]["random"][
    "KOR"
]

LUNA, GEM, QWEN, LLAMA = "openai-gpt-6-luna", "gemini-3.5-flash-lite", "qwen3.5-9b", "llama-3.1-8b"
NAMES = {LUNA: "GPT-6-Luna", GEM: "Gemini 3.5 Flash-Lite", QWEN: "Qwen3.5-9B", LLAMA: "Llama-3.1-8B"}
ORDER = [LUNA, GEM, QWEN, LLAMA]


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def span(vals, fmt="{:.2f}") -> str:
    lo, hi = fmt.format(min(vals)), fmt.format(max(vals))
    return lo if lo == hi else f"{lo}–{hi}"


SIGN = {m: AUD[m]["sign_agreement"] for m in ORDER}
FP = {m: AUD[m]["median_fingerprint_spearman"] for m in ORDER}
DIV = {m: AUD[m]["median_diversity_ratio"] for m in ORDER}
SAI = {m: AUD[m]["median_sai"] for m in ORDER}
NEARCOPY = {m: AUD[m]["mean_near_copy_rate"] for m in ORDER}
NETV = {m: NET[m]["median"] for m in ORDER}
R0L, R2L = REV[(LUNA, 0)]["indep_classifier_acc"], REV[(LUNA, 2)]["indep_classifier_acc"]
R0Q, R1Q = REV[(QWEN, 0)]["indep_classifier_acc"], REV[(QWEN, 1)]["indep_classifier_acc"]
KEEP = [REV[(LUNA, 2)]["identity_jaccard"], REV[(QWEN, 1)]["identity_jaccard"]]
BON1 = {m: BON[(m, 1)]["indep_classifier_acc"] for m in (LUNA, GEM, QWEN)}
BON8 = {m: BON[(m, 8)]["indep_classifier_acc"] for m in (LUNA, GEM, QWEN)}
DIV1 = {m: BON[(m, 1)]["diversity_ratio"] for m in (LUNA, GEM, QWEN)}
DIV8 = {m: BON[(m, 8)]["diversity_ratio"] for m in (LUNA, GEM, QWEN)}
SPENT = S["resources"]["openai_usd_spent"]
A, B = SPEAKER_A, SPEAKER_B


def who(name: str) -> str:
    """How to refer to a speaker mid-sentence."""
    return "my partner" if name.startswith("Partner") else name.split(" (")[0]


def Who(name: str) -> str:
    w = who(name)
    return w[:1].upper() + w[1:]


# ------------------------------------------------------------------------------------------------- the talk (14 slides)
SLIDES = [
    dict(n=1, title="Title", speaker=A, time="0:30",
         say="Good morning, we are Group 44. Our question is simple: when an AI writes a “traditional” recipe, does it "
             "really cook like that cuisine? We checked it with the methods from this course, and we built five things on "
             "the way: a dataset, an audit, a way to steer the AI, a Python library and a web app."),
    dict(n=2, title="Do AI chefs cook like the cuisine they claim?", speaker=A, time="0:50",
         say="People ask chatbots for authentic recipes every day, but an AI learns from text, not from tasting food. "
             "Every cuisine has a fingerprint: the way it combines ingredients and flavors, and this lab has shown how to "
             "measure it. We look at three layers: flavor pairing, the key ingredients, and the category pattern.",
         point="Analogy: it is like handwriting. We do not check the words (dish names), we check the handwriting."),
    dict(n=3, title="What did we compare?", speaker=A, time="0:50",
         say=f"We compared {N_REAL:,} real recipes from CulinaryDB with {D['total_recipes']:,} recipes written by four AI "
             "models: GPT-6-Luna and Gemini (company models) and Qwen and Llama (open models). Every model got exactly "
             "the same prompts for every cuisine, and every call was saved. We mapped each ingredient line to CulinaryDB "
             f"with CulinaryDB's own method, which is {ALIAS:.0%} accurate.",
         point="This dataset is deliverable D1, LLMCuisine-22."),
    dict(n=4, title="How did we do it?", speaker=A, time="0:40",
         say="This is our pipeline. We start from real cuisines and first check that our code reproduces the published "
             "result. Then we generate AI recipes, map their ingredients, and measure AI and real cuisines with exactly "
             "the same code. Finally, we use what we learned to steer the AI. The bottom row shows the five deliverables."),
    dict(n=5, title="How do we measure food pairing?", speaker=A, time="0:50",
         say="For every recipe we count how many flavor molecules its ingredient pairs share. That number is Ns. Then we "
             "compare with a “blindfolded chef” who picks random ingredients from the same pantry. Italian recipes share "
             "more flavors than chance; Korean recipes share fewer. The gap is ΔNs, and Z is that gap in standard units.",
         point="Point at the black curve (real) against the grey shape (blindfolded chef)."),
    dict(n=6, title="Step 1: can we reproduce the published result?", speaker=A, time="0:40",
         say=f"Before testing any AI recipe, we checked our code against Singh and Bagler's paper. All "
             f"{REPL['sign_agreement']} cuisines get the same sign and the ranking matches with a correlation of "
             f"{REPL['spearman']:.2f}. Korea's top flavor driver is {KOREA_TOP['name'].lower()}, exactly as published. So "
             "our measuring tool works.",
         handover=f"Now {who(B)} will show what we found."),
    dict(n=7, title="Results at a glance: AI vs real cuisines", speaker=B, time="1:00",
         say="This chart is the whole audit in one picture. Grey is what real samples of the same size achieve, our fair "
             f"yardstick. On pairing direction, Llama matches the real cuisine in {pct(SIGN[LLAMA])} of cuisines, as "
             f"often as real samples ({pct(GEN_SIGN)}). On the fingerprint, the signature pairs and the variety of "
             "ingredients, AI reaches only about a third to a half of the real level.",
         point="Read the grey bar first, then each model."),
    dict(n=8, title="Do AI cuisines keep the culinary fingerprint?", speaker=B, time="0:50",
         say="Here is the fingerprint of Korean food, category by category. AI agrees with real cooking on some rows, "
             "such as fungus and legumes, and not on others, such as meat and additives. Across all cuisines, AI "
             f"fingerprints match the real ones at about {span(FP.values())}, against "
             f"{GEN['fingerprint_spearman']:.2f} for real samples. AI gets the broad strokes, not the fine details."),
    dict(n=9, title="Do AI chefs combine the right ingredients?", speaker=B, time="0:50",
         say="These are Korea's ten most distinctive ingredient pairs. Soybean sauce with sesame appears in every AI "
             "cuisine, but shrimp with fish or carrot with potato do not. Across all cuisines, AI shares about "
             f"{span(NETV.values())} of the real top pairs, against {NET['real@160']['median']:.2f} for real samples. And "
             "it finds the key ingredient: soybean sauce is number one for GPT-6-Luna and Qwen, as in real recipes."),
    dict(n=10, title="From audit to fix: a fidelity scorer", speaker=B, time="0:50",
         say="So we turned the audit into a fix: a fidelity scorer. It combines three signals: which cuisine the recipe "
             "looks like, whether its ingredient pairs appear together in real recipes, and whether its flavor sharing "
             f"is typical. It tells real recipes from impostors well: AUC {AUC['other_cuisines']['composite']:.2f} to "
             f"{AUC['random_null']['composite']:.2f}. We use it in two ways: keep the best of several recipes, or give "
             "the AI feedback to revise its recipe."),
    dict(n=11, title="Does fingerprint feedback make AI recipes more authentic?", speaker=B, time="1:00",
         say=f"And it works. With feedback, recipes recognised as the right cuisine rise from {pct(R0L)} to {pct(R2L)} "
             f"for GPT-6-Luna in two rounds, and from {pct(R0Q)} to {pct(R1Q)} for Qwen in one round, while keeping about "
             f"{span(KEEP, '{:.0%}')} of the original ingredients. Picking the best of eight also helps every model. "
             f"Example: {EX['before']['title']} moved from the {EX['before']['fidelity_percentile']:.0f}th to the "
             f"{EX['after']['fidelity_percentile']:.0f}th percentile after adding "
             f"{', '.join(EX['entities_added']).lower()}.",
         handover=f"{Who(A)} will now show our app live."),
    dict(n=12, title="FlavorPrint: audit any recipe in the browser (LIVE DEMO)", speaker=A, time="2:00",
         say="This is FlavorPrint, our web app. You paste a recipe, pick the cuisine it claims, and it shows the "
             "ingredient mapping, the flavor network, the fidelity score, a stereotype check and suggestions. "
             "[Switch to the browser and follow the demo steps in Part 3.]",
         point="If the app does not open, talk over the screenshot on the slide."),
    dict(n=13, title="Everything we promised, delivered", speaker=A, time="0:30",
         say="Everything we promised is delivered: the LLMCuisine-22 dataset, the fingerprint audit, the steering method, "
             "the FlavorAudit library and the FlavorPrint app. All of it is on GitHub with a README, and the link is on "
             "this slide.",
         handover=f"{Who(B)} will conclude."),
    dict(n=14, title="Conclusion", speaker=B, time="0:40",
         say="So, do AI chefs respect culinary fingerprints? Partly. They get the skeleton right: recipe size, signature "
             "ingredients, often the pairing direction and the key flavor drivers. They miss part of the fingerprint, many "
             "regional ingredient pairs, and variety. And the fingerprint itself can guide them closer. Next, we want a "
             "scale study, more models and human taste tests. Thank you, we are happy to take questions."),
]  # fmt: skip

# ------------------------------------------------------------------------------------------------------- live demo
DEMO = [
    ("Kimchi jjigae (already selected) → Audit recipe",
     "Headline: “reads as Korea”. Fidelity percentile and the four cards.",
     "“This genuine Korean stew scores above about two-thirds of real Korean recipes.”"),
    ("Ingredients tab", "The mapping table; the gochugaru line is “not found”.",
     "“Every line is mapped to a CulinaryDB ingredient; unknown ones are left out, not guessed.”"),
    ("Flavor network tab", "Thick edges = many shared flavor molecules.",
     "“Ns is simply the average edge in this network.”"),
    ("Example: Spaghetti, claimed Korean → Audit recipe", "Headline: “claimed as Korea but reads as Italy”; fidelity "
     "drops to the bottom.", "“An Italian dish claimed as Korean is caught immediately.”"),
    ("Suggestions tab", "Unusual ingredients (hard wheat, tomato) and what fits Korean cooking.",
     "“This is exactly the feedback our steering sends to the AI.”"),
    ("Example: Chana masala → Stereotype check tab", "Orange stereotype flag.",
     "“A recipe can be authentic and still a caricature: it uses many signature ingredients at once.”"),
    ("22-cuisine dashboard tab", "Replication card and the model × cuisine heatmap.",
     "“Every model on every cuisine, at a glance.” Then go back to the slides."),
]  # fmt: skip

# --------------------------------------------------------------------------------------- how to show each deliverable
DELIVERABLES = [
    ("D1", "LLMCuisine-22 dataset", "data/release/LLMCuisine-22/ (01_Recipe_Details.csv, DATASHEET.md)",
     f"{D['total_recipes']:,} AI recipes in CulinaryDB's own format, with every model call logged."),
    ("D2", "Fingerprint audit", "deliverables/Group44_BIO544_report.pdf and figures/",
     "The same measurements as for real cuisines, compared with real samples of equal size."),
    ("D3", "Fingerprint-guided steering", "slide 11; results/steering_example.json; steering/",
     "Feedback from the fingerprint makes recipes more recognisable while keeping the dish."),
    ("D4", "FlavorAudit library", "src/flavoraudit/, notebooks/02_audit_recipes.ipynb, or flavoraudit --help",
     "Every method as a tested Python package anyone can install."),
    ("D5", "FlavorPrint web app", "the live demo (app/streamlit_app.py)", "Anyone can audit a recipe in the browser."),
]  # fmt: skip

# ------------------------------------------------------------------------------------------ the ideas in simple words
GLOSSARY = [
    ("Culinary fingerprint", "The typical way a cuisine combines ingredients and flavors. Like handwriting: personal and "
     "measurable."),
    ("Ns (flavor sharing)", "For one recipe: the average number of flavor molecules shared by each pair of its "
     "ingredients."),
    ("Random null (“blindfolded chef”)", "Recipes built at random from the same cuisine's ingredients, with the same "
     "recipe sizes. Real recipes are compared with these."),
    ("The four null models", "Random; keep ingredient popularity (frequency); keep ingredient categories (category); keep "
     "both. If a pattern survives a null, that null does not explain it."),
    ("ΔNs and Z", "ΔNs = real Ns minus random Ns. Positive = “uniform” (share more, like Italy); negative = "
     "“contrasting” (share less, like Korea). Z is the same gap in standard units (Singh & Bagler 2018)."),
    ("Key ingredients (χ%)", "How much the cuisine's flavor sharing changes if one ingredient is removed. Korea: soybean "
     "sauce is number one."),
    ("Category fingerprint", "Shuffle one ingredient category at a time and see how flavor sharing changes (Jain, Rakhi "
     "& Bagler 2015)."),
    ("Stereotype index (SAI)", "Does the AI overuse a cuisine's signature ingredients? 1 = used as often as in real "
     "recipes."),
    ("Ingredient pairs (NPMI)", "Which pairs of ingredients appear together far more often than chance in a cuisine; "
     "the most distinctive pairs are its signatures."),
    ("Equal-size comparison", "Each AI cuisine has 160 recipes, so we compare it with real samples of 160 recipes: "
     "small samples behave differently, and this makes the comparison fair."),
    ("Copy-mutate", "A classic model of how cuisines evolve (copy a recipe, swap ingredients). Our baseline, not our "
     "contribution."),
    ("Fidelity scorer", "Our score for “how much does this recipe look like the cuisine it claims?”: a cuisine classifier "
     "plus ingredient co-occurrence plus flavor-sharing consistency."),
    ("Best-of-N", "Let the AI write N recipes and keep the one the scorer likes most."),
    ("Critique-and-revise", "Send the scorer's feedback back to the AI and ask it to fix the recipe while keeping the "
     "dish."),
    ("Independent classifier", "A separate model, trained on different data, that judges the results so we never grade "
     "our own scorer."),
    ("AUC", "How well a score separates real recipes from impostors: 0.5 = coin toss, 1 = perfect."),
]  # fmt: skip

# ---------------------------------------------------------------------------------------------------------------- Q&A
QA = [
    ("Why should we trust your pipeline?",
     f"Because we first reproduced Singh & Bagler (2018): {REPL['sign_agreement']}/{REPL['n']} cuisines with the same "
     f"sign, rank correlation {REPL['spearman']:.2f}, and the same top contributors (Korea: {KOREA_TOP['name']}). Only "
     "then did we look at AI recipes."),
    ("Your Korea Z is different from the paper. Why?",
     f"Ours is {Z_KOREA_OURS:.0f}, published {Z_KOREA_PUB:.0f}. FlavorDB has been updated since 2018, so the flavor "
     "molecules changed a little. The signs and the ranking of all 22 cuisines are the same, which is what matters."),
    ("Why compare 160 AI recipes with 160 real recipes?",
     "Because food-pairing numbers depend on sample size: a small sample loses rare ingredients. So each AI cuisine is "
     f"compared with real samples of the same size. Even real samples match the full cuisine's sign only "
     f"{pct(GEN_SIGN)} of the time, which is the realistic yardstick."),
    ("Do AI models make every cuisine pair flavors the same way?",
     "No. There is no common drift; the mistakes differ by model and by cuisine. Pairing direction matches the real "
     f"cuisine in {span([SIGN[m] for m in ORDER], '{:.0%}')} of cuisines, depending on the model."),
    ("Do AI models exaggerate stereotypes?",
     f"Not on average: the stereotype index is {span(SAI.values())} (1 = like real recipes). What they do is narrow the "
     f"cuisine: about {span(DIV.values())} of the ingredient variety of real samples."),
    ("Why is the real fingerprint similarity only 0.75 and not 1?",
     "Fingerprints are noisy with 160 recipes. So 0.75 is the best a real sample of that size reaches; AI reaches "
     f"{span(FP.values())}."),
    ("Which model is best?",
     f"It depends on the measure. Llama matches pairing direction most often ({pct(SIGN[LLAMA])}), Qwen keeps the most "
     f"variety ({DIV[QWEN]:.2f}), Gemini has the closest fingerprint ({FP[GEM]:.2f}, on the {AUD[GEM]['regions']} "
     "cuisines it covered), and GPT-6-Luna, given two rounds of feedback, reached the highest recognition. "
     "No model is best on everything."),
    ("How did you turn AI ingredient text into CulinaryDB ingredients?",
     f"With CulinaryDB's own mapping method, which we rebuilt and tested on CulinaryDB's lines: {ALIAS:.1%} correct. We "
     "did not use an AI for this step, so the mapping cannot add AI bias to what we measure."),
    ("How does the fidelity scorer work, and why is the classifier simple (linear)?",
     "Three signals with equal weights: a cuisine classifier, ingredient co-occurrence, and flavor-sharing "
     f"consistency. A linear model explains itself: it can name the ingredients that make a recipe untypical, which is "
     f"the feedback we give the AI. The right cuisine is in its top 3 for {pct(SC['test_top3'])} of recipes (22 cuisines)."),
    ("Did steering really work?",
     f"Yes. With feedback, recognition rose from {pct(R0L)} to {pct(R2L)} (GPT-6-Luna, 2 rounds) and {pct(R0Q)} to "
     f"{pct(R1Q)} (Qwen, 1 round), keeping about {span(KEEP, '{:.0%}')} of the original ingredients. Best-of-8 raised it "
     f"for every model (GPT-6-Luna {pct(BON1[LUNA])} to {pct(BON8[LUNA])}), but it also lowers variety, so feedback is "
     "the better method."),
    ("Aren't you grading the AI with your own scorer?",
     "No. The judge is an independent classifier, a different model trained on different data, and we also check "
     "measures that have nothing to do with the scorer (ingredient use, fingerprint, pairing direction)."),
    ("Are AI recipes just copies of web recipes?",
     f"No more than real recipes are: the near-copy rate is {span(NEARCOPY.values())} for the AI models and "
     f"{GEN['near_copy_rate']:.2f} for real recipes."),
    ("Why these four models?",
     "The proposal asked for at least three, with at least one company model and two open families. We used two of "
     f"each, all with the same prompts and settings. Total cost: ${SPENT:.2f}; the open models ran on free Kaggle GPUs."),
    ("What is new compared with earlier work?",
     "Earlier work measured real cuisines. We apply the lab's fingerprint framework to AI-written recipes for the first "
     f"time, across all 22 regions with {D['total_recipes']:,} recipes and four models, add a fair equal-size "
     "comparison, and use the fingerprint to steer the AI. Copy-mutate and the null models are methods we reuse, not "
     "our contributions."),
    ("What would you do next?",
     "A scale study (a smaller and a larger model of the same family), more model families, all 22 cuisines for "
     "Gemini (its free quota limited it to 8), and human taste tests."),
    ("Can someone reproduce this?",
     "Yes. Everything runs from scripts with fixed seeds, the prompts are frozen, every AI call is logged, and the code, "
     f"data, tests and app are on GitHub: {GITHUB}."),
]  # fmt: skip

# ------------------------------------------------------------------------------------------- numbers cheat sheet
NUMBERS = [
    ("Real recipes / cuisines", f"{N_REAL:,} / 22 (CulinaryDB)"),
    ("AI recipes / models", f"{D['total_recipes']:,} / {D['n_models']} (GPT-6-Luna, Gemini, Qwen, Llama)"),
    ("Ingredient mapping accuracy", f"{ALIAS:.1%}"),
    ("Replication", f"{REPL['sign_agreement']}/{REPL['n']} signs, ρ = {REPL['spearman']:.2f}; Korea #1: "
                    f"{KOREA_TOP['name']} ({KOREA_TOP['chi_percent']:.0f}%)"),
    ("Pairing direction matches", f"AI {span([SIGN[m] for m in ORDER], '{:.0%}')} · real samples {pct(GEN_SIGN)}"),
    ("Fingerprint similarity", f"AI {span(FP.values())} · real samples {GEN['fingerprint_spearman']:.2f}"),
    ("Signature pairs shared", f"AI {span(NETV.values())} · real samples {NET['real@160']['median']:.2f}"),
    ("Ingredient variety", f"AI {span(DIV.values())} of real (1.00)"),
    ("Stereotype index", f"AI {span(SAI.values())} (1 = like real)"),
    ("Scorer AUC", f"{AUC['other_cuisines']['composite']:.2f} / {AUC['copy_mutate']['composite']:.2f} / "
                   f"{AUC['random_null']['composite']:.2f} (other cuisines / copy-mutate / random)"),
    ("Feedback (critique-and-revise)", f"GPT-6-Luna {pct(R0L)} → {pct(R2L)}; Qwen {pct(R0Q)} → {pct(R1Q)}"),
    ("Best-of-8", ", ".join(f"{NAMES[m].split()[0]} {pct(BON1[m])} → {pct(BON8[m])}" for m in (LUNA, GEM, QWEN))),
    ("Spend", f"${SPENT:.2f} (OpenAI); everything else free"),
]  # fmt: skip
