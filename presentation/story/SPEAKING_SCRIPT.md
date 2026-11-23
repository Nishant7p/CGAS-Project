# Speaking script: Do AI Chefs Respect Culinary Fingerprints?

From the deck's speaker notes. [DEMO] and [SHOW] are cues for you, not for the slide. Slides 1-14 are the talk (about 40-50 s each, plus a 2-minute demo on slide 12: about 12 minutes); slides 15 onward are backups for questions.

## Slide 1: Do AI Chefs Respect Culinary Fingerprints?

[2023355 · 0:30]
Good morning, we are Group 44. Our question is simple: when an AI writes a “traditional” recipe, does it really cook like that cuisine? We checked it with the methods from this course.

## Slide 2: Do AI chefs cook like the cuisine they claim?

*On the slide:* A cuisine has a measurable fingerprint, so we can check AI recipes against it

[2023355 · 0:50]
People ask chatbots for authentic recipes every day, but an AI learns from text, not from tasting food. Every cuisine has a fingerprint: the way it combines ingredients and flavors, and this lab has shown how to measure it. We look at three layers: flavor pairing, the key ingredients, and the category pattern.

Tip: Analogy: it is like handwriting. We do not check the words (dish names), we check the handwriting.

## Slide 3: What we built: five deliverables

*On the slide:* As part of the project we built five things, all on GitHub

[2023355 · 0:50]
As part of this project we built five things. One: LLMCuisine-22, a dataset of 11,656 AI-written recipes for all 22 cuisines, with every AI call saved. Two: an audit that measures AI cuisines with the same flavor tools as real ones. Three: a steering method that uses the fingerprint to make AI recipes more authentic. Four: FlavorAudit, a Python library with all our methods. And five: FlavorPrint, a web app where anyone can check a recipe; we will show it live later.

Tip: All five are on GitHub; the link is on the first and last slides.

## Slide 4: 45,772

*On the slide:* 45,772 real recipes against 11,656 recipes written by 4 AI models

[2023355 · 0:50]
We compared 45,772 real recipes from CulinaryDB with 11,656 recipes written by four AI models: GPT-6-Luna and Gemini (company models) and Qwen and Llama (open models). Every model got exactly the same prompts for every cuisine, and every call was saved. We mapped each ingredient line to CulinaryDB with CulinaryDB's own method, which is 98% accurate.

Tip: This dataset is deliverable D1, LLMCuisine-22.

## Slide 5: How did we do it?

*On the slide:* One pipeline from real cuisines to audited and improved AI recipes

[2023355 · 0:40]
This is our pipeline. We start from real cuisines and first check that our code reproduces the published result. Then we generate AI recipes, map their ingredients, and measure AI and real cuisines with exactly the same code. Finally, we use what we learned to steer the AI. The bottom row shows the five deliverables.

Hand over: Now my teammate will explain how we measure food pairing and what we found.

## Slide 6: How do we measure food pairing?

*On the slide:* Compare real recipes with a ‘blindfolded chef’ who picks from the same pantry

[2023477 · 0:50]
For every recipe we count how many flavor molecules its ingredient pairs share. That number is Ns. Then we compare with a “blindfolded chef” who picks random ingredients from the same pantry. Italian recipes share more flavors than chance; Korean recipes share fewer. The gap is ΔNs, and Z is that gap in standard units.

Tip: Point at the black curve (real) against the grey shape (blindfolded chef).

## Slide 7: 22/22

*On the slide:* Yes: all 22 cuisines match Singh & Bagler (2018), rank correlation ρ = 0.98

[2023477 · 0:40]
Before testing any AI recipe, we checked our code against Singh and Bagler's paper. All 22 cuisines get the same sign and the ranking matches with a correlation of 0.98. Korea's top flavor driver is soybean sauce, exactly as published. So our measuring tool works.

## Slide 8: Results at a glance: AI vs real cuisines

*On the slide:* Each AI cuisine (160 recipes) is compared with genuine samples of 160 real recipes

[2023477 · 1:00]
This chart is the whole audit in one picture. Grey is what real samples of the same size achieve, our fair yardstick. On pairing direction, Llama matches the real cuisine in 82% of cuisines, as often as real samples (78%). On the fingerprint, the signature pairs and the variety of ingredients, AI reaches only about a third to a half of the real level.

Tip: Read the grey bar first, then each model.

## Slide 9: 0.28–0.37

*On the slide:* Partly: they agree on some categories and differ on others

[2023477 · 0:50]
Here is the fingerprint of Korean food, category by category. AI agrees with real cooking on some rows, such as fungus and legumes, and not on others, such as meat and additives. Across all cuisines, AI fingerprints match the real ones at about 0.28–0.37, against 0.75 for real samples. AI gets the broad strokes, not the fine details.

## Slide 10: 0.11–0.13

*On the slide:* They keep the classic pairs and miss many regional signatures

[2023477 · 0:50]
These are Korea's ten most distinctive ingredient pairs. Soybean sauce with sesame appears in every AI cuisine, but shrimp with fish or carrot with potato do not. Across all cuisines, AI shares about 0.11–0.13 of the real top pairs, against 0.30 for real samples. And it finds the key ingredient: soybean sauce is number one for GPT-6-Luna and Qwen, as in real recipes.

## Slide 11: From audit to fix: a fidelity scorer

*On the slide:* One score for “how much does this recipe look like the cuisine it claims?”

[2023477 · 0:50]
So we turned the audit into a fix: a fidelity scorer. It combines three signals: which cuisine the recipe looks like, whether its ingredient pairs appear together in real recipes, and whether its flavor sharing is typical. It tells real recipes from impostors well: AUC 0.87 to 0.94. We use it in two ways: keep the best of several recipes, or give the AI feedback to revise its recipe.

## Slide 12: Does fingerprint feedback make AI recipes more authentic?

*On the slide:* Yes: recipes recognised as their claimed cuisine rise from 45% to 58% after two rounds of feedback

[2023477 · 1:00]
And it works. With feedback, recipes recognised as the right cuisine rise from 45% to 58% for GPT-6-Luna in two rounds, and from 47% to 55% for Qwen in one round, while keeping about 77%–80% of the original ingredients. Picking the best of eight also helps every model. Example: Cochinillo Asado (Roast Suckling Pig) moved from the 17th to the 26th percentile after adding garlic, parsley, pepper.

Hand over: Now my teammate will show our app live.

## Slide 13: FlavorPrint: audit any recipe in the browser

*On the slide:* Paste a recipe, pick its cuisine, see its fingerprint

[2023355 · 2:00]
This is FlavorPrint, our web app. You paste a recipe, pick the cuisine it claims, and it shows the ingredient mapping, the flavor network, the fidelity score, a stereotype check and suggestions. [Switch to the browser and follow the demo steps in Part 3.]

DEMO STEPS:
1. Kimchi jjigae (already selected) → Audit recipe: “This genuine Korean stew scores above about two-thirds of real Korean recipes.”
2. Ingredients tab: “Every line is mapped to a CulinaryDB ingredient; unknown ones are left out, not guessed.”
3. Flavor network tab: “Ns is simply the average edge in this network.”
4. Example: Spaghetti as Korean → Audit recipe: “An Italian dish claimed as Korean is caught immediately.”
5. Suggestions tab: “This is exactly the feedback our steering sends to the AI.”
6. Example: Chana masala → Stereotype tab: “A recipe can be authentic and still a caricature: it uses many signature ingredients at once.”
7. 22-cuisine dashboard tab: “Every model on every cuisine, at a glance.” Then go back to the slides.

Tip: If the app does not open, talk over the screenshot on the slide.

## Slide 14: So, do AI chefs respect culinary fingerprints?

[2023355 · 0:40]
So, do AI chefs respect culinary fingerprints? Partly. They get the skeleton right: recipe size, signature ingredients, often the pairing direction and the key flavor drivers. They miss part of the fingerprint, many regional ingredient pairs, and variety. And the fingerprint itself can guide them closer. Next, we want a scale study, more models and human taste tests. Thank you, we are happy to take questions.

## Slide 15: Links: everything we can show

*On the slide:* Click any row  ·  github.com/Nishant7p/CGAS-Project

Jump here to open anything the examiners ask about. GitHub links need the repository pushed; the app link needs the app running on this laptop.

## Slide 16: Backup: FlavorAudit in code

*On the slide:* Everything above as a pip-installable package

All methods are packaged as FlavorAudit: typed, tested, documented, with a command-line tool. The data are downloaded separately (CulinaryDB and FlavorDB are CC BY-NC-SA).

## Slide 17: Backup: where do models get the pairing direction right?

*On the slide:* Green: same direction as the real cuisine

Each square is a model and a cuisine; compared at equal size with genuine samples.

## Slide 18: Backup: which ingredients drive flavor pairing?

*On the slide:* Korea: soybean sauce for real, GPT-6-Luna and Qwen

χ% = change in the cuisine's flavor sharing when one ingredient is removed (Singh & Bagler).

## Slide 19: Backup: how varied are AI cuisines?

*On the slide:* They follow the popular ingredients but lose the long tail

Diversity ratio = unique ingredients relative to genuine samples of the same size.

## Slide 20: Backup: why compare at equal size?

*On the slide:* Small samples lose rare ingredients, so we compare 160 vs 160

Food pairing depends on sample size, so every AI cuisine is compared with genuine samples of the same size.

## Slide 21: Backup: AI vs a copy-mutate model

*On the slide:* AI cuisines beat the classic evolution model on composition and fingerprint

Copy-mutate (Jain & Bagler; Tuwani et al.) is our baseline, not a contribution.

## Slide 22: Backup: does a home-cook persona help?

*On the slide:* The persona prompt changes some measures, by model

Each point is one cuisine: persona minus plain prompt.

## Slide 23: Backup: the full scorecard

*On the slide:* Every source on every measure (medians over cuisines)

Genuine samples set the ceiling; AI models sit between them and copy-mutate.

## Slide 24: Backup: best-of-N in detail

*On the slide:* More candidates, more recognisable recipes

Best-of-N reranks each model's own recipes; the judge is an independent classifier.

## Slide 25: Backup: ingredient networks by cuisine

*On the slide:* Top-100 pair overlap with the real cuisine, every region

Grey bars: the range of genuine 160-recipe samples; markers: AI models.
