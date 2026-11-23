# FlavorPrint: live demo (2 minutes)

## Launch (one command, from the repo root)

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

It opens http://localhost:8501. If port 8501 is busy, add `--server.port 8502`.

**Before the talk:** open the page once and wait for the results to appear. The first load on a fresh server takes about
5–7 s while the flavor network and scorer load. After that, each click re-renders in well under a second. Leave the
**Audit a recipe** tab open on **Kimchi jjigae** (the default). Browser zoom of about 110–125% reads well on a projector,
and the layout stays two-column down to about 640 px.

All numbers below are what the app shows with the artifacts in `app/artifacts/`. They were computed by the app's own
`audit()` and `what_if()` functions, not typed by hand.

## Script

| Time | Click | Point at | Say |
|---|---|---|---|
| 0:00 | (nothing; the page is open) | The headline under **FlavorPrint** | "FlavorPrint asks whether a recipe looks like the cuisine it claims, using the same aliaser, flavor network and fidelity scorer we used to audit 11,656 LLM-written recipes." |
| 0:10 | (Kimchi jjigae is selected) | The verdict band **"Kimchi jjigae reads as Korea, the cuisine it claims"** (teal bar) with the big **Fidelity percentile 66** on its gauge, then the four cards: **Korea 68%**, **Flavor sharing Ns 28.0**, **Signature ingredients 5/10**, **8/9 matched** | "Fidelity 66 means it scores above about two-thirds of 61 genuine Korean recipes that the scorer never saw in training." |
| 0:25 | Sub-tab **Ingredients** (already open) | Row 4, **gochugaru**: "not found in CulinaryDB" | "Every line is mapped to a CulinaryDB entity. Gochugaru has none, so it is left out of every score rather than guessed." |
| 0:35 | Sub-tab **Flavor sharing** | The red **this recipe 28.0** line against the grey real-recipe histogram, then the blue box **Z = −57** | "Korean cuisine pairs contrastingly: real recipes share fewer compounds than random ones built from the same ingredients, and this stew is at percentile 22 of real Korean recipes." |
| 0:50 | Sub-tab **Flavor network** | The thickest edge and the **Strongest pairs** panel: **Garlic + Cabbage, 106** | "Each edge is the number of flavor compounds two ingredients share; Ns is the average edge." |
| 1:00 | Example button **Spaghetti as Korean** | Verdict **"claimed as Korea but reads as Italy"** (orange bar), **Fidelity percentile 2**, **Italy 98%** | "Same app, an Italian dish claimed as Korean: it scores below 98% of genuine Korean recipes." |
| 1:10 | Sub-tab **Cuisine fidelity** | Component bars **−3.22, −2.10, −1.50**; **Hard wheat** and **Tomato** left of the dashed cut | "All three components sit below the typical genuine recipe, and the classifier names the two ingredients that pull it away from Korea." |
| 1:20 | Sub-tab **Suggestions** | **Unusual for this cuisine: Hard wheat, Tomato**; what-if **Fidelity percentile 10 (+8)** | "This is exactly the critique our critique-and-revise steering sends back to an LLM." |
| 1:30 | Example button **Chana masala** | Orange badge **Stereotype flag** in the verdict and the orange **Signature ingredients 6/10** card | "A recipe can be authentically Indian and still be a caricature." |
| 1:35 | Sub-tab **Stereotype** | Highlighted chips (**6 of 10** signature ingredients) and the bar chart: **98.4%** of real recipes use this many or fewer | "It uses 6 of the cuisine's 10 signature ingredients; at the level of a whole LLM cuisine, this is our Stereotype Amplification Index." |
| 1:45 | Top tab **22-cuisine dashboard** | Card **Replication 22/22** (ρ = 0.98), then the heatmap and the **78%** real-sample card | "We first reproduce Singh & Bagler's signs in all 22 regions. LLM cuisines match the real food-pairing sign in 38–82% of regions, against 78% for random 160-recipe samples of the real cuisines." |
| 2:00 | Back to the slides | | |

### Optional extras for questions

- **Same recipe, different claim:** on the spaghetti example, change **Claimed cuisine** to Italy. The example highlight clears and the verdict turns teal (“reads as Italy, the cuisine it claims”): the score is always relative to the claimed cuisine.

- **Paste your own recipe:** click **Clear**, type one ingredient per line, pick a **Claimed cuisine**, then click
  **Audit recipe**.
- **Dashboard → Region detail:** pick any region (default Korea) to see each model's Z and its top-5 flavor drivers
  (χ%) next to the real cuisine's.
- **About & methods:** the Ns and Z formulas, the fidelity components, data licences and references.

## If something goes wrong

| Symptom | Fix |
|---|---|
| Port 8501 busy | `.venv/bin/streamlit run app/streamlit_app.py --server.port 8502` |
| Page blank or still loading after about 15 s | Reload the browser tab once. |
| No laptop or projector | Show the screenshots in `app/screenshots/` (01 overview, 02 ingredient mapping, 03 flavor network, 04 fidelity, 05 suggestions, 06 dashboard). |

## Checks run before the talk

- `.venv/bin/python -m pytest -q tests/test_app.py` (6 tests) and `.venv/bin/ruff check app tests`.
- Headless browser run: every example, every sub-tab, the dashboard, About, a pasted recipe and a 390 px phone
  width, with no exceptions, error boxes or console errors and no horizontal page scroll.
