"""FlavorPrint (D5) smoke tests with Streamlit's AppTest.

They are skipped unless app/artifacts is built (app/build_artifacts.py). The headline numbers are HTML metric cards
(``data-metric="<key>"``), read back from the rendered Markdown.
"""

import re
from pathlib import Path

import pytest

ARTIFACTS = Path(__file__).resolve().parents[1] / "app" / "artifacts" / "core.pkl"
pytestmark = pytest.mark.skipif(not ARTIFACTS.exists(), reason="app artifacts not built")
CARD = re.compile(r'data-metric="(\w+)"><div class="fp-lab">[^<]*</div><div class="fp-val[^"]*">([^<]*)<')
SUBTABS = ["Ingredients", "Flavor sharing", "Flavor network", "Cuisine fidelity", "Stereotype", "Suggestions"]


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(ARTIFACTS.parents[1] / "streamlit_app.py"), default_timeout=180).run()


def cards(at) -> dict[str, str]:
    return {k: v for m in at.markdown for k, v in CARD.findall(m.value)}


def text(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_example_recipe_is_audited_without_errors(app):
    assert not app.exception
    assert not app.error
    c = cards(app)
    assert {"fidelity", "ns", "top", "matched"} <= set(c)
    assert c["top"] == "Korea"  # the default example is a Korean kimchi stew
    assert c["matched"] == "8/9"  # gochugaru has no CulinaryDB entity
    labels = [t.label for t in app.tabs]
    assert all(label in labels for label in SUBTABS)
    assert "not found in CulinaryDB" in text(app)  # the unmatched line is flagged in the mapping table


def test_off_cuisine_preset_gets_low_fidelity_and_flags(app):
    app.button_group(key="preset").set_value("spaghetti").run()
    assert not app.exception
    assert app.selectbox(key="claimed").value == "Korea"
    c = cards(app)
    assert float(c["fidelity"]) < 10  # fidelity percentile vs genuine Korean recipes
    assert c["top"] == "Italy"
    body = text(app)
    assert "Unusual for this cuisine" in body and "Hard wheat" in body


def test_pasted_recipe_with_claimed_cuisine(app):
    app.text_area(key="recipe_text").set_value(
        "400 g spaghetti\n800 g canned tomatoes\n3 tbsp olive oil\n2 cloves garlic\n1 handful basil\n40 g parmesan"
    ).run()
    app.selectbox(key="claimed").set_value("Italy").run()
    assert not app.exception
    c = cards(app)
    assert c["top"] == "Italy" and float(c["fidelity"]) > 10


def test_stereotype_flag_on_signature_heavy_recipe(app):
    app.button_group(key="preset").set_value("chana").run()
    assert not app.exception
    assert app.selectbox(key="claimed").value == "Indian Subcontinent"
    assert cards(app)["signature"] == "6/10"
    assert '<div class="fp-callout warn"><b>Stereotype flag.</b>' in text(app)


def test_dashboard_renders_with_reference_numbers(app):
    c = cards(app)
    assert c.get("replication") == "22/22"  # Singh & Bagler (2018) signs, from results/summary.json
    assert len(app.dataframe) >= 2  # scorecard and region drivers
    assert not app.exception


def test_too_few_ingredients_is_handled(app):
    app.text_area(key="recipe_text").set_value("1 unicorn horn").run()
    assert not app.exception
    assert any("Fewer than two lines" in w.value for w in app.warning)
