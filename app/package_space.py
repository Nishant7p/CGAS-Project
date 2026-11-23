"""Assemble the Hugging Face Space folder for FlavorPrint (D5). It writes local files only and publishes nothing.

Output: app/space/ containing README.md (the Space card), Dockerfile, requirements.txt (pinned to the versions this
repo runs, so the pickled scorer loads), streamlit_app.py, .streamlit/config.toml (theme), artifacts/ and a copy of
src/flavoraudit.
Deploying it (e.g. `huggingface-cli upload <user>/flavorprint app/space --repo-type space`) needs the user's OK.

Usage (repo root, after app/build_artifacts.py): .venv/bin/python app/package_space.py
"""

from __future__ import annotations

import importlib.metadata as md
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app"
SPACE = APP / "space"
JUNK = shutil.ignore_patterns("__pycache__", ".DS_Store")
PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "inflect",
    "networkx",
    "matplotlib",
    "streamlit",
    "plotly",
    "pyarrow",
]

CARD = """---
title: FlavorPrint
emoji: 🍲
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8501
license: cc-by-nc-sa-4.0
short_description: Does a recipe respect its cuisine's culinary fingerprint?
---

# FlavorPrint

Audit any recipe against the culinary fingerprints of the 22 world cuisines in CulinaryDB:

- flavor sharing (Ns), with the cuisine's real distribution;
- cuisine prediction and a fidelity score (classifier + ingredient co-occurrence + flavor-sharing consistency);
- stereotype flags and fingerprint-guided suggestions.

Part of *Do AI Chefs Respect Culinary Fingerprints?* (Group 44, BIO544 Computational Gastronomy, IIIT-Delhi, 2026).

- **Code:** MIT (`flavoraudit`).
- **Data:** the artifacts derive from CulinaryDB and FlavorDB (CoSyLab, IIIT-Delhi, CC BY-NC-SA 3.0) and are
  CC BY-NC-SA 4.0. Cite Singh & Bagler (2018) and Garg et al. (2018).
"""


def main() -> int:
    if not (APP / "artifacts" / "core.pkl").exists():
        raise SystemExit("run app/build_artifacts.py first")
    if SPACE.exists():
        shutil.rmtree(SPACE)
    SPACE.mkdir()
    (SPACE / "README.md").write_text(CARD, encoding="utf-8")
    shutil.copy(APP / "Dockerfile", SPACE / "Dockerfile")
    shutil.copy(APP / "streamlit_app.py", SPACE / "streamlit_app.py")
    shutil.copytree(APP / ".streamlit", SPACE / ".streamlit", ignore=JUNK)  # theme (config.toml)
    shutil.copytree(APP / "artifacts", SPACE / "artifacts", ignore=JUNK)
    shutil.copytree(REPO / "src" / "flavoraudit", SPACE / "flavoraudit", ignore=JUNK)
    reqs = [f"{p}=={md.version(p)}" for p in PACKAGES]
    (SPACE / "requirements.txt").write_text("\n".join(reqs) + "\n")
    size = sum(p.stat().st_size for p in SPACE.rglob("*") if p.is_file()) / 1e6
    print(f"Space folder: {SPACE.relative_to(REPO)} ({size:.1f} MB)\n" + "\n".join(reqs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
