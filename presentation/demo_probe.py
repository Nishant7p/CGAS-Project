"""Run FlavorPrint's own audit function on the demo recipes and print the results as JSON.

The demo script (demo_script.md) and the backup slide quote what the app will show on screen. Rather than typing
those values, build.py runs this file in a subprocess: it loads app/streamlit_app.py without starting Streamlit
(the trailing ``main()`` call is removed), reads the app's precomputed artifacts, and audits three recipes:

1. the app's Korean example, claimed as Korea (genuine);
2. the app's Italian example, claimed as Korea (off-cuisine);
3. the app's Indian example, claimed as Indian Subcontinent (stereotype flag).

Usage (repo root): .venv/bin/python presentation/demo_probe.py
"""

from __future__ import annotations

import ast
import json
import logging
import pickle
import re
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app" / "streamlit_app.py"
CORE = REPO / "app" / "artifacts" / "core.pkl"

SCENARIOS = (
    ("genuine", "Korea", "Korea"),
    ("off_cuisine", "Italy", "Korea"),
    ("stereotype", "Indian Subcontinent", "Indian Subcontinent"),
)


def load_app_namespace() -> dict:
    tree = ast.parse(APP.read_text())
    tree.body = [
        n
        for n in tree.body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", "") == "main")
    ]
    ns: dict = {"__file__": str(APP), "__name__": "flavorprint_probe"}
    exec(compile(tree, str(APP), "exec"), ns)  # noqa: S102 - our own app code, trailing main() removed
    return ns


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    ns = load_app_namespace()
    with open(CORE, "rb") as fh:
        core = pickle.load(fh)  # noqa: S301 - artifact built by app/build_artifacts.py in this repo
    m = re.search(r"if share >= ([\d.]+):", APP.read_text())
    flag_cut = float(m.group(1)) if m else None
    names = core["names"]
    out = {"stereotype_cut_percent": flag_cut, "scenarios": {}}
    for key, example, claimed in SCENARIOS:
        title, text = ns["EXAMPLES"][example]
        lines = ns["clean_lines"](text)
        res = ns["audit"](lines, claimed, core)
        table = res["table"]
        unmatched = table.loc[table["Match"] == "no match", "Ingredient line"].tolist()
        fb = res.get("feedback", {})
        share = 100 * res.get("signature_share_at_most", float("nan"))
        out["scenarios"][key] = {
            "example": example,
            "title": title,
            "claimed": claimed,
            "n_lines": len(lines),
            "matched": int((table["Match"] != "no match").sum()),
            "unmatched": unmatched,
            "fidelity_pct": res.get("fidelity_pct"),
            "ns": res.get("ns"),
            "ns_pct": res.get("ns_pct"),
            "top5": [(r, float(p)) for r, p in res.get("top5", [])],
            "components": {k: float(v) for k, v in res["components"].items()} if "components" in res else {},
            "atypical": [names[r] for r in fb.get("atypical", [])],
            "missing": [names[r] for r in fb.get("missing", [])],
            "ns_verdict": fb.get("ns_verdict"),
            "signature_used": res.get("signature_used", []),
            "signature_share_at_most": share,
            "stereotype_flag": bool(flag_cut is not None and share >= flag_cut),
            "z_real": float(core["regions"][claimed]["z_real"]),
        }
    json.dump(out, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
