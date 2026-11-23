"""Build the Week-14 presentation pack from the pipeline outputs, with one command.

    .venv/bin/python presentation/build.py              # everything
    .venv/bin/python presentation/build.py --no-poster  # skip the LaTeX poster

Outputs (presentation/): slides.pptx, poster.pdf, findings_summary.md, qa_prep.md, demo_script.md.
Intermediates: presentation/_build/ (formula images; poster.tex with the figures it includes; tectonic log).

Every number comes from presentation/facts.py, which reads results/summary.json, tables/*.csv,
data/interim/steering/scorer_metrics.json and figures/*.png (plus a few reference and design files listed there).
Re-run analysis/summarize.py first when new data arrive; anything still missing prints as "[pending: ...]".

Requirements: python-pptx in the project venv (`uv pip install --python .venv/bin/python python-pptx`; it is not in
pyproject.toml), and tectonic (https://tectonic-typesetting.github.io) on PATH for the poster.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import jinja2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import extras  # noqa: E402
from deck import pct_rng, rng_text  # noqa: E402
from deck2 import build_deck2  # noqa: E402
from facts import REPO, build_facts, f, is_pending, join_and, minus, n0, pct, pending, pv, sf, tag  # noqa: E402

BUILD = HERE / "_build"
TEMPLATES = HERE / "templates"


def run_demo_probe() -> dict | None:
    try:
        res = subprocess.run([sys.executable, str(HERE / "demo_probe.py")], capture_output=True, text=True,
                             timeout=600, cwd=REPO)  # fmt: skip
        return json.loads(res.stdout) if res.returncode == 0 and res.stdout.strip() else None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return None


def helpers() -> dict:
    return {"f": f, "sf": sf, "pct": pct, "n0": n0, "pv": pv, "minus": minus, "join_and": join_and,
            "pending": pending, "tag": tag, "rng_text": rng_text, "pct_rng": pct_rng, "is_pending": is_pending,
            "rng": lambda pair, nd=2: rng_text(pair[0], pair[1], nd), "pctr": lambda pair: pct_rng(pair[0], pair[1])}  # fmt: skip


PREP = HERE / "prep"
PREP_ORDER = ["00_overview", "01_speaking_script", "02_key_findings", "03_methods_explained", "04_whats_new",
              "05_demo_walkthrough", "06_professor_questions", "07_limitations", "08_closing"]  # fmt: skip
_LABELS = {"HOW TO READ": "How to read", "IF ASKED": "If asked", "RESULT": "Result", "RESULTS": "Results",
           "STATUS": "Status", "DEVIATION TO DISCLOSE": "Deviation to disclose", "DEVIATION": "Deviation",
           "CAVEAT": "Caveat", "PRIMARY TEST": "Primary test", "THE STORY": "The story", "LEFT": "Left",
           "RIGHT": "Right", "RIGHT, TOP": "Right, top", "RIGHT, BOTTOM": "Right, bottom", "TOP RIGHT": "Top right",
           "BOTTOM RIGHT": "Bottom right", "THE SCORER": "The scorer", "BEST-OF-N": "Best-of-N",
           "CONSEQUENCE FOR LLMs": "Consequence for LLMs", "FINAL": "Final", "PROVISIONAL": "Provisional",
           "WHAT IS NEW vs BASELINE": "What is new vs baseline", "PROPRIETARY VS OPEN": "Proprietary vs open",
           "SCALE": "Scale", "ONE-LINER": "One-liner", "HOW TO READ THE DECK": "How to read the deck", "DEMO": "Demo", "COPY-MUTATE": "Copy-mutate", "KEY": "Key"}  # fmt: skip


def _secs(notes: str) -> int:
    m = re.search(r"about (\d+) s", notes or "")
    return int(m.group(1)) if m else 30


def _say(notes: str) -> str:
    """Speaker notes -> Markdown paragraphs (drop the timing header, unwrap lines, soften the CAPS labels)."""
    text = re.sub(r"^WHAT TO SAY \(about \d+ s[^)]*\)\s*", "", (notes or "").strip())
    paras = [" ".join(ln.strip() for ln in p.splitlines()) for p in re.split(r"\n\s*\n", text)]
    out = []
    for p in paras:
        lines = re.split(r"(?<=[.)\]])\s+(?=[A-Z][A-Z ,\-/vs]{2,}[A-Z](?: \([^)]*\))?:)", p)
        for ln in lines:
            m = re.match(r"^([A-Z][A-Z ,\-/vs]{2,}[A-Z])((?: \([^)]*\))?):\s*(.*)$", ln)
            if m and m.group(1) in _LABELS:
                ln = f"**{_LABELS[m.group(1)]}{m.group(2)}:** {m.group(3)}"
            out.append(ln)
    return "\n\n".join(o for o in out if o.strip())


def _status_md(text: str) -> str:
    return re.sub(r"\[pending:\s*([^\]]*)\]", r"*(status: \1)*", text)


def render_prep(F: dict, demo: dict | None, outline: list[dict], example: dict | None) -> list[Path]:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader([TEMPLATES / "prep", TEMPLATES]),
                             undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)  # fmt: skip
    env.globals.update(helpers())
    env.filters["regex_secs"] = _secs
    env.filters["strip_say"] = _say
    PREP.mkdir(exist_ok=True)
    out = []
    for name in PREP_ORDER:
        text = env.get_template(f"{name}.md.j2").render(F=F, demo=demo, outline=outline, steering_example=example)
        text = _status_md(minus(re.sub(r"\n{3,}", "\n\n", text)))
        path = PREP / f"{name}.md"
        path.write_text(text)
        out.append(path)
    return out


def export_prep_pdf(paths: list[Path], F: dict) -> str:
    pandoc = shutil.which("pandoc") or ("/opt/homebrew/bin/pandoc" if Path("/opt/homebrew/bin/pandoc").exists() else None)
    tect = shutil.which("tectonic") or ("/opt/homebrew/bin/tectonic" if Path("/opt/homebrew/bin/tectonic").exists() else None)
    if not (pandoc and tect):
        return "not exported (pandoc or tectonic missing)"
    body = "\n\n\\newpage\n\n".join(p.read_text() for p in paths)
    combined = BUILD / "prep_pack.md"
    combined.write_text(body)
    pdf = PREP / "prep_pack.pdf"
    cmd = [pandoc, str(combined), "-o", str(pdf), f"--pdf-engine={tect}", "--toc", "--toc-depth=1",
           "-V", "mainfont=Arial Unicode MS", "-V", "monofont=Arial Unicode MS", "-V", "geometry:margin=2cm",
           "-V", "fontsize=11pt", "-V", "papersize=a4", "-V", "colorlinks=true", "-V", "linkcolor=black",
           "-M", "title=Rehearsal pack: Do AI Chefs Respect Culinary Fingerprints?",
           "-M", f"subtitle=Group 44 · BIO544 Computational Gastronomy · data {F['generated_date']}"]  # fmt: skip
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=BUILD)
    (BUILD / "prep_pack.log").write_text(res.stdout + "\n" + res.stderr)
    return pdf.name if res.returncode == 0 and pdf.exists() else f"export failed (see _build/prep_pack.log): {res.stderr[-300:]}"


def render_markdown(F: dict, demo: dict | None) -> list[Path]:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES), undefined=jinja2.StrictUndefined,
                             trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)  # fmt: skip
    env.globals.update(helpers())
    out = []
    for name in ("findings_summary.md", "qa_prep.md", "demo_script.md"):
        example = None
        with contextlib.suppress(OSError, json.JSONDecodeError):
            example = json.loads((REPO / "results" / "steering_example.json").read_text())
        text = env.get_template(f"{name}.j2").render(F=F, demo=demo, steering_example=example)
        text = re.sub(r"\n{3,}", "\n\n", text)
        path = HERE / name
        path.write_text(_status_md(minus(text)))
        out.append(path)
    return out


# ------------------------------------------------------------------------------------------------- poster (LaTeX)
_TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
                "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}  # fmt: skip


def tex(value) -> str:
    """Escape text for LaTeX; **bold** and [pending: ...] markers become LaTeX markup."""
    s = minus(str(value))
    parts = re.split(r"(\*\*.+?\*\*|\[pending:[^\]]*\])", s)
    out = []
    for p in parts:
        if not p:
            continue
        esc = "".join(_TEX_SPECIAL.get(ch, ch) for ch in (p[2:-2] if p.startswith("**") else p))
        if p.startswith("**"):
            out.append(r"\textbf{" + esc + "}")
        elif p.startswith("[pending:"):
            inner = "".join(_TEX_SPECIAL.get(ch, ch) for ch in p[len("[pending:"):-1].strip())
            out.append(r"\pend{" + inner + "}")
        else:
            out.append(esc)
    return "".join(out)


def build_poster(F: dict) -> tuple[Path | None, str]:
    pdir = BUILD / "poster"
    pdir.mkdir(parents=True, exist_ok=True)
    figs = {}
    for key, path in F["figures"].items():
        if path is not None:
            dst = pdir / f"{key}.png"
            shutil.copyfile(path, dst)
            figs[key] = dst.name
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES), undefined=jinja2.StrictUndefined,
                             block_start_string="<%", block_end_string="%>", variable_start_string="<<",
                             variable_end_string=">>", comment_start_string="<#", comment_end_string="#>",
                             trim_blocks=True, lstrip_blocks=True)  # fmt: skip
    env.globals.update(helpers())
    env.filters["tex"] = tex
    (pdir / "poster.tex").write_text(env.get_template("poster.tex.j2").render(F=F, figs=figs))
    exe = shutil.which("tectonic") or ("/opt/homebrew/bin/tectonic" if Path("/opt/homebrew/bin/tectonic").exists()
                                       else None)  # fmt: skip
    if exe is None:
        return None, "tectonic not found: poster.tex written to presentation/_build/poster/ but not compiled"
    res = subprocess.run([exe, "-X", "compile", "poster.tex"], cwd=pdir, capture_output=True, text=True, timeout=900)
    (pdir / "tectonic.log").write_text(res.stdout + "\n" + res.stderr)
    if res.returncode != 0 or not (pdir / "poster.pdf").exists():
        return None, f"tectonic failed (see presentation/_build/poster/tectonic.log): {res.stderr[-400:]}"
    out = HERE / "poster.pdf"
    shutil.copyfile(pdir / "poster.pdf", out)
    size = "unknown"
    if shutil.which("pdfinfo"):
        info = subprocess.run(["pdfinfo", str(out)], capture_output=True, text=True).stdout
        m = re.search(r"Page size:\s*(.+)", info)
        pages = re.search(r"Pages:\s*(\d+)", info)
        if m:
            size = m.group(1).strip() + (f", {pages.group(1)} page(s)" if pages else "")
    return out, size


def export_pdf(pptx: Path) -> str:
    exe = shutil.which("soffice") or ("/opt/homebrew/bin/soffice" if Path("/opt/homebrew/bin/soffice").exists() else None)
    if exe is None:
        return "not exported (LibreOffice soffice not found)"
    res = subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", str(pptx.parent), str(pptx)],
                         capture_output=True, text=True, timeout=600)  # fmt: skip
    out = pptx.with_suffix(".pdf")
    return str(out.name) if res.returncode == 0 and out.exists() else f"export failed: {res.stderr[-300:]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-poster", action="store_true")
    ap.add_argument("--poster-only", action="store_true", help="rebuild poster.pdf only (slides and prep untouched)")
    args = ap.parse_args()
    F = build_facts()
    if args.poster_only:
        poster, size = build_poster(F)
        print(f"data snapshot: {F['generated_utc']}  (results/summary.json)")
        print(f"poster.pdf: {size if poster else 'NOT BUILT — ' + size}")
        return 0
    demo = run_demo_probe()
    if demo is None:
        F["warnings"].append("demo probe failed (app artifacts?): the demo script shows [pending] values")
    BUILD.mkdir(exist_ok=True)
    X = {}
    for key, fn in (("counts", lambda: extras.data_counts(F["regions"]["all"])),
                    ("pair", lambda: extras.recipe_pair(F["complete_models"],
                                                        {k: m["name"] for k, m in F["models"].items()})),
                    ("network", extras.signature_network)):  # fmt: skip
        try:
            X[key] = fn()
        except Exception as exc:  # noqa: BLE001 - a missing extra must not stop the build
            X[key] = None
            F["warnings"].append(f"extras.{key} failed: {exc}")
    F["extras"] = X
    deck = build_deck2(F, HERE / "slides.pptx", BUILD / "assets", demo, X)
    pdf_note = export_pdf(HERE / "slides.pptx")
    md = render_markdown(F, demo)
    example = None
    with contextlib.suppress(OSError, json.JSONDecodeError):
        example = json.loads((REPO / "results" / "steering_example.json").read_text())
    prep = render_prep(F, demo, deck["outline"], example)
    prep_pdf = export_prep_pdf(prep, F)
    poster, size = (None, "skipped") if args.no_poster else build_poster(F)
    print(f"data snapshot: {F['generated_utc']}  (results/summary.json)")
    print(f"slides.pptx: {deck['slides']} slides = {deck['main']} main + {deck['backup']} backup")
    print(f"slides.pdf: {pdf_note}")
    print(f"poster.pdf: {size if poster else 'NOT BUILT — ' + size}")
    print("markdown:", ", ".join(p.name for p in md))
    print(f"prep/: {len(prep)} Markdown files + {prep_pdf}")
    print("pending items:")
    for p in F["pending"]:
        print(f"  - {p['item']}: {p['detail']}")
    for w in F["warnings"]:
        print(f"WARNING: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
