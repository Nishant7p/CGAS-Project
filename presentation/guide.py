"""Presenter guide PDF: the talk split between the two speakers, the live demo, how to show each deliverable, the ideas
in simple words, Q&A and a numbers cheat sheet. Content comes from presentation/talk_script.py (numbers from the
result files); typeset with tectonic (XeTeX).

Usage (repo root): .venv/bin/python presentation/guide.py   -> presentation/guide/Group44_Presenter_Guide.pdf
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import talk_script as T  # noqa: E402

OUT = HERE / "guide"


def tex(s: str) -> str:
    """Escape plain text for LaTeX (XeTeX handles the Unicode letters; arrows go through math mode)."""
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}  # fmt: skip
    out = "".join(rep.get(ch, ch) for ch in str(s))
    return out.replace("→", r"$\rightarrow$").replace("≥", r"$\geq$")


def path_tex(s: str) -> str:
    """tex() for file paths: allow line breaks after / and _ so long paths wrap inside narrow columns."""
    return tex(s).replace("/", r"/\allowbreak{}").replace(r"\_", r"\_\allowbreak{}")


def speaker_color(name: str) -> str:
    return "SpeakerA" if name == T.SPEAKER_A else "SpeakerB"


def build_tex() -> str:
    A, B = tex(T.SPEAKER_A), tex(T.SPEAKER_B)
    gh = T.GITHUB
    total_a = sum(_secs(s["time"]) for s in T.SLIDES if s["speaker"] == T.SPEAKER_A)
    total_b = sum(_secs(s["time"]) for s in T.SLIDES if s["speaker"] == T.SPEAKER_B)
    p = []
    p.append(r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2cm,top=1.8cm,bottom=2cm]{geometry}
\usepackage{fontspec}
\setmainfont{Helvetica Neue}
\newfontfamily\headfont{Georgia}
\usepackage{xcolor}
\definecolor{Navy}{HTML}{1B2631}\definecolor{Teal}{HTML}{0E7C86}\definecolor{Orange}{HTML}{D55E00}
\definecolor{Grey}{HTML}{5D6D7E}\definecolor{Card}{HTML}{F0F3F4}
\definecolor{SpeakerA}{HTML}{0E7C86}\definecolor{SpeakerB}{HTML}{D55E00}
\usepackage[most]{tcolorbox}
\usepackage{tabularx,booktabs,array,enumitem,titlesec,fancyhdr}
\usepackage[colorlinks=true,urlcolor=Teal,linkcolor=Navy]{hyperref}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\renewcommand{\arraystretch}{1.35}
\titleformat{\section}{\headfont\Large\bfseries\color{Navy}}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\bfseries\large\color{Teal}}{}{0em}{}
\titlespacing*{\section}{0pt}{14pt}{6pt}
\pagestyle{fancy}\fancyhf{}\renewcommand{\headrulewidth}{0pt}
\fancyfoot[L]{\small\color{Grey}Group 44 · BIO544 · Presenter guide}\fancyfoot[R]{\small\color{Grey}\thepage}
\newtcolorbox{slidebox}[2]{colback=Card,colframe=#1,boxrule=0pt,leftrule=4pt,arc=2pt,left=8pt,right=8pt,top=5pt,
  bottom=5pt,before skip=6pt,after skip=6pt,title={#2},coltitle=#1,fonttitle=\bfseries,colbacktitle=Card,
  attach title to upper,after title={\par\smallskip}}
\begin{document}
""")
    # ---- title
    p.append(
        r"{\headfont\fontsize{24}{28}\selectfont\bfseries\color{Navy} Do AI Chefs Respect Culinary Fingerprints?\par}"
    )
    p.append(r"\vspace{2pt}{\large\color{Teal} Presenter guide: script, live demo, deliverables and Q\&A\par}")
    p.append(rf"\vspace{{4pt}}{{\color{{Grey}} Group 44 · {A} and {B} · BIO544 Computational Gastronomy, IIIT-Delhi · "
             rf"Prof. Ganesh Bagler\par}}")  # fmt: skip
    p.append(rf"\vspace{{2pt}}{{\color{{Grey}} Code, data and app: \url{{{gh}}}\par}}")
    p.append(
        r"""\begin{tcolorbox}[colback=Card,colframe=Card,arc=2pt,left=8pt,right=8pt,top=6pt,bottom=6pt]
\textbf{How to use this guide.} (1) Read Part 1 together (5 minutes). (2) Each of you rehearses your own slides in
Part 2: the \textcolor{SpeakerA}{\textbf{teal}} boxes are """
        + A
        + r""", the \textcolor{SpeakerB}{\textbf{orange}}
boxes are """
        + B
        + r""". (3) """
        + tex(T.SPEAKER_A.split(" (")[0])
        + r""" runs the live demo (Part 3).
(4) Read the questions in Part 6 aloud to each other once.
\end{tcolorbox}"""
    )
    # ---- part 1
    p.append(r"\section{The project in one page}")
    p.append(r"\subsection{The question}")
    p.append(tex("When an AI writes a “traditional” recipe, does it cook like the cuisine it claims? Every cuisine has a "
                 "measurable culinary fingerprint (how it combines ingredients and flavors). We measured AI recipes with "
                 "the same tools Prof. Bagler's lab uses for real cuisines, and then used the fingerprint to make AI "
                 "recipes more authentic."))  # fmt: skip
    p.append(r"\subsection{What we did}")
    steps = [
        f"Checked our code against Singh & Bagler (2018): all {T.REPL['sign_agreement']} cuisines reproduced.",
        f"Asked {T.D['n_models']} AI models (GPT-6-Luna, Gemini, Qwen, Llama) for recipes from 22 world cuisines: "
        f"{T.D['total_recipes']:,} recipes, every call saved.",
        f"Mapped every ingredient to CulinaryDB with CulinaryDB's own method ({T.ALIAS:.0%} accurate).",
        "Measured AI and real cuisines with the same code: flavor pairing, key ingredients, category fingerprint, "
        "ingredient pairs, variety and stereotypes, always against real samples of the same size.",
        "Built a fidelity scorer and used it to steer the AI (best-of-N and critique-and-revise).",
    ]
    p.append(r"\begin{enumerate}[leftmargin=1.4em,itemsep=1pt]" + "".join(rf"\item {tex(x)}" for x in steps)
             + r"\end{enumerate}")  # fmt: skip
    p.append(r"\subsection{What we found}")
    finds = [
        ("Pairing direction is often right: ", f"up to {T.pct(max(T.SIGN.values()))} of cuisines (Llama), as often as "
         f"real samples ({T.pct(T.GEN_SIGN)})."),
        ("The fingerprint is only partly kept: ", f"similarity {T.span(T.FP.values())} vs "
         f"{T.GEN['fingerprint_spearman']:.2f} for real samples."),
        ("Classic pairs yes, regional signatures less: ", f"AI shares {T.span(T.NETV.values())} of the real top pairs "
         f"vs {T.NET['real@160']['median']:.2f}."),
        ("Less variety, no stereotype overuse: ", f"variety {T.span(T.DIV.values())} of real; stereotype index "
         f"{T.span(T.SAI.values())} (1 = like real)."),
        ("Fingerprint feedback helps: ", f"recognised as the right cuisine {T.pct(T.R0L)} → {T.pct(T.R2L)} "
         "(GPT-6-Luna, 2 rounds)."),
    ]  # fmt: skip
    p.append(r"\begin{itemize}[leftmargin=1.4em,itemsep=1pt]" + "".join(rf"\item \textbf{{{tex(a)}}}{tex(b)}"
             for a, b in finds) + r"\end{itemize}")  # fmt: skip
    p.append(r"\subsection{What we built (the five deliverables)}")
    p.append(r"\begin{itemize}[leftmargin=1.4em,itemsep=1pt]" + "".join(rf"\item \textbf{{{tex(c)} {tex(n)}:}} {tex(w)}"
             for c, n, _, w in T.DELIVERABLES) + r"\end{itemize}")  # fmt: skip
    # ---- part 2
    p.append(r"\section{Who says what (about 12 minutes)}")
    p.append(
        rf"{A}: about {total_a // 60} min {total_a % 60:02d} s (including the 2-minute demo). "
        rf"{B}: about {total_b // 60} min {total_b % 60:02d} s. Speak slowly; the numbers are on the slides."
    )
    rows = "".join(rf"{s['n']} & {tex(s['title'])} & \textcolor{{{speaker_color(s['speaker'])}}}{{\textbf{{"
                   rf"{tex(s['speaker'].split(' (')[0])}}}}} & {s['time']} \\" for s in T.SLIDES)  # fmt: skip
    p.append(r"\begin{tabularx}{\linewidth}{@{}r X l r@{}}\toprule \textbf{Slide} & \textbf{Title} & \textbf{Speaker} & "
             r"\textbf{Time} \\ \midrule " + rows + r"\bottomrule\end{tabularx}")  # fmt: skip
    for s in T.SLIDES:
        body = tex(s["say"])
        if s.get("point"):
            body += r"\par\smallskip{\color{Grey}\small " + tex(s["point"]) + "}"
        if s.get("handover"):
            body += r"\par\smallskip\textbf{Hand over:} \emph{" + tex(s["handover"]) + "}"
        head = f"Slide {s['n']} · {tex(s['title'])} — {tex(s['speaker'].split(' (')[0])} ({s['time']})"
        p.append(rf"\begin{{slidebox}}{{{speaker_color(s['speaker'])}}}{{{head}}}{body}\end{{slidebox}}")
    # ---- part 3
    p.append(r"\newpage\section{The live demo (slide 12, about 2 minutes)}")
    p.append(tex("Before the talk: double-click 04_Demo/run_demo.command in the final folder (or run the command in "
                 "Part 8). The app opens at http://localhost:8501; first load about 5 seconds. Click Audit recipe once "
                 "so it is ready. During the talk, switch to the browser on slide 12."))  # fmt: skip
    rows = "".join(rf"{k} & {tex(c)} & {tex(pt)} & {tex(say)} \\" for k, (c, pt, say) in enumerate(T.DEMO, 1))
    p.append(r"\begin{tabularx}{\linewidth}{@{}r >{\raggedright\arraybackslash}p{3.6cm} >{\raggedright\arraybackslash}X "
             r">{\raggedright\arraybackslash}X@{}}\toprule & \textbf{Click} & \textbf{Point at} & \textbf{Say} \\ \midrule "
             + rows + r"\bottomrule\end{tabularx}")  # fmt: skip
    p.append(r"\textbf{If the app does not open:} " + tex("stay on slide 12 and describe the screenshot; it shows the "
             "same screen. Never debug live."))  # fmt: skip
    # ---- part 4
    p.append(r"\section{How to show each deliverable (if asked)}")
    rows = "".join(rf"\textbf{{{tex(c)}}} & {tex(n)} & {path_tex(where)} & {tex(say)} \\"
                   for c, n, where, say in T.DELIVERABLES)  # fmt: skip
    p.append(r"\begin{tabularx}{\linewidth}{@{}l >{\raggedright\arraybackslash}p{2.8cm} "
             r">{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X@{}}\toprule & \textbf{Deliverable} & "
             r"\textbf{Open (GitHub repo)} & \textbf{Say} \\ \midrule " + rows + r"\bottomrule\end{tabularx}")  # fmt: skip
    # ---- part 5
    p.append(r"\newpage\section{The ideas in simple words}")
    p.append(r"\begin{description}[leftmargin=0pt,itemsep=3pt,font=\color{Teal}\bfseries]"
             + "".join(rf"\item[{tex(a)}] {tex(b)}" for a, b in T.GLOSSARY) + r"\end{description}")  # fmt: skip
    # ---- part 6
    p.append(r"\newpage\section{Questions the professor may ask}")
    p.append(tex("Short answers first; add detail only if asked. Either of you can answer; the numbers are in Part 7."))
    for k, (q, a) in enumerate(T.QA, 1):
        p.append(
            rf"\noindent\begin{{minipage}}{{\linewidth}}\textbf{{Q{k}. {tex(q)}}}\par\vspace{{-2pt}}{tex(a)}\end{{minipage}}\par\medskip"
        )
    # ---- part 7
    p.append(r"\newpage\section{Numbers cheat sheet}")
    rows = "".join(rf"{tex(a)} & {tex(b)} \\" for a, b in T.NUMBERS)
    p.append(
        r"\begin{tabularx}{\linewidth}{@{}>{\bfseries}p{5.2cm} X@{}}\toprule " + rows + r"\bottomrule\end{tabularx}"
    )
    # ---- part 8
    p.append(r"\section{Links and running the app}")
    p.append(rf"\textbf{{GitHub:}} \url{{{gh}}} (README, code, dataset, report, slides, poster).\par")
    p.append(
        r"\textbf{Run the app on our laptop:} double-click \texttt{04\_Demo/run\_demo.command} in the final folder, "
        r"or:\par\texttt{cd ai-chefs-fingerprints \&\& .venv/bin/streamlit run app/streamlit\_app.py}\par"
    )
    p.append(r"\textbf{Run the app from GitHub (any computer with Python 3.12):}\par"
             r"\texttt{git clone " + gh + r".git \&\& cd CGAS-Project}\par"
             r"\texttt{pip install -e . streamlit plotly}\par\texttt{streamlit run app/streamlit\_app.py}\par")  # fmt: skip
    p.append(r"\end{document}")
    return "\n\n".join(p)  # blank lines = paragraph breaks between blocks


def _secs(t: str) -> int:
    m, s = t.split(":")
    return int(m) * 60 + int(s)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    (OUT / "Group44_Presenter_Guide.tex").write_text(build_tex(), encoding="utf-8")
    tectonic = shutil.which("tectonic") or "/opt/homebrew/bin/tectonic"
    r = subprocess.run([tectonic, "Group44_Presenter_Guide.tex"], cwd=OUT, capture_output=True, text=True)
    missing = [ln for ln in (r.stdout + r.stderr).splitlines() if "Missing character" in ln or "error" in ln.lower()]
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        return 1
    print(f"{(OUT / 'Group44_Presenter_Guide.pdf').relative_to(HERE.parent)}; problems: {missing[:5] or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
