"""Shared figure style (CLAUDE.md figure rules): a question as the title, labelled axes with units, one colour per
source everywhere (Okabe-Ito, colour-blind safe, from configs/models.yaml), vector PDF plus 300-dpi PNG, numbered as
in the report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FIGURES = REPO / "figures"
PALETTE: dict[str, str] = yaml.safe_load((REPO / "configs" / "models.yaml").read_text())["palette"]
LABELS = {
    "real": "Real (CulinaryDB)",
    "openai-gpt-6-luna": "GPT-6-Luna",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "qwen3.5-9b": "Qwen3.5-9B",
    "qwen3.5-4b": "Qwen3.5-4B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "ministral-3-8b": "Ministral-3-8B",
    "copy-mutate": "Copy-mutate (ranked fitness)",
}
MARKERS = {
    "real": "D",
    "openai-gpt-6-luna": "o",
    "gemini-3.5-flash-lite": "s",
    "qwen3.5-9b": "^",
    "qwen3.5-4b": "v",
    "llama-3.1-8b": "P",
    "ministral-3-8b": "X",
    "copy-mutate": "h",
}


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,  # embed TrueType fonts (editable text in the PDF)
        }
    )


def color(source: str) -> str:
    return PALETTE.get(source, "#666666")


def label(source: str) -> str:
    return LABELS.get(source, source)


def save(fig, name: str) -> Path:
    """figures/<name>.pdf (vector) and figures/<name>.png (300 dpi)."""
    FIGURES.mkdir(exist_ok=True)
    pdf = FIGURES / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    plt.close(fig)
    return pdf
