"""Small drawing kit on top of python-pptx: theme, measured text boxes, figures, tables, tiles, notes.

Text is measured with the real Arial metrics (PIL) when the font file is available, so dynamic content (numbers
that change when new data arrive) is shrunk to fit its box instead of overflowing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from lxml import etree
from PIL import Image, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

FONT = "Arial"
MONO = "Courier New"
SLIDE_W, SLIDE_H = 13.333, 7.5


def rgb(hexstr: str) -> RGBColor:
    h = hexstr.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


@dataclass(frozen=True)
class Theme:
    ink: str = "#1B2430"
    muted: str = "#5A6472"
    accent: str = "#D55E00"  # Okabe-Ito vermillion (also "contrasting" in the figures)
    blue: str = "#0072B2"
    green: str = "#009E73"
    amber: str = "#9A5B00"  # pending markers
    panel: str = "#F2F4F7"
    panel2: str = "#E8EEF5"
    rule: str = "#CBD2DB"
    good: str = "#D7F0E5"
    bad: str = "#FBE3D5"
    white: str = "#FFFFFF"


T = Theme()

_FONT_FILES = {
    (False, False): ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
                     "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"],
    (True, False): ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
                    "C:/Windows/Fonts/arialbd.ttf"],
    (False, True): ["/System/Library/Fonts/Supplemental/Arial Italic.ttf", "C:/Windows/Fonts/ariali.ttf"],
    (True, True): ["/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf", "C:/Windows/Fonts/arialbi.ttf"],
}  # fmt: skip


@lru_cache(maxsize=256)
def _pil_font(size_pt: float, bold: bool, italic: bool):
    for path in _FONT_FILES.get((bold, italic), []) + _FONT_FILES[(False, False)]:
        if Path(path).exists():
            # measure at 10x resolution for sub-point accuracy (1 pt = 1/72 in; we work in points)
            return ImageFont.truetype(path, int(round(size_pt * 10)))
    return None


def text_width_pt(text: str, size_pt: float, bold: bool = False, italic: bool = False) -> float:
    font = _pil_font(size_pt, bold, italic)
    if font is None:
        return 0.52 * size_pt * len(text) * (1.07 if bold else 1.0)
    return font.getlength(text) / 10.0


# ---------------------------------------------------------------------------------------------------------------------
# Markup: **bold**, [pending: ...] (amber italic), `code` (monospace)
# ---------------------------------------------------------------------------------------------------------------------
_TOKEN = re.compile(r"(\*\*.+?\*\*|\[pending:[^\]]*\]|`[^`]+`)")


def segments(text: str) -> list[tuple[str, str]]:
    """Split marked-up text into (kind, text) with kind in {plain, bold, pending, code}."""
    out = []
    for part in _TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            out.append(("bold", part[2:-2]))
        elif part.startswith("[pending:"):
            out.append(("pending", part[len("[pending:") : -1].strip()))
        elif part.startswith("`") and part.endswith("`"):
            out.append(("code", part[1:-1]))
        else:
            out.append(("plain", part))
    return out


def plain(text: str) -> str:
    return "".join(t for _, t in segments(text))


def wrapped_lines(text: str, size_pt: float, width_pt: float, bold: bool = False) -> int:
    """Number of lines a paragraph needs at this size (greedy word wrap with real metrics)."""
    words = plain(text).split(" ")
    lines, cur = 1, ""
    for w in words:
        trial = f"{cur} {w}" if cur else w
        if text_width_pt(trial, size_pt, bold) <= width_pt:
            cur = trial
        else:
            if cur:
                lines += 1
            cur = w
    return lines


@dataclass
class Para:
    text: str
    size: float = 18
    bold: bool = False
    color: str = T.ink
    bullet: bool = False
    level: int = 0
    space_before: float = 4
    align: str = "left"
    italic: bool = False
    font: str = FONT
    line_spacing: float = 1.0


def paragraphs_height_pt(paras: list[Para], width_pt: float, scale: float = 1.0) -> float:
    total = 0.0
    for p in paras:
        size = p.size * scale
        indent = 16 * scale if p.bullet else 0
        n = wrapped_lines(p.text, size, width_pt - indent - 2, p.bold)
        total += n * size * 1.2 * p.line_spacing + p.space_before * scale
    return total


def fit_scale(paras: list[Para], w_in: float, h_in: float, min_scale: float = 0.6) -> float:
    """Largest scale <= 1 at which the paragraphs fit the box (inner margins of 0.05 in assumed)."""
    width_pt, height_pt = (w_in - 0.1) * 72, (h_in - 0.1) * 72
    scale = 1.0
    while scale > min_scale and paragraphs_height_pt(paras, width_pt, scale) > height_pt:
        scale -= 0.03
    return max(scale, min_scale)


# ---------------------------------------------------------------------------------------------------------------------
# Deck wrapper
# ---------------------------------------------------------------------------------------------------------------------
class Deck:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width = Inches(SLIDE_W)
        self.prs.slide_height = Inches(SLIDE_H)
        self.blank = self.prs.slide_layouts[6]
        self.meta: list[dict] = []  # one entry per slide: kind (main/backup), label

    # --- slides --------------------------------------------------------------------------------------------------
    def new(self, kind: str = "main", label: str = ""):
        slide = self.prs.slides.add_slide(self.blank)
        bg = slide.background.fill
        bg.solid()
        bg.fore_color.rgb = rgb(T.white)
        self.meta.append({"kind": kind, "label": label, "slide": slide})
        return slide

    def save(self, path: Path) -> None:
        self.prs.save(path)

    # --- text ----------------------------------------------------------------------------------------------------
    def text(self, slide, x, y, w, h, paras, anchor="top", fit=True, min_scale=0.6, fill=None, margin=0.05,
             line=None):  # fmt: skip
        if isinstance(paras, (str, Para)):
            paras = [paras]
        paras = [p if isinstance(p, Para) else Para(p) for p in paras]
        scale = fit_scale(paras, w, h, min_scale) if fit else 1.0
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame
        tf.word_wrap = True
        tf.auto_size = None
        for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, side, Inches(margin))
        tf.vertical_anchor = {"top": MSO_ANCHOR.TOP, "middle": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}[anchor]
        if fill:
            box.fill.solid()
            box.fill.fore_color.rgb = rgb(fill)
        if line:
            box.line.color.rgb = rgb(line)
            box.line.width = Pt(1)
        for i, p in enumerate(paras):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            self._fill_paragraph(para, p, scale)
        return box

    def _fill_paragraph(self, para, p: Para, scale: float) -> None:
        para.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[p.align]
        para.space_before = Pt(p.space_before * scale)
        para.line_spacing = p.line_spacing
        size = Pt(p.size * scale)
        for kind, txt in segments(p.text):
            run = para.add_run()
            run.text = txt
            font = run.font
            font.size = size
            font.name = MONO if kind == "code" else p.font
            font.bold = p.bold or kind == "bold"
            font.italic = p.italic or kind == "pending"
            font.color.rgb = rgb(T.muted if kind == "pending" else p.color)
        if p.bullet:
            indent = Emu(Inches(0.22 * scale + 0.25 * p.level))
            ppr = para._p.get_or_add_pPr()
            ppr.set("marL", str(int(indent)))
            ppr.set("indent", str(-int(Inches(0.22 * scale))))
            for tag in ("a:buNone", "a:buChar", "a:buAutoNum", "a:buFont", "a:buClr"):
                for el in ppr.findall(qn(tag)):
                    ppr.remove(el)
            buclr = etree.SubElement(ppr, qn("a:buClr"))
            clr = etree.SubElement(buclr, qn("a:srgbClr"))
            clr.set("val", T.accent.lstrip("#"))
            bufont = etree.SubElement(ppr, qn("a:buFont"))
            bufont.set("typeface", FONT)
            buchar = etree.SubElement(ppr, qn("a:buChar"))
            buchar.set("char", "•" if p.level == 0 else "–")

    # --- decorations ---------------------------------------------------------------------------------------------
    def rect(self, slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
        s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
        if line:
            s.line.color.rgb = rgb(line)
            s.line.width = Pt(1)
        else:
            s.line.fill.background()
        s.shadow.inherit = False
        return s

    def hline(self, slide, x, y, w, color=T.rule, weight=1.0):
        c = slide.shapes.add_connector(1, Inches(x), Inches(y), Inches(x + w), Inches(y))
        c.line.color.rgb = rgb(color)
        c.line.width = Pt(weight)
        return c

    # --- figures -------------------------------------------------------------------------------------------------
    def picture(self, slide, path: Path | None, x, y, w, h, align="center", valign="top", missing=""):
        if path is None or not Path(path).exists():
            self.text(slide, x, y, w, h, [Para(missing or "[pending: figure not yet generated]", 16,
                                                color=T.amber, italic=True, align="center")],
                      anchor="middle", fill=T.panel)  # fmt: skip
            return None
        with Image.open(path) as im:
            iw, ih = im.size
        ar = iw / ih
        if w / h > ar:
            ph, pw = h, h * ar
        else:
            pw, ph = w, w / ar
        px = x + {"left": 0, "center": (w - pw) / 2, "right": w - pw}[align]
        py = y + {"top": 0, "middle": (h - ph) / 2, "bottom": h - ph}[valign]
        return slide.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))

    # --- tables --------------------------------------------------------------------------------------------------
    def table(self, slide, x, y, col_w, rows, size=14, row_h=0.36, header_fill=T.ink, zebra=True, fills=None,
              aligns=None, bold_first_col=False, header_size=None):  # fmt: skip
        """rows[0] is the header. fills: dict {(r, c): hex}. aligns: per column 'l'/'c'/'r'."""
        n_r, n_c = len(rows), len(rows[0])
        shape = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(sum(col_w)), Inches(row_h * n_r))
        tbl = shape.table
        tbl_pr = shape._element.graphic.graphicData.tbl.tblPr
        style = tbl_pr.find(qn("a:tableStyleId"))
        if style is None:
            style = etree.SubElement(tbl_pr, qn("a:tableStyleId"))
        style.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"  # "No Style, No Grid"
        tbl.first_row = True
        tbl.horz_banding = False
        for c, wv in enumerate(col_w):
            tbl.columns[c].width = Inches(wv)
        heights = []
        for r in range(n_r):
            fs = (header_size or size) if r == 0 else size
            need = max(wrapped_lines(str(rows[r][c]), fs, (col_w[c] - 0.14) * 72, r == 0) for c in range(n_c))
            heights.append(max(row_h, need * fs * 1.18 / 72 + 0.1))
        self.last_table_h = sum(heights)
        for r in range(n_r):
            tbl.rows[r].height = Inches(heights[r])
            for c in range(n_c):
                cell = tbl.cell(r, c)
                cell.margin_left = cell.margin_right = Inches(0.06)
                cell.margin_top = cell.margin_bottom = Inches(0.03)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                val = rows[r][c]
                tf = cell.text_frame
                tf.word_wrap = True
                para = tf.paragraphs[0]
                al = (aligns or ["l"] * n_c)[c]
                para.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[al]
                is_header = r == 0
                fsize = (header_size or size) if is_header else size
                for kind, txt in segments(str(val)):
                    run = para.add_run()
                    run.text = txt
                    run.font.size = Pt(fsize)
                    run.font.name = MONO if kind == "code" else FONT
                    run.font.bold = is_header or kind == "bold" or (bold_first_col and c == 0)
                    run.font.italic = kind == "pending"
                    if is_header:
                        run.font.color.rgb = rgb(T.white)
                    else:
                        run.font.color.rgb = rgb(T.muted if kind == "pending" else T.ink)
                fill = None
                if is_header:
                    fill = header_fill
                elif fills and (r, c) in fills:
                    fill = fills[(r, c)]
                elif zebra and r % 2 == 0:
                    fill = T.panel
                if fill:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = rgb(fill)
                else:
                    cell.fill.background()
        return shape

    # --- number tiles --------------------------------------------------------------------------------------------
    def tile(self, slide, x, y, w, h, big, caption, color=T.accent, big_size=30, cap_size=14, fill=T.panel):
        self.rect(slide, x, y, w, h, fill)
        self.rect(slide, x, y, 0.07, h, color)
        big_h = big_size / 72 * 1.25
        self.text(slide, x + 0.15, y + 0.06, w - 0.22, big_h, [Para(big, big_size, bold=True, color=color)],
                  fit=True, min_scale=0.7)  # fmt: skip
        self.text(slide, x + 0.15, y + 0.06 + big_h, w - 0.22, h - big_h - 0.1,
                  [Para(caption, cap_size, color=T.ink, space_before=0)], fit=True, min_scale=0.7)  # fmt: skip

    # --- notes ---------------------------------------------------------------------------------------------------
    def notes(self, slide, text: str) -> None:
        slide.notes_slide.notes_text_frame.text = text.strip()
        for m in self.meta:
            if m["slide"] is slide:
                m["notes"] = text.strip()


def header(deck: Deck, slide, title: str, answer: str | None = None, title_size: float = 30) -> None:
    """Question title plus a one-line answer with a vermillion bar."""
    for m in deck.meta:
        if m["slide"] is slide:
            m["title"], m["answer"] = title, answer
    deck.text(slide, 0.55, 0.28, 12.25, 0.78, [Para(title, title_size, bold=True, color=T.ink, space_before=0)],
              anchor="bottom", min_scale=0.75)  # fmt: skip
    if answer:
        deck.rect(slide, 0.55, 1.14, 0.08, 0.62, T.accent)
        deck.text(slide, 0.72, 1.1, 12.05, 0.7, [Para(answer, 18, color=T.ink, space_before=0)], anchor="middle",
                  min_scale=0.72)  # fmt: skip


def footer(deck: Deck, stamp: str) -> None:
    mains = [m for m in deck.meta if m["kind"] == "main"]
    n_main = len(mains)
    b = 0
    for i, m in enumerate(deck.meta):
        slide = m["slide"]
        if i == 0:
            continue  # the title slide carries its own stamp
        deck.hline(slide, 0.55, 7.05, 12.25, T.rule, 0.75)
        left = "Group 44 · BIO544 Computational Gastronomy · IIIT-Delhi · Monsoon 2026"
        deck.text(slide, 0.55, 7.07, 7.2, 0.32, [Para(left, 10, color=T.muted, space_before=0)], fit=False)
        if m["kind"] == "main":
            right = f"data: {stamp}   ·   {mains.index(m) + 1} / {n_main}"
        else:
            b += 1
            right = f"data: {stamp}   ·   Backup B{b}"
        deck.text(slide, 7.8, 7.07, 5.0, 0.32, [Para(right, 10, color=T.muted, align="right", space_before=0)],
                  fit=False)  # fmt: skip
