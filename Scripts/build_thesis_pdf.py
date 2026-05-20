from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "thesis_draft.md"
OUTPUT = ROOT / "docs" / "thesis_draft_preview.pdf"

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def build_styles():
    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "BaseCN",
        parent=styles["Normal"],
        fontName="STSong-Light",
        fontSize=11.5,
        leading=19,
        textColor=colors.HexColor("#1B1F23"),
        alignment=TA_JUSTIFY,
    )
    styles.add(base)
    styles.add(
        ParagraphStyle(
            "TitleCN",
            parent=base,
            fontSize=18,
            leading=26,
            alignment=TA_CENTER,
            spaceAfter=14,
            textColor=colors.HexColor("#12395B"),
        )
    )
    styles.add(
        ParagraphStyle(
            "H1CN",
            parent=base,
            fontSize=16,
            leading=24,
            spaceBefore=14,
            spaceAfter=10,
            textColor=colors.HexColor("#12395B"),
        )
    )
    styles.add(
        ParagraphStyle(
            "H2CN",
            parent=base,
            fontSize=14,
            leading=22,
            spaceBefore=12,
            spaceAfter=8,
            textColor=colors.HexColor("#12395B"),
        )
    )
    styles.add(
        ParagraphStyle(
            "H3CN",
            parent=base,
            fontSize=12.5,
            leading=20,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#1B1F23"),
        )
    )
    styles.add(
        ParagraphStyle(
            "QuoteCN",
            parent=base,
            leftIndent=12,
            rightIndent=6,
            borderPadding=8,
            backColor=colors.HexColor("#EEF4FA"),
            borderColor=colors.HexColor("#12395B"),
            borderWidth=0.6,
            borderLeftWidth=2.2,
            spaceBefore=6,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            "SmallCN",
            parent=base,
            fontSize=10,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#5A6773"),
        )
    )
    return styles


def markdown_to_story():
    styles = build_styles()
    story = []
    image_re = re.compile(r"^!\[(.*?)\]\((.*?)\)\s*$")
    in_code = False
    code_lines: list[str] = []

    def flush_code():
        nonlocal code_lines
        if code_lines:
            story.append(
                Preformatted(
                    "\n".join(code_lines),
                    ParagraphStyle(
                        "CodeCN",
                        fontName="Courier",
                        fontSize=8.5,
                        leading=11,
                        leftIndent=8,
                        rightIndent=8,
                        backColor=colors.HexColor("#F7F7F7"),
                        borderPadding=6,
                    ),
                )
            )
            story.append(Spacer(1, 6))
            code_lines = []

    for raw in SOURCE.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not stripped:
            story.append(Spacer(1, 7))
            continue

        img_match = image_re.match(stripped)
        if img_match:
            caption, img_path = img_match.groups()
            path = (SOURCE.parent / img_path).resolve()
            if path.exists():
                img = Image(str(path))
                max_w = 165 * mm
                max_h = 105 * mm
                img.drawWidth, img.drawHeight = fit_size(img.imageWidth, img.imageHeight, max_w, max_h)
                story.append(img)
                story.append(Spacer(1, 4))
                story.append(Paragraph(escape(caption), styles["SmallCN"]))
                story.append(Spacer(1, 7))
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(escape(stripped[2:].strip()), styles["TitleCN"]))
            continue
        if stripped.startswith("## "):
            story.append(Paragraph(escape(stripped[3:].strip()), styles["H1CN"]))
            continue
        if stripped.startswith("### "):
            story.append(Paragraph(escape(stripped[4:].strip()), styles["H2CN"]))
            continue
        if stripped.startswith("#### "):
            story.append(Paragraph(escape(stripped[5:].strip()), styles["H3CN"]))
            continue
        if stripped.startswith("> "):
            story.append(Paragraph(escape(stripped[2:].strip()), styles["QuoteCN"]))
            continue
        if stripped.startswith("- ") or stripped.startswith("* "):
            story.append(Paragraph("• " + inline_markup(stripped[2:].strip()), styles["BaseCN"]))
            continue
        if re.match(r"^\d+\.\s+", stripped):
            story.append(Paragraph(inline_markup(stripped), styles["BaseCN"]))
            continue
        if stripped.startswith("|"):
            story.append(
                Preformatted(
                    stripped,
                    ParagraphStyle(
                        "TableLine",
                        fontName="Courier",
                        fontSize=8.8,
                        leading=11,
                        leftIndent=4,
                        rightIndent=4,
                        backColor=colors.HexColor("#F7F7F7"),
                    ),
                )
            )
            continue

        story.append(Paragraph(inline_markup(stripped), styles["BaseCN"]))

    flush_code()
    return story


def inline_markup(text: str) -> str:
    escaped = escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    return escaped


def fit_size(w: float, h: float, max_w: float, max_h: float) -> tuple[float, float]:
    ratio = min(max_w / w, max_h / h)
    return w * ratio, h * ratio


def add_page_number(canvas, doc):
    page = canvas.getPageNumber()
    canvas.setFont("STSong-Light", 9)
    canvas.setFillColor(colors.HexColor("#5A6773"))
    canvas.drawCentredString(A4[0] / 2, 12 * mm, f"第 {page} 页")


def build_pdf() -> Path:
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title="基于 PINN 与 FNO 的热方程可视化自动化平台设计与实现",
        author="Codex",
    )
    story = markdown_to_story()
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return OUTPUT


if __name__ == "__main__":
    out = build_pdf()
    print(out)
