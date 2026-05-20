from __future__ import annotations

import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "platform_explanation_mobile.md"
OUTPUT = ROOT / "docs" / "platform_explanation_mobile.pdf"

PAGE_WIDTH = 1240
PAGE_HEIGHT = 1754
MARGIN_X = 88
MARGIN_Y = 88
CONTENT_WIDTH = PAGE_WIDTH - MARGIN_X * 2
BOTTOM_SAFE = 96
LINE_GAP = 12
PARA_GAP = 24
SECTION_GAP = 34

FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")

COLOR_BG = "#F6F0E5"
COLOR_PANEL = "#FFFDF8"
COLOR_TEXT = "#1B324A"
COLOR_MUTED = "#5E738A"
COLOR_ACCENT = "#C45A15"
COLOR_LINE = "#E6D9C7"


def load_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


FONT_TITLE = load_font(40)
FONT_H1 = load_font(32)
FONT_H2 = load_font(28)
FONT_H3 = load_font(24)
FONT_BODY = load_font(22)
FONT_BULLET = load_font(22)
FONT_CAPTION = load_font(18)
FONT_FOOT = load_font(16)


def measure(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    dummy = Image.new("RGB", (10, 10), "white")
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    lines: list[str] = []
    for raw in text.splitlines():
        raw = raw.rstrip()
        if not raw:
            lines.append("")
            continue
        current = ""
        for ch in raw:
            trial = current + ch
            width, _ = measure(trial, font)
            if current and width > max_width:
                lines.append(current)
                current = ch
            else:
                current = trial
        if current:
            lines.append(current)
    return lines


def parse_markdown(path: Path) -> list[dict]:
    blocks: list[dict] = []
    image_re = re.compile(r"^!\[(.*?)\]\((.*?)\)\s*$")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            blocks.append({"type": "space", "size": PARA_GAP})
            continue

        image_match = image_re.match(line)
        if image_match:
            caption, img_path = image_match.groups()
            blocks.append({"type": "image", "caption": caption, "path": img_path})
            continue

        if line.startswith("# "):
            blocks.append({"type": "title", "text": line[2:].strip()})
            continue
        if line.startswith("## "):
            blocks.append({"type": "h1", "text": line[3:].strip()})
            continue
        if line.startswith("### "):
            blocks.append({"type": "h2", "text": line[4:].strip()})
            continue
        if line.startswith("- "):
            blocks.append({"type": "bullet", "text": line[2:].strip()})
            continue

        blocks.append({"type": "paragraph", "text": line})
    return blocks


def ensure_page(pages: list[Image.Image], current: Image.Image | None, y: int, needed: int) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
    if current is None or y + needed > PAGE_HEIGHT - BOTTOM_SAFE:
        if current is not None:
            pages.append(current)
        current = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), COLOR_BG)
        draw = ImageDraw.Draw(current)
        draw.rounded_rectangle(
            [36, 36, PAGE_WIDTH - 36, PAGE_HEIGHT - 36],
            radius=36,
            fill=COLOR_PANEL,
            outline=COLOR_LINE,
            width=2,
        )
        return current, draw, MARGIN_Y
    draw = ImageDraw.Draw(current)
    return current, draw, y


def draw_text_block(
    pages: list[Image.Image],
    current: Image.Image | None,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str,
    gap_after: int,
    bullet: bool = False,
) -> tuple[Image.Image, int]:
    max_width = CONTENT_WIDTH - (34 if bullet else 0)
    lines = wrap_text(text, font, max_width)
    if not lines:
        current, draw, y = ensure_page(pages, current, y, gap_after)
        return current, y + gap_after

    line_heights = [measure(line or "国", font)[1] for line in lines]
    total_height = sum(line_heights) + LINE_GAP * max(0, len(lines) - 1) + gap_after
    current, draw, y = ensure_page(pages, current, y, total_height)

    x = MARGIN_X + (34 if bullet else 0)
    if bullet:
        bullet_y = y + 4
        draw.ellipse([MARGIN_X + 6, bullet_y + 8, MARGIN_X + 18, bullet_y + 20], fill=COLOR_ACCENT)

    for line, line_height in zip(lines, line_heights):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height + LINE_GAP

    y += gap_after - LINE_GAP
    return current, y


def draw_image_block(
    pages: list[Image.Image],
    current: Image.Image | None,
    y: int,
    caption: str,
    img_path: str,
) -> tuple[Image.Image, int]:
    path = Path(img_path)
    if not path.exists():
        fallback = f"图片未找到：{img_path}"
        return draw_text_block(pages, current, y, fallback, FONT_CAPTION, "#B14B2A", PARA_GAP)

    image = Image.open(path).convert("RGB")
    max_image_width = CONTENT_WIDTH
    max_image_height = 720
    ratio = min(max_image_width / image.width, max_image_height / image.height)
    new_size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    caption_lines = wrap_text(caption, FONT_CAPTION, CONTENT_WIDTH)
    cap_height = 0
    if caption_lines:
        cap_height = sum(measure(line or "国", FONT_CAPTION)[1] for line in caption_lines) + LINE_GAP * max(0, len(caption_lines) - 1)
    needed = image.height + cap_height + SECTION_GAP

    current, draw, y = ensure_page(pages, current, y, needed)
    x = MARGIN_X + (CONTENT_WIDTH - image.width) // 2
    current.paste(image, (x, y))
    y += image.height + 14
    for line in caption_lines:
        draw.text((MARGIN_X, y), line, font=FONT_CAPTION, fill=COLOR_MUTED)
        y += measure(line or "国", FONT_CAPTION)[1] + LINE_GAP
    y += SECTION_GAP - LINE_GAP
    return current, y


def add_footer(page: Image.Image, page_num: int, total: int) -> None:
    draw = ImageDraw.Draw(page)
    footer = f"PINN 与 FNO 热方程展示平台说明  ·  第 {page_num} / {total} 页"
    width, height = measure(footer, FONT_FOOT)
    draw.text(((PAGE_WIDTH - width) / 2, PAGE_HEIGHT - 54 - height), footer, font=FONT_FOOT, fill=COLOR_MUTED)


def build_pdf() -> Path:
    blocks = parse_markdown(SOURCE)
    pages: list[Image.Image] = []
    current: Image.Image | None = None
    y = MARGIN_Y

    for block in blocks:
        if block["type"] == "space":
            current, draw, y = ensure_page(pages, current, y, block["size"])
            y += block["size"]
            continue

        if block["type"] == "title":
            current, y = draw_text_block(pages, current, y, block["text"], FONT_TITLE, COLOR_ACCENT, SECTION_GAP)
            continue
        if block["type"] == "h1":
            current, y = draw_text_block(pages, current, y, block["text"], FONT_H1, COLOR_TEXT, 18)
            continue
        if block["type"] == "h2":
            current, y = draw_text_block(pages, current, y, block["text"], FONT_H2, COLOR_TEXT, 16)
            continue
        if block["type"] == "paragraph":
            current, y = draw_text_block(pages, current, y, block["text"], FONT_BODY, COLOR_TEXT, PARA_GAP)
            continue
        if block["type"] == "bullet":
            current, y = draw_text_block(pages, current, y, block["text"], FONT_BULLET, COLOR_TEXT, 18, bullet=True)
            continue
        if block["type"] == "image":
            current, y = draw_image_block(pages, current, y, block["caption"], block["path"])
            continue

    if current is not None:
        pages.append(current)

    if not pages:
        raise RuntimeError("No pages were generated.")

    for idx, page in enumerate(pages, start=1):
        add_footer(page, idx, len(pages))

    first, *rest = pages
    first.save(OUTPUT, save_all=True, append_images=rest, resolution=160.0)
    return OUTPUT


if __name__ == "__main__":
    out = build_pdf()
    print(out)
