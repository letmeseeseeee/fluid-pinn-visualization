from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SOURCE = DOCS / "thesis_draft.md"
TARGET = DOCS / "thesis_draft.tex"


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def convert_inline(text: str) -> str:
    placeholders: list[str] = []

    def stash(value: str) -> str:
        token = f"PHPLACEHOLDER{len(placeholders)}PH"
        placeholders.append(value)
        return token

    def code_repl(match: re.Match[str]) -> str:
        value = escape_latex(match.group(1))
        return stash(rf"\texttt{{{value}}}")

    def bold_repl(match: re.Match[str]) -> str:
        value = escape_latex(match.group(1))
        return stash(rf"\textbf{{{value}}}")

    text = re.sub(r"`([^`]+)`", code_repl, text)
    text = re.sub(r"\*\*([^*]+)\*\*", bold_repl, text)
    text = escape_latex(text)

    for idx, value in enumerate(placeholders):
        text = text.replace(f"PHPLACEHOLDER{idx}PH", value)

    text = text.replace("…", r"\ldots{}")
    return text


def strip_heading_prefix(text: str) -> str:
    text = re.sub(r"^\d+(\.\d+)*\s*", "", text).strip()
    text = re.sub(r"^附录\s+[A-Z]\s*", "", text).strip()
    text = re.sub(r"^[A-Z]\.\d+\s*", "", text).strip()
    return text


def split_table_row(line: str) -> list[str]:
    parts = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return parts


def convert_table(table_lines: list[str], caption: str | None) -> str:
    rows = [split_table_row(line) for line in table_lines if not re.fullmatch(r"\|\s*[-:| ]+\|", line.strip())]
    if not rows:
        return ""
    col_count = max(len(row) for row in rows)
    spec = "".join([r">{\raggedright\arraybackslash}X" for _ in range(col_count)])
    out: list[str] = [r"\begin{table}[H]", r"\centering"]
    if caption:
        out.append(rf"\caption{{{convert_inline(caption)}}}")
    out.extend(
        [
            rf"\begin{{tabularx}}{{\textwidth}}{{{spec}}}",
            r"\toprule",
        ]
    )
    header = rows[0] + [""] * (col_count - len(rows[0]))
    out.append(" & ".join(convert_inline(cell) for cell in header) + r" \\")
    out.append(r"\midrule")
    for row in rows[1:]:
        padded = row + [""] * (col_count - len(row))
        out.append(" & ".join(convert_inline(cell) for cell in padded) + r" \\")
    out.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    return "\n".join(out)


def convert_figure(line: str) -> str:
    match = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
    if not match:
        return convert_inline(line)
    caption, path = match.groups()
    path = path.replace("\\", "/")
    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=0.92\textwidth]{{{escape_latex(path)}}}",
            rf"\caption{{{convert_inline(caption)}}}",
            r"\end{figure}",
            "",
        ]
    )


def convert_paragraph(lines: list[str]) -> str:
    text = " ".join(line.strip() for line in lines).strip()
    if not text:
        return ""
    return convert_inline(text) + "\n"


def convert_references(ref_lines: list[str]) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in ref_lines:
        if not line.strip():
            if current:
                paragraphs.append(" ".join(item.strip() for item in current))
                current = []
            continue
        current.append(line.strip())
    if current:
        paragraphs.append(" ".join(item.strip() for item in current))

    out = [r"\begin{thebibliography}{99}"]
    for para in paragraphs:
        match = re.match(r"\[(\d+)\]\s*(.+)", para)
        if not match:
            continue
        idx, body = match.groups()
        out.append(rf"\bibitem{{ref{idx}}} {convert_inline(body)}")
    out.extend([r"\end{thebibliography}", ""])
    return "\n".join(out)


def build_tex() -> str:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    title = "基于 PINN 与 FNO 的热方程可视化自动化平台设计与实现"
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()

    out: list[str] = [
        r"\documentclass[UTF8,a4paper,12pt]{ctexrep}",
        r"\usepackage{geometry}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{tabularx}",
        r"\usepackage{array}",
        r"\usepackage{float}",
        r"\usepackage{hyperref}",
        r"\usepackage{caption}",
        r"\usepackage{longtable}",
        r"\usepackage{enumitem}",
        r"\usepackage{setspace}",
        r"\usepackage{indentfirst}",
        r"\geometry{left=3cm,right=2.5cm,top=2.8cm,bottom=2.8cm}",
        r"\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue,citecolor=black}",
        r"\setlength{\parindent}{2em}",
        r"\onehalfspacing",
        r"\title{" + convert_inline(title) + "}",
        r"\author{【待补姓名】}",
        r"\date{2026年5月}",
        r"\begin{document}",
        r"\maketitle",
        "",
    ]

    paragraph: list[str] = []
    table_block: list[str] = []
    code_block: list[str] = []
    math_block: list[str] = []
    list_block: list[str] = []
    ordered_list = False
    pending_table_caption: str | None = None
    in_code = False
    in_math = False
    refs_mode = False
    ref_lines: list[str] = []
    appendix_mode = False
    toc_inserted = False
    skip_toc_body = False

    def flush_paragraph() -> None:
        nonlocal paragraph, pending_table_caption
        if not paragraph:
            return
        text = " ".join(item.strip() for item in paragraph).strip()
        paragraph = []
        if not text:
            return
        if re.fullmatch(r"\*\*表\s*[^*]+\*\*", text):
            pending_table_caption = text.strip("*")
            return
        if text.startswith(">"):
            return
        out.append(convert_paragraph([text]))

    def flush_list() -> None:
        nonlocal list_block, ordered_list
        if not list_block:
            return
        env = "enumerate" if ordered_list else "itemize"
        out.append(rf"\begin{{{env}}}")
        for item in list_block:
            item = re.sub(r"^\d+\.\s*", "", item)
            item = re.sub(r"^-\s*", "", item)
            out.append(rf"\item {convert_inline(item.strip())}")
        out.append(rf"\end{{{env}}}")
        out.append("")
        list_block = []
        ordered_list = False

    def flush_table() -> None:
        nonlocal table_block, pending_table_caption
        if not table_block:
            return
        out.append(convert_table(table_block, pending_table_caption))
        table_block = []
        pending_table_caption = None

    for raw_line in lines[1:]:
        line = raw_line.rstrip()

        if refs_mode:
            if line.startswith("## "):
                out.append(convert_references(ref_lines))
                ref_lines = []
                refs_mode = False
            else:
                ref_lines.append(line)
                continue

        if skip_toc_body:
            if line.startswith("## "):
                skip_toc_body = False
            else:
                continue

        if in_code:
            if line.strip().startswith("```"):
                out.extend([r"\begin{verbatim}", *code_block, r"\end{verbatim}", ""])
                code_block = []
                in_code = False
            else:
                code_block.append(line)
            continue

        if in_math:
            if line.strip() == "$$":
                out.extend([r"\[", *math_block, r"\]", ""])
                math_block = []
                in_math = False
            else:
                math_block.append(line)
            continue

        if line.strip().startswith("```"):
            flush_paragraph()
            flush_list()
            flush_table()
            in_code = True
            code_block = []
            continue

        if line.strip() == "$$":
            flush_paragraph()
            flush_list()
            flush_table()
            in_math = True
            math_block = []
            continue

        if not line.strip():
            flush_paragraph()
            flush_list()
            flush_table()
            continue

        if line.startswith("## 目录"):
            flush_paragraph()
            flush_list()
            flush_table()
            if not toc_inserted:
                out.extend([r"\tableofcontents", r"\clearpage", ""])
                toc_inserted = True
            skip_toc_body = True
            continue

        if line.startswith("## 参考文献"):
            flush_paragraph()
            flush_list()
            flush_table()
            refs_mode = True
            ref_lines = []
            continue

        if line.startswith("## 附录"):
            flush_paragraph()
            flush_list()
            flush_table()
            appendix_mode = True
            out.extend([r"\appendix", ""])
            continue

        if line.startswith("#"):
            flush_paragraph()
            flush_list()
            flush_table()
            level = len(line) - len(line.lstrip("#"))
            heading = line[level:].strip()
            plain_heading = strip_heading_prefix(heading)

            if heading == "中文摘要":
                out.extend([r"\chapter*{中文摘要}", r"\addcontentsline{toc}{chapter}{中文摘要}", ""])
                continue
            if heading == "Abstract":
                out.extend([r"\chapter*{Abstract}", r"\addcontentsline{toc}{chapter}{Abstract}", ""])
                continue
            if heading == "致谢":
                out.extend([r"\chapter*{致谢}", r"\addcontentsline{toc}{chapter}{致谢}", ""])
                continue

            if appendix_mode and level == 3:
                out.extend([rf"\chapter{{{convert_inline(plain_heading)}}}", ""])
                continue
            if appendix_mode and level == 4:
                out.extend([rf"\section{{{convert_inline(plain_heading)}}}", ""])
                continue

            if level == 2 and re.match(r"^\d+\s+", heading):
                out.extend([rf"\chapter{{{convert_inline(plain_heading)}}}", ""])
            elif level == 3:
                out.extend([rf"\section{{{convert_inline(plain_heading)}}}", ""])
            elif level == 4:
                out.extend([rf"\subsection{{{convert_inline(plain_heading)}}}", ""])
            else:
                out.extend([rf"\paragraph{{{convert_inline(plain_heading)}}}", ""])
            continue

        if line.strip().startswith("!["):
            flush_paragraph()
            flush_list()
            flush_table()
            out.append(convert_figure(line))
            continue

        if line.lstrip().startswith("|"):
            flush_paragraph()
            flush_list()
            table_block.append(line)
            continue

        if re.match(r"^\d+\.\s+", line.strip()):
            flush_paragraph()
            flush_table()
            if list_block and not ordered_list:
                flush_list()
            ordered_list = True
            list_block.append(line.strip())
            continue

        if re.match(r"^-\s+", line.strip()):
            flush_paragraph()
            flush_table()
            if list_block and ordered_list:
                flush_list()
            ordered_list = False
            list_block.append(line.strip())
            continue

        paragraph.append(line)

    flush_paragraph()
    flush_list()
    flush_table()
    if refs_mode:
        out.append(convert_references(ref_lines))

    out.append(r"\end{document}")
    return "\n".join(out) + "\n"


def main() -> None:
    TARGET.write_text(build_tex(), encoding="utf-8")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
