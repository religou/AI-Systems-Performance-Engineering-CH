#!/usr/bin/env python3
"""Normalize blank lines in Markdown for translation readiness.

Rules:
  1. At most one consecutive blank line anywhere.
  2. Horizontal rules: one blank line before and after, unless at document boundaries.
  3. Tables: one blank line before and after, unless at document boundaries.
  4. Headings: one blank line before, unless at document start.
  5. Preserve the blank line between YAML frontmatter and following content.
  6. Every paragraph has exactly one blank line before and after,
     unless at document boundaries.
  7. With --strip-details: remove HTML <details> blocks inserted by VLM image analysis,
     keeping image references but discarding the VLM-generated text content.

Protected regions (YAML frontmatter, fenced code blocks, display math blocks)
pass through verbatim internally.
"""

import re
import sys
from pathlib import Path


def _is_hr(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    for ch in ("-", "*", "_"):
        if all(c in (ch, " ") for c in s) and sum(1 for c in s if c == ch) >= 3:
            return True
    return False


def _is_heading(s: str) -> bool:
    return bool(re.match(r"^#{1,6}\s", s.strip()))


def _is_table(s: str) -> bool:
    return "|" in s


def _block_type(line: str) -> str:
    s = line.strip()
    if _is_hr(s):
        return "hr"
    if _is_heading(s):
        return "heading"
    if _is_table(s):
        return "table"
    return "para"


class Stats:
    def __init__(self):
        self.collapsed = 0       # consecutive blank-line groups collapsed
        self.added = 0           # structural blank lines inserted
        self.removed_edge = 0    # blank lines trimmed at edges


def normalize(text: str, stats: Stats | None = None) -> str:
    lines = text.splitlines(keepends=False)
    n = len(lines)
    if n == 0:
        return "\n"

    # ---- find YAML frontmatter range -----------------------------------
    yaml_start = -1
    yaml_end = -1
    i = 0
    while i < n and lines[i].strip() == "":
        i += 1
    if i < n and lines[i].strip() == "---":
        yaml_start = i
        j = i + 1
        while j < n:
            if lines[j].strip() == "---":
                yaml_end = j
                break
            j += 1

    # ---- classify each line as protected or not ------------------------
    in_fenced = False
    fence_char = ""
    in_math = False

    for i, line in enumerate(lines):
        s = line.strip()

        if s.startswith("```") or s.startswith("~~~"):
            if not in_fenced:
                in_fenced = True
                fence_char = s[:3]
            elif s.startswith(fence_char):
                in_fenced = False
                fence_char = ""

        if s == "$$":
            in_math = not in_math

    # ---- build output --------------------------------------------------
    out: list[str] = []
    # Track whether we are past the YAML frontmatter so we can preserve
    # its trailing blank line.
    past_yaml = False

    for i, line in enumerate(lines):
        s = line.strip()

        # YAML frontmatter: pass through verbatim (including internal blanks).
        if yaml_start <= i <= yaml_end:
            out.append(line)
            if i == yaml_end:
                past_yaml = True
            continue

        # Immediately after closing ---: skip blank-line insertion logic
        # for one line so the existing blank (or lack thereof) is preserved.
        if past_yaml:
            # The next non-blank line after ---: just append it.
            # The blank line between --- and content is already in the
            # original — if it's missing we add one; if present we keep it.
            past_yaml = False
            if s == "":
                out.append(line)
                continue
            # Non-blank right after ---: ensure one blank line separates them.
            if out and out[-1].strip() != "":
                out.append("")
            out.append(line)
            continue

        # Fenced code and display math blocks: verbatim.
        if s.startswith("```") or s.startswith("~~~"):
            if not in_fenced:
                in_fenced = True
                fence_char = s[:3]
            elif s.startswith(fence_char):
                in_fenced = False
                fence_char = ""
        if s == "$$":
            in_math = not in_math
        if in_fenced or in_math:
            out.append(line)
            continue

        # Blank line outside protected region: collapse.
        if s == "":
            if out and out[-1].strip() != "":
                out.append(line)
            else:
                if stats:
                    stats.collapsed += 1
            continue

        # Non-blank, non-protected line: insert blank separator if needed.
        if out and out[-1].strip() != "":
            prev_type = _block_type(out[-1])
            curr_type = _block_type(line)
            same_block = (prev_type == "table" and curr_type == "table") or (
                prev_type == "para" and curr_type == "para"
            )
            if not same_block:
                out.append("")
                if stats:
                    stats.added += 1

        out.append(line)

    result = "\n".join(out)

    if stats:
        # Count leading/trailing blank lines being trimmed
        stripped = result.strip()
        lead = len(result) - len(result.lstrip("\n"))
        trail = len(result) - len(result.rstrip("\n"))
        # Each blank line is one \n; leading edge: lines before first content
        if lead > 1:
            stats.removed_edge += lead - 1
        if trail > 1:
            stats.removed_edge += trail - 1

    return result.strip() + "\n"


def strip_details(text: str) -> str:
    """Remove HTML <details>...</details> blocks inserted by VLM image analysis.

    MinerU VLM backends wrap chart/image analysis results in <details> tags
    (e.g. ``<details><summary>line</summary>...mermaid code...</details>``).
    This function removes those blocks while keeping surrounding content.
    """
    return re.sub(r"<details>.*?</details>", "", text, flags=re.DOTALL)


def main() -> None:
    args = sys.argv[1:]
    strip = False
    if "--strip-details" in args:
        strip = True
        args.remove("--strip-details")

    if len(args) < 1:
        print(f"Usage: {sys.argv[0]} [--strip-details] <input> [output]", file=sys.stderr)
        sys.exit(2)

    src = Path(args[0])
    if not src.exists():
        print(f"Error: file not found: {src}", file=sys.stderr)
        sys.exit(1)

    text = src.read_text(encoding="utf-8")
    if strip:
        text = strip_details(text)
    stats = Stats()
    result = normalize(text, stats)

    dst = Path(args[1]) if len(args) >= 2 else src
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(result, encoding="utf-8")

    parts = [f"Normalized: {dst}"]
    if stats.collapsed:
        parts.append(f"{stats.collapsed} blank group(s) collapsed")
    if stats.added:
        parts.append(f"{stats.added} structural blank line(s) added")
    if stats.removed_edge:
        parts.append(f"{stats.removed_edge} edge blank line(s) trimmed")
    print(", ".join(parts), file=sys.stderr)


if __name__ == "__main__":
    main()
