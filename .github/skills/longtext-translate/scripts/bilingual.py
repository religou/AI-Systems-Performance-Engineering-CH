#!/usr/bin/env python3
"""Minimal bilingual Markdown assembly toolset.

Three subcommands keep the workflow mechanical while surfacing structure
problems before final assembly:

  dump FILE
      Parse a Markdown file into a JSON array of blocks. Each block
      records its index, kind, heading level, word count, and full
      Markdown text. Frontmatter is included as a special entry.

  analyze SOURCE TRANSLATION [-o REPORT.json] [--mapping-out MAPPING.json]
      Load two Markdown files (or prior dump JSON files), build a
      conservative candidate alignment, and report the places that need
      review. The optional mapping export writes a grouped mapping JSON
      object for use with ``join``.

  join MAPPING SOURCE TRANSLATION -o OUTPUT
      Stitch two Markdown files together according to *mapping* and
      write the interleaved result. ``join`` accepts both the legacy
      pair-array format and the grouped mapping object emitted by
      ``analyze``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from markdown_it import MarkdownIt


_md_parser = MarkdownIt("js-default", {"html": True})

MAPPING_VERSION = 2
MAPPING_KIND = "bilingual-mapping"


@dataclass(frozen=True)
class Block:
    index: int
    kind: str
    md: str
    words: int
    heading_level: int | None = None


@dataclass(frozen=True)
class BlockDocument:
    file: str
    frontmatter: str
    blocks: tuple[Block, ...]
    input_kind: str
    sha256: str


@dataclass(frozen=True)
class AlignmentStep:
    op: str
    src_indices: tuple[int, ...]
    tgt_indices: tuple[int, ...]
    cost: float
    reason: str
    confidence: str
    severity: str | None = None


@dataclass(frozen=True)
class MappingSide:
    path: str
    sha256: str
    block_count: int


@dataclass(frozen=True)
class MappingEntry:
    source: tuple[int, ...]
    translation: tuple[int, ...]
    review: dict[str, Any] | None = None


@dataclass(frozen=True)
class MappingDocument:
    version: int
    kind: str
    source: MappingSide | None
    translation: MappingSide | None
    entries: tuple[MappingEntry, ...]


# ---------------------------------------------------------------------------
# Markdown parsing helpers
# ---------------------------------------------------------------------------


def _count_words(text: str) -> int:
    cleaned = re.sub(r"[#*`\[\]()>|_~-]", " ", text)
    cjk = re.findall(r"[一-鿿㐀-䶿豈-﫿]", cleaned)
    latin = re.findall(r"[a-zA-Z0-9]+", cleaned)
    return len(cjk) + len(latin)


def _trim_boundary(text: str) -> str:
    return re.sub(r"\n+$", "", re.sub(r"^\n+", "", text))


def _normalize_newlines(text: str) -> str:
    return text.removeprefix("﻿").replace("\r\n", "\n").replace("\r", "\n")


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_frontmatter(content: str) -> tuple[str, str]:
    """Return (frontmatter, body)."""
    lines = content.split("\n")
    if not lines or lines[0] != "---":
        return "", content
    for idx in range(1, len(lines)):
        if lines[idx] in {"---", "..."}:
            return (
                "\n".join(lines[: idx + 1]),
                "\n".join(lines[idx + 1 :]).lstrip("\n"),
            )
    return "", content


def _token_kind(token_type: str) -> str:
    if token_type == "heading_open":
        return "heading"
    if token_type == "hr":
        return "thematicBreak"
    if token_type == "html_block":
        return "html"
    if token_type in {"fence", "code_block"}:
        return "code"
    return "flow"


def _heading_level(md: str) -> int | None:
    match = re.match(r"^(#{1,6})\s+", md)
    return len(match.group(1)) if match else None


def _normalize_heading_text(md: str) -> str:
    text = re.sub(r"^#{1,6}\s+", "", md).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _size_bucket(words: int) -> int:
    if words <= 12:
        return 0
    if words <= 35:
        return 1
    if words <= 90:
        return 2
    if words <= 180:
        return 3
    return 4


def _downgrade_confidence(confidence: str, next_level: str) -> str:
    order = {"high": 0, "medium": 1, "low": 2}
    return next_level if order[next_level] > order[confidence] else confidence


def _block_to_dict(block: Block) -> dict[str, Any]:
    return {
        "index": block.index,
        "kind": block.kind,
        "heading_level": block.heading_level,
        "words": block.words,
        "md": block.md,
    }


def parse_blocks(content: str) -> list[Block]:
    """Parse *content* into a list of blocks."""
    if not content.strip():
        return []

    lines = content.split("\n")
    blocks: list[Block] = []

    for token in _md_parser.parse(content, {}):
        if not token.map or token.level != 0:
            continue
        if token.nesting not in {0, 1}:
            continue

        start_line, end_line = token.map
        md = _trim_boundary("\n".join(lines[start_line:end_line]))
        if not md:
            continue

        kind = _token_kind(token.type)
        blocks.append(
            Block(
                index=len(blocks),
                kind=kind,
                heading_level=_heading_level(md) if kind == "heading" else None,
                words=_count_words(md),
                md=md,
            )
        )

    if not blocks:
        body = _trim_boundary(content)
        if body:
            blocks.append(
                Block(
                    index=0,
                    kind="flow",
                    heading_level=None,
                    words=_count_words(body),
                    md=body,
                )
            )

    return blocks


def _is_dump_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return isinstance(data, dict) and isinstance(data.get("blocks"), list)


def _document_from_dump(path: Path, text: str) -> BlockDocument:
    data = json.loads(text)
    raw_blocks = data.get("blocks")
    if not isinstance(raw_blocks, list):
        raise ValueError(f"Dump JSON missing blocks array: {path}")

    blocks: list[Block] = []
    for index, raw_block in enumerate(raw_blocks):
        if not isinstance(raw_block, dict):
            raise ValueError(f"Invalid dump block at position {index}: {path}")
        kind = raw_block.get("kind")
        md = raw_block.get("md")
        words = raw_block.get("words")
        heading_level = raw_block.get("heading_level")
        if kind not in {"heading", "flow", "code", "thematicBreak", "html"}:
            raise ValueError(f"Invalid block kind at position {index}: {kind!r}")
        if not isinstance(md, str):
            raise ValueError(f"Invalid block md at position {index}: {path}")
        if not isinstance(words, int) or words < 0:
            raise ValueError(f"Invalid block words at position {index}: {path}")
        if heading_level is not None and (not isinstance(heading_level, int) or heading_level < 1):
            raise ValueError(f"Invalid heading level at position {index}: {path}")
        blocks.append(
            Block(
                index=index,
                kind=kind,
                md=md,
                words=words,
                heading_level=heading_level,
            )
        )

    frontmatter = data.get("frontmatter", "")
    if frontmatter is not None and not isinstance(frontmatter, str):
        raise ValueError(f"Invalid frontmatter in dump JSON: {path}")

    return BlockDocument(
        file=str(path),
        frontmatter=frontmatter or "",
        blocks=tuple(blocks),
        input_kind="dump",
        sha256=_sha256_text(text),
    )


def _load_blocks(path: Path) -> BlockDocument:
    raw = _normalize_newlines(path.read_text(encoding="utf-8"))
    if _is_dump_json(raw):
        return _document_from_dump(path, raw)

    frontmatter, body = _extract_frontmatter(raw)
    return BlockDocument(
        file=str(path),
        frontmatter=frontmatter,
        blocks=tuple(parse_blocks(body)),
        input_kind="markdown",
        sha256=_sha256_text(raw),
    )


def _mapping_side_from_doc(doc: BlockDocument) -> MappingSide:
    return MappingSide(
        path=doc.file,
        sha256=doc.sha256,
        block_count=len(doc.blocks),
    )


def _mapping_side_to_dict(side: MappingSide | None) -> dict[str, Any] | None:
    if side is None:
        return None
    return {
        "path": side.path,
        "sha256": side.sha256,
        "block_count": side.block_count,
    }


def _mapping_entry_to_dict(entry: MappingEntry) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source": list(entry.source),
        "translation": list(entry.translation),
    }
    if entry.review is not None:
        result["review"] = entry.review
    return result


def _mapping_document_to_dict(mapping: MappingDocument) -> dict[str, Any]:
    return {
        "version": mapping.version,
        "kind": mapping.kind,
        "source": _mapping_side_to_dict(mapping.source),
        "translation": _mapping_side_to_dict(mapping.translation),
        "entries": [_mapping_entry_to_dict(entry) for entry in mapping.entries],
    }


# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------


def cmd_dump(path: Path) -> int:
    raw = _normalize_newlines(path.read_text(encoding="utf-8"))
    frontmatter, body = _extract_frontmatter(raw)
    blocks = parse_blocks(body)

    output: dict[str, Any] = {
        "file": str(path),
        "blocks": [_block_to_dict(block) for block in blocks],
    }
    if frontmatter:
        output["frontmatter"] = frontmatter

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def _match_step(src: Block, tgt: Block) -> AlignmentStep:
    cost = 0.4
    confidence = "high"
    severity: str | None = None
    reason = "same-order block pair"

    if src.kind == tgt.kind:
        if src.kind == "heading":
            cost = 0.1
            reason = "same-order heading anchor"
            if src.heading_level != tgt.heading_level:
                cost += 0.7
                confidence = "medium"
                severity = "low"
                reason = "heading levels differ but order stays aligned"
            if _normalize_heading_text(src.md) == _normalize_heading_text(tgt.md):
                cost = max(0.0, cost - 0.05)
        elif src.kind == "flow":
            cost = 0.6
            reason = "same-order flow block"
            bucket_gap = abs(_size_bucket(src.words) - _size_bucket(tgt.words))
            if bucket_gap >= 2:
                cost += 0.5
                confidence = "medium"
                severity = "low"
                reason = "same-order flow block with noticeable size difference"
            if bucket_gap >= 3:
                cost += 0.8
                confidence = "low"
                severity = "medium"
                reason = "same-order flow block with large size difference"
        else:
            cost = 0.2 if src.md == tgt.md else 0.4
            reason = f"same-order {src.kind} block"
    elif src.kind == "heading" or tgt.kind == "heading":
        cost = 6.0
        confidence = "low"
        severity = "high"
        reason = "heading/body mismatch"
    elif "flow" in {src.kind, tgt.kind}:
        cost = 4.5
        confidence = "low"
        severity = "high"
        reason = "different block kinds in same-order pair"
    else:
        cost = 2.6
        confidence = "low"
        severity = "medium"
        reason = "different non-flow block kinds in same-order pair"

    return AlignmentStep(
        op="match",
        src_indices=(src.index,),
        tgt_indices=(tgt.index,),
        cost=cost,
        reason=reason,
        confidence=confidence,
        severity=severity,
    )


def _merge_source_step(src_a: Block, src_b: Block, tgt: Block) -> AlignmentStep | None:
    if src_a.kind != "flow" or src_b.kind != "flow" or tgt.kind != "flow":
        return None

    cost = 2.4
    confidence = "medium"
    severity = "medium"
    reason = "two adjacent source flow blocks likely map to one translation block"

    if min(src_a.words, src_b.words) <= 25:
        cost -= 0.9
        reason = "short adjacent source flow block likely merged into one translation block"
    if max(src_a.words, src_b.words) <= 35:
        cost -= 0.2
    if src_a.words >= 150 and src_b.words >= 150:
        cost += 1.2
        confidence = "low"
        severity = "high"
        reason = "two long source flow blocks would need manual merge review"
    if tgt.words <= 20 and (src_a.words + src_b.words) >= 140:
        cost += 0.7
        confidence = _downgrade_confidence(confidence, "low")

    return AlignmentStep(
        op="source_merge",
        src_indices=(src_a.index, src_b.index),
        tgt_indices=(tgt.index,),
        cost=cost,
        reason=reason,
        confidence=confidence,
        severity=severity,
    )


def _split_translation_step(src: Block, tgt_a: Block, tgt_b: Block) -> AlignmentStep | None:
    if src.kind != "flow" or tgt_a.kind != "flow" or tgt_b.kind != "flow":
        return None

    cost = 2.4
    confidence = "medium"
    severity = "medium"
    reason = "one source flow block likely split into two translation flow blocks"

    if min(tgt_a.words, tgt_b.words) <= 25:
        cost -= 0.9
        reason = "short adjacent translation flow block likely came from a split"
    if max(tgt_a.words, tgt_b.words) <= 35:
        cost -= 0.2
    if tgt_a.words >= 150 and tgt_b.words >= 150:
        cost += 1.2
        confidence = "low"
        severity = "high"
        reason = "two long translation flow blocks would need manual split review"
    if src.words <= 20 and (tgt_a.words + tgt_b.words) >= 140:
        cost += 0.7
        confidence = _downgrade_confidence(confidence, "low")

    return AlignmentStep(
        op="translation_split",
        src_indices=(src.index,),
        tgt_indices=(tgt_a.index, tgt_b.index),
        cost=cost,
        reason=reason,
        confidence=confidence,
        severity=severity,
    )


def _source_only_step(src: Block) -> AlignmentStep:
    if src.kind == "heading":
        cost = 1.2
        confidence = "medium"
        severity = "low"
        reason = "source-only heading"
    elif src.kind == "flow":
        cost = 3.0 if src.words > 25 else 2.2
        confidence = "medium" if src.words <= 25 else "low"
        severity = "medium"
        reason = "source-only flow block"
    else:
        cost = 1.4
        confidence = "medium"
        severity = "low"
        reason = f"source-only {src.kind} block"

    return AlignmentStep(
        op="source_only",
        src_indices=(src.index,),
        tgt_indices=(),
        cost=cost,
        reason=reason,
        confidence=confidence,
        severity=severity,
    )


def _translation_only_step(tgt: Block) -> AlignmentStep:
    if tgt.kind == "heading":
        cost = 1.2
        confidence = "medium"
        severity = "low"
        reason = "translation-only heading"
    elif tgt.kind == "flow":
        cost = 3.0 if tgt.words > 25 else 2.2
        confidence = "medium" if tgt.words <= 25 else "low"
        severity = "medium"
        reason = "translation-only flow block"
    else:
        cost = 1.4
        confidence = "medium"
        severity = "low"
        reason = f"translation-only {tgt.kind} block"

    return AlignmentStep(
        op="translation_only",
        src_indices=(),
        tgt_indices=(tgt.index,),
        cost=cost,
        reason=reason,
        confidence=confidence,
        severity=severity,
    )


def _candidate_steps(
    src_blocks: Sequence[Block],
    tgt_blocks: Sequence[Block],
    src_pos: int,
    tgt_pos: int,
) -> list[tuple[int, int, AlignmentStep]]:
    candidates: list[tuple[int, int, AlignmentStep]] = []

    if src_pos < len(src_blocks) and tgt_pos < len(tgt_blocks):
        candidates.append(
            (
                src_pos + 1,
                tgt_pos + 1,
                _match_step(src_blocks[src_pos], tgt_blocks[tgt_pos]),
            )
        )

    if src_pos + 1 < len(src_blocks) and tgt_pos < len(tgt_blocks):
        step = _merge_source_step(src_blocks[src_pos], src_blocks[src_pos + 1], tgt_blocks[tgt_pos])
        if step is not None:
            candidates.append((src_pos + 2, tgt_pos + 1, step))

    if src_pos < len(src_blocks) and tgt_pos + 1 < len(tgt_blocks):
        step = _split_translation_step(src_blocks[src_pos], tgt_blocks[tgt_pos], tgt_blocks[tgt_pos + 1])
        if step is not None:
            candidates.append((src_pos + 1, tgt_pos + 2, step))

    if src_pos < len(src_blocks):
        candidates.append((src_pos + 1, tgt_pos, _source_only_step(src_blocks[src_pos])))

    if tgt_pos < len(tgt_blocks):
        candidates.append((src_pos, tgt_pos + 1, _translation_only_step(tgt_blocks[tgt_pos])))

    return candidates


def _align_blocks(src_blocks: Sequence[Block], tgt_blocks: Sequence[Block]) -> list[AlignmentStep]:
    src_count = len(src_blocks)
    tgt_count = len(tgt_blocks)
    inf = float("inf")

    dp = [[inf] * (tgt_count + 1) for _ in range(src_count + 1)]
    prev: list[list[tuple[int, int, AlignmentStep] | None]] = [
        [None] * (tgt_count + 1) for _ in range(src_count + 1)
    ]
    dp[0][0] = 0.0

    for src_pos in range(src_count + 1):
        for tgt_pos in range(tgt_count + 1):
            base_cost = dp[src_pos][tgt_pos]
            if base_cost == inf:
                continue
            for next_src, next_tgt, step in _candidate_steps(src_blocks, tgt_blocks, src_pos, tgt_pos):
                total_cost = base_cost + step.cost
                if total_cost < dp[next_src][next_tgt]:
                    dp[next_src][next_tgt] = total_cost
                    prev[next_src][next_tgt] = (src_pos, tgt_pos, step)

    steps: list[AlignmentStep] = []
    src_pos = src_count
    tgt_pos = tgt_count
    while src_pos > 0 or tgt_pos > 0:
        back = prev[src_pos][tgt_pos]
        if back is None:
            raise ValueError("Failed to build alignment path")
        prev_src, prev_tgt, step = back
        steps.append(step)
        src_pos, tgt_pos = prev_src, prev_tgt

    steps.reverse()
    return steps


def _legacy_mapping_rows_from_entry(entry: MappingEntry) -> list[list[int | None]]:
    src = list(entry.source)
    tgt = list(entry.translation)

    if not src and not tgt:
        return []
    if not src:
        return [[None, index] for index in tgt]
    if not tgt:
        return [[index, None] for index in src]
    if len(src) == 1 and len(tgt) == 1:
        return [[src[0], tgt[0]]]

    rows: list[list[int | None]] = []
    for index in src[:-1]:
        rows.append([index, None])
    rows.append([src[-1], tgt[0]])
    for index in tgt[1:]:
        rows.append([None, index])
    return rows


def _pairs_from_entry(entry: MappingEntry, reason: str, confidence: str) -> list[dict[str, Any]]:
    rows = _legacy_mapping_rows_from_entry(entry)
    return [
        {
            "source": source_index,
            "translation": translation_index,
            "reason": reason,
            "confidence": confidence,
        }
        for source_index, translation_index in rows
    ]


def _warning_from_step(step: AlignmentStep, entry_index: int) -> dict[str, Any] | None:
    if step.severity is None:
        return None

    if step.op == "source_merge":
        kind = "possible-merge"
    elif step.op == "translation_split":
        kind = "possible-split"
    elif step.op == "source_only":
        kind = "source-only"
    elif step.op == "translation_only":
        kind = "translation-only"
    elif step.op == "match":
        kind = "suspicious-pair"
    else:
        kind = step.op

    return {
        "kind": kind,
        "severity": step.severity,
        "entry_index": entry_index,
        "source_indices": list(step.src_indices),
        "translation_indices": list(step.tgt_indices),
        "reason": step.reason,
        "confidence": step.confidence,
    }


def _risk_level(warnings: Sequence[dict[str, Any]]) -> str:
    severities = {warning["severity"] for warning in warnings}
    if "high" in severities:
        return "high"
    if "medium" in severities:
        return "medium"
    return "low"


def _is_ready_warning(warning: dict[str, Any]) -> bool:
    return warning["severity"] in {"medium", "high"}


def _mapping_entry_from_step(step: AlignmentStep) -> MappingEntry:
    review = None
    if step.severity is not None:
        review = {
            "severity": step.severity,
            "reason": step.reason,
            "confidence": step.confidence,
        }
    return MappingEntry(
        source=tuple(step.src_indices),
        translation=tuple(step.tgt_indices),
        review=review,
    )


def _build_analysis_report(source_doc: BlockDocument, target_doc: BlockDocument) -> dict[str, Any]:
    steps = _align_blocks(source_doc.blocks, target_doc.blocks)
    entries: list[MappingEntry] = []
    pairs: list[dict[str, Any]] = []
    legacy_mapping: list[list[int | None]] = []
    warnings: list[dict[str, Any]] = []

    for entry_index, step in enumerate(steps):
        entry = _mapping_entry_from_step(step)
        entries.append(entry)
        pairs.extend(_pairs_from_entry(entry, step.reason, step.confidence))
        legacy_mapping.extend(_legacy_mapping_rows_from_entry(entry))
        warning = _warning_from_step(step, entry_index)
        if warning is not None:
            warnings.append(warning)

    mapping = MappingDocument(
        version=MAPPING_VERSION,
        kind=MAPPING_KIND,
        source=_mapping_side_from_doc(source_doc),
        translation=_mapping_side_from_doc(target_doc),
        entries=tuple(entries),
    )

    ready = not any(_is_ready_warning(warning) for warning in warnings)
    report = {
        "ready": ready,
        "risk_level": _risk_level(warnings),
        "stats": {
            "source_blocks": len(source_doc.blocks),
            "translation_blocks": len(target_doc.blocks),
            "auto_pairs": sum(1 for entry in entries if entry.source and entry.translation),
            "review_regions": sum(1 for warning in warnings if _is_ready_warning(warning)),
            "mapping_entries": len(entries),
        },
        "warnings": warnings,
        "pairs": pairs,
        "legacy_mapping": legacy_mapping,
        "mapping": _mapping_document_to_dict(mapping),
    }
    return report


def cmd_analyze(
    source_path: Path,
    translation_path: Path,
    output_path: Path | None,
    mapping_out: Path | None,
) -> int:
    source_doc = _load_blocks(source_path)
    target_doc = _load_blocks(translation_path)
    report = _build_analysis_report(source_doc, target_doc)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["report_written"] = str(output_path)

    if mapping_out is not None:
        mapping_out.parent.mkdir(parents=True, exist_ok=True)
        mapping_out.write_text(
            json.dumps(report["mapping"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report["mapping_written"] = str(mapping_out)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------


def _normalize_index_list(value: Any, field_name: str, entry_index: int) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"Mapping entry {entry_index} field {field_name!r} must be an array")
    normalized: list[int] = []
    for idx, item in enumerate(value):
        if not isinstance(item, int):
            raise ValueError(
                f"Mapping entry {entry_index} field {field_name!r} contains a non-integer at position {idx}"
            )
        if item < 0:
            raise ValueError(
                f"Mapping entry {entry_index} field {field_name!r} contains a negative index {item}"
            )
        normalized.append(item)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Mapping entry {entry_index} field {field_name!r} contains duplicates")
    if normalized != sorted(normalized):
        raise ValueError(f"Mapping entry {entry_index} field {field_name!r} must be in ascending order")
    return tuple(normalized)


def _parse_mapping_side(raw: Any, label: str) -> MappingSide | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"Mapping {label} metadata must be an object")
    path = raw.get("path", "")
    sha256 = raw.get("sha256", "")
    block_count = raw.get("block_count", None)
    if path is not None and not isinstance(path, str):
        raise ValueError(f"Mapping {label} path must be a string")
    if sha256 is not None and not isinstance(sha256, str):
        raise ValueError(f"Mapping {label} sha256 must be a string")
    if block_count is not None and (not isinstance(block_count, int) or block_count < 0):
        raise ValueError(f"Mapping {label} block_count must be a non-negative integer")
    return MappingSide(
        path=path or "",
        sha256=sha256 or "",
        block_count=0 if block_count is None else block_count,
    )


def _mapping_document_from_legacy(raw_entries: list[Any]) -> MappingDocument:
    entries: list[MappingEntry] = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError(f"Each legacy mapping entry must be [src_idx, tgt_idx], got: {entry}")
        src_idx, tgt_idx = entry
        source: tuple[int, ...]
        translation: tuple[int, ...]
        if src_idx is None:
            source = ()
        elif isinstance(src_idx, int) and src_idx >= 0:
            source = (src_idx,)
        else:
            raise ValueError(f"Legacy source index must be a non-negative integer or null, got: {src_idx!r}")

        if tgt_idx is None:
            translation = ()
        elif isinstance(tgt_idx, int) and tgt_idx >= 0:
            translation = (tgt_idx,)
        else:
            raise ValueError(
                f"Legacy translation index must be a non-negative integer or null, got: {tgt_idx!r}"
            )

        entries.append(MappingEntry(source=source, translation=translation, review=None))

    return MappingDocument(
        version=1,
        kind="legacy-pair-array",
        source=None,
        translation=None,
        entries=tuple(entries),
    )


def _mapping_document_from_grouped(raw: dict[str, Any]) -> MappingDocument:
    if raw.get("version") != MAPPING_VERSION:
        raise ValueError(f"Unsupported mapping version: {raw.get('version')}")
    if raw.get("kind") != MAPPING_KIND:
        raise ValueError(f"Unsupported mapping kind: {raw.get('kind')}")

    raw_entries = raw.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("Grouped mapping must contain an entries array")

    entries: list[MappingEntry] = []
    for entry_index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Mapping entry {entry_index} must be an object")
        source = _normalize_index_list(raw_entry.get("source", []), "source", entry_index)
        translation = _normalize_index_list(raw_entry.get("translation", []), "translation", entry_index)
        review = raw_entry.get("review")
        if review is not None and not isinstance(review, dict):
            raise ValueError(f"Mapping entry {entry_index} review must be an object if present")
        entries.append(MappingEntry(source=source, translation=translation, review=review))

    return MappingDocument(
        version=MAPPING_VERSION,
        kind=MAPPING_KIND,
        source=_parse_mapping_side(raw.get("source"), "source"),
        translation=_parse_mapping_side(raw.get("translation"), "translation"),
        entries=tuple(entries),
    )


def _load_mapping(mapping_path: Path) -> MappingDocument:
    text = mapping_path.read_text(encoding="utf-8")
    parsed = json.loads(_strip_code_fence(text))

    if isinstance(parsed, dict) and "mapping" in parsed and isinstance(parsed["mapping"], (dict, list)):
        parsed = parsed["mapping"]

    if isinstance(parsed, list):
        return _mapping_document_from_legacy(parsed)
    if isinstance(parsed, dict):
        return _mapping_document_from_grouped(parsed)

    raise ValueError(
        "Mapping must be either a legacy pair array or a grouped mapping object."
    )


def _validate_mapping_metadata(
    mapping: MappingDocument,
    source_doc: BlockDocument,
    target_doc: BlockDocument,
) -> list[str]:
    severe_messages: list[str] = []

    if mapping.source is not None:
        if mapping.source.sha256 and mapping.source.sha256 != source_doc.sha256:
            severe_messages.append("mapping source file has changed since the mapping was created")
        if mapping.source.block_count and mapping.source.block_count != len(source_doc.blocks):
            severe_messages.append("mapping source block count does not match the current source file")

    if mapping.translation is not None:
        if mapping.translation.sha256 and mapping.translation.sha256 != target_doc.sha256:
            severe_messages.append("mapping translation file has changed since the mapping was created")
        if mapping.translation.block_count and mapping.translation.block_count != len(target_doc.blocks):
            severe_messages.append("mapping translation block count does not match the current translation file")

    return severe_messages


def _audit_mapping(
    mapping: MappingDocument,
    source_doc: BlockDocument,
    target_doc: BlockDocument,
) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    used_src: set[int] = set()
    used_tgt: set[int] = set()
    severe_messages: list[str] = _validate_mapping_metadata(mapping, source_doc, target_doc)

    source_only_run = 0
    translation_only_run = 0

    for position, entry in enumerate(mapping.entries):
        source_indices = entry.source
        translation_indices = entry.translation

        for src_idx in source_indices:
            if src_idx >= len(source_doc.blocks):
                raise ValueError(f"Source index {src_idx} out of range (0–{len(source_doc.blocks) - 1})")
            if src_idx in used_src:
                severe_messages.append(f"duplicate source index {src_idx} in mapping")
            used_src.add(src_idx)

        for tgt_idx in translation_indices:
            if tgt_idx >= len(target_doc.blocks):
                raise ValueError(
                    f"Translation index {tgt_idx} out of range (0–{len(target_doc.blocks) - 1})"
                )
            if tgt_idx in used_tgt:
                severe_messages.append(f"duplicate translation index {tgt_idx} in mapping")
            used_tgt.add(tgt_idx)

        if not source_indices and translation_indices:
            source_only_run = 0
            translation_only_run += 1
        elif source_indices and not translation_indices:
            translation_only_run = 0
            source_only_run += 1
        else:
            source_only_run = 0
            translation_only_run = 0

        if source_only_run >= 4:
            warnings.append(
                {
                    "severity": "medium",
                    "kind": "long-source-only-run",
                    "entry_index": position,
                    "reason": "four or more consecutive source-only entries",
                }
            )
            source_only_run = 0

        if translation_only_run >= 4:
            warnings.append(
                {
                    "severity": "medium",
                    "kind": "long-translation-only-run",
                    "entry_index": position,
                    "reason": "four or more consecutive translation-only entries",
                }
            )
            translation_only_run = 0

        if source_indices and translation_indices:
            source_blocks = [source_doc.blocks[index] for index in source_indices]
            translation_blocks = [target_doc.blocks[index] for index in translation_indices]
            source_kinds = {block.kind for block in source_blocks}
            translation_kinds = {block.kind for block in translation_blocks}

            if source_kinds == {"flow"} and translation_kinds == {"flow"}:
                pass
            elif source_kinds == {"heading"} and translation_kinds == {"heading"}:
                if len(source_blocks) == len(translation_blocks) == 1:
                    if source_blocks[0].heading_level != translation_blocks[0].heading_level:
                        warnings.append(
                            {
                                "severity": "low",
                                "kind": "heading-level-mismatch",
                                "entry_index": position,
                                "source": list(source_indices),
                                "translation": list(translation_indices),
                                "reason": "paired headings use different levels",
                            }
                        )
                else:
                    warnings.append(
                        {
                            "severity": "low",
                            "kind": "multi-heading-entry",
                            "entry_index": position,
                            "source": list(source_indices),
                            "translation": list(translation_indices),
                            "reason": "entry groups multiple heading blocks together",
                        }
                    )
            elif source_kinds != translation_kinds:
                severe_messages.append(
                    f"mapping entry {position} pairs incompatible block kinds: "
                    f"source {sorted(source_kinds)} vs translation {sorted(translation_kinds)}"
                )

    skipped_src = [block.index for block in source_doc.blocks if block.index not in used_src]
    skipped_tgt = [block.index for block in target_doc.blocks if block.index not in used_tgt]

    skipped_src_non_heading = [index for index in skipped_src if source_doc.blocks[index].kind != "heading"]
    skipped_tgt_non_heading = [index for index in skipped_tgt if target_doc.blocks[index].kind != "heading"]

    if skipped_src_non_heading:
        warnings.append(
            {
                "severity": "medium" if len(skipped_src_non_heading) > 3 else "low",
                "kind": "skipped-source-blocks",
                "indices": skipped_src_non_heading,
                "reason": "source blocks omitted from final mapping",
            }
        )
    if skipped_tgt_non_heading:
        warnings.append(
            {
                "severity": "medium" if len(skipped_tgt_non_heading) > 3 else "low",
                "kind": "skipped-translation-blocks",
                "indices": skipped_tgt_non_heading,
                "reason": "translation blocks omitted from final mapping",
            }
        )

    if severe_messages:
        details = "; ".join(dict.fromkeys(severe_messages))
        raise ValueError(f"Mapping audit failed: {details}")

    return {
        "risk_level": _risk_level(warnings),
        "warnings": warnings,
        "skipped_source_blocks": skipped_src,
        "skipped_translation_blocks": skipped_tgt,
        "entries": len(mapping.entries),
    }


def _append_entry_blocks(parts: list[str], blocks: Sequence[Block], indices: Sequence[int]) -> None:
    for index in indices:
        parts.append(blocks[index].md)
        parts.append("")


def cmd_join(mapping_path: Path, source_path: Path, trans_path: Path, output_path: Path) -> int:
    mapping = _load_mapping(mapping_path)
    source_doc = _load_blocks(source_path)
    target_doc = _load_blocks(trans_path)
    audit = _audit_mapping(mapping, source_doc, target_doc)

    parts: list[str] = []
    frontmatter = source_doc.frontmatter or target_doc.frontmatter
    if frontmatter:
        parts.append(frontmatter)
        parts.append("")

    for entry in mapping.entries:
        _append_entry_blocks(parts, source_doc.blocks, entry.source)
        _append_entry_blocks(parts, target_doc.blocks, entry.translation)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bilingual = "\n".join(parts).rstrip("\n") + "\n"
    output_path.write_text(bilingual, encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(output_path),
                "entries": len(mapping.entries),
                "audit": audit,
            },
            ensure_ascii=False,
        )
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    ap = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name if sys.argv else "bilingual.py",
        description="Minimal bilingual Markdown assembly toolset.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    dump_ap = sub.add_parser("dump", help="Parse a Markdown file into a JSON block list")
    dump_ap.add_argument("file", help="Markdown file to parse")

    analyze_ap = sub.add_parser(
        "analyze",
        help="Analyze source/translation structure and emit a candidate grouped mapping",
    )
    analyze_ap.add_argument("source", help="Source Markdown or dump JSON file")
    analyze_ap.add_argument("translation", help="Translated Markdown or dump JSON file")
    analyze_ap.add_argument("-o", "--output", help="Write full analysis report JSON")
    analyze_ap.add_argument("--mapping-out", help="Write grouped mapping JSON object")

    join_ap = sub.add_parser("join", help="Stitch blocks per a mapping into bilingual output")
    join_ap.add_argument("mapping", help="Legacy or grouped mapping JSON file")
    join_ap.add_argument("source", help="Source Markdown or dump JSON file")
    join_ap.add_argument("translation", help="Translated Markdown or dump JSON file")
    join_ap.add_argument("-o", "--output", required=True, help="Output bilingual Markdown file")

    args = ap.parse_args(argv)

    try:
        if args.command == "dump":
            return cmd_dump(Path(args.file))
        if args.command == "analyze":
            return cmd_analyze(
                Path(args.source),
                Path(args.translation),
                Path(args.output) if args.output else None,
                Path(args.mapping_out) if args.mapping_out else None,
            )
        if args.command == "join":
            return cmd_join(
                Path(args.mapping),
                Path(args.source),
                Path(args.translation),
                Path(args.output),
            )
        ap.error(f"Unknown command: {args.command}")
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
