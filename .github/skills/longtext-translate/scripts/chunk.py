#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from markdown_it import MarkdownIt


PLAN_VERSION = 1
SPLITTER_VERSION = "markdown-block-v1"
PLAN_FILE_NAME = "chunk-plan.json"
PREVIEW_FILE_NAME = "chunk-preview.html"
PREVIEW_HOST = "127.0.0.1"
PREVIEW_IDLE_TIMEOUT_SECONDS = 10 * 60


@dataclass
class Block:
    kind: str
    md: str
    words: int
    heading_level: int | None = None


@dataclass(frozen=True)
class Segment:
    id: str
    kind: str
    md: str
    words: int
    heading_level: int | None = None


@dataclass
class SourceModel:
    source: Path
    source_sha256: str
    frontmatter: str
    segments: list[Segment]
    chunks: list[list[str]]
    max_words: int


parser = MarkdownIt("js-default", {"html": True})


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"preview", "materialize", "serve-preview"}
    if argv and argv[0] not in commands and argv[0] not in {"-h", "--help"}:
        argv = ["materialize", *argv]

    ap = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name,
        usage="%(prog)s {preview,materialize,serve-preview} <file> [options]",
    )
    subparsers = ap.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser("preview", help="Generate a chunk preview HTML and local confirmation server")
    add_common_args(preview)
    preview.add_argument("--no-server", action="store_true", help="Only write chunk-preview.html without starting the local server")

    materialize = subparsers.add_parser("materialize", help="Write final chunk files")
    add_common_args(materialize)
    materialize.add_argument("--plan-json", default=None, help="Explicit compact plan JSON")
    materialize.add_argument("--plan-file", default=None, metavar="FILE", help="Explicit compact plan JSON file")

    serve_preview_parser = subparsers.add_parser("serve-preview", help=argparse.SUPPRESS)
    serve_preview_parser.add_argument("--output-dir", required=True)
    serve_preview_parser.add_argument("--host", default=PREVIEW_HOST)
    serve_preview_parser.add_argument("--port", type=parse_positive_int, required=True)
    serve_preview_parser.add_argument("--idle-timeout", type=parse_positive_int, default=PREVIEW_IDLE_TIMEOUT_SECONDS)

    args = ap.parse_args(argv)

    try:
        if args.command == "preview":
            result = create_preview(args.file, args.max_words, args.output_dir, start_server=not args.no_server)
        elif args.command == "serve-preview":
            result = serve_preview(output_dir=args.output_dir, host=args.host, port=args.port, idle_timeout=args.idle_timeout)
        else:
            if args.plan_json and args.plan_file:
                ap.error("--plan-json and --plan-file are mutually exclusive")
            result = materialize_chunks(
                args.file,
                args.max_words,
                args.output_dir,
                plan_json=args.plan_json,
                plan_file=args.plan_file,
            )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False))
    return 0


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("file")
    ap.add_argument("--max-words", type=parse_positive_int, default=5000)
    ap.add_argument("--output-dir", default="")


def parse_positive_int(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid --max-words value: {value}") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Invalid --max-words value: {value}")
    return parsed


def create_preview(file: str, max_words: int = 5000, output_dir: str = "", start_server: bool = True) -> dict[str, object]:
    model = build_source_model(file, max_words)
    output_root = resolve_output_root(model.source, output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    preview_path = output_root / PREVIEW_FILE_NAME
    data = {
        "version": PLAN_VERSION,
        "source_name": model.source.name,
        "source_sha256": model.source_sha256,
        "splitter": SPLITTER_VERSION,
        "max_words": model.max_words,
        "confirm_endpoint": "/chunk-plan",
        "plan_file": PLAN_FILE_NAME,
        "segments": [
            {"id": segment.id, "kind": segment.kind, "words": segment.words, "text": segment.md}
            for segment in model.segments
        ],
        "chunks": model.chunks,
    }

    template_path = Path(__file__).with_name("chunk-preview.html")
    template = template_path.read_text(encoding="utf-8")
    if "__CHUNK_PREVIEW_DATA__" not in template:
        raise ValueError(f"Preview template missing data placeholder: {template_path}")

    preview_path.write_text(
        template.replace("__CHUNK_PREVIEW_DATA__", json_for_script(data)),
        encoding="utf-8",
    )

    result: dict[str, object] = {
        "source": file,
        "preview_file": str(preview_path),
    }
    if start_server:
        result.update(start_preview_server(output_root))
    return result


def materialize_chunks(
    file: str,
    max_words: int = 5000,
    output_dir: str = "",
    plan_json: str | None = None,
    plan_file: str | None = None,
) -> dict[str, object]:
    source = Path(file)
    output_root = resolve_output_root(source, output_dir)
    plan = read_adjusted_plan(plan_json=plan_json, plan_file=plan_file)
    plan_source = "explicit" if plan else "default"
    if plan is None:
        confirmed_plan = output_root / PLAN_FILE_NAME
        if confirmed_plan.exists():
            plan = read_adjusted_plan(plan_json=None, plan_file=str(confirmed_plan))
            plan_source = PLAN_FILE_NAME

    effective_max_words = int(plan.get("max_words", max_words)) if plan else max_words
    model = build_source_model(file, effective_max_words)
    chunks = validate_plan(plan, model) if plan else model.chunks
    chunk_dir = write_chunk_files(model, chunks, output_root)

    return {
        "source": file,
        "chunks": len(chunks),
        "output_dir": str(chunk_dir),
        "frontmatter": bool(model.frontmatter),
        "plan": plan_source,
        "chunk_index": build_chunk_index(model.segments, chunks),
    }


def start_preview_server(output_root: Path) -> dict[str, object]:
    port = find_free_port(PREVIEW_HOST)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "serve-preview",
        "--output-dir",
        str(output_root),
        "--host",
        PREVIEW_HOST,
        "--port",
        str(port),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    url = f"http://localhost:{port}/"
    wait_for_server(url)
    return {"preview_url": url}


def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def wait_for_server(url: str) -> None:
    deadline = time.time() + 5
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5):
                return
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            time.sleep(0.05)
    raise RuntimeError(f"Preview server did not start: {last_error}")


def serve_preview(output_dir: str, host: str, port: int, idle_timeout: int) -> dict[str, object]:
    output_root = Path(output_dir)
    preview_path = output_root / PREVIEW_FILE_NAME
    if not preview_path.exists():
        raise ValueError(f"Preview file not found: {preview_path}")

    handler = make_preview_handler(output_root)
    server = http.server.ThreadingHTTPServer((host, port), handler)
    server.timeout = 1
    server.last_activity = time.time()  # type: ignore[attr-defined]
    server.shutdown_requested = False  # type: ignore[attr-defined]
    stop_reason = "idle timeout"
    while time.time() - server.last_activity < idle_timeout:  # type: ignore[attr-defined]
        if server.shutdown_requested:  # type: ignore[attr-defined]
            stop_reason = "confirmed"
            break
        server.handle_request()
    server.server_close()
    return {"server": "stopped", "reason": stop_reason}


def make_preview_handler(output_root: Path) -> type[http.server.BaseHTTPRequestHandler]:
    class PreviewHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.server.last_activity = time.time()  # type: ignore[attr-defined]
            path = self.path.split("?", 1)[0]
            if path == "/heartbeat":
                self.send_response(204)
                self.end_headers()
                return
            if path not in {"/", f"/{PREVIEW_FILE_NAME}"}:
                self.send_error(404)
                return
            html = (output_root / PREVIEW_FILE_NAME).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def do_POST(self) -> None:
            self.server.last_activity = time.time()  # type: ignore[attr-defined]
            path = self.path.split("?", 1)[0]
            if path != "/chunk-plan":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 20 * 1024 * 1024:
                self.send_json(400, {"ok": False, "error": "Invalid plan size"})
                return
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("Request body must be a JSON object")
                confirm_default = bool(body.get("confirm_default"))
                raw_plan = body.get("plan")
                if confirm_default:
                    plan_path = output_root / PLAN_FILE_NAME
                    if plan_path.exists():
                        plan_path.unlink()
                    self.send_json(200, {"ok": True, "plan_file": None})
                    self.server.shutdown_requested = True  # type: ignore[attr-defined]
                    return
                if not isinstance(raw_plan, dict):
                    raise ValueError("Plan must be a JSON object")
                plan_path = output_root / PLAN_FILE_NAME
                plan_path.write_text(json.dumps(raw_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                self.send_json(200, {"ok": True, "plan_file": str(plan_path)})
                self.server.shutdown_requested = True  # type: ignore[attr-defined]
            except Exception as error:
                self.send_json(400, {"ok": False, "error": str(error)})

        def send_json(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return PreviewHandler


def resolve_output_root(source: Path, output_dir: str) -> Path:
    return Path(output_dir) if output_dir else source.parent


def build_source_model(file: str, max_words: int) -> SourceModel:
    source = Path(file)
    raw_content = normalize_newlines(source.read_text(encoding="utf-8"))
    frontmatter, body = extract_frontmatter(raw_content)
    blocks = parse_markdown(body)
    segments = build_segments(blocks, max_words)
    return SourceModel(
        source=source,
        source_sha256=sha256_text(raw_content),
        frontmatter=frontmatter,
        segments=segments,
        chunks=build_default_chunks(segments, max_words),
        max_words=max_words,
    )


def read_adjusted_plan(plan_json: str | None, plan_file: str | None) -> dict[str, Any] | None:
    if plan_file:
        text = Path(plan_file).read_text(encoding="utf-8")
    elif plan_json:
        text = plan_json
    else:
        return None

    parsed = json.loads(strip_code_fence(text))
    if not isinstance(parsed, dict):
        raise ValueError("Adjusted plan must be a JSON object")
    return parsed


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def validate_plan(plan: dict[str, Any], model: SourceModel) -> list[list[str]]:
    if plan.get("version") != PLAN_VERSION:
        raise ValueError(f"Unsupported plan version: {plan.get('version')}")
    if plan.get("source_sha256") != model.source_sha256:
        raise ValueError("Adjusted plan does not match the current source file")
    if plan.get("splitter") != SPLITTER_VERSION:
        raise ValueError(f"Unsupported splitter: {plan.get('splitter')}")
    if plan.get("max_words") != model.max_words:
        raise ValueError("Adjusted plan max_words does not match the current splitter settings")

    raw_chunks = plan.get("chunks")
    if not isinstance(raw_chunks, list):
        raise ValueError("Adjusted plan must contain a chunks array")

    chunks: list[list[str]] = []
    for index, raw_chunk in enumerate(raw_chunks, 1):
        if not isinstance(raw_chunk, list):
            raise ValueError(f"Chunk {index} must be an array of segment IDs")
        if not raw_chunk:
            raise ValueError(f"Chunk {index} is empty")
        chunk: list[str] = []
        for segment_id in raw_chunk:
            if not isinstance(segment_id, str):
                raise ValueError(f"Chunk {index} contains a non-string segment ID")
            chunk.append(segment_id)
        chunks.append(chunk)

    expected_ids = [segment.id for segment in model.segments]
    actual_ids = [segment_id for chunk in chunks for segment_id in chunk]
    if not expected_ids and not chunks:
        return chunks
    if actual_ids != expected_ids:
        raise ValueError("Adjusted plan must include every segment exactly once in the original order")

    return chunks


def write_chunk_files(model: SourceModel, chunks: list[list[str]], output_root: Path) -> Path:
    chunk_dir = output_root / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    for old_chunk in chunk_dir.glob("chunk-*.md"):
        if re.fullmatch(r"chunk-\d+\.md", old_chunk.name):
            old_chunk.unlink()

    frontmatter_path = chunk_dir / "frontmatter.md"
    if model.frontmatter:
        frontmatter_path.write_text(model.frontmatter, encoding="utf-8")
    elif frontmatter_path.exists():
        frontmatter_path.unlink()

    segment_map = {segment.id: segment for segment in model.segments}
    for index, chunk in enumerate(chunks, 1):
        text = "\n\n".join(segment_map[segment_id].md for segment_id in chunk)
        (chunk_dir / f"chunk-{index:02d}.md").write_text(text, encoding="utf-8")

    return chunk_dir


def normalize_newlines(text: str) -> str:
    return text.removeprefix("﻿").replace("\r\n", "\n").replace("\r", "\n")


def trim_boundary_blank_lines(text: str) -> str:
    return re.sub(r"\n+$", "", re.sub(r"^\n+", "", text))


def extract_frontmatter(content: str) -> tuple[str, str]:
    lines = content.split("\n")
    if not lines or lines[0] != "---":
        return "", content

    for index in range(1, len(lines)):
        if lines[index] in {"---", "..."}:
            return "\n".join(lines[: index + 1]), "\n".join(lines[index + 1 :]).lstrip("\n")

    return "", content


def parse_markdown(content: str) -> list[Block]:
    if not content.strip():
        return []

    lines = content.split("\n")
    blocks: list[Block] = []

    for token in parser.parse(content, {}):
        if not token.map or token.level != 0:
            continue
        if token.nesting not in {0, 1}:
            continue

        start_line, end_line = token.map
        md = trim_boundary_blank_lines("\n".join(lines[start_line:end_line]))
        if not md:
            continue

        blocks.append(make_block(token_type_to_block_kind(token.type), md, heading_level=heading_level_for_token(token.type, md)))

    if not blocks:
        body = trim_boundary_blank_lines(content)
        if body:
            blocks.append(make_block("flow", body))

    return blocks


def token_type_to_block_kind(token_type: str) -> str:
    if token_type == "heading_open":
        return "heading"
    if token_type == "hr":
        return "thematicBreak"
    if token_type == "html_block":
        return "html"
    if token_type in {"fence", "code_block"}:
        return "code"
    return "flow"


def heading_level_for_token(token_type: str, md: str) -> int | None:
    if token_type != "heading_open":
        return None
    match = re.match(r"^(#{1,6})\s+", md)
    return len(match.group(1)) if match else None


def make_block(kind: str, md: str, heading_level: int | None = None) -> Block:
    trimmed = trim_boundary_blank_lines(md)
    return Block(kind=kind, md=trimmed, words=count_words(trimmed), heading_level=heading_level)


def build_segments(blocks: list[Block], max_words_per_chunk: int) -> list[Segment]:
    segments: list[Segment] = []
    for block in blocks:
        for split_block in split_oversized_block(block, max_words_per_chunk):
            segments.append(
                Segment(
                    id=f"seg-{len(segments) + 1:06d}",
                    kind=split_block.kind,
                    md=split_block.md,
                    words=split_block.words,
                    heading_level=split_block.heading_level,
                )
            )
    return segments


def build_default_chunks(segments: list[Segment], max_words_per_chunk: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    current_ids: list[str] = []
    current_words = 0

    for segment in segments:
        if current_ids and current_words + segment.words > max_words_per_chunk:
            chunks.append(current_ids)
            current_ids = [segment.id]
            current_words = segment.words
        else:
            current_ids.append(segment.id)
            current_words += segment.words

    if current_ids:
        chunks.append(current_ids)

    return chunks


def split_oversized_block(block: Block, max_words_per_chunk: int) -> list[Block]:
    if block.words <= max_words_per_chunk:
        return [block]

    if block.kind in {"heading", "thematicBreak", "html", "code"}:
        return [block]

    lines = block.md.split("\n")
    if len(lines) <= 1:
        return [block]

    split_blocks: list[Block] = []
    buffer: list[str] = []
    buffer_words = 0

    for line in lines:
        line_words = count_words(line)
        if buffer_words + line_words > max_words_per_chunk and buffer:
            split_blocks.append(make_block(block.kind, "\n".join(buffer)))
            buffer = [line]
            buffer_words = line_words
            continue

        buffer.append(line)
        buffer_words += line_words

    if buffer:
        split_blocks.append(make_block(block.kind, "\n".join(buffer)))

    return split_blocks


def build_chunk_index(segments: list[Segment], chunks: list[list[str]]) -> list[dict[str, object]]:
    segment_map = {segment.id: segment for segment in segments}
    starts_at = chunk_start_paths(segments, chunks)
    index: list[dict[str, object]] = []
    for chunk_number, chunk in enumerate(chunks, 1):
        item: dict[str, object] = {
            "id": f"chunk-{chunk_number:02d}",
            "words": sum(segment_map[segment_id].words for segment_id in chunk),
        }
        if starts_at[chunk_number - 1]:
            item["starts_at"] = starts_at[chunk_number - 1]
        index.append(item)
    return index


def chunk_start_paths(segments: list[Segment], chunks: list[list[str]]) -> list[str | None]:
    chunk_start_ids = {chunk[0] for chunk in chunks if chunk}
    starts_at_by_id: dict[str, str | None] = {}
    heading_stack: list[tuple[int, str]] = []

    for segment in segments:
        if segment.kind == "heading" and segment.heading_level is not None:
            heading_stack = [(level, text) for level, text in heading_stack if level < segment.heading_level]
            heading_stack.append((segment.heading_level, heading_text(segment.md)))
        if segment.id in chunk_start_ids:
            starts_at_by_id[segment.id] = format_heading_path(heading_stack)

    return [starts_at_by_id.get(chunk[0]) if chunk else None for chunk in chunks]


def heading_text(md: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", md).strip()


def format_heading_path(heading_stack: list[tuple[int, str]]) -> str | None:
    if not heading_stack:
        return None
    return " > ".join(text for _, text in heading_stack[-2:] if text)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_for_script(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return encoded.replace("</", "<\\/").replace(" ", "\\u2028").replace(" ", "\\u2029")


def count_words(text: str) -> int:
    cleaned = re.sub(r"[#*`\[\]()>|_~-]", " ", text)
    cjk = re.findall(r"[一-鿿㐀-䶿豈-﫿]", cleaned)
    latin = re.findall(r"[a-zA-Z0-9]+", cleaned)
    return len(cjk) + len(latin)


if __name__ == "__main__":
    raise SystemExit(main())
