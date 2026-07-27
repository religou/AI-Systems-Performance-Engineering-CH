#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import hashlib
import http.server
import json
import re
import socket
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import autocorrect_py
from markdown_it import MarkdownIt


parser = MarkdownIt("commonmark", {"html": True}).enable("table")

CHINESE_CHARACTERS = r"一-鿿㐀-䶿豈-﫿"
CHINESE_RE = re.compile(rf"[{CHINESE_CHARACTERS}]")
CJK_EMPHASIS_PUNCT = "。！？；：、，" + "」』》〉】）…"
EMPHASIS_MARKERS = ("***", "___", "**", "__", "*", "_")
REFERENCE_DEFINITION_RE = re.compile(r"^\s{0,3}\[[^\]]+\]:")
FOOTNOTE_DEFINITION_RE = re.compile(r"^\s{0,3}\[\^[^\]]+\]:")
URL_RE = re.compile(r"https?://[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
NON_DOLLAR_MATH_DELIMITERS = (("$$", "$$"), (r"\(", r"\)"), (r"\[", r"\]"))
INLINE_FORMATTING_MARKERS = "*_~"
ASCII_WORD_CHARACTERS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
OPEN_QUOTE_CONTEXT_CHARS = set("([{<（【《「『〈〔［｛")
CLOSE_QUOTE_CONTEXT_CHARS = set(")]}>）】》」』〉〕］｝")
QUOTE_FOLLOWING_PUNCTUATION = set("，。！？；：、,.!?;:")
FULLWIDTH_PUNCTUATION = {
    ":": "：",
    "!": "！",
    "?": "？",
    ";": "；",
    ",": "，",
    "!?": "！？",
    "?!": "？！",
}
CJKISH_PUNCTUATION = set("，。！？；：、“”‘’（）【】《》「」『』")
CJK_PUNCTUATION_WITHOUT_SURROUNDING_SPACES = "，。！？；：（）【】《》「」『』“”‘’"
REPEATED_PUNCTUATION_RE = re.compile(r"([！？])\1+")
REPEATED_MIXED_PUNCTUATION_RE = re.compile(r"([！？]{3,})")
BLOCKQUOTE_PREFIX_RE = re.compile(r"^([ \t]{0,3}(?:>[ \t]?)+)")
FORBIDDEN_ANCESTORS = {
    "table_open",
    "thead_open",
    "tbody_open",
    "tr_open",
    "th_open",
    "td_open",
}
ALLOWED_CONTAINERS = {"heading_open", "paragraph_open"}
URL_TRAILING_PUNCTUATION = set(".,!?;:，。！？；：、'\"")
URL_TRAILING_BRACKETS = {
    ")": "(",
    "]": "[",
    "}": "{",
    "）": "（",
    "】": "【",
    "》": "《",
    "」": "「",
    "』": "『",
    "〉": "〈",
    "〕": "〔",
    "］": "［",
    "｝": "｛",
}
MATH_SIGNAL_RE = re.compile(r"[A-Za-z\\^_=<>+\-*/(){}\[\]]")

PREVIEW_FILE_NAME = "polish-preview.html"
PREVIEW_HOST = "127.0.0.1"
PREVIEW_IDLE_TIMEOUT_SECONDS = 10 * 60
PLACEHOLDER_MARKER = "__POLISH_PREVIEW_DATA__"


@dataclass
class ProtectedSpans:
    prefix: str
    suffix: str
    items: list[str]


def fix_cjk_emphasis(text: str) -> str:
    """Normalize simple emphasis spacing while keeping CommonMark parseable."""
    if not any(marker in text for marker in EMPHASIS_MARKERS):
        return text

    masked, protected = protect_inline_spans(text)
    for marker in EMPHASIS_MARKERS:
        masked = normalize_emphasis_spacing_for_marker(masked, marker)
        masked = add_space_after_cjk_emphasis(masked, marker)
    return restore_protected_spans(masked, protected)


def normalize_emphasis_spacing_for_marker(text: str, marker: str) -> str:
    pattern = emphasis_spacing_pattern(marker)

    def replace(match: re.Match[str]) -> str:
        normalized_content = match.group("content").strip(" \t")
        replacement = f"{marker}{normalized_content}{marker}"
        next_char = match.string[match.end():match.end() + 1]
        if (
            normalized_content
            and normalized_content[-1] in CJK_EMPHASIS_PUNCT
            and next_char
            and not next_char.isspace()
            and not is_unicode_punctuation(next_char)
        ):
            replacement += " "
        return replacement

    return pattern.sub(replace, text)


def add_space_after_cjk_emphasis(text: str, marker: str) -> str:
    pieces: list[str] = []
    cursor = 0

    for start, end in iter_emphasis_spans(text, marker):
        pieces.append(text[cursor:start])
        span = text[start:end]
        pieces.append(span)

        next_char = text[end:end + 1]
        if should_add_space_after_emphasis(span, next_char, marker):
            pieces.append(" ")
        cursor = end

    if cursor == 0:
        return text

    pieces.append(text[cursor:])
    return "".join(pieces)



def iter_emphasis_spans(text: str, marker: str):
    marker_length = len(marker)
    index = 0

    while index < len(text):
        start = text.find(marker, index)
        if start == -1:
            return
        if not is_marker_boundary(text, start, marker):
            index = start + 1
            continue

        content_start = start + marker_length
        end = find_emphasis_closing_marker(text, content_start, marker)
        if end is None:
            index = start + marker_length
            continue

        yield start, end + marker_length
        index = end + marker_length



def is_marker_boundary(text: str, index: int, marker: str) -> bool:
    marker_char = marker[0]
    before = text[index - 1:index] if index > 0 else ""
    after_index = index + len(marker)
    after = text[after_index:after_index + 1]
    return before != marker_char and after != marker_char



def find_emphasis_closing_marker(text: str, start: int, marker: str) -> int | None:
    index = start
    while True:
        end = text.find(marker, index)
        if end == -1:
            return None
        if not is_marker_boundary(text, end, marker):
            index = end + 1
            continue

        content = text[start:end]
        if is_simple_emphasis_content(content):
            return end
        index = end + 1



def is_simple_emphasis_content(content: str) -> bool:
    stripped = content.strip(" \t")
    if not stripped:
        return False
    if any(character in INLINE_FORMATTING_MARKERS or character == "\n" for character in content):
        return False
    return True



def should_add_space_after_emphasis(span: str, next_char: str, marker: str) -> bool:
    if not next_char or next_char.isspace() or is_unicode_punctuation(next_char):
        return False

    content = span[len(marker):-len(marker)]
    stripped = content.rstrip(" \t")
    return bool(stripped) and stripped[-1] in CJK_EMPHASIS_PUNCT


def emphasis_spacing_pattern(marker: str) -> re.Pattern[str]:
    marker_char = re.escape(marker[0])
    escaped_marker = re.escape(marker)
    excluded = re.escape(INLINE_FORMATTING_MARKERS)
    return re.compile(
        rf"(?<!{marker_char})"
        rf"{escaped_marker}"
        rf"(?P<leading>[ \t]+)"
        rf"(?P<content>[^{excluded}\n]*?\S[^{excluded}\n]*?)"
        rf"(?P<trailing>[ \t]+)"
        rf"{escaped_marker}"
        rf"(?!{marker_char})"
    )


def is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def is_unicode_punctuation(character: str) -> bool:
    return bool(character) and unicodedata.category(character).startswith("P")


def apply_spacing(text: str) -> str:
    """Apply CJK/English spacing to visible prose text."""
    return autocorrect_py.format(text)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    commands = {"preview", "serve-preview"}
    if not args or (args[0] not in commands and args[0] not in {"-h", "--help"}):
        return legacy_main(args)

    ap = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name,
        usage="%(prog)s {preview,serve-preview} <file> [options]",
    )
    subparsers = ap.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser("preview", help="Generate a diff preview HTML and local confirmation server")
    preview.add_argument("file")
    preview.add_argument("--output-dir", default="")
    preview.add_argument("--no-server", action="store_true", help="Only write polish-preview.html without starting the local server")

    serve_preview_parser = subparsers.add_parser("serve-preview", help=argparse.SUPPRESS)
    serve_preview_parser.add_argument("--output-dir", required=True)
    serve_preview_parser.add_argument("--file", required=True)
    serve_preview_parser.add_argument("--host", default=PREVIEW_HOST)
    serve_preview_parser.add_argument("--port", type=int, required=True)
    serve_preview_parser.add_argument("--idle-timeout", type=int, default=PREVIEW_IDLE_TIMEOUT_SECONDS)

    parsed = ap.parse_args(args)

    try:
        if parsed.command == "preview":
            result = create_preview(parsed.file, parsed.output_dir, start_server=not parsed.no_server)
        else:
            result = serve_preview(
                output_dir=parsed.output_dir,
                file_path=parsed.file,
                host=parsed.host,
                port=parsed.port,
                idle_timeout=parsed.idle_timeout,
            )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False))
    return 0


def legacy_main(args: list[str]) -> int:
    if args and args[0] in {"-h", "--help"}:
        print(f"Usage: {Path(sys.argv[0]).name} <file.md>")
        print(f"       {Path(sys.argv[0]).name} preview <file.md> [--output-dir <dir>] [--no-server]")
        return 0
    if len(args) != 1:
        print(f"Usage: {Path(sys.argv[0]).name} <file.md>", file=sys.stderr)
        return 1

    path = Path(args[0])
    original = path.read_text(encoding="utf-8")

    result = polish_markdown(original)

    if result != original:
        path.write_text(result, encoding="utf-8")

    print(f"Polished: {path}")
    return 0


# ── Preview mode ──


def create_preview(file: str, output_dir: str = "", start_server: bool = True) -> dict[str, object]:
    path = Path(file)
    original = path.read_text(encoding="utf-8")
    source_sha256 = sha256_text(original)
    polished = polish_markdown(original)

    output_root = Path(output_dir) if output_dir else path.parent
    output_root.mkdir(parents=True, exist_ok=True)

    diff_blocks = compute_diff(original, polished)
    total_changes = sum(1 for block in diff_blocks if block["type"] == "changed")

    data: dict[str, object] = {
        "source_name": path.name,
        "source_sha256": source_sha256,
        "confirm_endpoint": "/polish-apply",
        "total_changes": total_changes,
        "blocks": diff_blocks,
    }

    template_path = Path(__file__).with_name(PREVIEW_FILE_NAME)
    template = template_path.read_text(encoding="utf-8")
    if PLACEHOLDER_MARKER not in template:
        raise ValueError(f"Preview template missing data placeholder: {template_path}")

    preview_path = output_root / PREVIEW_FILE_NAME
    preview_path.write_text(
        template.replace(PLACEHOLDER_MARKER, json_for_script(data)),
        encoding="utf-8",
    )

    result: dict[str, object] = {
        "source": file,
        "preview_file": str(preview_path),
        "source_sha256": source_sha256,
        "total_changes": total_changes,
    }
    if start_server:
        result.update(start_preview_server(output_root, file))
    return result


def start_preview_server(output_root: Path, file: str) -> dict[str, object]:
    port = find_free_port(PREVIEW_HOST)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "serve-preview",
        "--output-dir",
        str(output_root),
        "--file",
        str(Path(file).resolve()),
        "--host",
        PREVIEW_HOST,
        "--port",
        str(port),
    ]
    subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    url = f"http://localhost:{port}/"
    wait_for_server(url)
    return {"preview_url": url}


def serve_preview(output_dir: str, file_path: str, host: str, port: int, idle_timeout: int) -> dict[str, object]:
    output_root = Path(output_dir)
    preview_path = output_root / PREVIEW_FILE_NAME
    if not preview_path.exists():
        raise ValueError(f"Preview file not found: {preview_path}")

    handler = make_preview_handler(output_root, file_path)
    server = http.server.ThreadingHTTPServer((host, port), handler)
    server.timeout = 1
    server.last_activity = time.time()  # type: ignore[attr-defined]
    server.shutdown_requested = False  # type: ignore[attr-defined]
    stop_reason = "idle timeout"
    while time.time() - server.last_activity < idle_timeout:  # type: ignore[attr-defined]
        if server.shutdown_requested:  # type: ignore[attr-defined]
            stop_reason = "applied"
            break
        server.handle_request()
    server.server_close()
    return {"server": "stopped", "reason": stop_reason}


def make_preview_handler(output_root: Path, file_path: str) -> type[http.server.BaseHTTPRequestHandler]:
    target_file = Path(file_path)

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
            if path != "/polish-apply":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 20 * 1024 * 1024:
                self.send_json(400, {"ok": False, "error": "Invalid request size"})
                return
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                source_sha256 = body.get("source_sha256", "")
                has_result_text = "result_text" in body
                result_text = body.get("result_text", "")
                if not target_file.exists():
                    self.send_json(400, {"ok": False, "error": "Source file no longer exists"})
                    return
                current = target_file.read_text(encoding="utf-8")
                if source_sha256 and sha256_text(current) != source_sha256:
                    self.send_json(409, {"ok": False, "error": "File has changed since preview was generated"})
                    return
                if has_result_text:
                    target_file.write_text(str(result_text), encoding="utf-8")
                else:
                    polished = polish_markdown(current)
                    if polished != current:
                        target_file.write_text(polished, encoding="utf-8")
                self.send_json(200, {"ok": True})
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


# ── Diff computation ──


def compute_diff(original: str, polished: str) -> list[dict[str, object]]:
    orig_lines = original.splitlines(keepends=True)
    pol_lines = polished.splitlines(keepends=True)
    sm = difflib.SequenceMatcher(None, orig_lines, pol_lines)

    blocks: list[dict[str, object]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        orig_text = "".join(orig_lines[i1:i2])
        pol_text = "".join(pol_lines[j1:j2])
        if tag == "equal":
            blocks.append({"type": "equal", "text": orig_text})
            continue
        blocks.append({
            "type": "changed",
            "segments": compute_char_diff(orig_text, pol_text),
            "original": orig_text,
            "polished": pol_text,
        })

    return blocks


def compute_char_diff(orig: str, pol: str) -> list[dict[str, object]]:
    sm = difflib.SequenceMatcher(None, orig, pol)
    segs: list[dict[str, object]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            segs.append({"type": "equal", "text": orig[i1:i2]})
        elif tag == "replace":
            segs.append({"type": "delete", "text": orig[i1:i2]})
            segs.append({"type": "insert", "text": pol[j1:j2]})
        elif tag == "delete":
            segs.append({"type": "delete", "text": orig[i1:i2]})
        elif tag == "insert":
            segs.append({"type": "insert", "text": pol[j1:j2]})
    return segs


# ── Server utilities ──


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_for_script(data: dict[str, object]) -> str:
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return encoded.replace("</", "<\\/").replace(" ", "\\u2028").replace(" ", "\\u2029")


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


def polish_markdown(content: str) -> str:
    had_trailing_newline = content.endswith(("\n", "\r"))
    normalized = normalize_newlines(content)
    frontmatter, body = extract_frontmatter(normalized)
    polished_body = polish_body(body)
    result = frontmatter + polished_body if frontmatter else polished_body
    if had_trailing_newline:
        return result if result.endswith("\n") else result + "\n"
    return result[:-1] if result.endswith("\n") else result


def polish_body(body: str) -> str:
    if not body.strip():
        return body

    lines = body.split("\n")
    replacements = collect_replacements(body, lines)
    if not replacements:
        return body

    output: list[str] = []
    cursor = 0
    for start_line, end_line, replacement in replacements:
        output.extend(lines[cursor:start_line])
        output.extend(replacement.split("\n"))
        cursor = end_line
    output.extend(lines[cursor:])
    return "\n".join(output)


def collect_replacements(body: str, lines: list[str]) -> list[tuple[int, int, str]]:
    replacements: list[tuple[int, int, str]] = []
    stack: list[str] = []

    for token in parser.parse(body):
        if token.type == "inline" and token.map and should_process_inline(stack):
            start_line, end_line = token.map
            original = "\n".join(lines[start_line:end_line])
            updated = polish_block(original)
            if updated != original:
                replacements.append((start_line, end_line, updated))

        if token.nesting == 1:
            stack.append(token.type)
        elif token.nesting == -1 and stack:
            stack.pop()

    return replacements


def should_process_inline(stack: list[str]) -> bool:
    if not stack or stack[-1] not in ALLOWED_CONTAINERS:
        return False
    return not any(token_type in FORBIDDEN_ANCESTORS for token_type in stack[:-1])


def polish_block(block: str) -> str:
    if not CHINESE_RE.search(block):
        return block
    if is_special_definition(block):
        return block

    prefixes = extract_blockquote_prefixes(block)
    if any(prefixes):
        lines = block.split("\n")
        contents = [line[len(prefix):] for line, prefix in zip(lines, prefixes)]
        polished_lines = run_polish_pipeline("\n".join(contents)).split("\n")
        if len(polished_lines) == len(prefixes):
            return "\n".join(prefix + content for prefix, content in zip(prefixes, polished_lines))

    return run_polish_pipeline(block)


def extract_blockquote_prefixes(block: str) -> list[str]:
    prefixes: list[str] = []
    for line in block.split("\n"):
        match = BLOCKQUOTE_PREFIX_RE.match(line)
        prefixes.append(match.group(1) if match else "")
    return prefixes


def run_polish_pipeline(block: str) -> str:
    emphasized = fix_cjk_emphasis(block)
    masked, protected = protect_inline_spans(emphasized)
    parts: list[str] = []
    last_index = 0
    pending_footnote_pad = False

    for match in iter_protected_placeholders(masked, protected):
        visible = polish_visible_text(masked[last_index:match.start()])
        protected_text = protected.items[int(match.group(1))]

        if pending_footnote_pad and visible and CHINESE_RE.fullmatch(visible[0]):
            visible = " " + visible
        pending_footnote_pad = False

        parts.append(visible)
        parts.append(protected_text)
        last_index = match.end()

        if is_footnote_reference(protected_text):
            pending_footnote_pad = True

    trailing = polish_visible_text(masked[last_index:])
    if pending_footnote_pad and trailing and CHINESE_RE.fullmatch(trailing[0]):
        trailing = " " + trailing
    parts.append(trailing)

    return "".join(parts)


def is_footnote_reference(text: str) -> bool:
    return text.startswith("[^") and text.endswith("]")


def is_special_definition(block: str) -> bool:
    stripped = block.lstrip()
    return bool(REFERENCE_DEFINITION_RE.match(stripped) or FOOTNOTE_DEFINITION_RE.match(stripped))


def protect_inline_spans(text: str) -> tuple[str, ProtectedSpans]:
    protected = ProtectedSpans(*build_placeholder_delimiters(text), items=[])
    pieces: list[str] = []
    index = 0

    while index < len(text):
        end = find_protected_span(text, index)
        if end is None:
            pieces.append(text[index])
            index += 1
            continue

        placeholder = make_placeholder(protected, len(protected.items))
        protected.items.append(text[index:end])
        pieces.append(placeholder)
        index = end

    return "".join(pieces), protected


def build_placeholder_delimiters(text: str) -> tuple[str, str]:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    prefix = f"POLISH{digest}A"
    suffix = f"B{digest}HSILOP"
    while prefix in text or suffix in text:
        prefix = f"{prefix}x"
        suffix = f"x{suffix}"
    return prefix, suffix


def make_placeholder(protected: ProtectedSpans, index: int) -> str:
    return f"{protected.prefix}{index}{protected.suffix}"


def protected_placeholder_pattern(protected: ProtectedSpans) -> re.Pattern[str]:
    return re.compile(re.escape(protected.prefix) + r"(\d+)" + re.escape(protected.suffix))


def iter_protected_placeholders(text: str, protected: ProtectedSpans) -> re.Match[str]:
    return protected_placeholder_pattern(protected).finditer(text)


def restore_protected_spans(text: str, protected: ProtectedSpans) -> str:
    return protected_placeholder_pattern(protected).sub(lambda match: protected.items[int(match.group(1))], text)


def find_protected_span(text: str, index: int) -> int | None:
    url_match = find_url_span_end(text, index)
    if url_match is not None:
        return url_match

    email_match = EMAIL_RE.match(text, index)
    if email_match:
        return email_match.end()

    math_match = find_math_span_end(text, index)
    if math_match is not None:
        return math_match

    quoted_match = find_ascii_quoted_english_span_end(text, index)
    if quoted_match is not None:
        return quoted_match

    current = text[index]
    if current == "`":
        return find_code_span_end(text, index)
    if current == "!" and index + 1 < len(text) and text[index + 1] == "[":
        return find_link_span_end(text, index)
    if current == "[":
        return find_link_span_end(text, index)
    if current == "<":
        return find_angle_span_end(text, index)
    return None


def find_url_span_end(text: str, start: int) -> int | None:
    match = URL_RE.match(text, start)
    if not match:
        return None

    end = match.end()
    while end > start:
        last = text[end - 1]
        if last in URL_TRAILING_PUNCTUATION:
            end -= 1
            continue
        if last in URL_TRAILING_BRACKETS:
            opener = URL_TRAILING_BRACKETS[last]
            candidate = text[start:end]
            if candidate.count(opener) < candidate.count(last):
                end -= 1
                continue
        break

    return end if end > start else None


def find_code_span_end(text: str, start: int) -> int | None:
    tick_count = 1
    while start + tick_count < len(text) and text[start + tick_count] == "`":
        tick_count += 1
    marker = "`" * tick_count
    end = text.find(marker, start + tick_count)
    if end == -1:
        return None
    return end + tick_count


def find_math_span_end(text: str, start: int) -> int | None:
    if text.startswith("$", start) and not text.startswith("$$", start):
        return find_dollar_math_span_end(text, start)

    for opener, closer in NON_DOLLAR_MATH_DELIMITERS:
        if not text.startswith(opener, start):
            continue
        end = find_unescaped_token(text, closer, start + len(opener))
        if end == -1:
            return None
        return end + len(closer)
    return None


def find_dollar_math_span_end(text: str, start: int) -> int | None:
    if start + 1 >= len(text) or text[start + 1].isspace():
        return None

    end = find_unescaped_token(text, "$", start + 1)
    while end != -1:
        content = text[start + 1:end]
        next_char = text[end + 1] if end + 1 < len(text) else ""
        if content and not content[-1].isspace() and not next_char.isdigit() and looks_like_dollar_math(content):
            return end + 1
        end = find_unescaped_token(text, "$", end + 1)
    return None


def looks_like_dollar_math(content: str) -> bool:
    stripped = content.strip()
    if not stripped or "\n" in stripped:
        return False
    if MATH_SIGNAL_RE.search(stripped):
        return True
    return bool(re.fullmatch(r"\d+(?:[.,]\d+)*", stripped))


def find_unescaped_token(text: str, token: str, start: int) -> int:
    index = text.find(token, start)
    while index != -1:
        if not is_escaped(text, index):
            return index
        index = text.find(token, index + 1)
    return -1


def find_ascii_quoted_english_span_end(text: str, start: int) -> int | None:
    quote = text[start]
    if quote not in {'"', "'"}:
        return None

    before = text[start - 1] if start > 0 else ""
    after = text[start + 1] if start + 1 < len(text) else ""
    if not after or not after.isascii() or not after.isalnum():
        return None
    if quote == "'" and before in ASCII_WORD_CHARACTERS:
        return None

    end = text.find(quote, start + 1)
    while end != -1:
        nxt = text[end + 1] if end + 1 < len(text) else ""
        content = text[start + 1:end]
        if "\n" in content:
            return None
        if looks_like_ascii_quoted_content(content):
            if quote == "'" and nxt in ASCII_WORD_CHARACTERS:
                end = text.find(quote, end + 1)
                continue
            return end + 1
        end = text.find(quote, end + 1)
    return None


def looks_like_ascii_quoted_content(content: str) -> bool:
    stripped = content.strip()
    return bool(stripped) and stripped.isascii() and bool(ASCII_LETTER_RE.search(stripped)) and not CHINESE_RE.search(stripped)


def find_link_span_end(text: str, start: int) -> int | None:
    offset = 1 if text[start] == "!" else 0
    if start + offset >= len(text) or text[start + offset] != "[":
        return None

    label_start = start + offset
    label_end = find_balanced_span_end(text, label_start, "[", "]")
    if label_end is None:
        return None
    if text[label_start:label_start + 2] == "[^":
        return label_end
    if label_end >= len(text):
        return label_end

    if text[label_end] == "(":
        return find_balanced_span_end(text, label_end, "(", ")")
    if text[label_end] == "[":
        return find_balanced_span_end(text, label_end, "[", "]")
    return label_end


def find_angle_span_end(text: str, start: int) -> int | None:
    end = text.find(">", start + 1)
    if end == -1:
        return None
    if "\n" in text[start:end]:
        return None
    return end + 1


def find_balanced_span_end(text: str, start: int, open_char: str, close_char: str) -> int | None:
    if text[start] != open_char:
        return None

    depth = 0
    index = start
    while index < len(text):
        current = text[index]
        if current == "\\":
            index += 2
            continue
        if current == open_char:
            depth += 1
        elif current == close_char:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def polish_visible_text(text: str) -> str:
    if not text:
        return text
    text = normalize_quotes(text)
    text = normalize_punctuation(text)
    text = apply_spacing(text)
    text = normalize_cjk_punctuation_spacing(text)
    text = normalize_quote_spacing_in_ascii_context(text)
    text = normalize_repeated_punctuation(text)
    return text


def normalize_quotes(text: str) -> str:
    if '"' not in text and "'" not in text:
        return text

    result: list[str] = []
    quote_stack: list[str] = []

    for index, current in enumerate(text):
        if current not in {'"', "'"}:
            result.append(current)
            continue

        if is_prime_like(text, index):
            result.append(current)
            continue

        if current == "'" and is_apostrophe(text, index, quote_stack):
            result.append(current)
            continue

        if should_close_quote(text, index, current, quote_stack):
            mark = quote_stack.pop()
            result.append("”" if mark == "“" else "’")
            continue

        mark = opening_quote_mark(current, quote_stack)
        quote_stack.append(mark)
        result.append(mark)

    return "".join(result)


def is_apostrophe(text: str, index: int, quote_stack: list[str]) -> bool:
    left = nearest_significant_char(text, index - 1, -1)
    right = nearest_significant_char(text, index + 1, 1)
    if left in ASCII_WORD_CHARACTERS and right in ASCII_WORD_CHARACTERS:
        return True
    if quote_stack:
        return False
    return left in ASCII_WORD_CHARACTERS and right not in ASCII_WORD_CHARACTERS


def is_prime_like(text: str, index: int) -> bool:
    current = text[index]
    left = nearest_significant_char(text, index - 1, -1)
    right = nearest_significant_char(text, index + 1, 1)
    if left.isdigit() and current in {"'", '"'}:
        return True
    if current == '"' and left in ASCII_WORD_CHARACTERS and right in ASCII_WORD_CHARACTERS:
        return True
    return False


def should_close_quote(text: str, index: int, current: str, quote_stack: list[str]) -> bool:
    if not quote_stack:
        return False

    top = quote_stack[-1]
    if current == '"' and top != "“":
        return False
    if current == "'" and top != "‘":
        return False

    left = nearest_significant_char(text, index - 1, -1)
    right = nearest_significant_char(text, index + 1, 1)

    if not right:
        return True
    if right in {'"', "'"}:
        return True
    if (
        right.isspace()
        or right in QUOTE_FOLLOWING_PUNCTUATION
        or right in OPEN_QUOTE_CONTEXT_CHARS
        or right in CLOSE_QUOTE_CONTEXT_CHARS
    ):
        return True
    if left.isspace() or left in OPEN_QUOTE_CONTEXT_CHARS:
        return False
    return bool(left)


def opening_quote_mark(current: str, quote_stack: list[str]) -> str:
    if current == "'":
        return "‘"
    return "“" if not quote_stack else "‘"


def nearest_significant_char(text: str, index: int, step: int) -> str:
    while 0 <= index < len(text):
        current = text[index]
        if current in INLINE_FORMATTING_MARKERS:
            index += step
            continue
        return current
    return ""


def normalize_punctuation(text: str) -> str:
    pieces: list[str] = []
    index = 0
    has_chinese = bool(CHINESE_RE.search(text))

    while index < len(text):
        pair = text[index:index + 2]
        if pair in {"!?", "?!"} and should_fullwidth(text, index, 2, pair, has_chinese):
            pieces.append(FULLWIDTH_PUNCTUATION[pair])
            index += 2
            continue

        current = text[index]
        if current in {":", "!", "?", ";", ","} and should_fullwidth(text, index, 1, current, has_chinese):
            pieces.append(FULLWIDTH_PUNCTUATION[current])
        else:
            pieces.append(current)
        index += 1

    return "".join(pieces)


def normalize_cjk_punctuation_spacing(text: str) -> str:
    text = re.sub(
        rf"[ \t]+([{re.escape(CJK_PUNCTUATION_WITHOUT_SURROUNDING_SPACES)}])",
        r"\1",
        text,
    )
    text = re.sub(
        rf"([{re.escape(CJK_PUNCTUATION_WITHOUT_SURROUNDING_SPACES)}])[ \t]+",
        r"\1",
        text,
    )
    return text


def normalize_quote_spacing_in_ascii_context(text: str) -> str:
    text = re.sub(r"(?<=[A-Za-z0-9])([“‘])(?=[A-Za-z])", r" \1", text)
    text = re.sub(r"(?<=[A-Za-z])([”’])(?=[A-Za-z0-9])", r"\1 ", text)
    return text


def normalize_repeated_punctuation(text: str) -> str:
    text = REPEATED_PUNCTUATION_RE.sub(r"\1", text)
    return REPEATED_MIXED_PUNCTUATION_RE.sub(collapse_mixed_punctuation, text)


def collapse_mixed_punctuation(match: re.Match[str]) -> str:
    sequence = match.group(1)
    has_question = "？" in sequence
    has_exclamation = "！" in sequence
    if has_question and has_exclamation:
        question_index = sequence.index("？")
        exclamation_index = sequence.index("！")
        return "？！" if question_index < exclamation_index else "！？"
    if has_question:
        return "？"
    if has_exclamation:
        return "！"
    return sequence


def should_fullwidth(text: str, index: int, width: int, token: str, has_chinese: bool) -> bool:
    if not has_chinese:
        return False
    left = nearest_visible_char(text, index - 1, -1)
    right = nearest_visible_char(text, index + width, 1)
    if token == ":" and left.isdigit() and right.isdigit():
        return False
    if token == "," and left.isdigit() and right.isdigit():
        return False
    if is_cjkish(left) or is_cjkish(right):
        return True
    if token in {"!?", "?!", "!", "?"} and not right:
        return True
    return False


def nearest_visible_char(text: str, index: int, step: int) -> str:
    while 0 <= index < len(text):
        current = text[index]
        if current.isspace() or current in INLINE_FORMATTING_MARKERS:
            index += step
            continue
        return current
    return ""


def is_cjkish(character: str) -> bool:
    return bool(character) and (bool(CHINESE_RE.fullmatch(character)) or character in CJKISH_PUNCTUATION)


def normalize_newlines(text: str) -> str:
    return text.removeprefix("﻿").replace("\r\n", "\n").replace("\r", "\n")


def extract_frontmatter(content: str) -> tuple[str, str]:
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\n") != "---":
        return "", content

    consumed = len(lines[0])
    for line in lines[1:]:
        consumed += len(line)
        if line.rstrip("\n") in {"---", "..."}:
            return content[:consumed], content[consumed:]

    return "", content


if __name__ == "__main__":
    raise SystemExit(main())
