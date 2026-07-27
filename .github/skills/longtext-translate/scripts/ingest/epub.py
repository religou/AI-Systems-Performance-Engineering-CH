#!/usr/bin/env python3
"""Extract selected EPUB spine items into clean Markdown with local images."""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

OPS_NS = "http://www.idpf.org/2007/ops"
CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
OPF_NS = "http://www.idpf.org/2007/opf"
NCX_NS = "http://www.daisy.org/z3986/2005/ncx/"


def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def classes(elem: ET.Element) -> set[str]:
    return set((elem.attrib.get("class") or "").split())


def normalize_space(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def resolve_epub_path(current_dir: Path, src: str) -> str:
    parts: list[str] = []
    for part in (current_dir / src).as_posix().split("/"):
        if part == "..":
            if parts:
                parts.pop()
        elif part not in {"", "."}:
            parts.append(part)
    return "/".join(parts)


def read_spine(epub: zipfile.ZipFile) -> list[str]:
    container = ET.fromstring(epub.read("META-INF/container.xml"))
    rootfile = container.find(f".//{{{CONTAINER_NS}}}rootfile")
    if rootfile is None or "full-path" not in rootfile.attrib:
        raise ValueError("Cannot locate OPF rootfile in EPUB container.xml")

    opf_path = rootfile.attrib["full-path"]
    opf_dir = Path(opf_path).parent
    opf = ET.fromstring(epub.read(opf_path))

    manifest: dict[str, str] = {}
    for item in opf.findall(f".//{{{OPF_NS}}}manifest/{{{OPF_NS}}}item"):
        item_id = item.attrib.get("id")
        href = item.attrib.get("href")
        if item_id and href:
            manifest[item_id] = href

    spine: list[str] = []
    for itemref in opf.findall(f".//{{{OPF_NS}}}spine/{{{OPF_NS}}}itemref"):
        href = manifest.get(itemref.attrib.get("idref", ""))
        if href:
            spine.append((opf_dir / href).as_posix())
    return spine


def read_toc(epub: zipfile.ZipFile) -> list[tuple[int, str, str]] | None:
    """Parse EPUB TOC from NCX (EPUB 2) or NAV (EPUB 3).

    Returns list of (level, label, href) where level is 0-based (0 = top-level
    heading, 1 = sub-heading, etc.). Returns None if no TOC is found.
    """
    container = ET.fromstring(epub.read("META-INF/container.xml"))
    rootfile = container.find(f".//{{{CONTAINER_NS}}}rootfile")
    if rootfile is None or "full-path" not in rootfile.attrib:
        return None

    opf_path = rootfile.attrib["full-path"]
    opf_dir = Path(opf_path).parent
    opf = ET.fromstring(epub.read(opf_path))

    # Collect manifest items
    manifest: dict[str, tuple[str, str, str]] = {}
    for item in opf.findall(f".//{{{OPF_NS}}}manifest/{{{OPF_NS}}}item"):
        item_id = item.attrib.get("id")
        href = item.attrib.get("href")
        media_type = item.attrib.get("media-type", "")
        properties = item.attrib.get("properties", "")
        if item_id and href:
            manifest[item_id] = (href, media_type, properties)

    # Try EPUB 3 NAV first
    for _item_id, (href, _media_type, properties) in manifest.items():
        if "nav" in (properties or "").split():
            nav_full = (opf_dir / href).as_posix()
            if nav_full in epub.namelist():
                entries = _parse_nav(epub.read(nav_full))
                if entries:
                    return entries

    # Try EPUB 2 NCX: spine/@toc or manifest media-type match
    ncx_href: str | None = None
    spine_elem = opf.find(f".//{{{OPF_NS}}}spine")
    if spine_elem is not None:
        ncx_id = spine_elem.attrib.get("toc")
        if ncx_id and ncx_id in manifest:
            ncx_href = manifest[ncx_id][0]

    if not ncx_href:
        for _item_id, (href, media_type, _properties) in manifest.items():
            if media_type == "application/x-dtbncx+xml":
                ncx_href = href
                break

    if ncx_href:
        ncx_full = (opf_dir / ncx_href).as_posix()
        if ncx_full in epub.namelist():
            return _parse_ncx(epub.read(ncx_full))

    return None


def _parse_ncx(content: bytes) -> list[tuple[int, str, str]]:
    """Parse EPUB 2 NCX navMap into (level, label, href) entries."""
    root = ET.fromstring(content)
    nav_map = root.find(f".//{{{NCX_NS}}}navMap")
    if nav_map is None:
        return []

    entries: list[tuple[int, str, str]] = []

    def walk_navpoints(parent_elem: ET.Element, level: int) -> None:
        for nav_point in parent_elem.findall(f"{{{NCX_NS}}}navPoint"):
            label_elem = nav_point.find(f"{{{NCX_NS}}}navLabel/{{{NCX_NS}}}text")
            content_elem = nav_point.find(f"{{{NCX_NS}}}content")

            if label_elem is not None and label_elem.text:
                label = normalize_space(label_elem.text)
                href = content_elem.attrib.get("src", "") if content_elem is not None else ""
                entries.append((level, label, href))

            walk_navpoints(nav_point, level + 1)

    walk_navpoints(nav_map, 0)
    return entries


def _parse_nav(content: bytes) -> list[tuple[int, str, str]]:
    """Parse EPUB 3 NAV XHTML into (level, label, href) entries."""
    root = ET.fromstring(content)

    # Find <nav epub:type="toc"> (allow multi-token values like "toc landmarks")
    nav_elem = None
    for elem in root.iter():
        if local_name(elem.tag) != "nav":
            continue
        for key, val in elem.attrib.items():
            if local_name(key) == "type" and "toc" in val.split():
                nav_elem = elem
                break
        if nav_elem is not None:
            break

    if nav_elem is None:
        return []

    entries: list[tuple[int, str, str]] = []

    def walk_ol(ol_elem: ET.Element, level: int) -> None:
        for child in ol_elem:
            if local_name(child.tag) != "li":
                continue
            label = ""
            href = ""
            nested_ol = None
            for sub in child:
                sub_tag = local_name(sub.tag)
                if sub_tag == "a":
                    label = normalize_space("".join(sub.itertext()))
                    href = sub.attrib.get("href", "")
                elif sub_tag == "span" and not label:
                    label = normalize_space("".join(sub.itertext()))
                elif sub_tag == "ol" and nested_ol is None:
                    nested_ol = sub

            if label:
                entries.append((level, label, href))
            if nested_ol is not None:
                walk_ol(nested_ol, level + 1)

    # Find the first <ol> inside the nav
    for child in nav_elem:
        if local_name(child.tag) == "ol":
            walk_ol(child, 0)
            break

    return entries


def is_note(elem: ET.Element) -> bool:
    epub_type = elem.attrib.get(f"{{{OPS_NS}}}type")
    class_text = " ".join(classes(elem))
    return (
        local_name(elem.tag) == "aside"
        or epub_type == "footnote"
        or "footnote" in class_text
    )


def make_converter(epub: zipfile.ZipFile, image_dir: Path, image_prefix: str):
    image_dir.mkdir(parents=True, exist_ok=True)

    def copy_image(current_dir: Path, src: str) -> str:
        if not src or src.startswith(("http://", "https://", "data:")):
            return src
        internal_path = resolve_epub_path(current_dir, src)
        image_name = Path(internal_path).name
        if internal_path in epub.namelist():
            (image_dir / image_name).write_bytes(epub.read(internal_path))
            return f"{image_prefix}/{image_name}"
        return src

    def inline(elem: ET.Element, current_dir: Path) -> str:
        if is_note(elem):
            return elem.tail or ""

        tag = local_name(elem.tag)
        elem_classes = classes(elem)

        if tag == "a" and (
            elem_classes & {"note", "noteref"}
            or "Endnotes.xhtml" in elem.attrib.get("href", "")
        ):
            return elem.tail or ""

        if tag == "img":
            alt = elem.attrib.get("alt", "")
            src = copy_image(current_dir, elem.attrib.get("src", ""))
            return f"\n\n![{alt}]({src})\n\n" + (elem.tail or "")

        content = elem.text or ""
        content += "".join(inline(child, current_dir) for child in list(elem))

        if tag in {"em", "i"} and content.strip():
            content = f"*{content.strip()}*"
        elif tag in {"strong", "b"} and content.strip():
            content = f"**{content.strip()}**"
        elif tag == "br":
            content = "\n"

        return content + (elem.tail or "")

    def block(elem: ET.Element | None, current_dir: Path) -> list[str]:
        if elem is None or is_note(elem):
            return []

        tag = local_name(elem.tag)
        if tag in {"script", "style", "hr", "nav"}:
            return []

        if tag == "img":
            alt = elem.attrib.get("alt", "")
            src = copy_image(current_dir, elem.attrib.get("src", ""))
            return [f"![{alt}]({src})"]

        if re.fullmatch(r"h[1-6]", tag):
            text = normalize_space(inline(elem, current_dir))
            return [f"{'#' * int(tag[1])} {text}"] if text else []

        if tag == "p":
            text = inline(elem, current_dir)
            return [
                normalize_space(chunk)
                for chunk in re.split(r"\n\s*\n", text)
                if normalize_space(chunk)
            ]

        if tag == "li":
            text = normalize_space(inline(elem, current_dir))
            return [f"- {text}"] if text else []

        if tag == "blockquote":
            items: list[str] = []
            for child in list(elem):
                items.extend(block(child, current_dir))
            if not items:
                text = normalize_space(inline(elem, current_dir))
                items = [text] if text else []
            return ["> " + item.replace("\n", "\n> ") for item in items]

        items: list[str] = []
        if elem.text and elem.text.strip():
            items.append(normalize_space(elem.text))
        for child in list(elem):
            items.extend(block(child, current_dir))
        if items:
            return items

        text = normalize_space(inline(elem, current_dir))
        return [text] if text else []

    return block


def body_of(xhtml: bytes) -> ET.Element:
    root = ET.fromstring(xhtml.decode("utf-8", errors="replace"))
    return next((elem for elem in root.iter() if local_name(elem.tag) == "body"), root)


def validate(markdown_path: Path, markdown: str) -> tuple[int, int, int]:
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown)
    missing = 0
    for ref in refs:
        if ref.startswith(("http://", "https://", "data:")):
            continue
        if not (markdown_path.parent / ref).exists():
            missing += 1
    html_tags = len(re.findall(r"<[^>]+>", markdown))
    replacement_chars = markdown.count("�")
    return missing, html_tags, replacement_chars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epub", type=Path, help="EPUB file path")
    parser.add_argument("--list-spine", action="store_true", help="List EPUB spine and exit")
    parser.add_argument("--list-toc", action="store_true", help="List EPUB table of contents tree and exit")
    parser.add_argument("--href", action="append", default=[], help="EPUB href to extract, e.g. Text/Chapter1.xhtml")
    parser.add_argument("--spine-start", type=int, help="Inclusive 0-based spine start index")
    parser.add_argument("--spine-end", type=int, help="Inclusive 0-based spine end index")
    parser.add_argument("--output", type=Path, help="Output Markdown path")
    parser.add_argument("--image-dir-name", default="images", help="Image directory name beside the Markdown file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with zipfile.ZipFile(args.epub) as epub:
        spine = read_spine(epub)
        if args.list_spine:
            for index, href in enumerate(spine):
                print(f"{index}\t{href}")
            return 0

        if args.list_toc:
            toc = read_toc(epub)
            if toc is None:
                print("No TOC (NCX or NAV) found in this EPUB.", file=sys.stderr)
                return 1
            for level, label, href in toc:
                prefix = "#" * (level + 1)
                print(f"{prefix} {label}  ← {href}")
            return 0

        if not args.output:
            print("--output is required unless --list-spine is used", file=sys.stderr)
            return 2

        selected: list[str] = []
        if args.href:
            selected_set: set[str] = set()
            for requested in args.href:
                matches = [
                    href
                    for href in spine
                    if href == requested or href.removeprefix("OEBPS/") == requested
                ]
                if matches:
                    selected_set.update(matches)
                elif requested in epub.namelist():
                    selected_set.add(requested)
                elif f"OEBPS/{requested}" in epub.namelist():
                    selected_set.add(f"OEBPS/{requested}")
            selected = [href for href in spine if href in selected_set]
            selected.extend(href for href in selected_set if href not in spine)
        elif args.spine_start is not None and args.spine_end is not None:
            selected = spine[args.spine_start : args.spine_end + 1]
        else:
            print("Provide --href or both --spine-start and --spine-end", file=sys.stderr)
            return 2

        if not selected:
            print("No spine items selected", file=sys.stderr)
            return 1

        args.output.parent.mkdir(parents=True, exist_ok=True)
        image_dir = args.output.parent / args.image_dir_name
        block = make_converter(epub, image_dir, args.image_dir_name)

        blocks: list[str] = []
        for href in selected:
            internal = href if href in epub.namelist() else f"OEBPS/{href}"
            if internal not in epub.namelist():
                print(f"Warning: missing EPUB item {href}", file=sys.stderr)
                continue
            body = body_of(epub.read(internal))
            blocks.extend(block(body, Path(internal).parent))

    markdown = "\n\n".join(item for item in blocks if item.strip())
    markdown = re.sub(r"(?<=[A-Za-z0-9,.;:!?])\*(?=[A-Za-z0-9])", "* ", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"
    args.output.write_text(markdown, encoding="utf-8")

    missing, html_tags, replacement_chars = validate(args.output, markdown)
    print(f"wrote: {args.output}")
    print(f"selected_items: {len(selected)}")
    print(f"image_refs: {markdown.count('![')}")
    print(f"missing_images: {missing}")
    print(f"html_tags: {html_tags}")
    print(f"replacement_chars: {replacement_chars}")
    return 1 if missing or replacement_chars else 0


if __name__ == "__main__":
    raise SystemExit(main())
