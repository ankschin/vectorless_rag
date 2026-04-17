"""
Build a hierarchical TOC index from a PDF without embeddings or external APIs.

Usage:
    python toc_index.py                  # index all PDFs in data/
    python toc_index.py path/to/doc.pdf  # index a specific file

Output: workspace/<stem>.json  {title, toc, pages, source, total_pages}
"""

import json
import os
import sys
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

WORKSPACE = Path("workspace")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ── PDF extraction ─────────────────────────────────────────────────────────────

def extract_pages(pdf_path: Path) -> dict[int, str]:
    doc = fitz.open(str(pdf_path))
    pages = {i + 1: page.get_text() for i, page in enumerate(doc)}
    doc.close()
    return pages


def try_pdf_outline(pdf_path: Path, total_pages: int) -> list[dict] | None:
    """Return built-in PDF bookmarks as a tree, or None if absent."""
    doc = fitz.open(str(pdf_path))
    flat = doc.get_toc()  # [[level, title, page], ...]
    doc.close()
    if not flat:
        return None
    return _flat_to_tree(flat, total_pages)


def _flat_to_tree(flat: list[list], total_pages: int) -> list[dict]:
    # Annotate end pages
    nodes = []
    for i, (level, title, page_start) in enumerate(flat):
        page_end = total_pages
        for j in range(i + 1, len(flat)):
            if flat[j][0] <= level:
                page_end = flat[j][2] - 1
                break
        nodes.append({"_level": level, "title": title, "page_start": page_start,
                       "page_end": max(page_start, page_end), "children": []})

    root: list[dict] = []
    stack: list[tuple[int, dict]] = []
    for node in nodes:
        level = node.pop("_level")
        while stack and stack[-1][0] >= level:
            stack.pop()
        (stack[-1][1]["children"] if stack else root).append(node)
        stack.append((level, node))
    return root


# ── LLM-based TOC builder ──────────────────────────────────────────────────────

def build_toc_with_llm(pages: dict[int, str], client: Groq) -> list[dict]:
    """Infer a hierarchical TOC from page content using an LLM."""
    # Send only the first ~200 chars of each page so the prompt stays small
    previews = []
    for pnum in sorted(pages):
        snippet = pages[pnum].strip()[:200].replace("\n", " ")
        previews.append(f"Page {pnum}: {snippet}")

    preview_text = "\n".join(previews)
    if len(preview_text) > 20000:
        preview_text = preview_text[:20000] + "\n... (truncated)"

    last_page = max(pages)
    prompt = f"""Analyze the following document page previews and produce a hierarchical Table of Contents.

{preview_text}

Return ONLY a valid JSON array. Each node must have:
  "title"      : section title (string)
  "page_start" : first page of section (int)
  "page_end"   : last page of section (int, last node ends at {last_page})
  "children"   : array of sub-section nodes (same shape, may be empty)

No prose, no markdown fences, just the raw JSON array."""

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=4096,
    )
    text = resp.choices[0].message.content.strip()
    # Strip optional ```json fences
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ── Main indexing function ─────────────────────────────────────────────────────

def index_pdf(pdf_path: Path, force: bool = False) -> Path:
    WORKSPACE.mkdir(exist_ok=True)
    out_path = WORKSPACE / f"{pdf_path.stem}.json"

    if out_path.exists() and not force:
        print(f"[skip] {pdf_path.name} already indexed → {out_path}")
        return out_path

    print(f"Indexing {pdf_path.name} ...")
    pages = extract_pages(pdf_path)
    total = max(pages)
    print(f"  Extracted {total} pages.")

    toc = try_pdf_outline(pdf_path, total)
    if toc:
        source = "pdf_outline"
        print(f"  Using built-in PDF outline ({len(toc)} top-level entries).")
    else:
        source = "llm"
        print("  No outline found — building TOC with LLM ...")
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        toc = build_toc_with_llm(pages, client)
        print(f"  LLM produced {len(toc)} top-level entries.")

    index = {
        "title": pdf_path.stem,
        "source": source,
        "total_pages": total,
        "toc": toc,
        "pages": {str(k): v for k, v in pages.items()},
    }
    out_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print(f"  Saved -> {out_path}\n")
    return out_path


if __name__ == "__main__":
    force = "--force" in sys.argv
    paths = [Path(a) for a in sys.argv[1:] if not a.startswith("--")]

    if not paths:
        paths = sorted(Path("data").glob("*.pdf"))
        if not paths:
            print("No PDFs in data/. Pass a PDF path as argument.")
            sys.exit(1)

    for pdf in paths:
        index_pdf(pdf, force=force)
