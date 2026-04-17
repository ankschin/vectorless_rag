"""
Query a TOC-indexed document without embeddings.

The pipeline:
  1. LLM reads the TOC tree and picks relevant page ranges
  2. Fetch the raw page text for those ranges from the local index
  3. LLM generates the final answer from that context

Usage:
    python toc_query.py "What is this document about?"
    python toc_query.py --doc 2400MP_OM_English "How do I set up the projector?"
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

WORKSPACE = Path("workspace")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ── Index helpers ──────────────────────────────────────────────────────────────

def list_docs() -> list[str]:
    return [p.stem for p in WORKSPACE.glob("*.json") if p.stem != "index_map"]


def load_index(stem: str) -> dict:
    path = WORKSPACE / f"{stem}.json"
    if not path.exists():
        raise FileNotFoundError(f"No index for '{stem}'. Run toc_index.py first.")
    return json.loads(path.read_text())


def pick_doc(doc_name: Optional[str]) -> str:
    docs = list_docs()
    if not docs:
        raise RuntimeError("No indexed documents found. Run toc_index.py first.")
    if doc_name:
        stem = Path(doc_name).stem
        if stem not in docs:
            raise KeyError(f"'{stem}' not in index. Available: {docs}")
        return stem
    if len(docs) == 1:
        return docs[0]
    raise ValueError(f"Multiple docs indexed — specify --doc. Available: {docs}")


# ── TOC utilities ──────────────────────────────────────────────────────────────

def slim_toc(nodes: list[dict]) -> list[dict]:
    """Strip everything except title and page range to keep the prompt small."""
    result = []
    for n in nodes:
        node = {
            "title": n.get("title", ""),
            "page_start": n.get("page_start"),
            "page_end": n.get("page_end"),
        }
        children = slim_toc(n.get("children", []))
        if children:
            node["children"] = children
        result.append(node)
    return result


def fetch_pages(index: dict, page_ranges: list[str]) -> str:
    """Return concatenated page text for the requested ranges."""
    wanted: set[int] = set()
    for r in page_ranges:
        r = r.strip()
        if "-" in r:
            a, b = r.split("-", 1)
            wanted.update(range(int(a), int(b) + 1))
        else:
            wanted.add(int(r))

    store = index["pages"]
    parts = []
    for pnum in sorted(wanted):
        text = store.get(str(pnum), "").strip()
        if text:
            parts.append(f"[Page {pnum}]\n{text}")
    return "\n\n---\n\n".join(parts)


# ── Core query pipeline ────────────────────────────────────────────────────────

def query(question: str, doc_stem: str, client: Groq) -> str:
    index = load_index(doc_stem)
    toc_json = json.dumps(slim_toc(index["toc"]), indent=2)
    if len(toc_json) > 24000:
        toc_json = toc_json[:24000] + "\n... (truncated)"

    # ── Step 1: Navigate the TOC ───────────────────────────────────────────────
    print("  [1/3] LLM navigating TOC ...")
    nav_resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{
            "role": "user",
            "content": f"""You are navigating a document's Table of Contents to find where the answer lives.

Question: {question}

Table of Contents:
{toc_json}

Return ONLY valid JSON (no prose, no fences):
{{
  "thinking": "<one sentence reasoning>",
  "page_ranges": ["<start>-<end>", ...]
}}

List only the page ranges most likely to contain the answer.""",
        }],
        temperature=0,
        max_tokens=512,
    )
    nav_text = nav_resp.choices[0].message.content.strip()
    if "```" in nav_text:
        nav_text = nav_text.split("```")[1].lstrip("json").strip()
    nav = json.loads(nav_text)

    thinking = nav.get("thinking", "")
    page_ranges: list[str] = nav.get("page_ranges", [])
    print(f"  Reasoning : {thinking[:200]}")
    print(f"  Pages     : {page_ranges}")

    if not page_ranges:
        return "No relevant sections identified in the document structure."

    # ── Step 2: Fetch page content ─────────────────────────────────────────────
    print("  [2/3] Fetching page content ...")
    context = fetch_pages(index, page_ranges)
    if not context:
        return "Could not retrieve content for the identified page ranges."

    # ── Step 3: Generate answer ────────────────────────────────────────────────
    print("  [3/3] Generating answer ...")
    answer_resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{
            "role": "user",
            "content": f"""Answer the question using only the document excerpts below.
If the answer is not present, say so clearly.

Question: {question}

Document excerpts:
{context[:12000]}

Answer:""",
        }],
        temperature=0,
        max_tokens=1024,
    )
    return answer_resp.choices[0].message.content.strip()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a TOC-indexed document")
    parser.add_argument("question", nargs="*", default=["What is this document about?"])
    parser.add_argument("--doc", default=None, help="Document stem (required if multiple are indexed)")
    args = parser.parse_args()

    question = " ".join(args.question)
    stem = pick_doc(args.doc)

    print(f"\nDocument : {stem}")
    print(f"Question : {question}\n")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    answer = query(question, stem, client)
    print(f"\nAnswer:\n{answer}")
