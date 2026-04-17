"""
Query TOC-indexed documents without embeddings.

Pipeline (multi-doc):
  1. LLM reads all TOC trees and picks relevant page ranges per document
  2. Fetch the raw page text for those ranges from each index
  3. LLM synthesises a final answer with source attribution

Usage:
    python toc_query.py "What is this document about?"
    python toc_query.py --doc llm_fallacy "What fallacies are discussed?"
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

WORKSPACE = Path("workspace")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

index_map = {}

# ── Index helpers ──────────────────────────────────────────────────────────────

def list_docs() -> list[str]:
    return [p.stem for p in sorted(WORKSPACE.glob("*.json")) if p.stem != "index_map"]


def load_index(stem: str) -> dict:
    path = WORKSPACE / f"{stem}.json"
    if not path.exists():
        raise FileNotFoundError(f"No index for '{stem}'. Run toc_index.py first.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="cp1252"))


# ── TOC utilities ──────────────────────────────────────────────────────────────

def slim_toc(nodes: list[dict]) -> list[dict]:
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

def query_multi(question: str, doc_stems: list[str], client: Groq) -> str:
    indexes = {stem: load_index(stem) for stem in doc_stems}

    # Build a combined TOC block for all docs
    toc_block_parts = []
    for stem, index in indexes.items():
        toc_json = json.dumps(slim_toc(index["toc"]), indent=2)
        if len(toc_json) > 12000:
            toc_json = toc_json[:12000] + "\n... (truncated)"
        toc_block_parts.append(f'Document: "{stem}"\n{toc_json}')
    toc_block = "\n\n===\n\n".join(toc_block_parts)

    # ── Step 1: Navigate all TOCs ──────────────────────────────────────────────
    print(f"  [1/3] LLM navigating {len(doc_stems)} TOC(s) ...")
    nav_resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{
            "role": "user",
            "content": f"""You are navigating multiple documents' Tables of Contents to find where the answer lives.

Question: {question}

Tables of Contents:
{toc_block}

Return ONLY valid JSON (no prose, no fences):
{{
  "thinking": "<one sentence reasoning>",
  "matches": [
    {{"doc": "<document stem>", "page_ranges": ["<start>-<end>", ...]}},
    ...
  ]
}}

Only include documents that are likely to contain relevant information. Omit irrelevant documents entirely.""",
        }],
        temperature=0,
        max_tokens=768,
    )
    nav_text = nav_resp.choices[0].message.content.strip()
    if "```" in nav_text:
        nav_text = nav_text.split("```")[1].lstrip("json").strip()
    nav = json.loads(nav_text)

    thinking = nav.get("thinking", "")
    matches: list[dict] = nav.get("matches", [])
    print(f"  Reasoning : {thinking[:200]}")
    for m in matches:
        print(f"  Doc '{m['doc']}' → pages {m['page_ranges']}")

    if not matches:
        return "No relevant sections identified across the indexed documents."

    # ── Step 2: Fetch page content from each matched doc ───────────────────────
    print("  [2/3] Fetching page content ...")
    context_parts = []
    for m in matches:
        stem = m["doc"]
        if stem not in indexes:
            continue
        text = fetch_pages(indexes[stem], m["page_ranges"])
        if text:
            context_parts.append(f"### Source: {stem}\n\n{text}")

    if not context_parts:
        return "Could not retrieve content for the identified page ranges."

    combined_context = "\n\n" + "\n\n".join(context_parts)
    if len(combined_context) > 20000:
        combined_context = combined_context[:20000] + "\n... (truncated)"

    # ── Step 3: Generate answer ────────────────────────────────────────────────
    print("  [3/3] Generating answer ...")
    answer_resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{
            "role": "user",
            "content": f"""Answer the question using only the document excerpts below.
Cite the source document name when referencing information from it.
If the answer is not present in any excerpt, say so clearly.

Question: {question}

Document excerpts:
{combined_context}

Answer:""",
        }],
        temperature=0,
        max_tokens=1024,
    )
    return answer_resp.choices[0].message.content.strip()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query TOC-indexed documents")
    parser.add_argument("question", nargs="*", default=["What is this document about?"])
    parser.add_argument("--doc", default=None, help="Restrict to a single document stem")
    args = parser.parse_args()

    question = " ".join(args.question)
    docs = list_docs()
    if not docs:
        raise RuntimeError("No indexed documents found. Run toc_index.py first.")

    if args.doc:
        stem = Path(args.doc).stem
        if stem not in docs:
            raise KeyError(f"'{stem}' not indexed. Available: {docs}")
        docs = [stem]

    print(f"\nDocuments : {docs}")
    print(f"Question  : {question}\n")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    answer = query_multi(question, docs, client)
    print(f"\nAnswer:\n{answer}")
