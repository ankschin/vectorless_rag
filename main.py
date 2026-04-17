"""
CLI entry point for the vectorless RAG pipeline.

Usage:
  python main.py index                        # index all PDFs in data/
  python main.py index path/to/doc.pdf        # index a specific file
  python main.py query "your question"        # query (auto-picks if one doc)
  python main.py query "your question" --doc report.pdf
"""

import argparse
import sys
from pathlib import Path


def cmd_index(args):
    from toc_index import index_pdf
    paths = [Path(p) for p in args.files] if args.files else sorted(Path("data").glob("*.pdf"))
    if not paths:
        print("No PDFs found in data/. Pass a path as argument.")
        sys.exit(1)
    for pdf in paths:
        index_pdf(pdf, force=args.force)


def cmd_query(args):
    from toc_query import query, pick_doc
    from groq import Groq
    import os
    from dotenv import load_dotenv
    load_dotenv()

    stem = pick_doc(args.doc)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print(f"\nDocument : {stem}")
    print(f"Question : {args.question}\n")
    answer = query(args.question, stem, client)
    print(f"\nAnswer:\n{answer}")


def main():
    parser = argparse.ArgumentParser(description="Vectorless RAG — hierarchical TOC retrieval")
    sub = parser.add_subparsers(dest="command", required=True)

    idx = sub.add_parser("index", help="Index PDFs in data/")
    idx.add_argument("files", nargs="*", help="PDF paths (defaults to all in data/)")
    idx.add_argument("--force", action="store_true", help="Re-index even if already done")

    q = sub.add_parser("query", help="Ask a question against indexed documents")
    q.add_argument("question", help="Natural language question")
    q.add_argument("--doc", default=None, help="Document stem (required if multiple indexed)")

    args = parser.parse_args()
    {"index": cmd_index, "query": cmd_query}[args.command](args)


if __name__ == "__main__":
    main()
