# Vectorless RAG

Retrieval-Augmented Generation without embeddings or vector databases. Instead of similarity search, an LLM navigates a hierarchical Table of Contents to find relevant sections, then reads only those pages to answer the question.

## How it works

```
PDF  -->  extract pages  -->  build TOC tree  -->  workspace/<doc>.json
                                                          |
question  -->  LLM picks page ranges from TOC  -->  fetch pages  -->  LLM answers
```

No embeddings. No vector DB. Just structured navigation.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile   # optional, this is the default
```

## Usage

Place PDF files in `data/`.

**Index:**
```bash
python toc_index.py                    # index all PDFs in data/
python toc_index.py data/mydoc.pdf     # index a specific file
python toc_index.py --force            # re-index even if already done
```

**Query:**
```bash
python toc_query.py "What is this document about?"
python toc_query.py --doc mydoc "How do I configure X?"
```

**Via main.py:**
```bash
python main.py index
python main.py index data/mydoc.pdf --force
python main.py query "What is this document about?"
python main.py query "How do I configure X?" --doc mydoc
```

## Project structure

```
data/               PDFs to index
workspace/          Index files (gitignore this)
  <stem>.json       TOC tree + full page text
toc_index.py        Indexing pipeline
toc_query.py        Query pipeline
main.py             CLI wrapper
```

## Index file format

`workspace/<stem>.json`:

```json
{
  "title": "document stem",
  "source": "pdf_outline | llm",
  "total_pages": 49,
  "toc": [
    {
      "title": "Chapter 1",
      "page_start": 1,
      "page_end": 10,
      "children": [
        { "title": "1.1 Overview", "page_start": 1, "page_end": 4, "children": [] }
      ]
    }
  ],
  "pages": {
    "1": "page text...",
    "2": "page text..."
  }
}
```

`source` is `pdf_outline` when the PDF has built-in bookmarks (no LLM call needed for indexing), and `llm` when the TOC was inferred from page content.
