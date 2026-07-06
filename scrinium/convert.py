#!/usr/bin/env python3
"""
Lightweight docling converter — runs as subprocess.
Converts a single document to Markdown, outputs to stdout.
Does NOT import torch until called — zero startup cost.
"""
import sys
import json
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: convert.py <filepath>"}))
        sys.exit(1)

    filepath = Path(sys.argv[1])
    if not filepath.exists():
        print(json.dumps({"error": f"File not found: {filepath}"}))
        sys.exit(1)

    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        print(json.dumps({"error": "docling not installed"}))
        sys.exit(1)

    converter = DocumentConverter()
    doc = converter.convert(str(filepath))
    md = doc.document.export_to_markdown()
    pages = len(doc.pages)
    result = {"markdown": md, "pages": pages}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
