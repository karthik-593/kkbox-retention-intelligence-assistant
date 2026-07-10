"""Document loaders: turn corpus files into a flat list of {text, source, doc_type, page} records.

One page = one record for PDFs (page granularity is finer than the eventual chunk, so the chunker
still does the real splitting across page boundaries where needed). One row = one record for a
tabular file, since each row (one offer-eligibility rule) is already a self-contained retrievable
unit rather than something that benefits from further splitting.
"""
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd


def load_pdf(path, doc_type: str) -> list[dict]:
    doc = fitz.open(path)
    records = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            records.append({"text": text, "source": Path(path).name, "doc_type": doc_type, "page": i + 1})
    doc.close()
    return records


def load_csv(path, doc_type: str, text_columns: list[str] | None = None) -> list[dict]:
    df = pd.read_csv(path)
    cols = text_columns or df.columns.tolist()
    records = []
    for i, row in df.iterrows():
        text = "; ".join(f"{c}: {row[c]}" for c in cols if pd.notna(row[c]) and str(row[c]) != "")
        records.append({"text": text, "source": Path(path).name, "doc_type": doc_type, "page": i + 1})
    return records
