"""Section parsing and dataset loading utilities for HR policy documents."""
import json
import re
from pathlib import Path

from .config import LABEL2ID


def split_sections(text: str) -> list[dict]:
    """Split a policy document into structured sections (title + body).

    Supports:
    - Blank-line separated paragraphs/blocks
    - Numbered or titled headings like 'Section 1: ...' or '1. ...'
    """
    cleaned = text.replace("\r\n", "\n").strip()
    if not cleaned:
        return []

    raw_blocks = re.split(r"\n\s*\n+", cleaned)
    sections = []
    for idx, block in enumerate(raw_blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if len(lines) > 1 and (
            re.match(r"^(section\s+\d+|\d+\.|#+\s)", lines[0], re.IGNORECASE)
            or len(lines[0].split()) <= 10
        ):
            title = re.sub(r"^#+\s*", "", lines[0])
            body = " ".join(lines[1:])
        else:
            title = f"Section {idx}"
            body = " ".join(lines)

        full_text = f"{title}: {body}" if title != f"Section {idx}" else body
        sections.append({
            "index": idx,
            "title": title,
            "body": body,
            "full_text": full_text,
        })
    return sections


def load_dataset_jsonl(path: str | Path) -> list[dict]:
    """Load labeled HR policy records from a JSONL file."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        category = row["category"]
        if category not in LABEL2ID:
            raise ValueError(f"Unknown category: {category}")
        records.append({
            "text": row["text"].strip(),
            "category": category,
            "label": LABEL2ID[category],
        })
    return records
