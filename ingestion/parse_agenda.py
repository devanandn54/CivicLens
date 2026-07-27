"""
Parse a Charlotte zoning agenda PDF with Docling and store agenda items
(item_number, title) in Postgres.

Usage:
    python ingestion/parse_agenda.py --meeting-id 1 --pdf data/raw/agendas/meeting_1_agenda.pdf

Parsing note:
Docling's PDF layout reconstruction merges the right-margin item number into
the section header text, e.g. "Rezoning Petition: 2025-131 by Queen City
Land, LLC 3." -> the "3." at the end is the agenda item number, not part of
the title. Section headers like "Zoning Committee Recommendation:" and
category dividers (CONSENT/DECISIONS/HEARINGS) also come through as
SECTION_HEADER, so we anchor on the "Rezoning Petition:" prefix rather than
just "ends in a number" to avoid false matches (e.g. boilerplate consent
text that also happens to end in a numeral).
"""

import argparse
import os
import re

import psycopg
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItemLabel

load_dotenv()

ITEM_RE = re.compile(r"^(Rezoning Petition:\s*.+?)\s+(\d+)\.$")


def extract_items(pdf_path: str) -> list[dict]:
    doc = DocumentConverter().convert(pdf_path).document
    items = []
    for text_item in doc.texts:
        if text_item.label != DocItemLabel.SECTION_HEADER:
            continue
        m = ITEM_RE.match(text_item.text.strip())
        if not m:
            continue
        items.append({"item_number": m.group(2), "title": m.group(1)})
    return items


def save(meeting_id: int, items: list[dict]) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM agenda_items WHERE meeting_id = %s", (meeting_id,))
        cur.executemany(
            "INSERT INTO agenda_items (meeting_id, item_number, title, category) VALUES (%s, %s, %s, %s)",
            [(meeting_id, i["item_number"], i["title"], "zoning") for i in items],
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting-id", type=int, required=True)
    ap.add_argument("--pdf", required=True)
    args = ap.parse_args()

    print("1/2 parsing PDF with Docling...")
    items = extract_items(args.pdf)
    print(f"2/2 saving {len(items)} agenda items for meeting {args.meeting_id}...")
    save(args.meeting_id, items)
    for i in items:
        print(f"  {i['item_number']:>3}. {i['title']}")


if __name__ == "__main__":
    main()
