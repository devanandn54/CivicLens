"""
Week 3 — Chunk diarized segments by speaker turn and embed with BGE-small
(local, free). Skeleton is here now so the pipeline shape is visible from
day one; flesh out as you reach Week 3.

Chunking strategy (write about this):
- Group consecutive segments by the SAME speaker into one chunk (a "turn"),
  capped at ~1200 chars so chunks stay retrieval-sized.
- Keep meeting_id, speaker, start_s, end_s on every chunk -> citations.
- Later (Week 2+): attach agenda_item_id when the agenda parser can map
  timestamp ranges to agenda items.

Usage:
    python ingestion/embed.py [--meeting-id 1]
"""

import argparse
import os

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

MAX_CHUNK_CHARS = 1200


def build_chunks(segments: list[tuple]) -> list[dict]:
    """segments: rows of (speaker, start_s, end_s, text) ordered by start_s."""
    chunks: list[dict] = []
    current: dict | None = None
    for speaker, start_s, end_s, text in segments:
        same_speaker = current is not None and current["speaker"] == speaker
        fits = current is not None and len(current["text"]) + len(text) < MAX_CHUNK_CHARS
        if same_speaker and fits:
            current["text"] += " " + text
            current["end_s"] = end_s
        else:
            if current:
                chunks.append(current)
            current = {"speaker": speaker, "start_s": start_s, "end_s": end_s, "text": text}
    if current:
        chunks.append(current)
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting-id", type=int, default=None, help="default: all diarized meetings")
    args = ap.parse_args()

    model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id FROM meetings
            WHERE status = 'diarized' AND (%(mid)s IS NULL OR id = %(mid)s)
            """,
            {"mid": args.meeting_id},
        )
        meeting_ids = [r[0] for r in cur.fetchall()]

        for mid in meeting_ids:
            cur.execute(
                """
                SELECT speaker, start_s, end_s, text FROM segments
                WHERE meeting_id = %s ORDER BY start_s
                """,
                (mid,),
            )
            chunks = build_chunks(cur.fetchall())
            print(f"meeting {mid}: {len(chunks)} chunks, embedding...")
            vectors = model.encode(
                [c["text"] for c in chunks],
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            cur.execute("DELETE FROM chunks WHERE meeting_id = %s AND source = 'transcript'", (mid,))
            cur.executemany(
                """
                INSERT INTO chunks (meeting_id, source, speaker, start_s, end_s, text, embedding)
                VALUES (%s, 'transcript', %s, %s, %s, %s, %s)
                """,
                [
                    (mid, c["speaker"], c["start_s"], c["end_s"], c["text"], vec.tolist())
                    for c, vec in zip(chunks, vectors)
                ],
            )
            cur.execute("UPDATE meetings SET status = 'embedded' WHERE id = %s", (mid,))

    print("done. Start the API and ask a question.")


if __name__ == "__main__":
    main()
