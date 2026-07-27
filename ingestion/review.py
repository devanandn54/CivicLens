"""
Review a diarized meeting: print a readable transcript, or print
speaker/label stats.

Usage:
    python ingestion/review.py transcript --meeting-id 1 [--start-s 0] [--end-s 600]
    python ingestion/review.py stats --meeting-id 1
"""

import argparse
import os

import psycopg
from dotenv import load_dotenv

load_dotenv()


def fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fetch_segments(cur, meeting_id: int, start_s: float | None, end_s: float | None):
    cur.execute(
        """
        SELECT start_s, end_s, speaker, text FROM segments
        WHERE meeting_id = %(mid)s
          AND (%(start)s IS NULL OR start_s >= %(start)s)
          AND (%(end)s IS NULL OR start_s < %(end)s)
        ORDER BY start_s
        """,
        {"mid": meeting_id, "start": start_s, "end": end_s},
    )
    return cur.fetchall()


def print_transcript(cur, meeting_id: int, start_s: float | None, end_s: float | None) -> None:
    rows = fetch_segments(cur, meeting_id, start_s, end_s)
    for seg_start, _seg_end, speaker, text in rows:
        label = speaker or "UNKNOWN"
        print(f"[{fmt_ts(seg_start)}] {label:<14} {text.strip()}")


def print_stats(cur, meeting_id: int) -> None:
    cur.execute(
        "SELECT speaker, start_s, end_s FROM segments WHERE meeting_id = %s",
        (meeting_id,),
    )
    rows = cur.fetchall()
    total = len(rows)
    unlabeled = sum(1 for speaker, _, _ in rows if speaker is None)
    durations = [end_s - start_s for _, start_s, end_s in rows]
    avg_len = sum(durations) / total if total else 0.0

    per_speaker: dict[str, int] = {}
    for speaker, _, _ in rows:
        label = speaker or "UNKNOWN"
        per_speaker[label] = per_speaker.get(label, 0) + 1

    print(f"meeting {meeting_id}: {total} segments, {unlabeled} unlabeled, "
          f"avg segment length {avg_len:.2f}s")
    print()
    print(f"{'speaker':<14} {'segments':>8}")
    for label, count in sorted(per_speaker.items(), key=lambda kv: -kv[1]):
        print(f"{label:<14} {count:>8}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    tp = sub.add_parser("transcript", help="print a readable transcript")
    tp.add_argument("--meeting-id", type=int, required=True)
    tp.add_argument("--start-s", type=float, default=None, help="only show segments starting at/after this time")
    tp.add_argument("--end-s", type=float, default=None, help="only show segments starting before this time")

    sp = sub.add_parser("stats", help="print segment/speaker stats")
    sp.add_argument("--meeting-id", type=int, required=True)

    args = ap.parse_args()

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        if args.command == "transcript":
            print_transcript(cur, args.meeting_id, args.start_s, args.end_s)
        elif args.command == "stats":
            print_stats(cur, args.meeting_id)


if __name__ == "__main__":
    main()
