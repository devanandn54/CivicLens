"""
Week 1 — Ingest a meeting: download audio, transcribe with Groq Whisper,
store raw segments in Postgres and JSON on disk.

Usage:
    python ingestion/ingest.py --url "https://www.youtube.com/watch?v=XXXX" \
        --meeting-title "Charlotte City Council 2026-07-06" \
        [--meeting-date 2026-07-06]

Notes:
- Groq's audio endpoint caps file size (~25MB free tier), so we chunk the
  audio into ~10-minute segments with ffmpeg and offset timestamps. This is
  the same pattern you'd use in production for any provider limit.
- Diarization is a separate pass: run diarize.py after this.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CHUNK_SECONDS = 600  # 10-minute audio slices


def download_audio(url: str, out_dir: Path) -> Path:
    """Download best audio and convert to 16kHz mono m4a (small, ASR-friendly)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / "%(id)s.%(ext)s")
    subprocess.run(
        [
            "yt-dlp",
            "-f", "bestaudio",
            "--extract-audio",
            "--audio-format", "m4a",
            "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000",
            "-o", out_template,
            url,
        ],
        check=True,
    )
    # yt-dlp names the file by video id; grab the newest m4a
    files = sorted(out_dir.glob("*.m4a"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit("Download produced no audio file")
    return files[-1]


def split_audio(audio_path: Path) -> list[Path]:
    """Slice audio into CHUNK_SECONDS pieces so each fits provider limits."""
    part_dir = audio_path.parent / f"{audio_path.stem}_parts"
    part_dir.mkdir(exist_ok=True)
    pattern = str(part_dir / "part_%03d.m4a")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
            "-c", "copy", pattern,
        ],
        check=True,
        capture_output=True,
    )
    return sorted(part_dir.glob("part_*.m4a"))


def transcribe(parts: list[Path]) -> list[dict]:
    """Transcribe each slice with Groq Whisper; offset timestamps by slice index."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    model = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")
    segments: list[dict] = []

    for i, part in enumerate(parts):
        offset = i * CHUNK_SECONDS
        print(f"  transcribing {part.name} (offset {offset}s)...")
        with open(part, "rb") as f:
            resp = client.audio.transcriptions.create(
                file=(part.name, f.read()),
                model=model,
                response_format="verbose_json",  # includes segment timestamps
            )
        for seg in resp.segments:
            segments.append(
                {
                    "start_s": round(seg["start"] + offset, 2),
                    "end_s": round(seg["end"] + offset, 2),
                    "text": seg["text"].strip(),
                }
            )
    return segments


def save(meeting_title: str, meeting_date: str | None, url: str, segments: list[dict]) -> int:
    """Insert meeting + segments into Postgres, and mirror to JSON for diarize.py."""
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO meetings (title, meeting_date, source_url, status)
            VALUES (%s, %s, %s, 'transcribed')
            ON CONFLICT (source_url) DO UPDATE SET status = 'transcribed'
            RETURNING id
            """,
            (meeting_title, meeting_date, url),
        )
        meeting_id = cur.fetchone()[0]
        cur.execute("DELETE FROM segments WHERE meeting_id = %s", (meeting_id,))
        cur.executemany(
            "INSERT INTO segments (meeting_id, start_s, end_s, text) VALUES (%s, %s, %s, %s)",
            [(meeting_id, s["start_s"], s["end_s"], s["text"]) for s in segments],
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"meeting_{meeting_id}.json"
    out.write_text(json.dumps({"meeting_id": meeting_id, "segments": segments}, indent=2))
    print(f"  saved {len(segments)} segments -> {out}")
    return meeting_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--meeting-title", required=True)
    ap.add_argument("--meeting-date", default=None)
    args = ap.parse_args()

    print("1/3 downloading audio...")
    audio = download_audio(args.url, RAW_DIR)
    print("2/3 splitting + transcribing...")
    parts = split_audio(audio)
    segments = transcribe(parts)
    print("3/3 saving...")
    meeting_id = save(args.meeting_title, args.meeting_date, args.url, segments)
    print(f"Done. meeting_id={meeting_id}. Next: python ingestion/diarize.py --meeting-id {meeting_id} --audio {audio}")


if __name__ == "__main__":
    main()
