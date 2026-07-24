"""
Week 1 — Diarize a meeting's audio with pyannote and align speaker labels
onto the Whisper segments already in Postgres.

Usage:
    python ingestion/diarize.py --meeting-id 1 --audio data/raw/VIDEOID.m4a

Prereqs:
- Accept the model gate at hf.co/pyannote/speaker-diarization-3.1
- HF_TOKEN in .env

This runs on CPU. A 2-hour meeting can take a while — run it overnight.
That's fine: this is the OFFLINE pipeline by design.

Progress / status:
- Every phase is logged with a timestamp and flush=True, since stdout is
  fully buffered (not line-buffered) once redirected to a file, so a plain
  `print` can silently sit in a buffer for hours and never reach the log.
  Run with `python -u` (or PYTHONUNBUFFERED=1) as a second layer of defense.
- A JSON status file is written to data/processed/diarize_status_<meeting_id>.json
  after every phase transition, so you can check progress without tailing
  the log: `cat data/processed/diarize_status_<meeting_id>.json`

The alignment problem (the instructive part):
Whisper gives you text segments with timestamps. pyannote gives you speaker
turns with timestamps. They never line up exactly. Strategy here: assign each
Whisper segment the speaker whose turn overlaps it the most. Simple, robust,
and you'll find its failure modes (crosstalk, short interjections) yourself —
write them down for your blog post.
"""

import argparse
import json
import os
import time
import traceback

import psycopg
from dotenv import load_dotenv
from pyannote.audio import Pipeline

load_dotenv()


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def status_path(meeting_id: int) -> str:
    return f"data/processed/diarize_status_{meeting_id}.json"


def write_status(meeting_id: int, stage: str, **extra) -> None:
    payload = {
        "meeting_id": meeting_id,
        "stage": stage,
        "pid": os.getpid(),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra,
    }
    path = status_path(meeting_id)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, default=str)  # never let an odd value (e.g. numpy types) crash the run
    os.replace(tmp, path)  # atomic — readers never see a half-written file


class StepLogger:
    """pyannote progress hook that logs plain timestamped lines instead of
    drawing a terminal progress bar (rich's bar redraws with \\r, which is
    unreadable via tail/nohup logs) and mirrors progress into the status file."""

    def __init__(self, meeting_id: int):
        self.meeting_id = meeting_id
        self._step = None

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        # pyannote/numpy hand back numpy int64 here, which json.dump can't serialize
        completed = 1 if completed is None else int(completed)
        total = 1 if total is None else int(total)
        if step_name != self._step:
            self._step = step_name
            log(f"diarization step started: {step_name}")
        if completed >= total:
            log(f"diarization step done: {step_name} ({completed}/{total})")
            write_status(self.meeting_id, f"diarizing:{step_name}", step_completed=completed, step_total=total)
        elif total > 1 and completed % max(total // 10, 1) == 0:
            log(f"diarization step progress: {step_name} {completed}/{total}")
            write_status(self.meeting_id, f"diarizing:{step_name}", step_completed=completed, step_total=total)


def diarize(audio_path: str, meeting_id: int):
    log("loading pyannote pipeline...")
    write_status(meeting_id, "loading_model")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=os.environ["HF_TOKEN"],
    )
    log("pipeline loaded, running diarization (CPU — be patient)...")
    write_status(meeting_id, "diarizing")
    hook = StepLogger(meeting_id)
    return pipeline(audio_path, hook=hook)


def dominant_speaker(turns: list[tuple[float, float, str]], start: float, end: float) -> str | None:
    """Speaker whose turns overlap [start, end] the most."""
    overlap: dict[str, float] = {}
    for t_start, t_end, spk in turns:
        o = min(end, t_end) - max(start, t_start)
        if o > 0:
            overlap[spk] = overlap.get(spk, 0.0) + o
    return max(overlap, key=overlap.get) if overlap else None


def run(meeting_id: int, audio_path: str) -> None:
    log(f"starting diarization for meeting {meeting_id} (pid {os.getpid()}, audio={audio_path})")
    write_status(meeting_id, "starting")

    output = diarize(audio_path, meeting_id)
    # pyannote.audio 4.x wraps the result in DiarizeOutput; the Annotation
    # itself (with .itertracks) is on .speaker_diarization
    annotation = output.speaker_diarization
    turns = [
        (segment.start, segment.end, label)
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]
    n_speakers = len({t[2] for t in turns})
    log(f"found {n_speakers} speakers across {len(turns)} turns")
    write_status(meeting_id, "aligning_segments", speakers=n_speakers, turns=len(turns))

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, start_s, end_s FROM segments WHERE meeting_id = %s",
            (meeting_id,),
        )
        rows = cur.fetchall()
        updates = [
            (dominant_speaker(turns, start_s, end_s), seg_id)
            for seg_id, start_s, end_s in rows
        ]
        cur.executemany("UPDATE segments SET speaker = %s WHERE id = %s", updates)
        cur.execute(
            "UPDATE meetings SET status = 'diarized' WHERE id = %s",
            (meeting_id,),
        )
    labeled = sum(1 for spk, _ in updates if spk)
    log(f"labeled {labeled}/{len(updates)} segments. Next: python ingestion/embed.py")
    write_status(meeting_id, "done", labeled=labeled, total=len(updates))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting-id", type=int, required=True)
    ap.add_argument("--audio", required=True)
    args = ap.parse_args()

    try:
        run(args.meeting_id, args.audio)
    except Exception as exc:
        log(f"FAILED: {exc}")
        write_status(args.meeting_id, "failed", error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
