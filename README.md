# CivicLens

AI meeting intelligence over local government meetings. Transcribes and diarizes
public city council meetings, parses agenda PDFs, and answers questions with
timestamped, speaker-attributed citations.

**Design constraint: runs at $0/month.** Heavy compute (transcription, diarization,
embedding) runs locally as batch jobs; only the lightweight query path runs in the
cloud on free tiers.

## Architecture

```
                         OFFLINE (your laptop)
  YouTube ──yt-dlp──> audio ──Groq Whisper──> transcript segments
                                │
                        pyannote diarization
                                │
                     speaker-aligned segments ──BGE embeddings──> Postgres/pgvector
  Agenda PDFs ──Docling──> chunks ──────────────────────────────────────┘

                         ONLINE (Render + Vercel + Supabase)
  Query ──> FastAPI ──> hybrid retrieval (vector + FTS) ──> Groq LLM ──> cited answer
```

## Quickstart

```bash
# 1. copy env template and fill in keys (all free-tier)
cp .env.example .env

# 2. start local Postgres with pgvector
docker compose up -d

# 3. apply schema
make schema

# 4. install python deps
cd ingestion && pip install -r requirements.txt && cd ..

# 5. ingest your first meeting
python ingestion/ingest.py --url "https://www.youtube.com/watch?v=<MEETING_ID>" --meeting-title "City Council 2026-07-06"

# 6. embed + load into pgvector
python ingestion/embed.py

# 7. ask a question
cd api && pip install -r requirements.txt
uvicorn app.main:app --reload
curl "localhost:8000/ask?q=What+was+decided+about+the+rezoning"
```

## Free-tier accounts you need

| Service | What for | Link |
|---|---|---|
| Groq | Whisper transcription + Llama LLM | console.groq.com |
| Google AI Studio | Gemini Flash (router + eval judge) | aistudio.google.com |
| Hugging Face | pyannote model access (accept gate, create token) | huggingface.co |
| Supabase | hosted Postgres + pgvector (deploy target) | supabase.com |
| Langfuse | tracing (add in Week 8) | cloud.langfuse.com |

## Repo layout

```
ingestion/   offline pipeline: download, transcribe, diarize, chunk, embed
api/         FastAPI query service (retrieval + generation)
db/          schema.sql (Postgres + pgvector)
evals/       golden dataset + eval runner (start Week 4, expand Week 9)
web/         Next.js frontend (Week 11 — stub for now)
.github/     CI eval gate (Week 10)
```

## Roadmap

- [x] Week 1: audio pipeline (ingest.py, diarize.py)
- [ ] Week 2: PDF parsing + agenda-item chunking
- [ ] Week 3: embeddings + pgvector retrieval
- [ ] Week 4: naive RAG endpoint + 25-question golden set baseline
- [ ] Week 5: hybrid search (RRF) + reranker
- [ ] Week 6: LangGraph router/retrieval/synthesis graph
- [ ] Week 7: vote extraction to SQL (structured outputs)
- [ ] Week 8: batch ingestion queue + Langfuse
- [ ] Weeks 9-12: eval harness, CI gates, deploy, frontend, write-up
