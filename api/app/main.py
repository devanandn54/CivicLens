"""
Week 4 — Naive RAG endpoint. Deliberately simple: retrieve top-k by cosine
similarity, generate an answer with citations. Weeks 5-6 replace retrieval
with hybrid+rerank and the linear flow with a LangGraph graph.

Run:  uvicorn app.main:app --reload
Try:  curl "localhost:8000/ask?q=What+was+decided+about+the+rezoning"
"""

import os

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from groq import Groq
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()

app = FastAPI(title="CivicLens API")

# Loaded once at startup — embedding query text locally is free and fast
_embedder = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
_llm = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM_PROMPT = """You answer questions about local government meetings using ONLY the
provided transcript excerpts. Rules:
- Every factual claim must cite its source as [meeting_id @ MM:SS, speaker].
- If the excerpts don't contain the answer, say so plainly. Never guess.
- Quote sparingly; prefer paraphrase with citation."""


class Citation(BaseModel):
    meeting_id: int
    speaker: str | None
    start_s: float
    text: str


class Answer(BaseModel):
    answer: str
    citations: list[Citation]


def retrieve(query: str, k: int = 6) -> list[dict]:
    vec = _embedder.encode(query, normalize_embeddings=True).tolist()
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT meeting_id, speaker, start_s, text,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vec, vec, k),
        )
        cols = ["meeting_id", "speaker", "start_s", "text", "score"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def format_context(hits: list[dict]) -> str:
    lines = []
    for h in hits:
        mm, ss = divmod(int(h["start_s"]), 60)
        lines.append(
            f"[meeting {h['meeting_id']} @ {mm:02d}:{ss:02d}, {h['speaker'] or 'unknown'}]: {h['text']}"
        )
    return "\n\n".join(lines)


@app.get("/ask", response_model=Answer)
def ask(q: str = Query(..., min_length=3)) -> Answer:
    hits = retrieve(q)
    completion = _llm.chat.completions.create(
        model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpts:\n\n{format_context(hits)}\n\nQuestion: {q}"},
        ],
        temperature=0.1,
    )
    return Answer(
        answer=completion.choices[0].message.content,
        citations=[
            Citation(
                meeting_id=h["meeting_id"],
                speaker=h["speaker"],
                start_s=h["start_s"],
                text=h["text"][:200],
            )
            for h in hits
        ],
    )


@app.get("/health")
def health() -> dict:
    return {"ok": True}
