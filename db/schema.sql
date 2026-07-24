-- CivicLens schema
-- Design note: chunks join back to segments/agenda_items so answers can cite
-- meeting + timestamp + speaker. This relational linkage is WHY pgvector was
-- chosen over a standalone vector store.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS meetings (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title        TEXT NOT NULL,
    body         TEXT NOT NULL DEFAULT 'city_council',  -- city_council | school_board | county
    meeting_date DATE,
    source_url   TEXT UNIQUE,          -- YouTube / Granicus URL
    duration_s   INT,
    status       TEXT NOT NULL DEFAULT 'pending',  -- pending|transcribed|diarized|embedded|failed
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per speaker-attributed transcript segment
CREATE TABLE IF NOT EXISTS segments (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    meeting_id  BIGINT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    speaker     TEXT,                  -- diarization label e.g. SPEAKER_02 (map to names later)
    start_s     REAL NOT NULL,
    end_s       REAL NOT NULL,
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_segments_meeting ON segments(meeting_id);

-- Agenda items parsed from meeting packets (Week 2)
CREATE TABLE IF NOT EXISTS agenda_items (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    meeting_id  BIGINT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    item_number TEXT,
    title       TEXT NOT NULL,
    category    TEXT                   -- zoning|budget|public_safety|education|other (Week 6 tagger)
);

-- Retrieval unit: grouped speaker turns, optionally linked to an agenda item.
-- 384 dims = BAAI/bge-small-en-v1.5
CREATE TABLE IF NOT EXISTS chunks (
    id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    meeting_id     BIGINT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    agenda_item_id BIGINT REFERENCES agenda_items(id),
    source         TEXT NOT NULL DEFAULT 'transcript',  -- transcript | document
    speaker        TEXT,
    start_s        REAL,
    end_s          REAL,
    text           TEXT NOT NULL,
    embedding      vector(384),
    tsv            tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX IF NOT EXISTS idx_chunks_meeting ON chunks(meeting_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN(tsv);
-- HNSW index for vector search (build after bulk load for speed)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops);

-- Week 7: structured vote extraction lands here
CREATE TABLE IF NOT EXISTS motions (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    meeting_id  BIGINT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    agenda_item_id BIGINT REFERENCES agenda_items(id),
    description TEXT NOT NULL,
    outcome     TEXT,                  -- passed|failed|tabled
    votes_for   INT,
    votes_against INT,
    detail      JSONB                  -- per-member votes when extractable
);
