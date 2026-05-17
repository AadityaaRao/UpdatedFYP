-- backend/database/schema.sql
-- ─────────────────────────────────────────────────────────────────
-- VQA Guider — SQLite schema
-- Run via db.init_db() at application startup.
-- All ids are UUID4 strings (TEXT PRIMARY KEY).
-- Foreign keys are enforced via PRAGMA foreign_keys = ON (set per connection).
-- ─────────────────────────────────────────────────────────────────
-- ── videos ───────────────────────────────────────────────────────
-- One row per uploaded video file.
-- INSERT OR IGNORE guards against duplicate uploads of the same video_id.
CREATE TABLE IF NOT EXISTS videos (
    id                TEXT    PRIMARY KEY,   -- UUID4 (video_id from upload)
    path              TEXT    NOT NULL,      -- absolute path on disk
    original_filename TEXT    NOT NULL,      -- as supplied by the uploader
    created_at        TEXT    NOT NULL       -- ISO-8601 UTC
);
-- ── queries ───────────────────────────────────────────────────────
-- One row per question string submitted.
-- A question is re-used across different videos (same question asked
-- about different videos creates separate results rows but can share
-- a query row via the unique_question index below).
CREATE TABLE IF NOT EXISTS queries (
    id         TEXT    PRIMARY KEY,          -- UUID4 (query_id)
    question   TEXT    NOT NULL,             -- raw question string (as submitted)
    created_at TEXT    NOT NULL              -- ISO-8601 UTC
);
-- Index to quickly look up an existing query by its text
CREATE INDEX IF NOT EXISTS idx_queries_question
    ON queries (question);
-- ── results ───────────────────────────────────────────────────────
-- One row per (video, question) inference run.
-- routing_json stores {"action": 0.52, "tracking": 0.31, "scene": 0.17}
-- from_cache = 1 means the answer was served from pickle cache, not re-inferred.
CREATE TABLE IF NOT EXISTS results (
    id           TEXT     PRIMARY KEY,       -- UUID4 (result_id)
    video_id     TEXT     NOT NULL,          -- FK → videos.id
    query_id     TEXT     NOT NULL,          -- FK → queries.id
    answer       TEXT     NOT NULL,          -- AI-generated answer string
    routing_json TEXT     NOT NULL,          -- JSON-encoded task routing dict
    from_cache   INTEGER  NOT NULL DEFAULT 0,-- 0 = fresh inference, 1 = cache hit
    created_at   TEXT     NOT NULL,          -- ISO-8601 UTC
    FOREIGN KEY (video_id) REFERENCES videos(id)  ON DELETE CASCADE,
    FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE CASCADE
);
-- Index for fast result lookup by video (useful for history queries)
CREATE INDEX IF NOT EXISTS idx_results_video_id
    ON results (video_id);

-- Edu-VQAGuider v2 tables. Extra columns on videos are added by db.init_db()
-- because SQLite cannot reliably add existing columns from this script on all
-- bundled versions.
CREATE TABLE IF NOT EXISTS video_chunks (
    id              TEXT    PRIMARY KEY,
    video_id        TEXT    NOT NULL,
    start_time      REAL    NOT NULL,
    end_time        REAL    NOT NULL,
    transcript_text TEXT    NOT NULL DEFAULT '',
    visual_summary  TEXT,
    frame_paths_json TEXT   NOT NULL DEFAULT '[]',
    embedding_path  TEXT,
    created_at      TEXT   NOT NULL,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_video_chunks_video_id
    ON video_chunks (video_id);

CREATE TABLE IF NOT EXISTS edu_results (
    id              TEXT    PRIMARY KEY,
    video_id        TEXT    NOT NULL,
    question        TEXT    NOT NULL,
    direct_answer   TEXT    NOT NULL,
    answer          TEXT    NOT NULL,
    route_json      TEXT    NOT NULL,
    evidence_json   TEXT    NOT NULL,
    planner_source  TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_edu_results_video_id
    ON edu_results (video_id);
