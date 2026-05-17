"""
backend/database/db.py
────────────────────────────────────────────────────────────
SQLite CRUD layer for VQA Guider.
Design rules:
  • No ORM — plain sqlite3 from the stdlib.
  • Every public function opens its own connection and closes it
    in a finally block. This is safe, simple, and plays well with
    FastAPI's thread-pool execution of sync route handlers.
  • Foreign key enforcement is ON per connection (SQLite default is OFF).
  • WAL journal mode for better concurrent read performance.
  • All queries are parameterized — no string interpolation.
  • Row results are returned as plain dicts (via sqlite3.Row), not
    as custom objects, so routes can serialize them directly.
Functions:
    get_connection()   → sqlite3.Connection  (caller must close)
    init_db()          → None   (creates tables from schema.sql)
    insert_video()     → None
    insert_query()     → str    (returns generated query_id)
    insert_result()    → None
    get_result()       → dict | None
    get_results_for_video() → list[dict]
"""
from __future__ import annotations
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from backend.utils.logger import get_logger
from config import SQLITE_PATH
logger = get_logger(__name__)
# Schema file lives next to this module
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"
# ── Helpers ───────────────────────────────────────────────────
def _now() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain Python dict."""
    return dict(row)
# ══════════════════════════════════════════════════════════════
# Connection management
# ══════════════════════════════════════════════════════════════
def get_connection() -> sqlite3.Connection:
    """
    Open and return a new SQLite connection.
    Settings applied per connection:
      • row_factory = sqlite3.Row  → access columns by name
      • PRAGMA foreign_keys = ON   → enforce FK constraints
      • PRAGMA journal_mode = WAL  → concurrent reads without blocking writes
    The caller is responsible for closing the connection.
    Use in a try/finally block:
        conn = get_connection()
        try:
            ...
        finally:
            conn.close()
    """
    conn = sqlite3.connect(
        database=str(SQLITE_PATH),
        check_same_thread=False,   # safe: each call gets its own connection
    )
    conn.row_factory = sqlite3.Row
    # Enable FK enforcement (SQLite defaults to OFF)
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL mode: readers don't block writers, writers don't block readers
    conn.execute("PRAGMA journal_mode = WAL")
    return conn
# ══════════════════════════════════════════════════════════════
# Initialisation
# ══════════════════════════════════════════════════════════════
def init_db() -> None:
    """
    Create all tables and indexes from schema.sql.
    Safe to call multiple times — all statements use IF NOT EXISTS.
    Called once during application startup (main.py lifespan).
    """
    if not _SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {_SCHEMA_PATH}")
    schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    conn = get_connection()
    try:
        # executescript commits any pending transaction automatically
        conn.executescript(schema_sql)
        _ensure_video_columns(conn)
        logger.info("Database initialised: %s", SQLITE_PATH)
    except sqlite3.Error as exc:
        logger.exception("Failed to initialise database: %s", exc)
        raise
    finally:
        conn.close()


def _ensure_video_columns(conn: sqlite3.Connection) -> None:
    """Add v2 metadata columns to an existing videos table if needed."""
    existing = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(videos)").fetchall()
    }
    columns = {
        "duration_sec": "REAL",
        "status": "TEXT NOT NULL DEFAULT 'ready'",
        "processing_error": "TEXT",
    }
    for name, decl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE videos ADD COLUMN {name} {decl}")
    conn.commit()
# ══════════════════════════════════════════════════════════════
# videos table
# ══════════════════════════════════════════════════════════════
def insert_video(
    video_id: str,
    path: str,
    original_filename: str,
    duration_sec: float | None = None,
    status: str = "ready",
) -> None:
    """
    Insert a new video record.
    Uses INSERT OR IGNORE so calling this function a second time with
    the same video_id is a no-op (idempotent). This matters because
    ask_question can be called multiple times for the same video.
    Args:
        video_id:          UUID string (from upload route)
        path:              Absolute path where the file is stored on disk
        original_filename: Filename as provided by the uploader
    """
    sql = """
        INSERT OR IGNORE INTO videos
            (id, path, original_filename, duration_sec, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    conn = get_connection()
    try:
        conn.execute(sql, (video_id, path, original_filename, duration_sec, status, _now()))
        conn.commit()
        logger.debug("insert_video: id=%s path=%s", video_id, path)
    except sqlite3.Error as exc:
        logger.exception("insert_video failed: %s", exc)
        raise
    finally:
        conn.close()


def update_video_processing(
    video_id: str,
    status: str,
    duration_sec: float | None = None,
    processing_error: str | None = None,
) -> None:
    sql = """
        UPDATE videos
        SET status = ?,
            duration_sec = COALESCE(?, duration_sec),
            processing_error = ?
        WHERE id = ?
    """
    conn = get_connection()
    try:
        conn.execute(sql, (status, duration_sec, processing_error, video_id))
        conn.commit()
    finally:
        conn.close()


def get_video(video_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def delete_video_chunks(video_id: str) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM video_chunks WHERE video_id = ?", (video_id,))
        conn.commit()
    finally:
        conn.close()


def insert_video_chunk(
    chunk_id: str,
    video_id: str,
    start_time: float,
    end_time: float,
    transcript_text: str = "",
    visual_summary: str | None = None,
    frame_paths: list[str] | None = None,
    embedding_path: str | None = None,
) -> None:
    sql = """
        INSERT INTO video_chunks
            (id, video_id, start_time, end_time, transcript_text, visual_summary,
             frame_paths_json, embedding_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    conn = get_connection()
    try:
        conn.execute(
            sql,
            (
                chunk_id,
                video_id,
                start_time,
                end_time,
                transcript_text or "",
                visual_summary,
                json.dumps(frame_paths or []),
                embedding_path,
                _now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_video_chunks(video_id: str) -> list[dict]:
    sql = "SELECT * FROM video_chunks WHERE video_id = ? ORDER BY start_time ASC"
    conn = get_connection()
    try:
        rows = conn.execute(sql, (video_id,)).fetchall()
        chunks = []
        for row in rows:
            data = _row_to_dict(row)
            data["frame_paths"] = json.loads(data.pop("frame_paths_json") or "[]")
            chunks.append(data)
        return chunks
    finally:
        conn.close()


def insert_edu_result(
    result_id: str,
    video_id: str,
    question: str,
    direct_answer: str,
    answer: str,
    route: dict,
    evidence: list[dict],
    planner_source: str,
) -> None:
    sql = """
        INSERT INTO edu_results
            (id, video_id, question, direct_answer, answer, route_json,
             evidence_json, planner_source, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    conn = get_connection()
    try:
        conn.execute(
            sql,
            (
                result_id,
                video_id,
                question,
                direct_answer,
                answer,
                json.dumps(route),
                json.dumps(evidence),
                planner_source,
                _now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_edu_result(result_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM edu_results WHERE id = ?", (result_id,)).fetchone()
        if row is None:
            return None
        data = _row_to_dict(row)
        data["route"] = json.loads(data.pop("route_json"))
        data["evidence"] = json.loads(data.pop("evidence_json"))
        return data
    finally:
        conn.close()
# ══════════════════════════════════════════════════════════════
# queries table
# ══════════════════════════════════════════════════════════════
def insert_query(question: str) -> str:
    """
    Insert a new query record and return the generated query_id.
    A new UUID is generated for every call — this means one question
    asked multiple times creates multiple query rows, giving a full
    audit trail of every inference request.
    Args:
        question: Raw question string as submitted by the user
    Returns:
        query_id: UUID string for use in the results table
    """
    query_id = str(uuid.uuid4())
    sql = """
        INSERT INTO queries (id, question, created_at)
        VALUES (?, ?, ?)
    """
    conn = get_connection()
    try:
        conn.execute(sql, (query_id, question, _now()))
        conn.commit()
        logger.debug("insert_query: id=%s question='%s...'", query_id, question[:40])
        return query_id
    except sqlite3.Error as exc:
        logger.exception("insert_query failed: %s", exc)
        raise
    finally:
        conn.close()
# ══════════════════════════════════════════════════════════════
# results table
# ══════════════════════════════════════════════════════════════
def insert_result(
    result_id: str,
    video_id: str,
    query_id: str,
    answer: str,
    routing_dict: dict,
    from_cache: bool = False,
) -> None:
    """
    Insert a result record linking a video to a query and its answer.
    Args:
        result_id:    UUID string (generated in ask_question route)
        video_id:     FK → videos.id
        query_id:     FK → queries.id  (returned by insert_query)
        answer:       AI-generated answer string
        routing_dict: {"action": float, "tracking": float, "scene": float}
        from_cache:   True if the answer came from the pickle cache
    """
    routing_json = json.dumps(routing_dict)
    sql = """
        INSERT INTO results
            (id, video_id, query_id, answer, routing_json, from_cache, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    conn = get_connection()
    try:
        conn.execute(
            sql,
            (
                result_id,
                video_id,
                query_id,
                answer,
                routing_json,
                int(from_cache),
                _now(),
            ),
        )
        conn.commit()
        logger.debug(
            "insert_result: id=%s video_id=%s query_id=%s from_cache=%s",
            result_id, video_id, query_id, from_cache,
        )
    except sqlite3.Error as exc:
        logger.exception("insert_result failed: %s", exc)
        raise
    finally:
        conn.close()
def get_result(result_id: str) -> Optional[dict]:
    """
    Fetch a single result row joined with its query (for the question text).
    Returns a plain dict with keys:
        result_id, video_id, question, answer,
        task_routing (dict), from_cache (bool), created_at (str)
    Returns None if no row matches result_id.
    """
    sql = """
        SELECT
            r.id          AS result_id,
            r.video_id,
            q.question,
            r.answer,
            r.routing_json,
            r.from_cache,
            r.created_at
        FROM results r
        JOIN queries q ON q.id = r.query_id
        WHERE r.id = ?
    """
    conn = get_connection()
    try:
        row = conn.execute(sql, (result_id,)).fetchone()
        if row is None:
            logger.debug("get_result: not found — id=%s", result_id)
            return None
        data = _row_to_dict(row)
        # Deserialize routing JSON → dict
        data["task_routing"] = json.loads(data.pop("routing_json"))
        data["from_cache"] = bool(data["from_cache"])
        return data
    except sqlite3.Error as exc:
        logger.exception("get_result failed: %s", exc)
        raise
    finally:
        conn.close()
def get_results_for_video(video_id: str, limit: int = 50) -> list[dict]:
    """
    Fetch up to `limit` results for a given video, newest first.
    Useful for building a query history view in the frontend.
    Returns a list of dicts (same schema as get_result).
    """
    sql = """
        SELECT
            r.id          AS result_id,
            r.video_id,
            q.question,
            r.answer,
            r.routing_json,
            r.from_cache,
            r.created_at
        FROM results r
        JOIN queries q ON q.id = r.query_id
        WHERE r.video_id = ?
        ORDER BY r.created_at DESC
        LIMIT ?
    """
    conn = get_connection()
    try:
        rows = conn.execute(sql, (video_id, limit)).fetchall()
        results = []
        for row in rows:
            data = _row_to_dict(row)
            data["task_routing"] = json.loads(data.pop("routing_json"))
            data["from_cache"] = bool(data["from_cache"])
            results.append(data)
        logger.debug(
            "get_results_for_video: video_id=%s returned %d rows", video_id, len(results)
        )
        return results
    except sqlite3.Error as exc:
        logger.exception("get_results_for_video failed: %s", exc)
        raise
    finally:
        conn.close()
