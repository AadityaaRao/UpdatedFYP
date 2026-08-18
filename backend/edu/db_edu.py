"""
backend/edu/db_edu.py
────────────────────────────────────────────────────────────
SQLite CRUD layer for Edu-VQAGuider v2 tables.

Tables managed (defined in schema.sql):
    video_chunks  — per-chunk metadata, transcript, frame paths
    edu_results   — v2 QA results with route + evidence

Also extends the v1 `videos` table with additional columns
for v2 processing status.

Design follows the same patterns as backend/database/db.py:
    • No ORM — plain sqlite3
    • Each function opens/closes its own connection
    • Parameterized queries only
    • Returns plain dicts
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from backend.database.db import get_connection, _now, _row_to_dict
from backend.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# Videos table extensions
# ══════════════════════════════════════════════════════════════

def ensure_video_v2_columns() -> None:
    """
    Add v2-specific columns to the videos table if they don't exist.
    Safe to call multiple times.

    Added columns:
        duration_sec       REAL
        status             TEXT (pending/transcribing/indexing/ready/error)
        processing_error   TEXT
    """
    conn = get_connection()
    try:
        # SQLite doesn't have IF NOT EXISTS for ALTER TABLE,
        # so we catch the "duplicate column" error silently.
        for col_sql in [
            "ALTER TABLE videos ADD COLUMN duration_sec REAL DEFAULT 0.0",
            "ALTER TABLE videos ADD COLUMN status TEXT DEFAULT 'pending'",
            "ALTER TABLE videos ADD COLUMN processing_error TEXT",
        ]:
            try:
                conn.execute(col_sql)
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower():
                    pass  # Column already exists — safe to ignore
                else:
                    raise
        conn.commit()
        logger.debug("Video v2 columns ensured")
    finally:
        conn.close()


def update_video_status(
    video_id: str,
    status: str,
    duration_sec: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """
    Update the processing status of a video.

    Args:
        video_id:     UUID of the video
        status:       One of: pending, transcribing, indexing, ready, error
        duration_sec: Video duration (set once during processing)
        error:        Error message (set when status='error')
    """
    conn = get_connection()
    try:
        if duration_sec is not None:
            conn.execute(
                "UPDATE videos SET status = ?, duration_sec = ?, processing_error = ? WHERE id = ?",
                (status, duration_sec, error, video_id),
            )
        else:
            conn.execute(
                "UPDATE videos SET status = ?, processing_error = ? WHERE id = ?",
                (status, error, video_id),
            )
        conn.commit()
        logger.debug("Video %s status -> %s", video_id, status)
    except sqlite3.Error as exc:
        logger.exception("update_video_status failed: %s", exc)
        raise
    finally:
        conn.close()


def get_video_status(video_id: str) -> Optional[dict]:
    """
    Get the current status of a video.

    Returns:
        Dict with keys: id, path, original_filename, duration_sec,
                        status, processing_error, created_at
        None if video not found.
    """
    sql = """
        SELECT id, path, original_filename, duration_sec,
               status, processing_error, created_at
        FROM videos WHERE id = ?
    """
    conn = get_connection()
    try:
        row = conn.execute(sql, (video_id,)).fetchone()
        if row is None:
            return None
        return _row_to_dict(row)
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════
# Video Chunks table
# ══════════════════════════════════════════════════════════════

def insert_chunks(chunks_data: list[dict]) -> int:
    """
    Batch insert chunk metadata.

    Args:
        chunks_data: List of dicts from ChunkMeta.to_db_row()

    Returns:
        Number of rows inserted
    """
    sql = """
        INSERT OR REPLACE INTO video_chunks
            (id, video_id, start_time, end_time, transcript_text,
             visual_summary, frame_paths_json, embedding_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    now = _now()
    conn = get_connection()
    try:
        rows = [
            (
                d["id"], d["video_id"], d["start_time"], d["end_time"],
                d.get("transcript_text", ""),
                d.get("visual_summary"),
                d.get("frame_paths_json", "[]"),
                d.get("embedding_path"),
                now,
            )
            for d in chunks_data
        ]
        conn.executemany(sql, rows)
        conn.commit()
        logger.debug("Inserted %d chunks for video %s", len(rows), chunks_data[0]["video_id"] if chunks_data else "?")
        return len(rows)
    except sqlite3.Error as exc:
        logger.exception("insert_chunks failed: %s", exc)
        raise
    finally:
        conn.close()


def get_chunks_for_video(video_id: str) -> list[dict]:
    """
    Fetch all chunks for a video, ordered by start_time.

    Returns:
        List of dicts with all video_chunks columns.
        frame_paths_json is parsed into a Python list.
    """
    sql = """
        SELECT * FROM video_chunks
        WHERE video_id = ?
        ORDER BY start_time ASC
    """
    conn = get_connection()
    try:
        rows = conn.execute(sql, (video_id,)).fetchall()
        results = []
        for row in rows:
            data = _row_to_dict(row)
            # Parse JSON fields
            data["frame_paths"] = json.loads(data.get("frame_paths_json", "[]"))
            results.append(data)
        return results
    finally:
        conn.close()


def update_chunk_transcript(chunk_id: str, transcript_text: str) -> None:
    """Update the transcript text for a single chunk."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE video_chunks SET transcript_text = ? WHERE id = ?",
            (transcript_text, chunk_id),
        )
        conn.commit()
    finally:
        conn.close()


def update_chunk_visual_summary(chunk_id: str, visual_summary: str) -> None:
    """Update the visual summary for a single chunk (from VLM frame captioning)."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE video_chunks SET visual_summary = ? WHERE id = ?",
            (visual_summary, chunk_id),
        )
        conn.commit()
    finally:
        conn.close()


def count_chunks(video_id: str) -> int:
    """Count the number of chunks for a video."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM video_chunks WHERE video_id = ?",
            (video_id,),
        ).fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════
# Edu Results table
# ══════════════════════════════════════════════════════════════

def insert_edu_result(
    result_id: str,
    video_id: str,
    question: str,
    direct_answer: str,
    detailed_answer: str,
    route_json: str,
    evidence_json: str,
    planner_source: str,
) -> None:
    """
    Insert a v2 QA result.

    Args:
        result_id:       UUID for this result
        video_id:        FK → videos.id
        question:        Original question text
        direct_answer:   Concise 1-2 sentence answer
        detailed_answer: Full explanatory answer
        route_json:      JSON string of RouteInfo
        evidence_json:   JSON string of evidence chunks
        planner_source:  "learned" or "fallback"
    """
    sql = """
        INSERT INTO edu_results
            (id, video_id, question, direct_answer, answer,
             route_json, evidence_json, planner_source, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    conn = get_connection()
    try:
        conn.execute(sql, (
            result_id, video_id, question, direct_answer, detailed_answer,
            route_json, evidence_json, planner_source, _now(),
        ))
        conn.commit()
        logger.debug("Inserted edu_result %s", result_id)
    except sqlite3.Error as exc:
        logger.exception("insert_edu_result failed: %s", exc)
        raise
    finally:
        conn.close()


def get_edu_result(result_id: str) -> Optional[dict]:
    """
    Fetch a single edu result by ID.

    Returns:
        Dict with parsed route_json and evidence_json, or None.
    """
    sql = "SELECT * FROM edu_results WHERE id = ?"
    conn = get_connection()
    try:
        row = conn.execute(sql, (result_id,)).fetchone()
        if row is None:
            return None
        data = _row_to_dict(row)
        data["route"] = json.loads(data.pop("route_json"))
        data["evidence"] = json.loads(data.pop("evidence_json"))
        return data
    finally:
        conn.close()


def get_edu_results_for_video(
    video_id: str, limit: int = 50
) -> list[dict]:
    """
    Fetch recent edu results for a video (question history).

    Returns:
        List of dicts, newest first.
    """
    sql = """
        SELECT * FROM edu_results
        WHERE video_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """
    conn = get_connection()
    try:
        rows = conn.execute(sql, (video_id, limit)).fetchall()
        results = []
        for row in rows:
            data = _row_to_dict(row)
            data["route"] = json.loads(data.pop("route_json"))
            data["evidence"] = json.loads(data.pop("evidence_json"))
            results.append(data)
        return results
    finally:
        conn.close()
