"""
backend/edu/schemas.py
────────────────────────────────────────────────────────────
Pydantic request/response schemas for Edu-VQAGuider v2 API.

Endpoints served:
    POST /api/v2/videos              → EduUploadResponse
    POST /api/v2/videos/{id}/transcript  → TranscriptResponse
    GET  /api/v2/videos/{id}/status  → VideoStatusResponse
    POST /api/v2/videos/{id}/ask     → EduAskResponse
    GET  /api/v2/results/{id}        → EduResultResponse
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────

class EduRoute(str, Enum):
    """Question-intent categories produced by the Edu-VQAGuider planner."""
    concept   = "concept"
    procedure = "procedure"
    temporal  = "temporal"
    visual    = "visual"
    summary   = "summary"


class PlannerSource(str, Enum):
    """How the route was determined."""
    learned  = "learned"
    fallback = "fallback"


class VideoStatus(str, Enum):
    """Processing lifecycle of an uploaded educational video."""
    pending      = "pending"       # uploaded, not yet processed
    transcribing = "transcribing"  # Whisper / manual transcript in progress
    indexing     = "indexing"      # embeddings + frame extraction
    ready        = "ready"         # fully processed, queryable
    error        = "error"         # processing failed


# ── Shared sub-schemas ────────────────────────────────────────

class ChunkInfo(BaseModel):
    """Metadata for a single video chunk returned as evidence."""
    chunk_id: str
    start_time: float = Field(..., description="Chunk start in seconds")
    end_time: float   = Field(..., description="Chunk end in seconds")
    transcript_text: str
    selected_frame_path: Optional[str] = Field(
        None, description="Path to the CLIP-selected best frame for this chunk"
    )
    relevance_score: float = Field(
        0.0, description="Cosine similarity score for this chunk"
    )


class RouteInfo(BaseModel):
    """Planner output: which route was selected and how."""
    route: EduRoute
    confidence: float = Field(..., ge=0.0, le=1.0)
    planner_source: PlannerSource
    all_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Scores for all 5 routes from the planner"
    )


# ══════════════════════════════════════════════════════════════
# POST /api/v2/videos
# ══════════════════════════════════════════════════════════════

class EduUploadResponse(BaseModel):
    """Returned after a successful educational video upload."""
    video_id: str
    original_filename: str
    duration_sec: float
    num_chunks: int
    status: VideoStatus


# ══════════════════════════════════════════════════════════════
# POST /api/v2/videos/{id}/transcript
# ══════════════════════════════════════════════════════════════

class ManualTranscriptRequest(BaseModel):
    """Request body for manual transcript upload."""
    transcript_text: str = Field(
        ..., min_length=10,
        description="Plain text or SRT-format transcript"
    )
    format: str = Field(
        "plain", description="'plain' or 'srt'"
    )

    @field_validator("transcript_text")
    @classmethod
    def transcript_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Transcript must not be blank")
        return v


class TranscriptResponse(BaseModel):
    """Returned after transcript is processed and aligned to chunks."""
    video_id: str
    num_chunks_with_text: int
    status: VideoStatus


# ══════════════════════════════════════════════════════════════
# GET /api/v2/videos/{id}/status
# ══════════════════════════════════════════════════════════════

class VideoStatusResponse(BaseModel):
    """Current processing status of a video."""
    video_id: str
    status: VideoStatus
    duration_sec: float
    num_chunks: int
    has_transcript: bool
    has_embeddings: bool
    processing_error: Optional[str] = None


# ══════════════════════════════════════════════════════════════
# POST /api/v2/videos/{id}/ask
# ══════════════════════════════════════════════════════════════

class EduAskRequest(BaseModel):
    """Request body for asking a question about a processed educational video."""
    question: str = Field(
        ..., min_length=3, max_length=500,
        description="Natural language question about the video content"
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question must not be blank or whitespace-only")
        return v.strip()


class EduAskResponse(BaseModel):
    """Full response for an educational VQA query."""
    result_id: str
    video_id: str
    question: str
    direct_answer: str = Field(
        ..., description="Concise 1-2 sentence answer"
    )
    detailed_answer: str = Field(
        ..., description="Full explanatory answer with evidence"
    )
    route: RouteInfo
    evidence_chunks: list[ChunkInfo] = Field(
        default_factory=list,
        description="Retrieved chunks used as evidence"
    )
    confidence_level: str = Field(
        "high",
        description="Answer confidence level: 'high', 'medium', or 'low'"
    )


# ══════════════════════════════════════════════════════════════
# GET /api/v2/results/{id}
# ══════════════════════════════════════════════════════════════

class EduResultResponse(BaseModel):
    """Stored result fetched by ID."""
    result_id: str
    video_id: str
    question: str
    direct_answer: str
    detailed_answer: str
    route: RouteInfo
    evidence_chunks: list[ChunkInfo]
    created_at: str


# ══════════════════════════════════════════════════════════════
# GET /api/v2/videos/{id}/history
# ══════════════════════════════════════════════════════════════

class HistoryItem(BaseModel):
    """Summary of a past query for the question history list."""
    result_id: str
    question: str
    direct_answer: str
    route: EduRoute
    created_at: str


class HistoryResponse(BaseModel):
    """List of past queries for a video."""
    video_id: str
    items: list[HistoryItem]


# ══════════════════════════════════════════════════════════════
# Evaluation / Comparison Baseline Schemas
# ══════════════════════════════════════════════════════════════

class BaselineAskRequest(BaseModel):
    """Request body for baseline query."""
    question: str


class BaselineAskResponse(BaseModel):
    """Response body for baseline query."""
    answer: str


class DirectGenerateRequest(BaseModel):
    """Request body for direct generation prompt."""
    prompt: str


class DirectGenerateResponse(BaseModel):
    """Response body for direct generation prompt."""
    answer: str

