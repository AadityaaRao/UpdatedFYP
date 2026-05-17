"""
backend/models/schemas.py
────────────────────────────────────────────────────────────
Pydantic request and response schemas for all API endpoints.
Keeping schemas here (not inside route files) means:
  • Routes stay thin — just validation + delegation
  • Schemas are importable for tests without starting the server
  • A single source of truth for the API contract
"""
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
# ══════════════════════════════════════════════════════════════
# POST /upload_video
# ══════════════════════════════════════════════════════════════
class UploadVideoResponse(BaseModel):
    """Returned after a successful video upload."""
    video_id: str = Field(..., description="Unique identifier for the uploaded video")
    original_filename: str = Field(..., description="Original name of the uploaded file")
    stored_filename: str = Field(..., description="Filename as saved on disk (video_id + ext)")
    size_bytes: int = Field(..., description="File size in bytes")
# ══════════════════════════════════════════════════════════════
# POST /ask_question
# ══════════════════════════════════════════════════════════════
class AskQuestionRequest(BaseModel):
    """Request body for asking a question about an uploaded video."""
    video_id: str = Field(
        ...,
        min_length=1,
        description="video_id returned by POST /upload_video",
    )
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural language question about the video",
    )
    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank or whitespace-only")
        return v.strip()
    @field_validator("video_id")
    @classmethod
    def video_id_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("video_id must not be blank")
        return v.strip()
class TaskRoutingSchema(BaseModel):
    """Softmax-normalized routing probabilities across three task heads."""
    action: float = Field(..., ge=0.0, le=1.0, description="Action recognition weight")
    tracking: float = Field(..., ge=0.0, le=1.0, description="Object tracking weight")
    scene: float = Field(..., ge=0.0, le=1.0, description="Scene understanding weight")
class AskQuestionResponse(BaseModel):
    """Response returned after a successful VQA inference."""
    result_id: str = Field(..., description="Unique ID for this result (used with GET /result/{id})")
    video_id: str = Field(..., description="Video this result belongs to")
    question: str = Field(..., description="The question that was asked")
    answer: str = Field(..., description="AI-generated answer")
    task_routing: TaskRoutingSchema = Field(..., description="Task routing probabilities")
    from_cache: bool = Field(False, description="True if the result was served from cache")
# ══════════════════════════════════════════════════════════════
# GET /result/{result_id}
# ══════════════════════════════════════════════════════════════
class ResultResponse(BaseModel):
    """Stored result fetched by ID."""
    result_id: str
    video_id: str
    question: str
    answer: str
    task_routing: TaskRoutingSchema
    created_at: str = Field(..., description="ISO-8601 timestamp of when the result was created")
# ══════════════════════════════════════════════════════════════
# Shared error schema
# ══════════════════════════════════════════════════════════════
class ErrorResponse(BaseModel):
    """Standard error envelope used across all endpoints."""
    detail: str
    error_code: str = "INTERNAL_ERROR"


class EduVideoCreateResponse(BaseModel):
    video_id: str
    status: str
    message: str


class EduVideoStatusResponse(BaseModel):
    video_id: str
    status: str
    progress: float = Field(..., ge=0.0, le=1.0)
    chunks_total: int = 0
    chunks_processed: int = 0
    duration_sec: float | None = None
    error: str | None = None


class EduAskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def edu_question_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class EduRouteSchema(BaseModel):
    primary_intent: str
    evidence_types: list[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    planner_source: str = "fallback"


class EduEvidenceSchema(BaseModel):
    chunk_id: str
    start_time: float
    end_time: float
    transcript_excerpt: str
    visual_summary: str | None = None
    frame_paths: list[str] = Field(default_factory=list)
    score: float | None = None


class EduAskResponse(BaseModel):
    result_id: str
    video_id: str
    question: str
    direct_answer: str
    answer: str
    route: EduRouteSchema
    evidence: list[EduEvidenceSchema]


class EduResultResponse(EduAskResponse):
    created_at: str
