"""
backend/routes/video.py
────────────────────────────────────────────────────────────
POST /api/v1/upload_video
Responsibilities (this file only):
  ✓ Receive multipart file upload
  ✓ Validate file extension
  ✓ Generate a unique video_id (UUID4)
  ✓ Save file to /uploads/{video_id}{ext}
  ✓ Persist record to SQLite via db.insert_video()
  ✓ Return UploadVideoResponse
NOT here:
  ✗ Model inference   → vqa_service.py
  ✗ Cache logic       → cache_service.py
"""
from __future__ import annotations
import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request, UploadFile, status
from backend.database import db
from backend.models.schemas import UploadVideoResponse
from backend.utils.logger import get_logger
from config import MAX_UPLOAD_SIZE_MB, UPLOADS_DIR
logger = get_logger(__name__)
router = APIRouter(tags=["Video"])
# Allowed video MIME types and extensions
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".avi", ".webm", ".mov", ".mkv"})
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024
# ── Helpers ───────────────────────────────────────────────────
def _validate_extension(filename: str) -> str:
    """
    Return the lowercase extension or raise HTTP 400 for unsupported types.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )
    return ext
def _save_upload(upload: UploadFile, dest: Path) -> int:
    """
    Stream-copy an UploadFile to disk, return bytes written.
    Uses shutil.copyfileobj to avoid loading the full file into memory.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    with dest.open("wb") as out_file:
        shutil.copyfileobj(upload.file, out_file)
        bytes_written = out_file.tell()
    return bytes_written
def resolve_video_path(video_id: str) -> Path:
    """
    Find the actual file for a given video_id.
    Saved as uploads/{video_id}{ext} so we glob for the first match.
    Returns the Path if found, raises HTTP 404 otherwise.
    Exported so query.py can reuse it without duplication.
    """
    matches = list(UPLOADS_DIR.glob(f"{video_id}.*"))
    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No video found for video_id='{video_id}'. Upload the video first.",
        )
    return matches[0]
# ── Route ─────────────────────────────────────────────────────
@router.post(
    "/upload_video",
    response_model=UploadVideoResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a video file",
    response_description="Unique video_id for subsequent /ask_question calls",
)
async def upload_video(request: Request, file: UploadFile) -> UploadVideoResponse:
    """
    Accept a video upload and persist it to the uploads directory.
    - Validates the file extension (mp4 / avi / webm / mov / mkv).
    - Generates a UUIDv4 as the `video_id` — stable, collision-free.
    - Saves the file as `uploads/{video_id}{ext}` for deterministic lookup.
    - Does **not** run inference; inference happens via `/ask_question`.
    Returns the `video_id` which must be included in the subsequent
    `/ask_question` request.
    """
    if file.filename is None or file.filename.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in the upload.",
        )
    # 1. Validate extension
    ext = _validate_extension(file.filename)
    # 2. Generate unique ID
    video_id = str(uuid.uuid4())
    stored_filename = f"{video_id}{ext}"
    dest_path = UPLOADS_DIR / stored_filename
    logger.info("Receiving upload: '%s' → %s", file.filename, dest_path.name)
    # 3. Save to disk
    try:
        size_bytes = _save_upload(file, dest_path)
    except Exception as exc:
        logger.exception("Failed to save uploaded file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded video. Please try again.",
        ) from exc
    finally:
        await file.close()
    # 4. Basic size sanity check (after write, since we don't read Content-Length)
    if size_bytes == 0:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file appears to be empty.",
        )
    if size_bytes > MAX_UPLOAD_BYTES:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({size_bytes / 1e6:.1f} MB). Max: {MAX_UPLOAD_SIZE_MB} MB.",
        )
    logger.info(
        "Upload saved: video_id=%s | size=%.2f MB",
        video_id, size_bytes / 1e6,
    )
    # 5. Persist to database
    # INSERT OR IGNORE means re-uploading the same video_id is a safe no-op.
    # A DB failure here does NOT abort the response — the file is on disk
    # and can be re-registered on the next ask_question call.
    try:
        db.insert_video(
            video_id=video_id,
            path=str(dest_path.resolve()),
            original_filename=file.filename,
        )
    except Exception as exc:
        logger.error("DB insert_video failed (non-fatal): %s", exc)
    return UploadVideoResponse(
        video_id=video_id,
        original_filename=file.filename,
        stored_filename=stored_filename,
        size_bytes=size_bytes,
    )
