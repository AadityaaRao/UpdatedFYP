"""
frontend/api_client.py
────────────────────────────────────────────────────────────
Thin HTTP client for the VQA Guider backend API.
Responsibilities:
  • All requests.post / requests.get calls live here — none in app.py
  • Returns typed Python dicts on success
  • Raises descriptive APIError exceptions on failure
  • Configurable base URL (local dev vs deployed)
app.py imports only:
    upload_video()   → {"video_id": str, ...}
    ask_question()   → {"answer": str, "task_routing": {...}, "result_id": str}
    get_result()     → same as ask_question response
    health_check()   → {"status": str, "model_ready": bool, ...}
"""
from __future__ import annotations
import requests
# ── Custom exception ──────────────────────────────────────────
class APIError(Exception):
    """
    Raised when the backend returns an error or is unreachable.
    Carries a user-friendly message that can be shown directly in the UI.
    """
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
# ── Shared request helpers ────────────────────────────────────
_TIMEOUT_UPLOAD  = 120   # seconds — large file uploads
_TIMEOUT_INFER   = 300   # seconds — model inference can be slow
_TIMEOUT_FETCH   = 10    # seconds — lightweight DB fetch
def _handle_response(resp: requests.Response) -> dict:
    """
    Raise APIError for non-2xx responses; otherwise return parsed JSON.
    """
    if resp.status_code == 503:
        raise APIError(
            "Backend model is still initializing. Please wait a moment and try again.",
            status_code=503,
        )
    if resp.status_code == 404:
        detail = _extract_detail(resp)
        raise APIError(f"Not found: {detail}", status_code=404)
    if resp.status_code == 413:
        raise APIError("Video file is too large. Please upload a smaller file.", status_code=413)
    if resp.status_code == 422:
        detail = _extract_detail(resp)
        raise APIError(f"Unprocessable video: {detail}", status_code=422)
    if not resp.ok:
        detail = _extract_detail(resp)
        raise APIError(f"Server error ({resp.status_code}): {detail}", status_code=resp.status_code)
    try:
        return resp.json()
    except ValueError as exc:
        raise APIError("Backend returned an unexpected response format.") from exc
def _extract_detail(resp: requests.Response) -> str:
    """Pull 'detail' from FastAPI error JSON, or fall back to raw text."""
    try:
        return resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text[:200]
# ── Public API functions ──────────────────────────────────────
def upload_video(base_url: str, video_bytes: bytes, filename: str) -> dict:
    """
    POST /api/v1/upload_video
    Args:
        base_url:    Backend base URL, e.g. "http://localhost:8000"
        video_bytes: Raw file bytes from st.file_uploader
        filename:    Original filename (used for extension detection)
    Returns:
        {"video_id": str, "original_filename": str, "size_bytes": int, ...}
    Raises:
        APIError on any HTTP error or connectivity problem.
    """
    url = f"{base_url.rstrip('/')}/api/v1/upload_video"
    try:
        resp = requests.post(
            url,
            files={"file": (filename, video_bytes, "video/mp4")},
            timeout=_TIMEOUT_UPLOAD,
        )
    except requests.exceptions.ConnectionError:
        raise APIError(
            f"Cannot connect to backend at {base_url}. "
            "Is the server running?"
        )
    except requests.exceptions.Timeout:
        raise APIError("Upload timed out. Try a shorter or smaller video.")
    return _handle_response(resp)
def ask_question(base_url: str, video_id: str, question: str) -> dict:
    """
    POST /api/v1/ask_question
    Args:
        base_url:  Backend base URL
        video_id:  UUID returned by upload_video()
        question:  Natural language question string
    Returns:
        {
            "result_id": str,
            "video_id":  str,
            "question":  str,
            "answer":    str,
            "task_routing": {"action": float, "tracking": float, "scene": float},
            "from_cache": bool,
        }
    Raises:
        APIError on any HTTP error or connectivity problem.
    """
    url = f"{base_url.rstrip('/')}/api/v1/ask_question"
    try:
        resp = requests.post(
            url,
            json={"video_id": video_id, "question": question},
            timeout=_TIMEOUT_INFER,
        )
    except requests.exceptions.ConnectionError:
        raise APIError(
            f"Cannot connect to backend at {base_url}. "
            "Is the server running?"
        )
    except requests.exceptions.Timeout:
        raise APIError(
            "Inference timed out. The model may still be loading — "
            "please try again in a moment."
        )
    return _handle_response(resp)
def get_result(base_url: str, result_id: str) -> dict:
    """
    GET /api/v1/result/{result_id}
    Args:
        base_url:  Backend base URL
        result_id: UUID returned by ask_question()
    Returns:
        Same schema as ask_question() response, with added "created_at".
    Raises:
        APIError on any HTTP error or connectivity problem.
    """
    url = f"{base_url.rstrip('/')}/api/v1/result/{result_id}"
    try:
        resp = requests.get(url, timeout=_TIMEOUT_FETCH)
    except requests.exceptions.ConnectionError:
        raise APIError(f"Cannot connect to backend at {base_url}.")
    except requests.exceptions.Timeout:
        raise APIError("Request timed out.")
    return _handle_response(resp)
def health_check(base_url: str) -> dict:
    """
    GET /health
    Returns the server + model status dict, or raises APIError.
    Used by the sidebar to show live server status.
    """
    url = f"{base_url.rstrip('/')}/health"
    try:
        resp = requests.get(url, timeout=5)
        return _handle_response(resp)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        raise APIError(f"Backend offline or unreachable at {base_url}.")