"""
frontend/edu_api_client.py
────────────────────────────────────────────────────────────
HTTP client for Edu-VQAGuider v2 API.

All v2 API calls live here — none in the Streamlit app.

API endpoints:
    POST /api/v2/videos                    -> upload_edu_video()
    POST /api/v2/videos/{id}/transcript    -> upload_transcript()
    POST /api/v2/videos/{id}/transcribe    -> auto_transcribe()
    GET  /api/v2/videos/{id}/status        -> get_video_status()
    POST /api/v2/videos/{id}/ask           -> ask_edu_question()
    GET  /api/v2/results/{id}              -> get_edu_result()
    GET  /api/v2/videos/{id}/history       -> get_video_history()
    GET  /health                           -> health_check()
"""
from __future__ import annotations
import requests


class APIError(Exception):
    """Raised when the backend returns an error or is unreachable."""
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


_TIMEOUT_UPLOAD = 120
_TIMEOUT_TRANSCRIBE = 600  # Whisper can be slow on long videos
_TIMEOUT_INFER = 300
_TIMEOUT_FETCH = 10


def _handle(resp: requests.Response) -> dict:
    """Parse response or raise APIError."""
    if resp.status_code == 503:
        raise APIError("Backend is still initializing.", 503)
    if resp.status_code == 404:
        raise APIError(f"Not found: {_detail(resp)}", 404)
    if resp.status_code == 413:
        raise APIError("File too large.", 413)
    if resp.status_code == 422:
        raise APIError(f"Unprocessable: {_detail(resp)}", 422)
    if not resp.ok:
        raise APIError(f"Error ({resp.status_code}): {_detail(resp)}", resp.status_code)
    try:
        return resp.json()
    except ValueError:
        raise APIError("Unexpected response format.")


def _detail(resp: requests.Response) -> str:
    try:
        return resp.json().get("detail", resp.text[:200])
    except ValueError:
        return resp.text[:200]


# ══════════════════════════════════════════════════════════════
# Video Management
# ══════════════════════════════════════════════════════════════

def upload_edu_video(base_url: str, video_bytes: bytes, filename: str) -> dict:
    """POST /api/v2/videos — upload a video for processing."""
    url = f"{base_url.rstrip('/')}/api/v2/videos"
    try:
        resp = requests.post(
            url,
            files={"file": (filename, video_bytes, "video/mp4")},
            timeout=_TIMEOUT_UPLOAD,
        )
    except requests.exceptions.ConnectionError:
        raise APIError(f"Cannot connect to {base_url}. Is the server running?")
    except requests.exceptions.Timeout:
        raise APIError("Upload timed out.")
    return _handle(resp)


def upload_transcript(base_url: str, video_id: str, transcript_text: str) -> dict:
    """POST /api/v2/videos/{id}/transcript — provide manual transcript."""
    url = f"{base_url.rstrip('/')}/api/v2/videos/{video_id}/transcript"
    try:
        resp = requests.post(
            url,
            json={"transcript_text": transcript_text, "format": "plain"},
            timeout=_TIMEOUT_TRANSCRIBE,
        )
    except requests.exceptions.ConnectionError:
        raise APIError(f"Cannot connect to {base_url}.")
    except requests.exceptions.Timeout:
        raise APIError("Transcript processing timed out.")
    return _handle(resp)


def auto_transcribe(base_url: str, video_id: str) -> dict:
    """POST /api/v2/videos/{id}/transcribe — auto-transcribe with Whisper."""
    url = f"{base_url.rstrip('/')}/api/v2/videos/{video_id}/transcribe"
    try:
        resp = requests.post(url, timeout=_TIMEOUT_TRANSCRIBE)
    except requests.exceptions.ConnectionError:
        raise APIError(f"Cannot connect to {base_url}.")
    except requests.exceptions.Timeout:
        raise APIError("Transcription timed out. The video may be too long.")
    return _handle(resp)


def get_video_status(base_url: str, video_id: str) -> dict:
    """GET /api/v2/videos/{id}/status — check processing status."""
    url = f"{base_url.rstrip('/')}/api/v2/videos/{video_id}/status"
    try:
        resp = requests.get(url, timeout=_TIMEOUT_FETCH)
    except requests.exceptions.ConnectionError:
        raise APIError(f"Cannot connect to {base_url}.")
    return _handle(resp)


# ══════════════════════════════════════════════════════════════
# Question Answering
# ══════════════════════════════════════════════════════════════

def ask_edu_question(base_url: str, video_id: str, question: str) -> dict:
    """POST /api/v2/videos/{id}/ask — ask a question."""
    url = f"{base_url.rstrip('/')}/api/v2/videos/{video_id}/ask"
    try:
        resp = requests.post(
            url,
            json={"question": question},
            timeout=_TIMEOUT_INFER,
        )
    except requests.exceptions.ConnectionError:
        raise APIError(f"Cannot connect to {base_url}.")
    except requests.exceptions.Timeout:
        raise APIError("Answer generation timed out.")
    return _handle(resp)


def get_edu_result(base_url: str, result_id: str) -> dict:
    """GET /api/v2/results/{id} — fetch stored result."""
    url = f"{base_url.rstrip('/')}/api/v2/results/{result_id}"
    try:
        resp = requests.get(url, timeout=_TIMEOUT_FETCH)
    except requests.exceptions.ConnectionError:
        raise APIError(f"Cannot connect to {base_url}.")
    return _handle(resp)


def get_video_history(base_url: str, video_id: str) -> dict:
    """GET /api/v2/videos/{id}/history — question history."""
    url = f"{base_url.rstrip('/')}/api/v2/videos/{video_id}/history"
    try:
        resp = requests.get(url, timeout=_TIMEOUT_FETCH)
    except requests.exceptions.ConnectionError:
        raise APIError(f"Cannot connect to {base_url}.")
    return _handle(resp)


# ══════════════════════════════════════════════════════════════
# System
# ══════════════════════════════════════════════════════════════

def health_check(base_url: str) -> dict:
    """GET /health — server status."""
    url = f"{base_url.rstrip('/')}/health"
    try:
        resp = requests.get(url, timeout=5)
        return _handle(resp)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        raise APIError(f"Backend offline at {base_url}.")
