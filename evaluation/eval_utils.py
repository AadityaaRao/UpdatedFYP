from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests


EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
VIDEOS_DIR = EVAL_DIR / "videos"
GENERATED_DIR = EVAL_DIR / "generated"
REPORT_JSON = EVAL_DIR / "report.json"
REPORT_MD = EVAL_DIR / "report.md"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".webm", ".mov", ".mkv"}
QUESTION_CATEGORIES = [
    ("factual", 3),
    ("reasoning", 2),
    ("temporal", 1),
    ("visual/multimodal", 2),
    ("unanswerable", 2),
]


class EvaluationError(RuntimeError):
    pass


def ensure_dirs() -> None:
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def discover_videos(video_dir: Path = VIDEOS_DIR) -> list[Path]:
    if not video_dir.exists():
        return []
    return sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def video_key(video_path_or_name: str | Path) -> str:
    return Path(video_path_or_name).stem


def video_generated_dir(video_path_or_name: str | Path) -> Path:
    return GENERATED_DIR / video_key(video_path_or_name)


def health_check(base_url: str) -> dict:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/health", timeout=8)
    except requests.RequestException as exc:
        raise EvaluationError(
            f"Backend is not reachable at {base_url}. Start it with: "
            "uvicorn backend.main:app --host 0.0.0.0 --port 8000"
        ) from exc
    return handle_response(resp)


def handle_response(resp: requests.Response) -> dict:
    if resp.ok:
        try:
            return resp.json()
        except ValueError as exc:
            raise EvaluationError("Backend returned non-JSON response.") from exc

    try:
        detail = resp.json().get("detail", resp.text[:300])
    except ValueError:
        detail = resp.text[:300]
    raise EvaluationError(f"Backend error {resp.status_code}: {detail}")


def upload_video(base_url: str, video_path: Path) -> dict:
    url = f"{base_url.rstrip('/')}/api/v2/videos"
    with video_path.open("rb") as f:
        try:
            resp = requests.post(
                url,
                files={"file": (video_path.name, f, "video/mp4")},
                timeout=180,
            )
        except requests.RequestException as exc:
            raise EvaluationError(f"Video upload failed for {video_path.name}: {exc}") from exc
    return handle_response(resp)


def auto_transcribe(base_url: str, video_id: str) -> dict:
    url = f"{base_url.rstrip('/')}/api/v2/videos/{video_id}/transcribe"
    try:
        resp = requests.post(url, timeout=900)
    except requests.RequestException as exc:
        raise EvaluationError(f"Auto-transcription failed for video_id={video_id}: {exc}") from exc
    return handle_response(resp)


def ask_question(base_url: str, video_id: str, question: str) -> tuple[dict | None, float, str | None]:
    url = f"{base_url.rstrip('/')}/api/v2/videos/{video_id}/ask"
    start = time.perf_counter()
    try:
        resp = requests.post(url, json={"question": question}, timeout=360)
        latency = time.perf_counter() - start
        return handle_response(resp), latency, None
    except Exception as exc:
        latency = time.perf_counter() - start
        return None, latency, str(exc)


def direct_generate(base_url: str, prompt: str) -> str:
    url = f"{base_url.rstrip('/')}/api/v2/direct_generate"
    try:
        resp = requests.post(url, json={"prompt": prompt}, timeout=360)
    except requests.RequestException as exc:
        raise EvaluationError(f"Direct generation endpoint failed: {exc}") from exc
    data = handle_response(resp)
    answer = data.get("answer", "")
    if not answer or "Qwen is not loaded" in answer or "Generation Placeholder" in answer:
        raise EvaluationError(
            "The backend is reachable, but the LLM generation endpoint is not loaded. "
            "Run this on the college machine with the v2 models available."
        )
    return answer


def extract_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1).strip())

    start_positions = [pos for pos in (text.find("["), text.find("{")) if pos >= 0]
    if not start_positions:
        raise ValueError("No JSON object or array found in model output.")
    start = min(start_positions)
    end = max(text.rfind("]"), text.rfind("}"))
    if end <= start:
        raise ValueError("Incomplete JSON in model output.")
    return json.loads(text[start:end + 1])

