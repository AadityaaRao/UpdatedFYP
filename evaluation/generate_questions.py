from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval_utils import (
    GENERATED_DIR,
    PROJECT_ROOT,
    QUESTION_CATEGORIES,
    VIDEOS_DIR,
    EvaluationError,
    auto_transcribe,
    direct_generate,
    discover_videos,
    ensure_dirs,
    extract_json,
    health_check,
    read_json,
    upload_video,
    video_generated_dir,
    video_key,
    write_json,
)

sys.path.insert(0, str(PROJECT_ROOT))


def get_chunks(video_id: str) -> list[dict]:
    from backend.edu.db_edu import get_chunks_for_video

    chunks = get_chunks_for_video(video_id)
    return [
        {
            "start_time": c.get("start_time"),
            "end_time": c.get("end_time"),
            "transcript_text": c.get("transcript_text", ""),
            "visual_summary": c.get("visual_summary", ""),
            "frame_paths": c.get("frame_paths", []),
        }
        for c in chunks
    ]


def build_evidence_text(chunks: list[dict], max_chars: int = 18000) -> str:
    parts: list[str] = []
    total = 0
    for chunk in chunks:
        text = (chunk.get("transcript_text") or "").strip()
        visual = (chunk.get("visual_summary") or "").strip()
        if not text and not visual:
            continue
        section = f"[{chunk.get('start_time')}s - {chunk.get('end_time')}s]\nTranscript: {text}"
        if visual:
            section += f"\nVisual: {visual}"
        if total + len(section) > max_chars:
            break
        parts.append(section)
        total += len(section)
    return "\n\n".join(parts)


def question_prompt(video_name: str, evidence_text: str) -> str:
    return f"""
You are creating a human-reviewable evaluation set for Edu-VQAGuider.
Use ONLY the provided transcript/visual evidence from video "{video_name}".

Generate exactly 10 questions with this category distribution:
- 3 factual
- 2 reasoning
- 1 temporal
- 2 visual/multimodal
- 2 unanswerable

Rules:
- Questions must be answerable from the evidence, except unanswerable questions.
- Unanswerable questions must ask for information not present in the video.
- Ground truth for unanswerable questions must clearly say the video does not provide enough information.
- Include concrete timestamp ranges and evidence snippets when possible.
- Return JSON only: an array of 10 objects.

Each object must have:
{{
  "id": "stable_id",
  "video": "{video_name}",
  "category": "factual|reasoning|temporal|visual/multimodal|unanswerable",
  "question": "...",
  "ground_truth": "...",
  "timestamp": "...",
  "evidence": "...",
  "verified": false
}}

Evidence:
{evidence_text}
""".strip()


def normalize_questions(raw: list[dict], video_name: str) -> list[dict]:
    expected = {category: count for category, count in QUESTION_CATEGORIES}
    counts = {category: 0 for category, _ in QUESTION_CATEGORIES}
    normalized: list[dict] = []

    for item in raw:
        category = str(item.get("category", "")).strip().lower()
        if category == "visual":
            category = "visual/multimodal"
        if category not in expected or counts[category] >= expected[category]:
            continue
        counts[category] += 1
        qid = f"{video_key(video_name)}_{category.replace('/', '_')}_{counts[category]}"
        normalized.append({
            "id": str(item.get("id") or qid),
            "video": video_name,
            "category": category,
            "question": str(item.get("question", "")).strip(),
            "ground_truth": str(item.get("ground_truth", "")).strip(),
            "timestamp": str(item.get("timestamp", "")).strip(),
            "evidence": str(item.get("evidence", "")).strip(),
            "verified": False,
        })

    missing = {cat: expected[cat] - counts[cat] for cat in expected if counts[cat] != expected[cat]}
    if len(normalized) != 10 or missing:
        raise EvaluationError(f"Question generation did not produce the required distribution. Missing: {missing}")
    if any(not q["question"] or not q["ground_truth"] for q in normalized):
        raise EvaluationError("Generated JSON contains blank question or ground_truth fields.")
    return normalized


def ensure_video_processed(base_url: str, video_path: Path, force_upload: bool) -> str:
    out_dir = video_generated_dir(video_path)
    state_path = out_dir / "video_state.json"
    state = read_json(state_path, default={}) or {}

    if state.get("video_id") and not force_upload:
        return state["video_id"]

    upload = upload_video(base_url, video_path)
    video_id = upload["video_id"]
    auto_transcribe(base_url, video_id)
    write_json(state_path, {
        "video": video_path.name,
        "video_id": video_id,
        "upload_response": upload,
        "transcription": "auto",
    })
    return video_id


def generate_for_video(base_url: str, video_path: Path, force: bool, force_upload: bool) -> list[dict]:
    out_dir = video_generated_dir(video_path)
    questions_path = out_dir / "questions.json"
    if questions_path.exists() and not force:
        print(f"Skipping {video_path.name}: questions already exist ({questions_path})")
        return read_json(questions_path, default=[])

    print(f"Processing {video_path.name}")
    video_id = ensure_video_processed(base_url, video_path, force_upload)
    chunks = get_chunks(video_id)
    evidence_text = build_evidence_text(chunks)
    if not evidence_text.strip():
        raise EvaluationError(
            f"No transcript/visual evidence found for {video_path.name}. "
            "The video must be transcribed before content-grounded questions can be generated."
        )

    answer = direct_generate(base_url, question_prompt(video_path.name, evidence_text))
    raw = extract_json(answer)
    if not isinstance(raw, list):
        raise EvaluationError("Question generator returned JSON, but not an array.")
    questions = normalize_questions(raw, video_path.name)
    write_json(questions_path, questions)
    print(f"Saved {len(questions)} draft questions to {questions_path}")
    return questions


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate content-grounded evaluation questions.")
    parser.add_argument("--backend", default="http://localhost:8000")
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--force", action="store_true", help="Regenerate questions even if questions.json exists.")
    parser.add_argument("--force-upload", action="store_true", help="Upload/transcribe even if a saved video_id exists.")
    args = parser.parse_args()

    ensure_dirs()
    videos = discover_videos(args.videos_dir)
    if not videos:
        print(f"No videos found in {args.videos_dir}. Add videos later and rerun this command.")
        return 0

    try:
        health = health_check(args.backend)
    except EvaluationError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Backend online. v2_edu_ready={health.get('v2_edu_ready')} device={health.get('device')}")

    all_questions: list[dict] = []
    failures: list[str] = []
    for video in videos:
        try:
            all_questions.extend(generate_for_video(args.backend, video, args.force, args.force_upload))
        except Exception as exc:
            failures.append(f"{video.name}: {exc}")
            print(f"ERROR: {video.name}: {exc}")

    if all_questions:
        write_json(GENERATED_DIR / "all_questions.json", all_questions)
        print(f"Combined question file written to {GENERATED_DIR / 'all_questions.json'}")

    if failures:
        print("Some videos failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
