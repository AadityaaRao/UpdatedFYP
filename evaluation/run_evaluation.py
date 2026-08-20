from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

from eval_utils import (
    GENERATED_DIR,
    REPORT_JSON,
    REPORT_MD,
    EvaluationError,
    direct_generate,
    ensure_dirs,
    extract_json,
    health_check,
    read_json,
    ask_question,
    video_generated_dir,
    video_key,
    write_json,
)


def load_video_id_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    data = read_json(path, default={}) or {}
    return {str(k): str(v) for k, v in data.items()}


def load_questions(allow_unverified: bool) -> dict[str, list[dict]]:
    by_video: dict[str, list[dict]] = {}
    for qpath in sorted(GENERATED_DIR.glob("*/questions.json")):
        questions = read_json(qpath, default=[]) or []
        selected = [q for q in questions if allow_unverified or q.get("verified") is True]
        if selected:
            video_name = selected[0].get("video") or f"{qpath.parent.name}.mp4"
            by_video[str(video_name)] = selected
    return by_video


def evidence_text_from_response(data: dict) -> str:
    parts = []
    for chunk in data.get("evidence_chunks") or []:
        start = chunk.get("start_time")
        end = chunk.get("end_time")
        text = (chunk.get("transcript_text") or "").strip()
        frame = chunk.get("selected_frame_path")
        parts.append(f"[{start}s - {end}s] {text}" + (f" Frame: {frame}" if frame else ""))
    return "\n".join(parts)


def timestamps_from_response(data: dict) -> list[dict]:
    return [
        {
            "start_time": chunk.get("start_time"),
            "end_time": chunk.get("end_time"),
            "chunk_id": chunk.get("chunk_id"),
            "selected_frame_path": chunk.get("selected_frame_path"),
        }
        for chunk in data.get("evidence_chunks") or []
    ]


def result_from_api(question: dict, data: dict | None, latency: float, error: str | None) -> dict:
    route = (data or {}).get("route") or {}
    return {
        "question_id": question.get("id"),
        "question": question.get("question"),
        "category": question.get("category"),
        "ground_truth": question.get("ground_truth"),
        "actual_answer": (data or {}).get("direct_answer") or (data or {}).get("detailed_answer") or "",
        "detailed_answer": (data or {}).get("detailed_answer") or "",
        "evidence": evidence_text_from_response(data or {}),
        "timestamps": timestamps_from_response(data or {}),
        "route": route.get("route"),
        "planner_source": route.get("planner_source"),
        "route_confidence": route.get("confidence"),
        "latency_seconds": round(latency, 3),
        "error": error,
    }


def judge_prompt(result: dict) -> str:
    return f"""
You are an answer-quality evaluator for educational video QA.
Score the actual answer against the expected answer and retrieved evidence.

Return JSON only with integer scores 0, 1, or 2:
{{
  "correctness": 0,
  "relevance": 0,
  "groundedness": 0,
  "completeness": 0,
  "reason": "brief explanation"
}}

Rubric:
- Correctness: 0 incorrect, 1 partially correct, 2 fully correct.
- Relevance: 0 irrelevant, 1 partially relevant, 2 directly answers the question.
- Groundedness: 0 unsupported/hallucinated, 1 partially supported, 2 fully supported by video/evidence.
- Completeness: 0 major missing info, 1 partial, 2 complete.
- For unanswerable questions, the correct behavior is saying the information is not available in the video. Confident unsupported answers should score poorly.

Category: {result.get("category")}
Question: {result.get("question")}
Expected answer: {result.get("ground_truth")}
Actual answer: {result.get("actual_answer")}
Retrieved evidence: {result.get("evidence")}
""".strip()


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def heuristic_judge(result: dict, reason: str = "Heuristic fallback judge used.") -> dict:
    if result.get("error"):
        return {
            "correctness": 0,
            "relevance": 0,
            "groundedness": 0,
            "completeness": 0,
            "reason": f"Question failed with error: {result.get('error')}",
        }

    answer = result.get("actual_answer", "")
    expected = result.get("ground_truth", "")
    answer_words = words(answer)
    expected_words = words(expected)
    overlap = len(answer_words & expected_words) / max(1, len(expected_words))
    unavailable = any(
        phrase in answer.lower()
        for phrase in ["not available", "not provided", "does not provide", "cannot determine", "not enough information"]
    )

    if result.get("category") == "unanswerable":
        correctness = 2 if unavailable else 0
        groundedness = 2 if unavailable else 0
        relevance = 2 if answer.strip() else 0
        completeness = 2 if unavailable else 0
    else:
        correctness = 2 if overlap >= 0.55 else 1 if overlap >= 0.25 else 0
        relevance = 2 if answer.strip() else 0
        groundedness = 2 if result.get("evidence") and overlap >= 0.25 else 1 if result.get("evidence") else 0
        completeness = 2 if overlap >= 0.55 else 1 if overlap >= 0.25 else 0

    return {
        "correctness": correctness,
        "relevance": relevance,
        "groundedness": groundedness,
        "completeness": completeness,
        "reason": reason,
    }


def normalize_judgement(data: dict) -> dict:
    out = {}
    for key in ("correctness", "relevance", "groundedness", "completeness"):
        value = int(data.get(key, 0))
        out[key] = max(0, min(2, value))
    out["total_score"] = sum(out.values())
    out["reason"] = str(data.get("reason", "")).strip()
    return out


def evaluate_result(result: dict, base_url: str | None, dry_run: bool) -> dict:
    if dry_run or not base_url or result.get("error"):
        return normalize_judgement(heuristic_judge(result, "Dry-run heuristic judge used."))
    try:
        raw = direct_generate(base_url, judge_prompt(result))
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError("Judge did not return a JSON object.")
        return normalize_judgement(data)
    except Exception as exc:
        return normalize_judgement(heuristic_judge(result, f"LLM judge unavailable; heuristic fallback used. {exc}"))


def load_video_id(video_name: str, explicit_map: dict[str, str]) -> str | None:
    if video_name in explicit_map:
        return explicit_map[video_name]
    if video_key(video_name) in explicit_map:
        return explicit_map[video_key(video_name)]
    state = read_json(video_generated_dir(video_name) / "video_state.json", default={}) or {}
    return state.get("video_id")


def existing_results(path: Path) -> dict[str, dict]:
    rows = read_json(path, default=[]) or []
    return {str(r.get("question_id")): r for r in rows}


def dry_run_questions() -> dict[str, list[dict]]:
    return {
        "dry_run_demo.mp4": [
            {
                "id": "dry_factual_1",
                "video": "dry_run_demo.mp4",
                "category": "factual",
                "question": "What topic is introduced in the demo evidence?",
                "ground_truth": "The demo evidence introduces matrix notation.",
                "timestamp": "00:10",
                "evidence": "Demo evidence only.",
                "verified": False,
            },
            {
                "id": "dry_unanswerable_1",
                "video": "dry_run_demo.mp4",
                "category": "unanswerable",
                "question": "What is the instructor's home address?",
                "ground_truth": "The video does not provide enough information to answer this.",
                "timestamp": "",
                "evidence": "No such information is present.",
                "verified": False,
            },
        ]
    }


def dry_run_result(question: dict) -> dict:
    if question["category"] == "unanswerable":
        answer = "The video does not provide enough information to answer that."
    else:
        answer = "The demo evidence introduces matrix notation."
    return {
        "question_id": question["id"],
        "question": question["question"],
        "category": question["category"],
        "ground_truth": question["ground_truth"],
        "actual_answer": answer,
        "detailed_answer": answer,
        "evidence": question["evidence"],
        "timestamps": [],
        "route": "dry_run",
        "planner_source": "dry_run",
        "route_confidence": None,
        "latency_seconds": 0.0,
        "error": None,
    }


def run_video(video_name: str, questions: list[dict], base_url: str | None, video_id: str | None, dry_run: bool) -> list[dict]:
    out_dir = video_generated_dir(video_name)
    results_path = out_dir / "results.json"
    previous = existing_results(results_path)
    rows: list[dict] = []

    for idx, question in enumerate(questions, 1):
        qid = str(question.get("id"))
        if qid in previous and previous[qid].get("evaluation"):
            rows.append(previous[qid])
            print(f"  [{idx}/{len(questions)}] reused {qid}")
            continue

        print(f"  [{idx}/{len(questions)}] {qid}")
        if dry_run:
            result = dry_run_result(question)
        elif not video_id or not base_url:
            result = result_from_api(question, None, 0.0, "Missing video_id. Run generate_questions.py first or pass --video-id-map.")
        else:
            data, latency, error = ask_question(base_url, video_id, question["question"])
            result = result_from_api(question, data, latency, error)

        result["evaluation"] = evaluate_result(result, base_url, dry_run)
        rows.append(result)
        write_json(results_path, rows)

    return rows


def pct(avg: float) -> float:
    return round((avg / 2.0) * 100, 2) if not math.isnan(avg) else 0.0


def summarize(rows: list[dict], total_questions: int, dry_run: bool) -> dict:
    evaluated = [r for r in rows if r.get("evaluation")]
    failures = [r for r in evaluated if r["evaluation"].get("total_score", 0) < 6 or r.get("error")]

    def avg_metric(metric: str, items: list[dict]) -> float:
        if not items:
            return 0.0
        return sum(r["evaluation"].get(metric, 0) for r in items) / len(items)

    overall_avg = sum(r["evaluation"].get("total_score", 0) for r in evaluated) / len(evaluated) if evaluated else 0.0

    per_video = {}
    per_category = {}
    for key, group_by in (("video", per_video), ("category", per_category)):
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in evaluated:
            groups[str(row.get(key, "unknown"))].append(row)
        for name, items in groups.items():
            group_by[name] = {
                "questions": len(items),
                "average_score": round(sum(i["evaluation"]["total_score"] for i in items) / len(items), 2),
                "failures": [
                    {
                        "question_id": i.get("question_id"),
                        "question": i.get("question"),
                        "score": i["evaluation"].get("total_score", 0),
                        "reason": i["evaluation"].get("reason", ""),
                    }
                    for i in items if i["evaluation"].get("total_score", 0) < 6 or i.get("error")
                ],
            }

    return {
        "dry_run": dry_run,
        "overall": {
            "total_questions": total_questions,
            "questions_evaluated": len(evaluated),
            "average_score": round(overall_avg, 2),
            "correctness_percent": pct(avg_metric("correctness", evaluated)),
            "relevance_percent": pct(avg_metric("relevance", evaluated)),
            "groundedness_percent": pct(avg_metric("groundedness", evaluated)),
            "completeness_percent": pct(avg_metric("completeness", evaluated)),
        },
        "per_video": per_video,
        "per_category": per_category,
        "failed_questions": failures,
        "worst_questions": sorted(evaluated, key=lambda r: r["evaluation"].get("total_score", 0))[:10],
    }


def write_markdown_report(summary: dict) -> None:
    overall = summary["overall"]
    lines = [
        "# Edu-VQAGuider Answer Quality Report",
        "",
        f"Dry run: {summary['dry_run']}",
        "",
        "## Overall",
        f"- Total questions: {overall['total_questions']}",
        f"- Questions evaluated: {overall['questions_evaluated']}",
        f"- Average score: {overall['average_score']} / 8",
        f"- Correctness: {overall['correctness_percent']}%",
        f"- Relevance: {overall['relevance_percent']}%",
        f"- Groundedness: {overall['groundedness_percent']}%",
        f"- Completeness: {overall['completeness_percent']}%",
        "",
        "## Per Video",
    ]
    for video, data in summary["per_video"].items():
        lines.append(f"- {video}: {data['average_score']} / 8 across {data['questions']} questions; failures={len(data['failures'])}")

    lines.extend(["", "## Per Category"])
    for category, data in summary["per_category"].items():
        lines.append(f"- {category}: {data['average_score']} / 8 across {data['questions']} questions; failures={len(data['failures'])}")

    lines.extend(["", "## Failed Questions"])
    for row in summary["failed_questions"]:
        ev = row["evaluation"]
        lines.extend([
            f"### {row.get('question_id')} ({row.get('category')})",
            f"- Question: {row.get('question')}",
            f"- Expected: {row.get('ground_truth')}",
            f"- Actual: {row.get('actual_answer')}",
            f"- Score: {ev.get('total_score')} / 8",
            f"- Reason: {ev.get('reason')}",
            f"- Evidence: {row.get('evidence') or 'None'}",
            "",
        ])

    lines.extend(["", "## Worst Performing Questions"])
    for row in summary["worst_questions"]:
        lines.append(f"- {row.get('question_id')}: {row['evaluation'].get('total_score')} / 8 - {row.get('question')}")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run automated answer-quality evaluation.")
    parser.add_argument("--backend", default="http://localhost:8000")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--video-id-map", type=Path, default=None, help="JSON mapping video filename/stem to backend video_id.")
    args = parser.parse_args()

    ensure_dirs()
    if args.dry_run:
        question_map = load_questions(allow_unverified=True) or dry_run_questions()
        base_url = None
        print("Dry-run mode: backend calls are skipped and results are marked dry-run.")
    else:
        try:
            health = health_check(args.backend)
        except EvaluationError as exc:
            print(f"ERROR: {exc}")
            return 1
        print(f"Backend online. v2_edu_ready={health.get('v2_edu_ready')} device={health.get('device')}")
        question_map = load_questions(args.allow_unverified)
        base_url = args.backend

    if not question_map:
        print("No questions selected. Mark questions as verified=true or rerun with --allow-unverified.")
        return 0

    explicit_map = load_video_id_map(args.video_id_map)
    all_rows: list[dict] = []
    total_questions = sum(len(qs) for qs in question_map.values())

    for video_name, questions in question_map.items():
        print(f"Evaluating {video_name} ({len(questions)} questions)")
        video_id = None if args.dry_run else load_video_id(video_name, explicit_map)
        rows = run_video(video_name, questions, base_url, video_id, args.dry_run)
        for row in rows:
            row["video"] = video_name
        all_rows.extend(rows)

    summary = summarize(all_rows, total_questions, args.dry_run)
    write_json(REPORT_JSON, summary)
    write_markdown_report(summary)
    print(f"Report written to {REPORT_JSON} and {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
