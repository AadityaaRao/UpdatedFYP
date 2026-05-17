from __future__ import annotations

import math
import re
from collections import Counter

from backend.database import db
from backend.services.edu_planner import EduRoute
from config import EDU_SUMMARY_TOP_K, EDU_TOP_K


_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokens(text: str) -> list[str]:
    return [tok.lower() for tok in _WORD_RE.findall(text or "") if len(tok) > 2]


def _cosine_counts(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[key] * b.get(key, 0) for key in a)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def retrieve_chunks(video_id: str, question: str, route: EduRoute, top_k: int | None = None) -> list[dict]:
    chunks = db.get_video_chunks(video_id)
    if not chunks:
        return []
    requested_k = top_k or (EDU_SUMMARY_TOP_K if route.primary_intent == "summary" else EDU_TOP_K)
    query_vec = Counter(_tokens(question))
    scored = []
    for chunk in chunks:
        chunk_text = " ".join(
            [
                chunk.get("transcript_text") or "",
                chunk.get("visual_summary") or "",
            ]
        )
        score = _cosine_counts(query_vec, Counter(_tokens(chunk_text)))
        if route.primary_intent == "visual" and chunk.get("frame_paths"):
            score += 0.05
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [(score, chunk) for score, chunk in scored[:requested_k]]

    if route.primary_intent == "temporal" and selected:
        top_chunk = selected[0][1]
        idx = next((i for i, chunk in enumerate(chunks) if chunk["id"] == top_chunk["id"]), 0)
        neighbor_indices = [i for i in [idx - 1, idx, idx + 1] if 0 <= i < len(chunks)]
        selected_ids = {chunk["id"] for _, chunk in selected}
        for i in neighbor_indices:
            chunk = chunks[i]
            if chunk["id"] not in selected_ids:
                selected.append((0.01, chunk))

    evidence = []
    for score, chunk in selected:
        evidence.append(
            {
                "chunk_id": chunk["id"],
                "start_time": float(chunk["start_time"]),
                "end_time": float(chunk["end_time"]),
                "transcript_excerpt": chunk.get("transcript_text") or "",
                "visual_summary": chunk.get("visual_summary"),
                "frame_paths": chunk.get("frame_paths") or [],
                "score": round(float(score), 4),
            }
        )
    return evidence
