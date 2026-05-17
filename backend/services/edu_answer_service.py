from __future__ import annotations

from backend.services.edu_planner import EduRoute


def build_grounded_answer(question: str, route: EduRoute, evidence: list[dict]) -> tuple[str, str]:
    """Deterministic MVP answer builder. Replace with local Qwen after scaffold."""
    snippets = [item.get("transcript_excerpt", "").strip() for item in evidence]
    snippets = [snippet for snippet in snippets if snippet]
    if snippets:
        direct = _direct_answer(route.primary_intent, snippets[0])
    else:
        direct = "I could not find enough transcript evidence to answer confidently."

    if route.primary_intent == "summary":
        body = _summary_answer(snippets)
    elif route.primary_intent == "procedure":
        body = _procedure_answer(snippets)
    elif route.primary_intent == "temporal":
        body = _temporal_answer(evidence)
    elif route.primary_intent == "visual":
        body = _visual_answer(snippets, evidence)
    else:
        body = _concept_answer(question, snippets)
    return direct, body


def _direct_answer(intent: str, snippet: str) -> str:
    clean = " ".join(snippet.split())
    if len(clean) > 220:
        clean = clean[:217].rsplit(" ", 1)[0] + "..."
    return f"The most relevant {intent} evidence is: {clean}"


def _concept_answer(question: str, snippets: list[str]) -> str:
    context = _joined(snippets)
    return (
        f"Based on the retrieved lecture segment, the answer to '{question}' is grounded in this context: "
        f"{context}\n\n"
        "Interpretation: the system found this part of the video as the strongest evidence. "
        "A local instruction model can be connected here to turn the evidence into a fuller explanation."
    )


def _procedure_answer(snippets: list[str]) -> str:
    context = _joined(snippets)
    return (
        "The relevant procedure appears in the retrieved evidence. Key steps or actions mentioned are:\n"
        f"1. {context}\n\n"
        "Use the timestamped evidence below to verify the exact sequence in the video."
    )


def _temporal_answer(evidence: list[dict]) -> str:
    parts = []
    for item in evidence:
        parts.append(
            f"{_fmt_time(item['start_time'])}-{_fmt_time(item['end_time'])}: "
            f"{item.get('transcript_excerpt') or 'visual segment selected'}"
        )
    return "Timeline evidence:\n" + "\n".join(f"- {part}" for part in parts)


def _visual_answer(snippets: list[str], evidence: list[dict]) -> str:
    frame_count = sum(len(item.get("frame_paths") or []) for item in evidence)
    return (
        f"The visual route selected {frame_count} frame(s) from the relevant segment(s). "
        f"Transcript context: {_joined(snippets)}"
    )


def _summary_answer(snippets: list[str]) -> str:
    return "Summary of the retrieved video evidence:\n" + _joined(snippets)


def _joined(snippets: list[str]) -> str:
    text = " ".join(" ".join(snippet.split()) for snippet in snippets)
    return text[:1600] if text else "No transcript text was available for the retrieved segments."


def _fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"
