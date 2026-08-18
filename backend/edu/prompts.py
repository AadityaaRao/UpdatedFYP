"""
backend/edu/prompts.py
────────────────────────────────────────────────────────────
Route-specific prompt templates for Edu-VQAGuider.

Each route produces a DIFFERENT prompt and answer structure.
This is what makes the planner functional, not decorative.

Usage:
    from backend.edu.prompts import build_prompt
    prompt = build_prompt(route="concept", question=q, evidence=ev)
"""
from __future__ import annotations


# ── Route-specific prompt templates ───────────────────────────
# {evidence} = concatenated transcript snippets from retrieved chunks
# {question} = the user's question
# {frame_description} = optional, used by visual route

ROUTE_PROMPTS: dict[str, str] = {
    "concept": (
        "You are a friendly, expert educational AI tutor helping a student understand a lecture.\n"
        "The student is asking about a concept or definition.\n\n"
        "You are provided with lecture frames and transcript evidence from the relevant sections.\n"
        "FIRST, examine the attached lecture frames for any relevant diagrams, equations, or text "
        "on slides that relate to the concept. THEN, use the transcript evidence to understand "
        "what the instructor said.\n\n"
        "Combine BOTH your visual observations and the transcript to write a natural, "
        "conversational explanation. Speak directly to the student. Smoothly incorporate the "
        "definition, why it matters, and any visual aids or analogies the instructor used.\n\n"
        "If the evidence does not contain enough information, say so honestly.\n\n"
        "--- Transcript Evidence ---\n{evidence}\n--- End Evidence ---\n\n"
        "Question: {question}\n\n"
        "Detailed Answer:"
    ),

    "procedure": (
        "You are a friendly, expert educational AI tutor helping a student understand a lecture.\n"
        "The student is asking about a process, method, or how to do something.\n\n"
        "You are provided with lecture frames and transcript evidence from the relevant sections.\n"
        "FIRST, examine the attached lecture frames for any step-by-step instructions, formulas, "
        "code, or diagrams shown on slides or the board. THEN, read the transcript evidence.\n\n"
        "Combine BOTH your visual observations and the transcript to explain the step-by-step "
        "method clearly. If the frames show steps or code that the instructor didn't fully "
        "verbalize, include those details. Speak directly to the student.\n\n"
        "If the evidence does not contain enough information, say so honestly.\n\n"
        "--- Transcript Evidence ---\n{evidence}\n--- End Evidence ---\n\n"
        "Question: {question}\n\n"
        "Detailed Answer:"
    ),

    "temporal": (
        "You are a friendly, expert educational AI tutor helping a student understand a lecture.\n"
        "The student is asking about when something happens or the order of events.\n\n"
        "You are provided with lecture frames and transcript evidence from the relevant sections.\n"
        "Examine the attached frames for any timelines, sequence diagrams, or ordered content "
        "shown visually. Use BOTH the visual content and the transcript to explain the timeline, "
        "what was discussed before and after, and how the sequence builds understanding.\n\n"
        "Speak directly to the student. Do not just copy-paste content as headings.\n\n"
        "If the evidence does not contain enough information, say so honestly.\n\n"
        "--- Transcript Evidence ---\n{evidence}\n--- End Evidence ---\n\n"
        "Question: {question}\n\n"
        "Detailed Answer:"
    ),

    "visual": (
        "You are a friendly, expert educational AI tutor helping a student understand a lecture.\n"
        "The student is asking about something shown visually (a diagram, slide, or equation).\n\n"
        "You are provided with lecture frames and transcript evidence.\n"
        "This is a VISUAL question, so the images are critical.\n\n"
        "STEP 1: Carefully examine EACH attached lecture frame. Describe what you see — "
        "any text on slides, equations on the board, diagrams, graphs, code, or other visual "
        "content.\n"
        "STEP 2: Read the transcript evidence below to understand the instructor's explanation.\n"
        "STEP 3: Combine your visual observations with the transcript to write a detailed, "
        "natural, conversational answer.\n\n"
        "You MUST reference specific visual content you observed in the frames.\n\n"
        "If the evidence does not contain enough information, say so honestly.\n\n"
        "--- Transcript Evidence ---\n{evidence}\n--- End Evidence ---\n\n"
        "Question: {question}\n\n"
        "Detailed Answer:"
    ),

    "summary": (
        "You are a friendly, expert educational AI tutor helping a student understand a lecture.\n"
        "The student wants an overview or summary of a topic covered in the lecture.\n\n"
        "You are provided with lecture frames and transcript evidence from across the lecture.\n"
        "Examine the attached frames for any key slides, diagrams, or structural content that "
        "helps outline the main topics. Use BOTH the visual content and the transcript to write "
        "a natural, conversational summary.\n\n"
        "Speak directly to the student. Highlight the main points and key takeaways in an "
        "engaging way, showing how the ideas connect.\n\n"
        "If the evidence does not contain enough information, say so honestly.\n\n"
        "--- Transcript Evidence ---\n{evidence}\n--- End Evidence ---\n\n"
        "Question: {question}\n\n"
        "Detailed Answer:"
    ),
}


# ── Direct answer prompt (used for all routes) ───────────────
DIRECT_ANSWER_PROMPT = (
    "Given this detailed answer to a student's question, "
    "write a concise 1-2 sentence direct answer.\n\n"
    "Question: {question}\n"
    "Detailed Answer: {detailed_answer}\n\n"
    "Direct Answer (1-2 sentences):"
)


def build_prompt(
    route: str,
    question: str,
    evidence: str,
) -> str:
    """
    Build a route-specific prompt from evidence and question.

    Args:
        route:    One of: concept, procedure, temporal, visual, summary
        question: The user's question
        evidence: Concatenated transcript text from retrieved chunks

    Returns:
        Formatted prompt string ready for LLM input
    """
    template = ROUTE_PROMPTS.get(route, ROUTE_PROMPTS["concept"])
    return template.format(
        question=question,
        evidence=evidence,
    )


def build_direct_answer_prompt(question: str, detailed_answer: str) -> str:
    """
    Build a prompt to condense a detailed answer into 1-2 sentences.
    """
    return DIRECT_ANSWER_PROMPT.format(
        question=question,
        detailed_answer=detailed_answer,
    )


# ── Route-specific retrieval parameters ──────────────────────

ROUTE_RETRIEVAL_CONFIG: dict[str, dict] = {
    "concept": {
        "top_k": 3,
        "use_clip_boost": False,
        "use_temporal_neighbors": False,
    },
    "procedure": {
        "top_k": 3,
        "use_clip_boost": False,
        "use_temporal_neighbors": False,
    },
    "temporal": {
        "top_k": 1,  # retrieve top-1, then expand to neighbors
        "use_clip_boost": False,
        "use_temporal_neighbors": True,  # add chunk before + after
    },
    "visual": {
        "top_k": 3,
        "use_clip_boost": True,   # boost CLIP frame similarity in ranking
        "use_temporal_neighbors": False,
    },
    "summary": {
        "top_k": 5,  # broader coverage for summaries
        "use_clip_boost": False,
        "use_temporal_neighbors": False,
    },
}
