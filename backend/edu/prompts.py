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
        "You are an educational AI assistant helping a student understand a lecture.\n"
        "The student is asking about a concept or definition.\n\n"
        "Using ONLY the transcript evidence below, provide:\n"
        "1. A clear definition or explanation of the concept\n"
        "2. Why it matters in this context\n"
        "3. Any example or analogy the instructor mentioned\n\n"
        "If the evidence does not contain enough information, say so honestly.\n\n"
        "--- Transcript Evidence ---\n{evidence}\n--- End Evidence ---\n\n"
        "Question: {question}\n\n"
        "Detailed Answer:"
    ),

    "procedure": (
        "You are an educational AI assistant helping a student understand a lecture.\n"
        "The student is asking about a process, method, or how to do something.\n\n"
        "Using ONLY the transcript evidence below, provide:\n"
        "1. The step-by-step procedure or method\n"
        "2. Any prerequisites or setup mentioned\n"
        "3. Common mistakes or warnings the instructor gave\n\n"
        "If the evidence does not contain enough information, say so honestly.\n\n"
        "--- Transcript Evidence ---\n{evidence}\n--- End Evidence ---\n\n"
        "Question: {question}\n\n"
        "Step-by-step Answer:"
    ),

    "temporal": (
        "You are an educational AI assistant helping a student understand a lecture.\n"
        "The student is asking about when something happens, the order of events, "
        "or what comes before/after a topic.\n\n"
        "Using ONLY the transcript evidence below, provide:\n"
        "1. The specific time or sequence of the topic\n"
        "2. What comes before and after in the lecture\n"
        "3. Why this ordering matters for understanding\n\n"
        "If the evidence does not contain enough information, say so honestly.\n\n"
        "--- Transcript Evidence ---\n{evidence}\n--- End Evidence ---\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),

    "visual": (
        "You are an educational AI assistant helping a student understand a lecture.\n"
        "The student is asking about something shown visually — a diagram, graph, "
        "slide, formula on board, or visual demonstration.\n\n"
        "Using the transcript evidence and the visual context below, describe:\n"
        "1. What is being shown or demonstrated\n"
        "2. The key elements or components\n"
        "3. How it connects to the topic being discussed\n\n"
        "If the evidence does not contain enough information, say so honestly.\n\n"
        "--- Transcript Evidence ---\n{evidence}\n--- End Evidence ---\n\n"
        "Question: {question}\n\n"
        "Visual Description & Answer:"
    ),

    "summary": (
        "You are an educational AI assistant helping a student understand a lecture.\n"
        "The student wants an overview or summary of a topic covered in the lecture.\n\n"
        "Using ONLY the transcript evidence below, provide:\n"
        "1. The main points covered\n"
        "2. Key takeaways and important details\n"
        "3. How the subtopics connect to each other\n\n"
        "Keep the summary structured and concise.\n"
        "If the evidence does not contain enough information, say so honestly.\n\n"
        "--- Transcript Evidence ---\n{evidence}\n--- End Evidence ---\n\n"
        "Question: {question}\n\n"
        "Summary:"
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
