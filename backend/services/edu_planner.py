from __future__ import annotations

from dataclasses import dataclass


ROUTE_EVIDENCE = {
    "concept": ["transcript"],
    "procedure": ["transcript", "frames"],
    "temporal": ["transcript"],
    "visual": ["transcript", "frames"],
    "summary": ["transcript"],
}


@dataclass(frozen=True)
class EduRoute:
    primary_intent: str
    evidence_types: list[str]
    confidence: float
    planner_source: str = "fallback"

    def to_dict(self) -> dict:
        return {
            "primary_intent": self.primary_intent,
            "evidence_types": self.evidence_types,
            "confidence": self.confidence,
            "planner_source": self.planner_source,
        }


class EduVQAGuiderPlanner:
    """MVP planner. A learned classifier can replace _rule_route later."""

    def route(self, question: str) -> EduRoute:
        intent, confidence = self._rule_route(question)
        return EduRoute(
            primary_intent=intent,
            evidence_types=ROUTE_EVIDENCE[intent],
            confidence=confidence,
            planner_source="fallback",
        )

    def _rule_route(self, question: str) -> tuple[str, float]:
        q = question.lower()
        if any(word in q for word in ["summarize", "summary", "overview", "main points"]):
            return "summary", 0.82
        if any(word in q for word in ["formula", "written", "slide", "board", "text"]):
            return "visual", 0.72
        if any(word in q for word in ["shown", "visible", "diagram", "graph", "image", "object"]):
            return "visual", 0.78
        if any(word in q for word in ["before", "after", "when", "where", "timestamp"]):
            return "temporal", 0.80
        if any(word in q for word in ["steps", "procedure", "process", "how", "method"]):
            return "procedure", 0.80
        if any(word in q for word in ["why", "explain", "reason", "concept", "meaning"]):
            return "concept", 0.78
        return "concept", 0.55


planner = EduVQAGuiderPlanner()
