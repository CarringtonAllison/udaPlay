"""ReportFormatter — structures agent answers into text, JSON, or both."""
from __future__ import annotations

from typing import Union


_VALID_MODES = {"text", "json", "both"}


class ReportFormatter:
    """Formats the agent's final answer into clean output.

    Modes
    -----
    ``"text"``
        Returns a plain-text string with answer prose and a citations section.
    ``"json"``
        Returns a dict with structured fields suitable for API consumption.
    ``"both"``
        Returns a dict with both ``"text"`` and ``"json"`` keys.
    """

    def format(
        self,
        answer: str,
        citations: list,
        mode: str = "both",
        confidence: float = 0.0,
        query: str = "",
    ) -> Union[str, dict]:
        """Format an agent answer.

        Parameters
        ----------
        answer : str
            The natural language answer text.
        citations : list
            List of citation dicts with keys ``title``, ``url``, ``source``.
        mode : str
            Output mode: ``"text"``, ``"json"``, or ``"both"``.
        confidence : float
            The agent's confidence score (0–1) to include in JSON output.
        query : str
            The original user query (included in JSON output).

        Returns
        -------
        str or dict
            Formatted output matching the requested mode.
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Choose from: {sorted(_VALID_MODES)}"
            )

        if mode == "text":
            return self._format_text(answer, citations)
        if mode == "json":
            return self._format_json(answer, citations, confidence, query)
        # "both"
        return {
            "text": self._format_text(answer, citations),
            "json": self._format_json(answer, citations, confidence, query),
        }

    # ------------------------------------------------------------------
    # Private formatters
    # ------------------------------------------------------------------

    def _format_text(self, answer: str, citations: list) -> str:
        lines = [answer.strip()]

        if citations:
            lines.append("\n**Sources:**")
            for cit in citations:
                title = _get(cit, "title", "Unknown")
                url = _get(cit, "url")
                source = _get(cit, "source", "internal")
                if url:
                    lines.append(f"  - {title} ({url}) [{source}]")
                else:
                    lines.append(f"  - {title} [{source}]")

        return "\n".join(lines)

    def _format_json(
        self,
        answer: str,
        citations: list,
        confidence: float,
        query: str,
    ) -> dict:
        # Extract game titles from citations for the games_mentioned list
        games_mentioned = []
        for cit in citations:
            title = _get(cit, "title")
            if title and title not in games_mentioned:
                games_mentioned.append(title)

        sources = []
        for cit in citations:
            sources.append({
                "title": _get(cit, "title"),
                "url": _get(cit, "url"),
                "source": _get(cit, "source", "internal"),
            })

        return {
            "query": query,
            "answer_summary": answer.strip(),
            "games_mentioned": games_mentioned,
            "confidence": round(confidence, 4),
            "sources": sources,
        }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get(obj, key: str, default=None):
    """Get a value from a dict or object attribute, returning *default* if missing."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
