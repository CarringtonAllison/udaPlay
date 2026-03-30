"""AgentMemory — session and conversation history management."""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_CLAUDE_MODEL = "claude-sonnet-4-6"

_SUMMARY_SYSTEM = (
    "You are a conversation summarizer for a video game assistant. "
    "Summarize the conversation history concisely in 1-3 sentences, "
    "focusing on which games were discussed and what was learned."
)


class AgentMemory:
    """In-memory store for per-session conversation history.

    Each session is identified by a ``session_id`` string.  Within a session,
    each ``AgentContext`` turn is appended in order so the agent can
    reconstruct what was discussed.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list] = {}

    # ------------------------------------------------------------------
    # Record & retrieve
    # ------------------------------------------------------------------

    def record_turn(self, session_id: str, ctx) -> None:
        """Append an ``AgentContext`` to the session history."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(ctx)

    def get_session_history(self, session_id: str) -> list:
        """Return the ordered list of ``AgentContext`` objects for a session."""
        return self._sessions.get(session_id, [])

    def clear_session(self, session_id: str) -> None:
        """Remove all turns for a session."""
        self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    def get_topics_discussed(self, session_id: str) -> list[str]:
        """Return unique game titles mentioned across all turns in a session.

        Extracts titles from ``ctx.final_answer`` text.  This is a
        best-effort extraction; for precise tracking the agent should
        populate ``ctx.citations``.
        """
        history = self.get_session_history(session_id)
        titles: list[str] = []
        for ctx in history:
            for citation in (ctx.citations or []):
                title = citation.get("title") if isinstance(citation, dict) else getattr(citation, "title", None)
                if title and title not in titles:
                    titles.append(title)
        return titles

    def get_conversation_summary(self, session_id: str, llm_client) -> str:
        """Ask Claude to summarise the session history into 1-3 sentences.

        Returns an empty string if the session has no history, without
        making an LLM call.
        """
        history = self.get_session_history(session_id)
        if not history:
            return ""

        # Build a compact Q&A transcript
        turns = []
        for ctx in history:
            turns.append(f"User: {ctx.query}")
            if ctx.final_answer:
                turns.append(f"Assistant: {ctx.final_answer[:300]}")
        transcript = "\n".join(turns)

        try:
            response = llm_client.messages.create(
                model=_CLAUDE_MODEL,
                max_tokens=256,
                system=_SUMMARY_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": f"Summarise this conversation:\n\n{transcript}",
                    }
                ],
            )
            return response.content[0].text.strip()
        except Exception as exc:
            logger.warning("get_conversation_summary failed: %s", exc)
            return transcript[:500]
