"""Tests for AgentMemory."""
import pytest
from unittest.mock import MagicMock


def _make_ctx(query: str, answer: str = "Some answer", turn: int = 1):
    """Build a minimal AgentContext-like object for testing."""
    from src.memory import AgentMemory
    ctx = MagicMock()
    ctx.query = query
    ctx.final_answer = answer
    ctx.turn_count = turn
    ctx.citations = []
    return ctx


class TestAgentMemory:

    def test_record_turn_stores_context(self):
        from src.memory import AgentMemory
        memory = AgentMemory()
        ctx = _make_ctx("Who made Elden Ring?")
        memory.record_turn("session-1", ctx)
        history = memory.get_session_history("session-1")
        assert len(history) == 1
        assert history[0].query == "Who made Elden Ring?"

    def test_multiple_turns_preserved(self):
        from src.memory import AgentMemory
        memory = AgentMemory()
        for i, q in enumerate(["Q1?", "Q2?", "Q3?"]):
            memory.record_turn("session-1", _make_ctx(q, turn=i + 1))
        history = memory.get_session_history("session-1")
        assert len(history) == 3
        assert [ctx.query for ctx in history] == ["Q1?", "Q2?", "Q3?"]

    def test_sessions_are_isolated(self):
        from src.memory import AgentMemory
        memory = AgentMemory()
        memory.record_turn("session-A", _make_ctx("Question for A"))
        memory.record_turn("session-B", _make_ctx("Question for B"))
        assert len(memory.get_session_history("session-A")) == 1
        assert len(memory.get_session_history("session-B")) == 1
        assert memory.get_session_history("session-A")[0].query == "Question for A"

    def test_empty_session_returns_empty_list(self):
        from src.memory import AgentMemory
        memory = AgentMemory()
        assert memory.get_session_history("nonexistent") == []

    def test_get_topics_discussed_returns_game_titles(self):
        from src.memory import AgentMemory
        memory = AgentMemory()
        ctx1 = _make_ctx("Who made Elden Ring?", answer="Elden Ring was made by FromSoftware.")
        ctx2 = _make_ctx("What platforms is Hollow Knight on?", answer="Hollow Knight is on PC and Switch.")
        memory.record_turn("session-1", ctx1)
        memory.record_turn("session-1", ctx2)
        topics = memory.get_topics_discussed("session-1")
        assert isinstance(topics, list)

    def test_get_conversation_summary_calls_llm(self):
        from src.memory import AgentMemory
        memory = AgentMemory()
        memory.record_turn("session-1", _make_ctx("Who made Elden Ring?", "FromSoftware made it."))
        mock_llm = MagicMock()
        summary_block = MagicMock()
        summary_block.type = "text"
        summary_block.text = "The user asked about Elden Ring's developer."
        mock_response = MagicMock()
        mock_response.content = [summary_block]
        mock_llm.messages.create.return_value = mock_response
        summary = memory.get_conversation_summary("session-1", mock_llm)
        assert isinstance(summary, str)
        assert len(summary) > 0
        mock_llm.messages.create.assert_called_once()

    def test_get_conversation_summary_empty_session_returns_empty_string(self):
        from src.memory import AgentMemory
        memory = AgentMemory()
        mock_llm = MagicMock()
        summary = memory.get_conversation_summary("empty-session", mock_llm)
        assert summary == ""
        mock_llm.messages.create.assert_not_called()

    def test_clear_session(self):
        from src.memory import AgentMemory
        memory = AgentMemory()
        memory.record_turn("session-1", _make_ctx("Q1"))
        memory.clear_session("session-1")
        assert memory.get_session_history("session-1") == []
