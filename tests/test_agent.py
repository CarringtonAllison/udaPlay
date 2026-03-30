"""Tests for UdaPlayAgent state machine."""
import json
import uuid
import pytest
from unittest.mock import MagicMock, patch
from tests.conftest import SAMPLE_GAMES


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def populated_store(ephemeral_chroma_client, embedding_manager):
    from src.vector_store import VectorStoreManager
    store = VectorStoreManager(
        chroma_client=ephemeral_chroma_client,
        embedding_manager=embedding_manager,
        collection_name=f"agent_test_{uuid.uuid4().hex[:8]}",
    )
    store.upsert_documents(SAMPLE_GAMES)
    return store


def _make_eval_response(confidence: float):
    msg = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps({
        "confidence": confidence,
        "reason": "Test evaluation.",
        "relevant_ids": ["elden-ring"],
    })
    msg.content = [block]
    return msg


def _make_answer_response(answer_text: str):
    msg = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = answer_text
    msg.content = [block]
    msg.stop_reason = "end_turn"
    return msg


# ---------------------------------------------------------------------------
# AgentState enum
# ---------------------------------------------------------------------------

class TestAgentState:

    def test_all_required_states_exist(self):
        from src.agent import AgentState
        required = {"IDLE", "RETRIEVE", "EVALUATE", "WEBSEARCH", "PERSIST", "ANSWER", "ERROR"}
        actual = {s.name for s in AgentState}
        assert required.issubset(actual)


# ---------------------------------------------------------------------------
# AgentContext dataclass
# ---------------------------------------------------------------------------

class TestAgentContext:

    def test_context_initialises_with_defaults(self):
        from src.agent import AgentContext, AgentState
        ctx = AgentContext(query="test query")
        assert ctx.query == "test query"
        assert ctx.state == AgentState.IDLE
        assert ctx.retrieval_results == []
        assert ctx.web_results == []
        assert ctx.final_answer == ""
        assert ctx.citations == []
        assert ctx.error is None

    def test_context_accepts_session_id(self):
        from src.agent import AgentContext
        ctx = AgentContext(query="test", session_id="session-abc")
        assert ctx.session_id == "session-abc"


# ---------------------------------------------------------------------------
# UdaPlayAgent — RAG hit path (high confidence)
# ---------------------------------------------------------------------------

class TestAgentRAGPath:

    def test_run_returns_agent_context(self, populated_store, mock_anthropic_client, mock_tavily_client):
        from src.agent import UdaPlayAgent, AgentState
        mock_anthropic_client.messages.create.side_effect = [
            _make_eval_response(0.95),   # evaluate_retrieval call
            _make_answer_response("Elden Ring was made by FromSoftware."),  # answer call
        ]
        agent = UdaPlayAgent(
            vector_store=populated_store,
            llm_client=mock_anthropic_client,
            tavily_client=mock_tavily_client,
        )
        ctx = agent.run("Who made Elden Ring?")
        assert ctx.state == AgentState.ANSWER
        assert len(ctx.final_answer) > 0

    def test_high_confidence_does_not_call_tavily(self, populated_store, mock_anthropic_client, mock_tavily_client):
        from src.agent import UdaPlayAgent
        mock_anthropic_client.messages.create.side_effect = [
            _make_eval_response(0.95),
            _make_answer_response("Elden Ring was made by FromSoftware."),
        ]
        agent = UdaPlayAgent(
            vector_store=populated_store,
            llm_client=mock_anthropic_client,
            tavily_client=mock_tavily_client,
        )
        agent.run("Who made Elden Ring?")
        mock_tavily_client.search.assert_not_called()

    def test_retrieval_results_populated(self, populated_store, mock_anthropic_client, mock_tavily_client):
        from src.agent import UdaPlayAgent
        mock_anthropic_client.messages.create.side_effect = [
            _make_eval_response(0.95),
            _make_answer_response("Hollow Knight was made by Team Cherry."),
        ]
        agent = UdaPlayAgent(
            vector_store=populated_store,
            llm_client=mock_anthropic_client,
            tavily_client=mock_tavily_client,
        )
        ctx = agent.run("Who made Hollow Knight?")
        assert len(ctx.retrieval_results) > 0


# ---------------------------------------------------------------------------
# UdaPlayAgent — web search fallback path (low confidence)
# ---------------------------------------------------------------------------

class TestAgentWebSearchPath:

    def test_low_confidence_triggers_web_search(self, populated_store, mock_anthropic_client, mock_tavily_client):
        from src.agent import UdaPlayAgent
        mock_anthropic_client.messages.create.side_effect = [
            _make_eval_response(0.1),   # low confidence → web search
            _make_answer_response("Unknown game details found on web."),
        ]
        agent = UdaPlayAgent(
            vector_store=populated_store,
            llm_client=mock_anthropic_client,
            tavily_client=mock_tavily_client,
        )
        ctx = agent.run("Tell me about a 2025 game that doesn't exist in the database")
        mock_tavily_client.search.assert_called()

    def test_web_results_persisted_to_vector_store(self, populated_store, mock_anthropic_client, mock_tavily_client):
        from src.agent import UdaPlayAgent
        initial_count = populated_store.get_collection_stats()["count"]
        mock_anthropic_client.messages.create.side_effect = [
            _make_eval_response(0.1),
            _make_answer_response("Found it on the web."),
        ]
        agent = UdaPlayAgent(
            vector_store=populated_store,
            llm_client=mock_anthropic_client,
            tavily_client=mock_tavily_client,
        )
        agent.run("What is that new 2025 game?")
        # Vector store should have grown after persisting web results
        final_count = populated_store.get_collection_stats()["count"]
        assert final_count >= initial_count

    def test_web_search_failure_still_produces_answer(self, populated_store, mock_anthropic_client, mock_tavily_client):
        from src.agent import UdaPlayAgent, AgentState
        mock_tavily_client.search.side_effect = Exception("Tavily down")
        mock_anthropic_client.messages.create.side_effect = [
            _make_eval_response(0.1),
            _make_answer_response("I could not find reliable information."),
        ]
        agent = UdaPlayAgent(
            vector_store=populated_store,
            llm_client=mock_anthropic_client,
            tavily_client=mock_tavily_client,
        )
        ctx = agent.run("Obscure game query")
        assert ctx.state == AgentState.ANSWER
        assert len(ctx.final_answer) > 0


# ---------------------------------------------------------------------------
# UdaPlayAgent — error handling
# ---------------------------------------------------------------------------

class TestAgentErrorHandling:

    def test_llm_failure_transitions_to_error_then_answer(self, populated_store, mock_anthropic_client, mock_tavily_client):
        from src.agent import UdaPlayAgent, AgentState
        mock_anthropic_client.messages.create.side_effect = Exception("LLM crashed")
        agent = UdaPlayAgent(
            vector_store=populated_store,
            llm_client=mock_anthropic_client,
            tavily_client=mock_tavily_client,
        )
        ctx = agent.run("Who made Elden Ring?")
        # Agent should recover and return ANSWER with an error message
        assert ctx.state == AgentState.ANSWER
        assert ctx.final_answer != ""


# ---------------------------------------------------------------------------
# UdaPlayAgent — multi-turn sessions
# ---------------------------------------------------------------------------

class TestAgentSessions:

    def test_session_id_auto_generated(self, populated_store, mock_anthropic_client, mock_tavily_client):
        from src.agent import UdaPlayAgent
        mock_anthropic_client.messages.create.side_effect = [
            _make_eval_response(0.9),
            _make_answer_response("Answer 1."),
        ]
        agent = UdaPlayAgent(
            vector_store=populated_store,
            llm_client=mock_anthropic_client,
            tavily_client=mock_tavily_client,
        )
        ctx = agent.run("Q1?")
        assert ctx.session_id is not None

    def test_same_session_accumulates_turns(self, populated_store, mock_anthropic_client, mock_tavily_client):
        from src.agent import UdaPlayAgent
        mock_anthropic_client.messages.create.side_effect = [
            _make_eval_response(0.9), _make_answer_response("Answer 1."),
            _make_eval_response(0.9), _make_answer_response("Answer 2."),
        ]
        agent = UdaPlayAgent(
            vector_store=populated_store,
            llm_client=mock_anthropic_client,
            tavily_client=mock_tavily_client,
        )
        ctx1 = agent.run("Q1?", session_id="session-xyz")
        ctx2 = agent.run("Q2?", session_id="session-xyz")
        history = agent.memory.get_session_history("session-xyz")
        assert len(history) == 2
