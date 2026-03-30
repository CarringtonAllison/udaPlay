"""Tests for agent tools: retrieve_game, evaluate_retrieval, game_web_search."""
import json
import uuid
import pytest
from unittest.mock import MagicMock
from tests.conftest import SAMPLE_GAME, SAMPLE_GAME_2, SAMPLE_GAMES


# ---------------------------------------------------------------------------
# Shared fixture: a populated VectorStoreManager
# ---------------------------------------------------------------------------

@pytest.fixture
def populated_store(ephemeral_chroma_client, embedding_manager):
    from src.vector_store import VectorStoreManager
    store = VectorStoreManager(
        chroma_client=ephemeral_chroma_client,
        embedding_manager=embedding_manager,
        collection_name=f"games_{uuid.uuid4().hex[:8]}",
    )
    store.upsert_documents(SAMPLE_GAMES)
    return store


# ---------------------------------------------------------------------------
# Tool 1: retrieve_game
# ---------------------------------------------------------------------------

class TestRetrieveGame:

    def test_returns_list_of_retrieval_results(self, populated_store):
        from src.tools import retrieve_game
        results = retrieve_game("open world action game", populated_store)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_returns_at_most_n_results(self, populated_store):
        from src.tools import retrieve_game
        results = retrieve_game("game", populated_store, n_results=1)
        assert len(results) <= 1

    def test_results_have_relevance_score(self, populated_store):
        from src.tools import retrieve_game
        results = retrieve_game("RPG game", populated_store)
        for r in results:
            assert hasattr(r, "relevance_score")
            assert 0.0 <= r.relevance_score <= 1.0

    def test_correct_game_surfaced(self, populated_store):
        from src.tools import retrieve_game
        results = retrieve_game("Metroidvania underground bug kingdom", populated_store, n_results=2)
        top_ids = [r.game_id for r in results]
        assert "hollow-knight" in top_ids

    def test_empty_store_returns_empty_list(self, ephemeral_chroma_client, embedding_manager):
        from src.vector_store import VectorStoreManager
        from src.tools import retrieve_game
        empty_store = VectorStoreManager(
            chroma_client=ephemeral_chroma_client,
            embedding_manager=embedding_manager,
            collection_name=f"empty_{uuid.uuid4().hex[:8]}",
        )
        results = retrieve_game("anything", empty_store)
        assert results == []


# ---------------------------------------------------------------------------
# Tool 2: evaluate_retrieval
# ---------------------------------------------------------------------------

def _make_eval_response(confidence: float, reason: str = "Good match.", relevant_ids=None):
    payload = {
        "confidence": confidence,
        "reason": reason,
        "relevant_ids": relevant_ids or ["elden-ring"],
    }
    msg = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(payload)
    msg.content = [block]
    return msg


class TestEvaluateRetrieval:

    def test_high_confidence_does_not_trigger_web_search(self, populated_store, mock_anthropic_client):
        from src.tools import retrieve_game, evaluate_retrieval
        mock_anthropic_client.messages.create.return_value = _make_eval_response(0.95)
        results = retrieve_game("Elden Ring", populated_store)
        eval_result = evaluate_retrieval("Who made Elden Ring?", results, mock_anthropic_client)
        assert eval_result.confidence == pytest.approx(0.95)
        assert eval_result.should_web_search is False

    def test_low_confidence_triggers_web_search(self, populated_store, mock_anthropic_client):
        from src.tools import retrieve_game, evaluate_retrieval
        mock_anthropic_client.messages.create.return_value = _make_eval_response(0.2)
        results = retrieve_game("obscure game from 2025", populated_store)
        eval_result = evaluate_retrieval("Tell me about a 2025 MMORPG", results, mock_anthropic_client)
        assert eval_result.should_web_search is True

    def test_threshold_boundary_at_0_65(self, populated_store, mock_anthropic_client):
        from src.tools import retrieve_game, evaluate_retrieval
        results = retrieve_game("game", populated_store)
        # At exactly threshold: not a web search
        mock_anthropic_client.messages.create.return_value = _make_eval_response(0.65)
        eval_result = evaluate_retrieval("query", results, mock_anthropic_client, threshold=0.65)
        assert eval_result.should_web_search is False

    def test_returns_evaluation_result_dataclass(self, populated_store, mock_anthropic_client):
        from src.tools import retrieve_game, evaluate_retrieval
        from src.tools import EvaluationResult
        mock_anthropic_client.messages.create.return_value = _make_eval_response(0.8)
        results = retrieve_game("game", populated_store)
        eval_result = evaluate_retrieval("query", results, mock_anthropic_client)
        assert isinstance(eval_result, EvaluationResult)
        assert hasattr(eval_result, "confidence")
        assert hasattr(eval_result, "reason")
        assert hasattr(eval_result, "should_web_search")
        assert hasattr(eval_result, "sufficient_results")

    def test_invalid_llm_response_returns_zero_confidence(self, populated_store, mock_anthropic_client):
        from src.tools import retrieve_game, evaluate_retrieval
        bad_msg = MagicMock()
        bad_block = MagicMock()
        bad_block.type = "text"
        bad_block.text = "this is not json"
        bad_msg.content = [bad_block]
        mock_anthropic_client.messages.create.return_value = bad_msg
        results = retrieve_game("game", populated_store)
        eval_result = evaluate_retrieval("query", results, mock_anthropic_client)
        assert eval_result.confidence == 0.0
        assert eval_result.should_web_search is True

    def test_empty_results_returns_zero_confidence(self, mock_anthropic_client):
        from src.tools import evaluate_retrieval
        eval_result = evaluate_retrieval("query", [], mock_anthropic_client)
        assert eval_result.confidence == 0.0
        assert eval_result.should_web_search is True


# ---------------------------------------------------------------------------
# Tool 3: game_web_search
# ---------------------------------------------------------------------------

class TestGameWebSearch:

    def test_returns_list_of_web_search_results(self, mock_tavily_client):
        from src.tools import game_web_search
        results = game_web_search("Elden Ring release date", mock_tavily_client)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_web_search_result_has_required_fields(self, mock_tavily_client):
        from src.tools import game_web_search, WebSearchResult
        results = game_web_search("Elden Ring", mock_tavily_client)
        for r in results:
            assert isinstance(r, WebSearchResult)
            assert hasattr(r, "title")
            assert hasattr(r, "url")
            assert hasattr(r, "content")
            assert hasattr(r, "score")

    def test_query_prepends_video_game_prefix(self, mock_tavily_client):
        from src.tools import game_web_search
        game_web_search("FIFA 21 release date", mock_tavily_client)
        call_kwargs = mock_tavily_client.search.call_args
        query_sent = call_kwargs[1].get("query") or call_kwargs[0][0]
        assert "video game" in query_sent.lower() or "FIFA 21" in query_sent

    def test_tavily_failure_returns_empty_list(self, mock_tavily_client):
        from src.tools import game_web_search
        mock_tavily_client.search.side_effect = Exception("rate limited")
        results = game_web_search("any game", mock_tavily_client)
        assert results == []
