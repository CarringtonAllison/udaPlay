"""Tests for VectorStoreManager."""
import uuid
import pytest
from tests.conftest import SAMPLE_GAME, SAMPLE_GAME_2, SAMPLE_GAMES, validate_game_schema


class TestVectorStoreManager:
    """Tests for VectorStoreManager class."""

    @pytest.fixture
    def manager(self, ephemeral_chroma_client, embedding_manager):
        """Return a VectorStoreManager with a unique collection name per test.

        Using a unique collection name ensures complete isolation even when
        chromadb.EphemeralClient shares process-level state between instances.
        """
        from src.vector_store import VectorStoreManager
        unique_name = f"games_{uuid.uuid4().hex[:8]}"
        return VectorStoreManager(
            chroma_client=ephemeral_chroma_client,
            embedding_manager=embedding_manager,
            collection_name=unique_name,
        )

    # ------------------------------------------------------------------
    # upsert_documents
    # ------------------------------------------------------------------

    def test_upsert_returns_count(self, manager):
        """upsert_documents should return the number of documents upserted."""
        count = manager.upsert_documents(SAMPLE_GAMES)
        assert count == len(SAMPLE_GAMES)

    def test_upsert_idempotent(self, manager):
        """Upserting the same documents twice should not increase count."""
        manager.upsert_documents(SAMPLE_GAMES)
        manager.upsert_documents(SAMPLE_GAMES)
        stats = manager.get_collection_stats()
        assert stats["count"] == len(SAMPLE_GAMES)

    def test_upsert_single_game(self, manager):
        """Single-document upsert should work."""
        count = manager.upsert_documents([SAMPLE_GAME])
        assert count == 1
        stats = manager.get_collection_stats()
        assert stats["count"] == 1

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    def test_query_returns_results(self, manager):
        """query() should return a non-empty list of RetrievalResult objects."""
        manager.upsert_documents(SAMPLE_GAMES)
        results = manager.query("open world RPG action game", n_results=2)
        assert len(results) >= 1

    def test_query_result_has_required_fields(self, manager):
        """Each RetrievalResult should have game_id, title, relevance_score, metadata."""
        manager.upsert_documents(SAMPLE_GAMES)
        results = manager.query("action game", n_results=2)
        for r in results:
            assert hasattr(r, "game_id")
            assert hasattr(r, "title")
            assert hasattr(r, "relevance_score")
            assert hasattr(r, "metadata")
            assert hasattr(r, "distance")
            assert hasattr(r, "document")

    def test_query_relevance_score_between_0_and_1(self, manager):
        """relevance_score must be in [0, 1]."""
        manager.upsert_documents(SAMPLE_GAMES)
        results = manager.query("indie platformer game")
        for r in results:
            assert 0.0 <= r.relevance_score <= 1.0

    def test_query_respects_n_results(self, manager):
        """query() should return at most n_results documents."""
        manager.upsert_documents(SAMPLE_GAMES)
        results = manager.query("video game", n_results=1)
        assert len(results) <= 1

    def test_query_semantic_relevance(self, manager):
        """The most relevant result for 'Metroidvania insect kingdom' should be Hollow Knight."""
        manager.upsert_documents(SAMPLE_GAMES)
        results = manager.query("Metroidvania insect kingdom underground", n_results=2)
        assert len(results) >= 1
        top = results[0]
        assert top.game_id == "hollow-knight"

    # ------------------------------------------------------------------
    # get_collection_stats
    # ------------------------------------------------------------------

    def test_stats_empty_collection(self, manager):
        """Stats on empty collection should show count=0."""
        stats = manager.get_collection_stats()
        assert stats["count"] == 0
        assert "collection_name" in stats

    def test_stats_after_upsert(self, manager):
        """Stats count should match number of unique documents upserted."""
        manager.upsert_documents(SAMPLE_GAMES)
        stats = manager.get_collection_stats()
        assert stats["count"] == len(SAMPLE_GAMES)

    # ------------------------------------------------------------------
    # delete_by_source
    # ------------------------------------------------------------------

    def test_delete_by_source_removes_matching_docs(self, manager):
        """delete_by_source('internal') should remove all internal docs."""
        manager.upsert_documents(SAMPLE_GAMES)
        deleted = manager.delete_by_source("internal")
        assert deleted == len(SAMPLE_GAMES)
        stats = manager.get_collection_stats()
        assert stats["count"] == 0

    def test_delete_by_source_nonexistent_returns_zero(self, manager):
        """delete_by_source on a source with no matches should return 0."""
        manager.upsert_documents(SAMPLE_GAMES)
        deleted = manager.delete_by_source("web_search")
        assert deleted == 0
        stats = manager.get_collection_stats()
        assert stats["count"] == len(SAMPLE_GAMES)
