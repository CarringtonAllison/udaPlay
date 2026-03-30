"""Tests for EmbeddingManager."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestEmbeddingManager:
    """Tests for EmbeddingManager class."""

    def test_encode_returns_list_of_floats(self):
        """encode() should return a list of floats (embedding vector)."""
        from src.embeddings import EmbeddingManager
        mgr = EmbeddingManager(backend="sentence-transformers")
        result = mgr.encode("Elden Ring is an open world RPG")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)

    def test_encode_returns_384_dims_for_minilm(self):
        """all-MiniLM-L6-v2 produces 384-dimensional vectors."""
        from src.embeddings import EmbeddingManager
        mgr = EmbeddingManager(backend="sentence-transformers")
        result = mgr.encode("test sentence")
        assert len(result) == 384

    def test_encode_batch_returns_list_of_vectors(self):
        """encode_batch() should return a list of embedding vectors."""
        from src.embeddings import EmbeddingManager
        mgr = EmbeddingManager(backend="sentence-transformers")
        texts = ["Elden Ring", "Hollow Knight", "Stardew Valley"]
        results = mgr.encode_batch(texts)
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(len(v) == 384 for v in results)

    def test_similar_texts_have_higher_cosine_similarity(self):
        """Semantically similar texts should have cosine similarity > dissimilar ones."""
        from src.embeddings import EmbeddingManager
        mgr = EmbeddingManager(backend="sentence-transformers")

        v1 = np.array(mgr.encode("action RPG game with open world"))
        v2 = np.array(mgr.encode("open world role playing adventure"))
        v3 = np.array(mgr.encode("cooking pasta recipe in the kitchen"))

        sim_similar = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        sim_dissimilar = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))

        assert sim_similar > sim_dissimilar

    def test_invalid_backend_raises_value_error(self):
        """Unsupported backend should raise ValueError."""
        from src.embeddings import EmbeddingManager
        with pytest.raises(ValueError, match="Unsupported backend"):
            EmbeddingManager(backend="unsupported-backend")

    def test_openai_backend_mocked(self):
        """OpenAI backend should call the openai client and return a vector."""
        from src.embeddings import EmbeddingManager

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

        with patch("src.embeddings.openai") as mock_openai:
            mock_openai.OpenAI.return_value.embeddings.create.return_value = mock_response
            mgr = EmbeddingManager(backend="openai", api_key="fake-key")
            result = mgr.encode("test")

        assert len(result) == 1536
        assert result[0] == pytest.approx(0.1)

    def test_dimension_property(self):
        """dimension property should return vector size."""
        from src.embeddings import EmbeddingManager
        mgr = EmbeddingManager(backend="sentence-transformers")
        assert mgr.dimension == 384
