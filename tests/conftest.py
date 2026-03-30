"""Shared pytest fixtures for UdaPlay test suite."""
import pytest
from unittest.mock import MagicMock, patch
import chromadb


# ---------------------------------------------------------------------------
# Sample game data
# ---------------------------------------------------------------------------

SAMPLE_GAME = {
    "game_id": "elden-ring",
    "title": "Elden Ring",
    "developer": "FromSoftware",
    "publisher": "Bandai Namco Entertainment",
    "release_date": "2022-02-25",
    "platforms": ["PlayStation 5", "PC", "Xbox Series X/S"],
    "genre": ["Action RPG", "Open World"],
    "description": (
        "Elden Ring is an open-world action RPG developed by FromSoftware "
        "in collaboration with author George R.R. Martin. Players explore the "
        "Lands Between seeking to become Elden Lord."
    ),
    "metacritic_score": 96,
    "esrb_rating": "M",
    "notable_features": ["Open world", "George R.R. Martin lore"],
    "source": "internal",
}

SAMPLE_GAME_2 = {
    "game_id": "hollow-knight",
    "title": "Hollow Knight",
    "developer": "Team Cherry",
    "publisher": "Team Cherry",
    "release_date": "2017-02-24",
    "platforms": ["PC", "Nintendo Switch", "PlayStation 4"],
    "genre": ["Metroidvania", "Action-Platformer"],
    "description": (
        "Hollow Knight is a hand-drawn Metroidvania set in a vast underground "
        "insect kingdom called Hallownest. Players control the Knight exploring "
        "interconnected caverns and battling challenging bosses."
    ),
    "metacritic_score": 87,
    "esrb_rating": "E10+",
    "notable_features": ["Hand-drawn art", "Challenging boss fights"],
    "source": "internal",
}

SAMPLE_GAMES = [SAMPLE_GAME, SAMPLE_GAME_2]


# ---------------------------------------------------------------------------
# ChromaDB — ephemeral (in-memory) client, no disk I/O in tests
# ---------------------------------------------------------------------------

@pytest.fixture
def ephemeral_chroma_client():
    """Return an in-memory ChromaDB client (fresh per test)."""
    return chromadb.EphemeralClient()


# ---------------------------------------------------------------------------
# Sentence-transformer embedding manager — loaded once per test session
# to avoid repeated network calls to HuggingFace hub.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def embedding_manager():
    """Return a session-scoped EmbeddingManager using sentence-transformers."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from src.embeddings import EmbeddingManager
    return EmbeddingManager(backend="sentence-transformers")


# ---------------------------------------------------------------------------
# Mock Anthropic client
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_anthropic_client():
    """Return a MagicMock that mimics the anthropic.Anthropic() interface."""
    client = MagicMock()
    # Default: messages.create returns a text block
    mock_message = MagicMock()
    mock_content_block = MagicMock()
    mock_content_block.type = "text"
    mock_content_block.text = '{"confidence": 0.9, "reason": "Strong match found.", "relevant_ids": ["elden-ring"]}'
    mock_message.content = [mock_content_block]
    mock_message.stop_reason = "end_turn"
    client.messages.create.return_value = mock_message
    return client


# ---------------------------------------------------------------------------
# Mock Tavily client
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_tavily_client():
    """Return a MagicMock that mimics the TavilyClient interface."""
    client = MagicMock()
    client.search.return_value = {
        "results": [
            {
                "title": "Elden Ring - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Elden_Ring",
                "content": (
                    "Elden Ring is a 2022 action RPG developed by FromSoftware "
                    "and published by Bandai Namco. It was released on February 25, 2022."
                ),
                "score": 0.92,
                "raw_content": None,
            },
            {
                "title": "Elden Ring Review - IGN",
                "url": "https://www.ign.com/articles/elden-ring-review",
                "content": "An extraordinary achievement in open-world design. Score: 10/10.",
                "score": 0.85,
                "raw_content": None,
            },
        ]
    }
    return client


# ---------------------------------------------------------------------------
# Schema validation helper (available in all tests via import)
# ---------------------------------------------------------------------------

REQUIRED_GAME_FIELDS = {
    "game_id", "title", "developer", "publisher", "release_date",
    "platforms", "genre", "description", "metacritic_score",
    "esrb_rating", "notable_features", "source",
}


def validate_game_schema(game: dict) -> bool:
    """Return True if game dict contains all required fields with non-None values."""
    return all(field in game and game[field] is not None for field in REQUIRED_GAME_FIELDS)
