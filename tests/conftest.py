"""Shared pytest fixtures for UdaPlay test suite."""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Add starter/ to path so `lib` imports work (lib lives at starter/lib/)
STARTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'starter'))
if STARTER_DIR not in sys.path:
    sys.path.insert(0, STARTER_DIR)


# ---------------------------------------------------------------------------
# Sample game data (matches course schema)
# ---------------------------------------------------------------------------

SAMPLE_GAME = {
    "Name": "Gran Turismo",
    "Platform": "PlayStation 1",
    "Genre": "Racing",
    "Publisher": "Sony Computer Entertainment",
    "Description": "A realistic racing simulator featuring a wide array of cars and tracks.",
    "YearOfRelease": 1997,
}

SAMPLE_GAME_2 = {
    "Name": "Super Mario 64",
    "Platform": "Nintendo 64",
    "Genre": "Platformer",
    "Publisher": "Nintendo",
    "Description": "A groundbreaking 3D platformer that set new standards for the genre.",
    "YearOfRelease": 1996,
}

SAMPLE_GAMES = [SAMPLE_GAME, SAMPLE_GAME_2]


# ---------------------------------------------------------------------------
# Mock OpenAI client
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_openai_client():
    """Return a MagicMock that mimics the openai.OpenAI() interface."""
    client = MagicMock()

    # Default: chat.completions.create returns a simple text response
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Gran Turismo was published by Sony Computer Entertainment."
    mock_message.tool_calls = None
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150
    )
    client.chat.completions.create.return_value = mock_response
    client.beta.chat.completions.parse.return_value = mock_response
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
                "title": "Top Games 2025",
                "url": "https://example.com/games-2025",
                "content": "The most anticipated games of 2025 include GTA VI and many others.",
                "score": 0.88,
            },
        ]
    }
    return client


# ---------------------------------------------------------------------------
# Schema validation helper
# ---------------------------------------------------------------------------

REQUIRED_GAME_FIELDS = {"Name", "Platform", "Genre", "Publisher", "Description", "YearOfRelease"}


def validate_game_schema(game: dict) -> bool:
    """Return True if game dict contains all required fields."""
    return all(field in game and game[field] is not None for field in REQUIRED_GAME_FIELDS)
