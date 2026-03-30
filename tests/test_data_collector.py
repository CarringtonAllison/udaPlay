"""Tests for GameDataCollector."""
import json
import os
import pytest
from unittest.mock import MagicMock, patch
from tests.conftest import REQUIRED_GAME_FIELDS, validate_game_schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tavily_results(content: str = "Elden Ring is a 2022 action RPG."):
    return {
        "results": [
            {
                "title": "Elden Ring - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Elden_Ring",
                "content": content,
                "score": 0.95,
                "raw_content": None,
            }
        ]
    }


def _make_claude_response(json_text: str):
    """Build a mock Anthropic messages.create return value."""
    msg = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = json_text
    msg.content = [block]
    return msg


_VALID_GAME_JSON = json.dumps({
    "game_id": "elden-ring",
    "title": "Elden Ring",
    "developer": "FromSoftware",
    "publisher": "Bandai Namco Entertainment",
    "release_date": "2022-02-25",
    "platforms": ["PlayStation 5", "PC"],
    "genre": ["Action RPG"],
    "description": "Elden Ring is an open-world action RPG developed by FromSoftware.",
    "metacritic_score": 96,
    "esrb_rating": "M",
    "notable_features": ["Open world"],
    "source": "internal",
})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def collector(mock_tavily_client, mock_anthropic_client, tmp_path):
    """Return a GameDataCollector with mocked external clients."""
    from src.data_collector import GameDataCollector
    output_path = str(tmp_path / "games.json")
    return GameDataCollector(
        tavily_client=mock_tavily_client,
        llm_client=mock_anthropic_client,
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGameDataCollector:

    # Slugify ----------------------------------------------------------------

    def test_slugify_basic(self, collector):
        assert collector._slugify("Elden Ring") == "elden-ring"

    def test_slugify_special_chars(self, collector):
        assert collector._slugify("Baldur's Gate 3") == "baldurs-gate-3"

    def test_slugify_colons_and_spaces(self, collector):
        assert collector._slugify("God of War: Ragnarök") == "god-of-war-ragnarok"

    # Schema validation -------------------------------------------------------

    def test_validate_game_schema_passes_valid(self):
        game = json.loads(_VALID_GAME_JSON)
        assert validate_game_schema(game) is True

    def test_validate_game_schema_fails_missing_field(self):
        game = json.loads(_VALID_GAME_JSON)
        del game["developer"]
        assert validate_game_schema(game) is False

    # _extract_with_claude ----------------------------------------------------

    def test_extract_with_claude_returns_dict(self, collector, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = _make_claude_response(_VALID_GAME_JSON)
        result = collector._extract_with_claude("Elden Ring", [{"content": "test"}])
        assert isinstance(result, dict)
        assert result["title"] == "Elden Ring"

    def test_extract_with_claude_handles_invalid_json(self, collector, mock_anthropic_client):
        """If Claude returns invalid JSON, extract_with_claude returns None."""
        mock_anthropic_client.messages.create.return_value = _make_claude_response("not valid json at all")
        result = collector._extract_with_claude("Elden Ring", [{"content": "test"}])
        assert result is None

    def test_extract_with_claude_handles_json_in_markdown_fence(self, collector, mock_anthropic_client):
        """Claude sometimes wraps JSON in markdown fences — should still parse."""
        fenced = f"```json\n{_VALID_GAME_JSON}\n```"
        mock_anthropic_client.messages.create.return_value = _make_claude_response(fenced)
        result = collector._extract_with_claude("Elden Ring", [{"content": "test"}])
        assert result is not None
        assert result["game_id"] == "elden-ring"

    # collect_game ------------------------------------------------------------

    def test_collect_game_returns_valid_schema(self, collector, mock_tavily_client, mock_anthropic_client):
        mock_tavily_client.search.return_value = _make_tavily_results()
        mock_anthropic_client.messages.create.return_value = _make_claude_response(_VALID_GAME_JSON)
        result = collector.collect_game("Elden Ring")
        assert result is not None
        assert validate_game_schema(result)

    def test_collect_game_returns_none_on_tavily_failure(self, collector, mock_tavily_client):
        mock_tavily_client.search.side_effect = Exception("API error")
        result = collector.collect_game("Elden Ring")
        assert result is None

    def test_collect_game_returns_none_on_claude_failure(self, collector, mock_tavily_client, mock_anthropic_client):
        mock_tavily_client.search.return_value = _make_tavily_results()
        mock_anthropic_client.messages.create.return_value = _make_claude_response("bad json")
        result = collector.collect_game("Elden Ring")
        assert result is None

    # collect_all -------------------------------------------------------------

    def test_collect_all_saves_to_file(self, collector, mock_tavily_client, mock_anthropic_client, tmp_path):
        mock_tavily_client.search.return_value = _make_tavily_results()
        mock_anthropic_client.messages.create.return_value = _make_claude_response(_VALID_GAME_JSON)
        results = collector.collect_all(["Elden Ring"], delay_seconds=0)
        assert len(results) == 1
        output_file = tmp_path / "games.json"
        assert output_file.exists()
        saved = json.loads(output_file.read_text())
        assert len(saved) == 1
        assert saved[0]["title"] == "Elden Ring"

    def test_collect_all_skips_duplicates(self, collector, mock_tavily_client, mock_anthropic_client):
        mock_tavily_client.search.return_value = _make_tavily_results()
        mock_anthropic_client.messages.create.return_value = _make_claude_response(_VALID_GAME_JSON)
        # Collect the same game twice
        collector.collect_all(["Elden Ring"], delay_seconds=0)
        results = collector.collect_all(["Elden Ring"], delay_seconds=0)
        # Should still only have 1 entry (deduplicated by game_id)
        assert len(results) == 1

    def test_collect_all_returns_failed_titles(self, collector, mock_tavily_client):
        mock_tavily_client.search.side_effect = Exception("Network error")
        results = collector.collect_all(["Unknown Game XYZ"], delay_seconds=0)
        assert len(results) == 0
        assert len(collector.failed_titles) == 1
        assert "Unknown Game XYZ" in collector.failed_titles
