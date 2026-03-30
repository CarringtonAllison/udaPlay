"""GameDataCollector — builds game JSON from Tavily search + Claude extraction."""
from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema & prompt
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "game_id", "title", "developer", "publisher", "release_date",
    "platforms", "genre", "description", "metacritic_score",
    "esrb_rating", "notable_features", "source",
]

_EXTRACTION_SYSTEM_PROMPT = (
    "You are a game data extraction assistant. "
    "Extract structured game metadata from web search results. "
    "Return ONLY valid JSON — no prose, no markdown fences."
)

_EXTRACTION_USER_TEMPLATE = """\
Game title: "{title}"

Search results:
{results}

Extract and return JSON with exactly these fields:
{{
  "game_id": "<url-slug, e.g. elden-ring>",
  "title": "<exact title>",
  "developer": "<studio name>",
  "publisher": "<publisher name>",
  "release_date": "<YYYY-MM-DD or 'Unknown'>",
  "platforms": ["<platform>"],
  "genre": ["<genre>"],
  "description": "<2-4 sentence factual summary>",
  "metacritic_score": <integer 0-100, or 0 if unknown>,
  "esrb_rating": "<E / E10+ / T / M / AO / Unknown>",
  "notable_features": ["<short feature>"],
  "source": "internal"
}}
"""

_CLAUDE_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# GameDataCollector
# ---------------------------------------------------------------------------

class GameDataCollector:
    """Collects game metadata via Tavily search + Claude JSON extraction.

    Parameters
    ----------
    tavily_client :
        An instantiated ``TavilyClient``.
    llm_client :
        An instantiated ``anthropic.Anthropic`` client.
    output_path : str
        Path where the collected ``games.json`` will be written.
    """

    def __init__(
        self,
        tavily_client,
        llm_client,
        output_path: str = "data/games/games.json",
    ) -> None:
        self._tavily = tavily_client
        self._llm = llm_client
        self._output_path = Path(output_path)
        self.failed_titles: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_game(self, title: str) -> Optional[dict]:
        """Search for a game title and extract structured metadata.

        Returns
        -------
        dict or None
            A game document matching the schema, or ``None`` if extraction
            failed.
        """
        try:
            search_results = self._search(title)
        except Exception as exc:
            logger.warning("Tavily search failed for '%s': %s", title, exc)
            return None

        game = self._extract_with_claude(title, search_results)
        if game is None:
            logger.warning("Claude extraction failed for '%s'", title)
            return None

        if not self._validate(game):
            logger.warning("Schema validation failed for '%s'", title)
            return None

        return game

    def collect_all(
        self,
        titles: list[str],
        delay_seconds: float = 1.0,
    ) -> list[dict]:
        """Collect metadata for each title in *titles*.

        Results are written to ``output_path``.  Existing entries with the
        same ``game_id`` are preserved (upsert semantics).

        Returns
        -------
        list[dict]
            All successfully collected games (including previously saved ones).
        """
        existing = self._load_existing()
        existing_ids = {g["game_id"] for g in existing}
        self.failed_titles = []

        new_games: list[dict] = []
        for title in titles:
            slug = self._slugify(title)
            if slug in existing_ids:
                logger.info("Skipping '%s' — already collected.", title)
                continue

            game = self.collect_game(title)
            if game is None:
                self.failed_titles.append(title)
            else:
                new_games.append(game)
                existing_ids.add(game["game_id"])

            if delay_seconds > 0 and title != titles[-1]:
                time.sleep(delay_seconds)

        all_games = existing + new_games
        self._save(all_games)
        return all_games

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search(self, title: str) -> list[dict]:
        """Run a Tavily search focused on game metadata."""
        query = f"{title} video game developer publisher release date platforms genre"
        response = self._tavily.search(query=query, max_results=3, search_depth="advanced")
        return response.get("results", [])

    def _extract_with_claude(self, title: str, search_results: list[dict]) -> Optional[dict]:
        """Ask Claude to extract game schema from search result snippets."""
        formatted = "\n\n".join(
            f"[{i+1}] {r.get('title', '')}\n{r.get('content', '')}"
            for i, r in enumerate(search_results)
        )
        user_msg = _EXTRACTION_USER_TEMPLATE.format(
            title=title,
            results=formatted,
        )

        try:
            response = self._llm.messages.create(
                model=_CLAUDE_MODEL,
                max_tokens=1024,
                system=_EXTRACTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw_text = response.content[0].text.strip()
        except Exception as exc:
            logger.warning("LLM call failed for '%s': %s", title, exc)
            return None

        return self._parse_json(raw_text)

    def _parse_json(self, text: str) -> Optional[dict]:
        """Parse JSON, stripping markdown fences if present."""
        # Strip ```json ... ``` or ``` ... ``` fences
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _validate(self, game: dict) -> bool:
        """Return True if all required fields are present and non-None."""
        return all(field in game and game[field] is not None for field in REQUIRED_FIELDS)

    def _slugify(self, title: str) -> str:
        """Convert a game title to a URL-safe slug."""
        # Normalise unicode (e.g. ö → o, ä → a)
        nfkd = unicodedata.normalize("NFKD", title)
        ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
        slug = ascii_str.lower()
        slug = re.sub(r"[^\w\s-]", "", slug)   # remove punctuation
        slug = re.sub(r"[\s_]+", "-", slug)      # spaces/underscores → hyphens
        slug = re.sub(r"-+", "-", slug)          # collapse multiple hyphens
        return slug.strip("-")

    def _load_existing(self) -> list[dict]:
        """Load already-collected games from output_path, or return []."""
        if self._output_path.exists():
            try:
                return json.loads(self._output_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save(self, games: list[dict]) -> None:
        """Write games list to output_path as JSON."""
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(
            json.dumps(games, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
