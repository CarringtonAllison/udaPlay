"""Agent tools: retrieve_game, evaluate_retrieval, game_web_search."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_CLAUDE_MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Output of the evaluate_retrieval tool."""
    confidence: float               # 0.0 – 1.0
    reason: str                     # LLM explanation
    should_web_search: bool         # True when confidence < threshold
    sufficient_results: list        # Subset of RetrievalResult deemed relevant


@dataclass
class WebSearchResult:
    """A single result from game_web_search."""
    title: str
    url: str
    content: str
    score: float
    raw_content: Optional[str] = None


# ---------------------------------------------------------------------------
# Tool 1: retrieve_game
# ---------------------------------------------------------------------------

def retrieve_game(
    query: str,
    vector_store,
    n_results: int = 5,
    genre_filter: Optional[str] = None,
) -> list:
    """Search the internal ChromaDB game database.

    Parameters
    ----------
    query : str
        Natural language question or search phrase.
    vector_store : VectorStoreManager
        The vector store to query.
    n_results : int
        Maximum number of results to return.
    genre_filter : str, optional
        If provided, restrict results to documents whose genre metadata
        contains this string (case-sensitive substring match via ``$contains``).

    Returns
    -------
    list[RetrievalResult]
        Semantically ranked results, highest relevance first.
    """
    where_filter = None
    if genre_filter:
        where_filter = {"genre": {"$contains": genre_filter}}

    return vector_store.query(
        query_text=query,
        n_results=n_results,
        where_filter=where_filter,
    )


# ---------------------------------------------------------------------------
# Tool 2: evaluate_retrieval
# ---------------------------------------------------------------------------

_EVAL_SYSTEM = (
    "You are a retrieval evaluation assistant for a video game knowledge base. "
    "Given a user query and a set of retrieved game documents, assess how well "
    "the documents answer the query. "
    "Respond with ONLY valid JSON — no prose, no markdown fences."
)

_EVAL_USER_TEMPLATE = """\
Query: {query}

Retrieved documents:
{documents}

Rate how well these documents answer the query and return JSON:
{{
  "confidence": <float 0.0 to 1.0>,
  "reason": "<brief explanation>",
  "relevant_ids": ["<game_id of relevant docs>"]
}}

Scoring guide:
- 0.9–1.0: Documents directly answer the query with high specificity
- 0.65–0.89: Documents are related and partially answer the query
- 0.3–0.64: Documents are tangentially related; key info may be missing
- 0.0–0.29: Documents do not answer the query
"""


def evaluate_retrieval(
    query: str,
    results: list,
    llm_client,
    threshold: float = 0.65,
) -> EvaluationResult:
    """Ask Claude to assess how well retrieved documents answer the query.

    Parameters
    ----------
    query : str
        The original user query.
    results : list[RetrievalResult]
        Documents returned by retrieve_game.
    llm_client :
        An instantiated ``anthropic.Anthropic`` client.
    threshold : float
        Confidence below this value triggers web search fallback.

    Returns
    -------
    EvaluationResult
    """
    if not results:
        return EvaluationResult(
            confidence=0.0,
            reason="No documents were retrieved to evaluate.",
            should_web_search=True,
            sufficient_results=[],
        )

    formatted_docs = "\n\n".join(
        f"[{r.game_id}] {r.title} (relevance: {r.relevance_score:.2f})\n{r.document[:300]}"
        for r in results
    )

    user_msg = _EVAL_USER_TEMPLATE.format(query=query, documents=formatted_docs)

    try:
        response = llm_client.messages.create(
            model=_CLAUDE_MODEL,
            max_tokens=512,
            system=_EVAL_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()
        parsed = _parse_json(raw)
    except Exception as exc:
        logger.warning("evaluate_retrieval LLM call failed: %s", exc)
        parsed = None

    if parsed is None:
        return EvaluationResult(
            confidence=0.0,
            reason="Evaluation failed — LLM response could not be parsed.",
            should_web_search=True,
            sufficient_results=[],
        )

    confidence = float(parsed.get("confidence", 0.0))
    reason = str(parsed.get("reason", ""))
    relevant_ids = set(parsed.get("relevant_ids", []))
    sufficient = [r for r in results if r.game_id in relevant_ids]

    return EvaluationResult(
        confidence=confidence,
        reason=reason,
        should_web_search=confidence < threshold,
        sufficient_results=sufficient,
    )


# ---------------------------------------------------------------------------
# Tool 3: game_web_search
# ---------------------------------------------------------------------------

def game_web_search(
    query: str,
    tavily_client,
    max_results: int = 5,
    search_depth: str = "advanced",
) -> list[WebSearchResult]:
    """Search the web for game information using the Tavily API.

    The query is automatically prefixed with ``"video game:"`` to bias results
    toward gaming content.

    Parameters
    ----------
    query : str
        The search query (game title, question, etc.).
    tavily_client :
        An instantiated ``TavilyClient``.
    max_results : int
        Maximum number of results to return.
    search_depth : str
        Tavily search depth: ``"basic"`` or ``"advanced"``.

    Returns
    -------
    list[WebSearchResult]
        Search results sorted by Tavily relevance score (highest first).
    """
    prefixed_query = f"video game: {query}"

    try:
        response = tavily_client.search(
            query=prefixed_query,
            max_results=max_results,
            search_depth=search_depth,
        )
    except Exception as exc:
        logger.warning("game_web_search failed for query '%s': %s", query, exc)
        return []

    raw_results = response.get("results", [])
    return [
        WebSearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            content=r.get("content", ""),
            score=float(r.get("score", 0.0)),
            raw_content=r.get("raw_content"),
        )
        for r in raw_results
    ]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> Optional[dict]:
    """Parse JSON from text, stripping markdown fences if present."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Claude tool-use schemas (for use with the messages API tools parameter)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "retrieve_game",
        "description": (
            "Search the internal game knowledge base for information relevant to a user query. "
            "Always call this tool first before considering a web search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "n_results": {"type": "integer", "default": 5},
                "genre_filter": {
                    "type": "string",
                    "description": "Optional genre to narrow results (e.g. 'RPG')",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "evaluate_retrieval",
        "description": (
            "Evaluate how well retrieved documents answer the user query. "
            "Returns a confidence score and whether web search is needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "threshold": {"type": "number", "default": 0.65},
            },
            "required": ["query"],
        },
    },
    {
        "name": "game_web_search",
        "description": (
            "Search the web for video game information using Tavily. "
            "Use only when the internal knowledge base has insufficient information."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Game-related search query"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
]
