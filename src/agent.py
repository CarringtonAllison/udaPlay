"""UdaPlayAgent — stateful agent with IDLE→RETRIEVE→EVALUATE→ANSWER/WEBSEARCH pipeline."""
from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.tools import retrieve_game, evaluate_retrieval, game_web_search, WebSearchResult
from src.memory import AgentMemory
from src.reporter import ReportFormatter

logger = logging.getLogger(__name__)

_CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
_CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))


# ---------------------------------------------------------------------------
# State machine types
# ---------------------------------------------------------------------------

class AgentState(Enum):
    IDLE = "idle"
    RETRIEVE = "retrieve"
    EVALUATE = "evaluate"
    WEBSEARCH = "websearch"
    PERSIST = "persist"
    ANSWER = "answer"
    ERROR = "error"


@dataclass
class AgentContext:
    """Shared state bag passed through every state transition."""
    query: str
    session_id: Optional[str] = None
    state: AgentState = AgentState.IDLE
    retrieval_results: list = field(default_factory=list)
    confidence_score: float = 0.0
    confidence_reason: str = ""
    web_results: list = field(default_factory=list)
    final_answer: str = ""
    citations: list = field(default_factory=list)
    answer_format: str = "both"         # "text" | "json" | "both"
    error: Optional[Exception] = None
    turn_count: int = 1


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

_ANSWER_SYSTEM = """\
You are UdaPlay, an expert video game research assistant.
Answer the user's question using the provided context documents.
Be accurate, concise, and cite which game entries informed your answer.
If information is incomplete, say so honestly."""

_ANSWER_USER_TEMPLATE = """\
User question: {query}

Context from knowledge base:
{context}

Provide a clear, factual answer (2-4 paragraphs).
End with a brief one-line summary prefixed with "Summary: "."""


class UdaPlayAgent:
    """Stateful video game research agent.

    Workflow
    --------
    IDLE → RETRIEVE → EVALUATE → ANSWER          (high confidence RAG hit)
    IDLE → RETRIEVE → EVALUATE → WEBSEARCH → PERSIST → ANSWER  (low confidence)

    Parameters
    ----------
    vector_store : VectorStoreManager
        The ChromaDB-backed game knowledge base.
    llm_client :
        An instantiated ``anthropic.Anthropic`` client.
    tavily_client :
        An instantiated ``TavilyClient``.
    confidence_threshold : float
        Minimum confidence to answer from RAG without web search.
    answer_format : str
        Default output format for all runs: ``"text"``, ``"json"``, or ``"both"``.
    """

    def __init__(
        self,
        vector_store,
        llm_client,
        tavily_client,
        confidence_threshold: float = _CONFIDENCE_THRESHOLD,
        answer_format: str = "both",
    ) -> None:
        self._vector_store = vector_store
        self._llm = llm_client
        self._tavily = tavily_client
        self._threshold = confidence_threshold
        self._default_format = answer_format
        self.memory = AgentMemory()
        self._reporter = ReportFormatter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        answer_format: Optional[str] = None,
    ) -> AgentContext:
        """Execute the agent pipeline for a single query.

        Parameters
        ----------
        query : str
            The user's question.
        session_id : str, optional
            Identifies a multi-turn conversation.  A new UUID is generated
            if not provided.
        answer_format : str, optional
            Override the default output format for this run.

        Returns
        -------
        AgentContext
            The final context object with ``final_answer``, ``citations``,
            and ``confidence_score`` populated.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        fmt = answer_format or self._default_format

        # Count turns in this session
        history = self.memory.get_session_history(session_id)
        turn = len(history) + 1

        ctx = AgentContext(
            query=query,
            session_id=session_id,
            state=AgentState.IDLE,
            answer_format=fmt,
            turn_count=turn,
        )

        # State machine loop
        while ctx.state not in (AgentState.ANSWER, AgentState.ERROR):
            ctx = self._step(ctx)

        # Ensure we always end in ANSWER (recover from ERROR)
        if ctx.state == AgentState.ERROR:
            ctx = self._handle_error_recovery(ctx)

        self.memory.record_turn(session_id, ctx)
        return ctx

    # ------------------------------------------------------------------
    # State dispatch
    # ------------------------------------------------------------------

    def _step(self, ctx: AgentContext) -> AgentContext:
        handlers = {
            AgentState.IDLE: self._handle_retrieve,
            AgentState.RETRIEVE: self._handle_evaluate,
            AgentState.EVALUATE: self._handle_after_evaluate,
            AgentState.WEBSEARCH: self._handle_persist,
            AgentState.PERSIST: self._handle_answer,
        }
        handler = handlers.get(ctx.state)
        if handler is None:
            ctx.state = AgentState.ANSWER
            return ctx
        return handler(ctx)

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_retrieve(self, ctx: AgentContext) -> AgentContext:
        """IDLE → RETRIEVE: Query the vector store."""
        try:
            results = retrieve_game(ctx.query, self._vector_store, n_results=5)
            ctx.retrieval_results = results
            ctx.state = AgentState.RETRIEVE
            logger.info("Retrieved %d results for query: %s", len(results), ctx.query)
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            ctx.error = exc
            ctx.retrieval_results = []
            ctx.state = AgentState.RETRIEVE  # continue to evaluate with empty results
        return ctx

    def _handle_evaluate(self, ctx: AgentContext) -> AgentContext:
        """RETRIEVE → EVALUATE: Score result quality."""
        try:
            eval_result = evaluate_retrieval(
                query=ctx.query,
                results=ctx.retrieval_results,
                llm_client=self._llm,
                threshold=self._threshold,
            )
            ctx.confidence_score = eval_result.confidence
            ctx.confidence_reason = eval_result.reason
            ctx.state = AgentState.EVALUATE
        except Exception as exc:
            logger.error("Evaluation failed: %s", exc)
            ctx.error = exc
            ctx.confidence_score = 0.0
            ctx.state = AgentState.EVALUATE
        return ctx

    def _handle_after_evaluate(self, ctx: AgentContext) -> AgentContext:
        """EVALUATE → ANSWER or WEBSEARCH based on confidence."""
        if ctx.confidence_score >= self._threshold:
            ctx.state = AgentState.PERSIST  # skip websearch, go straight to answer via persist
            return self._handle_answer(ctx)
        ctx.state = AgentState.WEBSEARCH
        return self._handle_websearch(ctx)

    def _handle_websearch(self, ctx: AgentContext) -> AgentContext:
        """EVALUATE → WEBSEARCH: Fall back to Tavily search."""
        try:
            web_results = game_web_search(ctx.query, self._tavily)
            ctx.web_results = web_results
            logger.info("Web search returned %d results", len(web_results))
        except Exception as exc:
            logger.warning("Web search failed: %s", exc)
            ctx.web_results = []
        ctx.state = AgentState.WEBSEARCH
        return ctx

    def _handle_persist(self, ctx: AgentContext) -> AgentContext:
        """WEBSEARCH → PERSIST: Save web results to vector store for long-term memory."""
        if ctx.web_results:
            try:
                docs = [_web_result_to_game_doc(r, ctx.query) for r in ctx.web_results]
                self._vector_store.upsert_documents(docs)
                logger.info("Persisted %d web results to vector store", len(docs))
            except Exception as exc:
                logger.warning("Persistence failed: %s", exc)
        ctx.state = AgentState.PERSIST
        return ctx

    def _handle_answer(self, ctx: AgentContext) -> AgentContext:
        """PERSIST → ANSWER: Generate the final answer using Claude."""
        try:
            # Build context from both RAG results and web results
            context_parts = []
            citations = []

            for r in ctx.retrieval_results[:3]:
                context_parts.append(
                    f"[Internal] {r.title}\n{r.document[:400]}"
                )
                citations.append({
                    "title": r.title,
                    "url": None,
                    "source": "internal",
                })

            for wr in ctx.web_results[:2]:
                context_parts.append(
                    f"[Web] {wr.title}\n{wr.content[:400]}\nSource: {wr.url}"
                )
                citations.append({
                    "title": wr.title,
                    "url": wr.url,
                    "source": "web_search",
                })

            context_text = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."

            user_msg = _ANSWER_USER_TEMPLATE.format(
                query=ctx.query,
                context=context_text,
            )

            response = self._llm.messages.create(
                model=_CLAUDE_MODEL,
                max_tokens=1024,
                system=_ANSWER_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw_answer = response.content[0].text.strip()
            ctx.final_answer = raw_answer
            ctx.citations = citations

        except Exception as exc:
            logger.error("Answer generation failed: %s", exc)
            ctx.error = exc
            ctx.final_answer = (
                f"I encountered an error generating a response. "
                f"Please try rephrasing your question. (Error: {exc})"
            )
            ctx.citations = []

        ctx.state = AgentState.ANSWER
        return ctx

    def _handle_error_recovery(self, ctx: AgentContext) -> AgentContext:
        """ERROR → ANSWER: Produce a graceful error message."""
        if not ctx.final_answer:
            ctx.final_answer = (
                "I'm sorry, I encountered an unexpected error processing your request. "
                "Please try again."
            )
        ctx.state = AgentState.ANSWER
        return ctx

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_formatted_answer(self, ctx: AgentContext) -> object:
        """Return the answer in the format specified by ctx.answer_format."""
        return self._reporter.format(
            answer=ctx.final_answer,
            citations=ctx.citations,
            mode=ctx.answer_format,
            confidence=ctx.confidence_score,
            query=ctx.query,
        )


# ---------------------------------------------------------------------------
# Helper: convert a WebSearchResult to a game document for persistence
# ---------------------------------------------------------------------------

def _web_result_to_game_doc(result: WebSearchResult, original_query: str) -> dict:
    """Normalise a web search result into the game schema for ChromaDB storage."""
    import re
    import time as _time

    # Create a slug from the title
    slug = re.sub(r"[^\w\s-]", "", result.title.lower())
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = slug.strip("-")[:50]

    return {
        "game_id": f"web-{slug}-{int(_time.time())}",
        "title": result.title,
        "developer": "Unknown",
        "publisher": "Unknown",
        "release_date": "Unknown",
        "platforms": ["Unknown"],
        "genre": ["Unknown"],
        "description": result.content[:800],
        "metacritic_score": 0,
        "esrb_rating": "Unknown",
        "notable_features": [],
        "source": "web_search",
        "source_url": result.url,
        "original_query": original_query,
    }
