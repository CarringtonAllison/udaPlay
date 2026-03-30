"""VectorStoreManager — ChromaDB persistence and semantic search."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import chromadb


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """A single document returned by a vector store query."""
    game_id: str
    title: str
    document: str
    metadata: dict
    distance: float
    relevance_score: float  # 1 - normalised_distance, in [0, 1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_embed_text(game: dict) -> str:
    """Construct the text string that will be embedded for a game document."""
    platforms = ", ".join(game.get("platforms") or [])
    genres = ", ".join(game.get("genre") or [])
    return (
        f"{game['title']} by {game['developer']}. "
        f"Genre: {genres}. "
        f"Platforms: {platforms}. "
        f"{game['description']}"
    )


def _game_to_metadata(game: dict) -> dict:
    """Flatten a game dict into ChromaDB-compatible metadata (no lists)."""
    return {
        "title": str(game.get("title", "")),
        "developer": str(game.get("developer", "")),
        "publisher": str(game.get("publisher", "")),
        "release_date": str(game.get("release_date", "")),
        "platforms": ", ".join(game.get("platforms") or []),
        "genre": ", ".join(game.get("genre") or []),
        "metacritic_score": int(game.get("metacritic_score") or 0),
        "esrb_rating": str(game.get("esrb_rating", "")),
        "notable_features": ", ".join(game.get("notable_features") or []),
        "source": str(game.get("source", "internal")),
        "source_url": str(game.get("source_url", "")),
    }


def _distance_to_relevance(distance: float, max_distance: float = 2.0) -> float:
    """Convert a ChromaDB L2 distance to a [0, 1] relevance score."""
    normalised = min(distance / max_distance, 1.0)
    return round(1.0 - normalised, 4)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class VectorStoreManager:
    """Manages a ChromaDB collection of game documents.

    Parameters
    ----------
    chroma_client : chromadb.ClientAPI
        A ChromaDB client (persistent or ephemeral).
    embedding_manager : EmbeddingManager
        Used to produce embeddings for documents and queries.
    collection_name : str
        Name of the ChromaDB collection.
    """

    COLLECTION_NAME = "games"

    def __init__(
        self,
        chroma_client: chromadb.ClientAPI,
        embedding_manager,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self._client = chroma_client
        self._embedding_mgr = embedding_manager
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "l2"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_documents(self, games: list[dict]) -> int:
        """Embed and upsert a list of game dicts into the collection.

        Idempotent — upserting the same game_id overwrites the existing record.

        Returns
        -------
        int
            Number of documents upserted.
        """
        if not games:
            return 0

        ids = [game["game_id"] for game in games]
        documents = [_build_embed_text(game) for game in games]
        embeddings = self._embedding_mgr.encode_batch(documents)
        metadatas = [_game_to_metadata(game) for game in games]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(games)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where_filter: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """Semantic search over the collection.

        Parameters
        ----------
        query_text : str
            Natural language query.
        n_results : int
            Maximum number of results to return.
        where_filter : dict, optional
            ChromaDB metadata filter (e.g. ``{"source": {"$eq": "internal"}}``).

        Returns
        -------
        list[RetrievalResult]
            Results ranked by relevance (highest first).
        """
        collection_count = self._collection.count()
        if collection_count == 0:
            return []

        actual_n = min(n_results, collection_count)
        query_embedding = self._embedding_mgr.encode(query_text)

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": actual_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        raw = self._collection.query(**kwargs)

        results: list[RetrievalResult] = []
        ids = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        for game_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            results.append(
                RetrievalResult(
                    game_id=game_id,
                    title=meta.get("title", game_id),
                    document=doc,
                    metadata=meta,
                    distance=dist,
                    relevance_score=_distance_to_relevance(dist),
                )
            )

        # Already sorted by distance (ascending) by ChromaDB, so relevance is descending
        return results

    def get_collection_stats(self) -> dict:
        """Return basic stats about the collection."""
        return {
            "collection_name": self._collection_name,
            "count": self._collection.count(),
        }

    def delete_by_source(self, source: str) -> int:
        """Delete all documents with the given source tag.

        Returns
        -------
        int
            Number of documents deleted.
        """
        # Fetch IDs of matching documents
        results = self._collection.get(
            where={"source": {"$eq": source}},
            include=["metadatas"],
        )
        ids = results.get("ids", [])
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)
