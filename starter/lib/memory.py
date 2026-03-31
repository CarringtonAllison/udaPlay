from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import copy

from lib.documents import Document, Corpus
from lib.vector_db import VectorStoreManager, QueryResult


class SessionNotFoundError(Exception):
    """Raised when attempting to access a session that doesn't exist"""
    pass


@dataclass
class ShortTermMemory():
    """Manage the history of objects across multiple sessions"""
    sessions: Dict[str, List[Any]] = field(default_factory=lambda: {})

    def __post_init__(self):
        self.create_session("default")

    def __str__(self) -> str:
        session_ids = list(self.sessions.keys())
        return f"Memory(sessions={session_ids})"

    def __repr__(self) -> str:
        return self.__str__()

    def create_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            return False
        self.sessions[session_id] = []
        return True

    def delete_session(self, session_id: str) -> bool:
        if session_id == "default":
            raise ValueError("Cannot delete the default session")
        if session_id not in self.sessions:
            return False
        del self.sessions[session_id]
        return True

    def _validate_session(self, session_id: str):
        if session_id not in self.sessions:
            raise SessionNotFoundError(f"Session '{session_id}' not found")

    def add(self, object: Any, session_id: Optional[str] = None):
        session_id = session_id or "default"
        self._validate_session(session_id)
        self.sessions[session_id].append(copy.deepcopy(object))

    def get_all_objects(self, session_id: Optional[str] = None) -> List[Any]:
        session_id = session_id or "default"
        self._validate_session(session_id)
        return [copy.deepcopy(obj) for obj in self.sessions[session_id]]

    def get_last_object(self, session_id: Optional[str] = None) -> Optional[Any]:
        objects = self.get_all_objects(session_id)
        return objects[-1] if objects else None

    def get_all_sessions(self) -> List[str]:
        return list(self.sessions.keys())

    def reset(self, session_id: Optional[str] = None):
        if session_id is None:
            for sid in self.sessions:
                self.sessions[sid] = []
        else:
            self._validate_session(session_id)
            self.sessions[session_id] = []

    def pop(self, session_id: Optional[str] = None) -> Optional[Any]:
        session_id = session_id or "default"
        self._validate_session(session_id)
        if not self.sessions[session_id]:
            return None
        return self.sessions[session_id].pop()


@dataclass
class MemoryFragment:
    """A single piece of memory information stored in the long-term memory system."""
    content: str
    owner: str
    namespace: str = "default"
    timestamp: int = field(default_factory=lambda: int(datetime.now().timestamp()))


@dataclass
class MemorySearchResult:
    """Container for the results of a memory search operation."""
    fragments: List[MemoryFragment]
    metadata: Dict


@dataclass
class TimestampFilter:
    """Filter criteria for time-based memory searches."""
    greater_than_value: int = None
    lower_than_value: int = None


class LongTermMemory:
    """Manages persistent memory storage and retrieval using vector embeddings."""

    def __init__(self, db: VectorStoreManager):
        self.vector_store = db.create_store("long_term_memory", force=True)

    def get_namespaces(self) -> List[str]:
        results = self.vector_store.get()
        namespaces = [r["metadatas"][0]["namespace"] for r in results]
        return namespaces

    def register(self, memory_fragment: MemoryFragment, metadata: Optional[Dict] = None):
        complete_metadata = {
            "owner": memory_fragment.owner,
            "namespace": memory_fragment.namespace,
            "timestamp": memory_fragment.timestamp,
        }
        if metadata:
            complete_metadata.update(metadata)

        self.vector_store.add(
            Document(
                content=memory_fragment.content,
                metadata=complete_metadata,
            )
        )

    def search(self, query_text: str, owner: str, limit: int = 3,
               timestamp_filter: Optional[TimestampFilter] = None,
               namespace: Optional[str] = "default") -> MemorySearchResult:
        where = {
            "$and": [
                {"namespace": {"$eq": namespace}},
                {"owner": {"$eq": owner}},
            ]
        }

        if timestamp_filter:
            if timestamp_filter.greater_than_value:
                where["$and"].append({"timestamp": {"$gt": timestamp_filter.greater_than_value}})
            if timestamp_filter.lower_than_value:
                where["$and"].append({"timestamp": {"$lt": timestamp_filter.lower_than_value}})

        result = self.vector_store.query(
            query_texts=[query_text],
            n_results=limit,
            where=where
        )

        fragments = []
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        for content, meta in zip(documents, metadatas):
            fragment = MemoryFragment(
                content=content,
                owner=meta.get("owner"),
                namespace=meta.get("namespace", "default"),
                timestamp=meta.get("timestamp")
            )
            fragments.append(fragment)

        return MemorySearchResult(
            fragments=fragments,
            metadata={"distances": result.get("distances", [[]])[0]}
        )
