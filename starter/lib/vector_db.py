from typing import List, Optional, Dict, Any, Union
from typing_extensions import TypedDict
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.models.Collection import Collection as ChromaCollection
from chromadb.api.types import EmbeddingFunction, QueryResult, GetResult

from lib.loaders import PDFLoader
from lib.documents import Document, Corpus


class VectorStore:
    """High-level interface for vector database operations using ChromaDB."""

    def __init__(self, chroma_collection: ChromaCollection):
        self._collection = chroma_collection

    def add(self, item: Union[Document, Corpus, List[Document]]):
        """Add documents to the vector store with automatic embedding generation."""
        if isinstance(item, Document):
            item = Corpus([item])
        elif isinstance(item, list):
            if not all(isinstance(doc, Document) for doc in item):
                raise TypeError("List must contain Document objects only.")
            item = Corpus(item)
        elif not isinstance(item, Corpus):
            raise TypeError("item must be Document, Corpus, or List[Document].")

        item_dict = item.to_dict()

        self._collection.add(
            documents=item_dict["contents"],
            ids=item_dict["ids"],
            metadatas=item_dict["metadatas"]
        )

    def query(self, query_texts: str | List[str], n_results: int = 3,
              where: Optional[Dict[str, Any]] = None,
              where_document: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Perform semantic similarity search against stored documents."""
        return self._collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=['documents', 'distances', 'metadatas']
        )

    def get(self, ids: Optional[List[str]] = None,
            where: Optional[Dict[str, Any]] = None,
            limit: Optional[int] = None) -> GetResult:
        """Retrieve documents by ID or metadata filters without similarity search."""
        return self._collection.get(
            ids=ids,
            where=where,
            limit=limit,
            include=['documents', 'metadatas']
        )


class VectorStoreManager:
    """Factory and lifecycle manager for ChromaDB vector stores."""

    def __init__(self, openai_api_key: str, chroma_path: str = "./chroma_db"):
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.chroma_path = chroma_path
        self.embedding_function = self._create_embedding_function(openai_api_key)

    def _create_embedding_function(self, api_key: str) -> EmbeddingFunction:
        embeddings_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key
        )
        return embeddings_fn

    def __repr__(self):
        return f"VectorStoreManager(path={self.chroma_path}):{self.chroma_client}"

    def get_store(self, name: str) -> Optional[VectorStore]:
        try:
            chroma_collection = self.chroma_client.get_collection(name)
            return VectorStore(chroma_collection)
        except Exception:
            return None

    def create_store(self, store_name: str, force: bool = False) -> VectorStore:
        if force:
            self.delete_store(store_name)

        try:
            chroma_collection = self.chroma_client.create_collection(
                name=store_name,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            print(f"Pass `force=True` or use `get_or_create_store` method")

        return VectorStore(chroma_collection)

    def get_or_create_store(self, store_name: str) -> VectorStore:
        chroma_collection = self.chroma_client.get_or_create_collection(
            name=store_name,
            embedding_function=self.embedding_function
        )
        return VectorStore(chroma_collection)

    def delete_store(self, store_name: str):
        try:
            self.chroma_client.delete_collection(name=store_name)
        except Exception:
            pass


class CorpusLoaderService:
    """Service for loading documents from various sources into vector stores."""

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.manager = vector_store_manager

    def load_pdf(self, store_name: str, pdf_path: str) -> VectorStore:
        """Load a PDF file into a vector store."""
        store = self.manager.get_or_create_store(store_name)
        print(f"VectorStore `{store_name}` ready!")

        loader = PDFLoader(pdf_path)
        document = loader.load()
        store.add(document)
        print(f"Pages from `{pdf_path}` added!")

        return store
