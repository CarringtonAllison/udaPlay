"""EmbeddingManager — wraps sentence-transformers or OpenAI embeddings."""
import os
from typing import Optional


# Lazy import sentinel so openai is only required when explicitly used
try:
    import openai as _openai_module
except ImportError:
    _openai_module = None  # type: ignore

# Expose for mocking in tests
openai = _openai_module


class EmbeddingManager:
    """Produces text embeddings using either sentence-transformers or OpenAI.

    Parameters
    ----------
    backend : str
        ``"sentence-transformers"`` (default) or ``"openai"``.
    model_name : str, optional
        Override the default model for the chosen backend.
    api_key : str, optional
        API key for the OpenAI backend.  Falls back to the ``OPENAI_API_KEY``
        environment variable when not supplied.
    """

    _SENTENCE_TRANSFORMERS_DEFAULT = "all-MiniLM-L6-v2"
    _OPENAI_DEFAULT = "text-embedding-3-small"
    _SUPPORTED_BACKENDS = {"sentence-transformers", "openai"}

    def __init__(
        self,
        backend: str = "sentence-transformers",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        if backend not in self._SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. "
                f"Choose from: {self._SUPPORTED_BACKENDS}"
            )
        self._backend = backend
        self._model = None
        self._openai_client = None
        self._dim: Optional[int] = None

        if backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            self._model_name = model_name or self._SENTENCE_TRANSFORMERS_DEFAULT
            self._model = SentenceTransformer(self._model_name)
            # Determine dimension by encoding a dummy string once
            self._dim = len(self._model.encode("ping").tolist())

        elif backend == "openai":
            if openai is None:
                raise ImportError(
                    "openai package is required for the 'openai' backend. "
                    "Run: pip install openai"
                )
            self._model_name = model_name or self._OPENAI_DEFAULT
            resolved_key = api_key or os.getenv("OPENAI_API_KEY")
            self._openai_client = openai.OpenAI(api_key=resolved_key)
            # OpenAI text-embedding-3-small → 1536 dims
            self._dim = 1536

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self._dim  # type: ignore[return-value]

    def encode(self, text: str) -> list[float]:
        """Embed a single text string.

        Parameters
        ----------
        text : str
            The text to embed.

        Returns
        -------
        list[float]
            A flat list of floats representing the embedding vector.
        """
        if self._backend == "sentence-transformers":
            return self._model.encode(text).tolist()

        # OpenAI path
        response = self._openai_client.embeddings.create(
            model=self._model_name,
            input=text,
        )
        return response.data[0].embedding

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.

        Returns
        -------
        list[list[float]]
            One embedding vector per input text.
        """
        if self._backend == "sentence-transformers":
            return [v.tolist() for v in self._model.encode(texts)]

        # OpenAI path — batches up to 2048 inputs per call
        response = self._openai_client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        # Sort by index to preserve input order
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]
