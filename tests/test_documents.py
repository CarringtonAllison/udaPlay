"""Tests for Document and Corpus classes."""
import pytest
from lib.documents import Document, Corpus


class TestDocument:

    def test_document_auto_generates_id(self):
        doc = Document(content="hello")
        assert doc.id is not None
        assert len(doc.id) > 0

    def test_document_stores_content(self):
        doc = Document(content="Gran Turismo is a racing game.")
        assert doc.content == "Gran Turismo is a racing game."

    def test_document_stores_metadata(self):
        meta = {"name": "Gran Turismo", "year": "1997"}
        doc = Document(content="text", metadata=meta)
        assert doc.metadata == meta

    def test_two_documents_have_unique_ids(self):
        doc1 = Document(content="a")
        doc2 = Document(content="b")
        assert doc1.id != doc2.id


class TestCorpus:

    def test_empty_corpus_has_length_zero(self):
        corpus = Corpus()
        assert len(corpus) == 0

    def test_corpus_accepts_documents_at_init(self):
        docs = [Document(content="a"), Document(content="b")]
        corpus = Corpus(docs)
        assert len(corpus) == 2

    def test_corpus_append(self):
        corpus = Corpus()
        corpus.append(Document(content="hello"))
        assert len(corpus) == 1

    def test_corpus_getitem(self):
        doc = Document(content="test")
        corpus = Corpus([doc])
        assert corpus[0].content == "test"

    def test_corpus_rejects_non_document(self):
        corpus = Corpus()
        with pytest.raises(TypeError):
            corpus.append("not a document")

    def test_corpus_to_dict_empty(self):
        corpus = Corpus()
        result = corpus.to_dict()
        assert result == {"contents": [], "metadatas": [], "ids": []}

    def test_corpus_to_dict_single(self):
        doc = Document(content="Gran Turismo", metadata={"name": "GT"})
        corpus = Corpus([doc])
        result = corpus.to_dict()
        assert result["contents"] == ["Gran Turismo"]
        assert result["metadatas"] == [{"name": "GT"}]
        assert result["ids"] == [doc.id]

    def test_corpus_to_dict_multiple(self):
        docs = [Document(content=f"doc {i}") for i in range(3)]
        corpus = Corpus(docs)
        result = corpus.to_dict()
        assert len(result["contents"]) == 3
        assert len(result["ids"]) == 3
