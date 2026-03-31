from typing import List
import pdfplumber
from lib.documents import Corpus, Document


class PDFLoader:
    """
    Document loader for extracting text content from PDF files.

    Each page of the PDF becomes a separate Document object, enabling
    page-level search and retrieval in RAG applications.

    Example:
        >>> loader = PDFLoader("research_paper.pdf")
        >>> corpus = loader.load()
        >>> print(f"Loaded {len(corpus)} pages")
    """
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def load(self) -> Corpus:
        corpus = Corpus()

        with pdfplumber.open(self.pdf_path) as pdf:
            for num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    corpus.append(
                        Document(
                            id=str(num),
                            content=text
                        )
                    )
        return corpus
