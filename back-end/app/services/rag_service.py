from dataclasses import dataclass
import hashlib
import logging
import os
import tempfile
from typing import Callable, List, Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.cache import SemanticFAQCache, semantic_faq_cache
from app.core.config import settings
from app.core.exceptions import DocumentIngestionError, NovaDeskError, VectorStoreError
from app.core.vectordb import get_embeddings, get_vector_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """Represents a single retrieved knowledge-base chunk with metadata."""

    doc_id: str
    content: str
    source: str
    page: int | str


@dataclass(frozen=True)
class RAGRetrievalResult:
    """
    Structured output of a RAG retrieval operation.
    Provides both `formatted_context` (for LLM system prompts) and `raw_contexts` /
    `retrieved_doc_ids` (required by DeepEval for Precision@K, Recall@K, MRR, and Faithfulness).
    """

    query: str
    chunks: List[RetrievedChunk]
    formatted_context: str

    @property
    def raw_contexts(self) -> List[str]:
        """List of raw chunk strings consumed by DeepEval `LLMTestCase.retrieval_context`."""
        return [chunk.content for chunk in self.chunks]

    @property
    def retrieved_doc_ids(self) -> List[str]:
        """List of deterministic chunk IDs for IR ranking metrics (MRR, Precision@K, Recall@K)."""
        return [chunk.doc_id for chunk in self.chunks]


@dataclass(frozen=True)
class IngestionResult:
    """Summary of a completed document ingestion into the vector store."""

    filename: str
    chunks_ingested: int
    chunk_ids: List[str]


@dataclass(frozen=True)
class FAQAnswerResult:
    """Result of answering an FAQ query (either from Semantic Cache or live RAG + LLM)."""

    answer: str
    raw_contexts: List[str]
    cache_hit: bool
    match_type: Optional[str] = None  # "exact" | "semantic" | None


class RAGService:
    """
    Dedicated service encapsulating Knowledge Base PDF ingestion, deterministic
    chunk deduplication, vector similarity retrieval, and Semantic FAQ Caching.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        cache: Optional[SemanticFAQCache] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        default_top_k: Optional[int] = None,
    ) -> None:
        self._vector_store = vector_store
        self.cache = cache if cache is not None else semantic_faq_cache
        self.chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP
        self.default_top_k = default_top_k or settings.RAG_TOP_K

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store

    @staticmethod
    def compute_chunk_id(source: str, page: int | str, chunk_index: int, content: str) -> str:
        """
        Generate a deterministic SHA-256 ID for a document chunk.
        Passing deterministic IDs to Pinecone makes re-uploading the same PDF an
        idempotent upsert instead of creating duplicate vectors.
        """
        normalized = f"{source.strip().lower()}|page:{page}|idx:{chunk_index}|{content.strip()}"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def format_chunks(chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into the prompt context string expected by the FAQ LLM."""
        if not chunks:
            return "No relevant FAQ documents found."
        formatted_sections = [
            f"Page Number: {chunk.page}\n Title: {chunk.source}\nContent: {chunk.content}"
            for chunk in chunks
        ]
        return "\n\n---\n\n".join(formatted_sections)

    def ingest_pdf(self, *, file_bytes: bytes, filename: str) -> IngestionResult:
        """
        Validate, parse, chunk, and deterministically upsert a PDF file into Pinecone.
        """
        if not filename or not filename.lower().endswith(".pdf"):
            raise DocumentIngestionError(
                "Only PDF files (.pdf) are supported for FAQ ingestion.",
                details={"filename": filename},
            )

        if not file_bytes:
            raise DocumentIngestionError(
                "The uploaded PDF file is empty.",
                details={"filename": filename},
            )

        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name

            loader = PyMuPDFLoader(tmp_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = filename

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            split_docs = text_splitter.split_documents(documents)
            if not split_docs:
                raise DocumentIngestionError(
                    "No extractable text chunks were found in the uploaded PDF.",
                    details={"filename": filename},
                )

            chunk_ids: List[str] = []
            for idx, doc in enumerate(split_docs):
                page = doc.metadata.get("page", 0)
                chunk_id = self.compute_chunk_id(
                    source=filename,
                    page=page,
                    chunk_index=idx,
                    content=doc.page_content,
                )
                doc.metadata["chunk_id"] = chunk_id
                chunk_ids.append(chunk_id)

            try:
                self.vector_store.add_documents(split_docs, ids=chunk_ids)
            except Exception as exc:
                logger.error(
                    "Failed to upsert %d chunks for '%s' into Pinecone: %s",
                    len(split_docs),
                    filename,
                    exc,
                    exc_info=True,
                )
                raise VectorStoreError(
                    "Failed to index document chunks into the knowledge base.",
                    details={"filename": filename},
                ) from exc

            # Invalidate the Semantic FAQ Cache so newly uploaded knowledge takes effect immediately
            self.cache.clear()

            logger.info("Ingested %d chunks from '%s' into Pinecone.", len(split_docs), filename)
            return IngestionResult(
                filename=filename,
                chunks_ingested=len(split_docs),
                chunk_ids=chunk_ids,
            )

        except NovaDeskError:
            raise
        except Exception as exc:
            logger.error("Failed to parse or ingest PDF '%s': %s", filename, exc, exc_info=True)
            raise DocumentIngestionError(
                f"Failed to process PDF '{filename}'.",
                details={"filename": filename},
            ) from exc
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError as rm_err:
                    logger.warning("Failed to remove temporary file %s: %s", tmp_path, rm_err)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RAGRetrievalResult:
        """
        Retrieve the top-K most relevant FAQ chunks for `query` and return a
        structured `RAGRetrievalResult` containing both raw chunks and formatted text.
        """
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return RAGRetrievalResult(
                query="",
                chunks=[],
                formatted_context="No relevant FAQ documents found.",
            )

        effective_k = top_k or self.default_top_k
        try:
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": effective_k},
            )
            retrieved_docs: List[Document] = retriever.invoke(cleaned_query)
        except Exception as exc:
            logger.error("Vector store retrieval failed for query '%s': %s", cleaned_query, exc, exc_info=True)
            raise VectorStoreError(
                "Unable to search the knowledge base at the moment.",
                details={"query": cleaned_query},
            ) from exc

        chunks: List[RetrievedChunk] = []
        for idx, doc in enumerate(retrieved_docs):
            source = doc.metadata.get("source", "N/A")
            page = doc.metadata.get("page", "N/A")
            doc_id = (
                doc.metadata.get("chunk_id")
                or getattr(doc, "id", None)
                or self.compute_chunk_id(source=str(source), page=page, chunk_index=idx, content=doc.page_content)
            )
            chunks.append(
                RetrievedChunk(
                    doc_id=str(doc_id),
                    content=doc.page_content,
                    source=str(source),
                    page=page,
                )
            )

        formatted_context = self.format_chunks(chunks)
        return RAGRetrievalResult(
            query=cleaned_query,
            chunks=chunks,
            formatted_context=formatted_context,
        )

    def answer_faq(
        self,
        *,
        query: str,
        generate_fn: Callable[[str], str],
    ) -> FAQAnswerResult:
        """
        Answer an FAQ query using the 2-stage Semantic Cache first:
        1. If the query hits the Exact or Semantic Cache (cosine similarity >= threshold),
           return the cached base answer and raw contexts immediately (0 Pinecone & 0 LLM calls).
        2. On a cache miss, retrieve from Pinecone, invoke `generate_fn(formatted_context)`
           to synthesize the base answer, store it in the Semantic Cache, and return it.
        """
        embed_fn = get_embeddings().embed_query
        cached_hit = self.cache.lookup(query, embed_fn=embed_fn)
        if cached_hit is not None:
            return FAQAnswerResult(
                answer=cached_hit.answer,
                raw_contexts=cached_hit.raw_contexts,
                cache_hit=True,
                match_type=cached_hit.match_type,
            )

        retrieval_result = self.retrieve(query)
        base_answer = generate_fn(retrieval_result.formatted_context)

        self.cache.store(
            query=query,
            answer=base_answer,
            raw_contexts=retrieval_result.raw_contexts,
            embed_fn=embed_fn,
        )
        return FAQAnswerResult(
            answer=base_answer,
            raw_contexts=retrieval_result.raw_contexts,
            cache_hit=False,
            match_type=None,
        )
