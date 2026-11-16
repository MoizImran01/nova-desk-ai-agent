from collections import OrderedDict
from dataclasses import dataclass
import logging
import re
from threading import RLock
import time
from typing import Any, Callable, Generic, List, Optional, TypeVar

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


def normalize_query_text(text: str) -> str:
    """
    Normalize a user query by lowercasing, stripping punctuation, and collapsing
    whitespace so minor phrasing differences ('How much is Botox?' vs 'how much is botox')
    hit the exact cache without needing an embedding API call.
    """
    lowered = (text or "").strip().lower()
    cleaned = re.sub(r"[^\w\s]", "", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


class TTLCache(Generic[T]):
    """
    Thread-safe, bounded In-Memory LRU + Time-To-Live (TTL) cache.
    Used for Tier-1 caching (e.g., Service catalog and query embeddings).
    """

    def __init__(self, *, max_size: int = 512, ttl_seconds: float = 900.0) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> Optional[T]:
        now = time.monotonic()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            expires_at, value = item
            if now >= expires_at:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: T) -> None:
        expires_at = time.monotonic() + self.ttl_seconds
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (expires_at, value)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


@dataclass(frozen=True)
class SemanticCacheEntry:
    """Stores a cached FAQ response alongside its normalized vector and raw RAG chunks."""

    original_query: str
    normalized_query: str
    unit_vector: np.ndarray
    answer: str
    raw_contexts: List[str]
    created_at: float


@dataclass(frozen=True)
class SemanticCacheHit:
    """Returned when either an exact or semantic similarity match is found."""

    answer: str
    raw_contexts: List[str]
    similarity_score: float
    matched_query: str
    match_type: str  # "exact" | "semantic"


class SemanticFAQCache:
    """
    Two-stage FAQ Cache (Exact Normalized Match + Cosine Similarity Vector Match):
    1. Stage 1 (Exact Normalized Match): Checks if `normalize_query_text(query)` matches
       an existing cached entry. Costs 0 API calls (no embedding, no Pinecone, no LLM).
    2. Stage 2 (Semantic Vector Match): Embeds the query (using the Tier-1 embedding
       cache first) and computes cosine similarity against all non-expired FAQ vectors.
       If the highest similarity >= `SEMANTIC_CACHE_THRESHOLD` (default 0.92), returns
       the cached base answer and raw contexts (skipping Pinecone and LLM generation).
    """

    def __init__(
        self,
        *,
        similarity_threshold: Optional[float] = None,
        ttl_seconds: float = 86400.0,
        max_entries: int = 500,
    ) -> None:
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.SEMANTIC_CACHE_THRESHOLD
        )
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: OrderedDict[str, SemanticCacheEntry] = OrderedDict()
        self._embedding_cache: TTLCache[np.ndarray] = TTLCache(
            max_size=1000,
            ttl_seconds=ttl_seconds,
        )
        self._lock = RLock()

    @staticmethod
    def _to_unit_vector(vector: List[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0.0:
            return arr
        return arr / norm

    def _evict_expired(self, now: float) -> None:
        expired_keys = [
            key
            for key, entry in self._entries.items()
            if (now - entry.created_at) >= self.ttl_seconds
        ]
        for key in expired_keys:
            del self._entries[key]

    def get_or_compute_unit_vector(
        self,
        normalized_query: str,
        embed_fn: Callable[[str], List[float]],
    ) -> np.ndarray:
        """Fetch the query's unit embedding vector from Tier-1 cache or compute it once."""
        cached_vec = self._embedding_cache.get(normalized_query)
        if cached_vec is not None:
            return cached_vec

        raw_vector = embed_fn(normalized_query)
        unit_vec = self._to_unit_vector(raw_vector)
        self._embedding_cache.set(normalized_query, unit_vec)
        return unit_vec

    def lookup(
        self,
        query: str,
        embed_fn: Callable[[str], List[float]],
    ) -> Optional[SemanticCacheHit]:
        """
        Attempt to resolve `query` from the cache:
        - First checks Stage 1 (exact normalized string match — 0 API calls).
        - Next checks Stage 2 (cosine similarity >= threshold).
        """
        normalized = normalize_query_text(query)
        if not normalized:
            return None

        now = time.monotonic()
        with self._lock:
            self._evict_expired(now)
            if not self._entries:
                return None

            # Stage 1: Exact normalized match (zero embedding API cost)
            exact_entry = self._entries.get(normalized)
            if exact_entry is not None:
                self._entries.move_to_end(normalized)
                logger.info(
                    "SemanticFAQCache [EXACT HIT] for query='%s'",
                    query,
                )
                return SemanticCacheHit(
                    answer=exact_entry.answer,
                    raw_contexts=list(exact_entry.raw_contexts),
                    similarity_score=1.0,
                    matched_query=exact_entry.original_query,
                    match_type="exact",
                )

            snapshot_entries = list(self._entries.values())

        # Stage 2: Semantic cosine similarity check
        try:
            query_unit_vec = self.get_or_compute_unit_vector(normalized, embed_fn)
        except Exception as exc:
            logger.warning("Failed to compute query embedding for semantic cache lookup: %s", exc)
            return None

        best_entry: Optional[SemanticCacheEntry] = None
        best_score: float = -1.0

        for entry in snapshot_entries:
            score = float(np.dot(query_unit_vec, entry.unit_vector))
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= self.similarity_threshold:
            with self._lock:
                if best_entry.normalized_query in self._entries:
                    self._entries.move_to_end(best_entry.normalized_query)
            logger.info(
                "SemanticFAQCache [SEMANTIC HIT] score=%.4f (threshold=%.2f) | query='%s' matched='%s'",
                best_score,
                self.similarity_threshold,
                query,
                best_entry.original_query,
            )
            return SemanticCacheHit(
                answer=best_entry.answer,
                raw_contexts=list(best_entry.raw_contexts),
                similarity_score=best_score,
                matched_query=best_entry.original_query,
                match_type="semantic",
            )

        logger.debug(
            "SemanticFAQCache [MISS] best_score=%.4f (threshold=%.2f) for query='%s'",
            best_score,
            self.similarity_threshold,
            query,
        )
        return None

    def store(
        self,
        *,
        query: str,
        answer: str,
        raw_contexts: List[str],
        embed_fn: Callable[[str], List[float]],
    ) -> None:
        """Store a base FAQ answer and its raw retrieval contexts in the semantic cache."""
        normalized = normalize_query_text(query)
        if not normalized or not answer:
            return

        try:
            unit_vec = self.get_or_compute_unit_vector(normalized, embed_fn)
        except Exception as exc:
            logger.warning("Skipping semantic cache store due to embedding error: %s", exc)
            return

        entry = SemanticCacheEntry(
            original_query=query,
            normalized_query=normalized,
            unit_vector=unit_vec,
            answer=answer,
            raw_contexts=list(raw_contexts),
            created_at=time.monotonic(),
        )
        with self._lock:
            if normalized in self._entries:
                self._entries.move_to_end(normalized)
            self._entries[normalized] = entry
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Invalidate all cached FAQ entries (called when new FAQ PDFs are uploaded)."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
        logger.info("Cleared %d entries from SemanticFAQCache.", count)


# ---------------------------------------------------------------------------
# Singleton Application Caches
# ---------------------------------------------------------------------------

# Tier-1 Cache: Spa Service Catalog (TTL: 15 minutes)
service_catalog_cache: TTLCache[List[Any]] = TTLCache(max_size=8, ttl_seconds=900.0)

# Tier-2 Cache: Semantic FAQ Response Cache (TTL: 24 hours)
semantic_faq_cache = SemanticFAQCache(
    similarity_threshold=settings.SEMANTIC_CACHE_THRESHOLD,
    ttl_seconds=86400.0,
    max_entries=500,
)
