"""
Phase 1: Custom DeepEval Judge Wrapper & RAG Evaluation Metrics.

This module provides:
1. `RateLimitedGeminiJudge` — A custom `DeepEvalBaseLLM` wrapper around
   `gemini-3.1-flash-lite` that enforces a strict per-minute pacing interval
   (default ~12.5 RPM) and retries automatically on HTTP 429 rate-limit responses.
2. Deterministic Information Retrieval (IR) metrics implementing DeepEval's
   `BaseMetric` interface (`PrecisionAtKMetric`, `RecallAtKMetric`, `MRRMetric`)
   scored against the ground-truth `evidence` IDs and quotes in `goldens.jsonl`.
3. `build_rag_metrics()` — Factory that assembles all 5 metrics:
   - Precision@K (Retrieval)
   - Recall@K (Retrieval)
   - MRR (Retrieval)
   - Faithfulness (Generation — judged by Gemini 3.1 Flash Lite)
   - Answer Relevancy (Generation — judged by Gemini 3.1 Flash Lite)
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Type

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.metrics import AnswerRelevancyMetric, BaseMetric, FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Rate-Limited Gemini Judge for DeepEval (`gemini-3.1-flash-lite`)
# ---------------------------------------------------------------------------

def _strip_markdown_json(text: str) -> str:
    """
    Strip optional ```json ... ``` markdown code fences if the judge LLM wraps
    a raw JSON response in backticks.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


class RateLimitedGeminiJudge(DeepEvalBaseLLM):
    """
    Custom DeepEval LLM wrapper for Google's `gemini-3.1-flash-lite`.

    Why this wrapper is necessary:
    - Google AI Studio's free tier enforces a strict 15 Requests Per Minute (RPM)
      ceiling on `gemini-3.1-flash-lite`.
    - A single DeepEval question makes ~5 internal judge LLM calls across
      `FaithfulnessMetric` and `AnswerRelevancyMetric`.
    - This wrapper serializes judge calls with a minimum interval (default 4.8s
      = 12.5 RPM) and automatically backs off if a 429 quota error ever occurs.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.1-flash-lite",
        google_api_key: Optional[str] = None,
        min_interval_seconds: float = 4.8,
        max_retries_on_429: int = 4,
    ) -> None:
        self._model_name = model_name
        self._google_api_key = google_api_key
        self._min_interval = min_interval_seconds
        self._max_retries = max_retries_on_429
        self._lock = threading.Lock()
        self._last_call_ts: float = 0.0
        self.total_calls: int = 0
        super().__init__(model_name)

    def load_model(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=self._model_name,
            google_api_key=self._google_api_key,
            temperature=0.0,
            max_retries=0,
            timeout=30.0,
        )

    def _wait_for_rate_limit_slot(self) -> None:
        """Ensure at least `self._min_interval` seconds elapse between judge calls."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_ts
            if elapsed < self._min_interval and self._last_call_ts > 0:
                sleep_for = self._min_interval - elapsed
                time.sleep(sleep_for)
            self._last_call_ts = time.monotonic()
            self.total_calls += 1

    def generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Any:
        """
        Synchronous generation entrypoint called by DeepEval metrics.
        When `schema` (a Pydantic `BaseModel` subclass) is supplied by DeepEval,
        we bind `.with_structured_output(schema)` so Gemini returns a validated
        Pydantic object directly.
        """
        chat_model = self.load_model()

        for attempt in range(self._max_retries + 1):
            self._wait_for_rate_limit_slot()
            try:
                if schema is not None:
                    structured_llm = chat_model.with_structured_output(schema)
                    result = structured_llm.invoke(prompt)
                    if result is not None:
                        return result
                    # Fallback if structured output returned None: ask for raw JSON and validate
                    raw_response = chat_model.invoke(prompt)
                    cleaned_json = _strip_markdown_json(str(raw_response.content))
                    return schema.model_validate_json(cleaned_json)

                raw_response = chat_model.invoke(prompt)
                return _strip_markdown_json(str(raw_response.content))

            except Exception as exc:
                err_text = str(exc)
                is_retriable = any(
                    token in err_text
                    for token in (
                        "429",
                        "RESOURCE_EXHAUSTED",
                        "Connection reset",
                        "Errno 54",
                        "Server disconnected",
                        "UNAVAILABLE",
                        "503",
                    )
                )
                if is_retriable and attempt < self._max_retries:
                    backoff = 10.0 * (attempt + 1)
                    logger.warning(
                        "Gemini judge hit transient error '%s' (attempt %d/%d). Sleeping %.1fs...",
                        err_text[:120],
                        attempt + 1,
                        self._max_retries,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue
                raise

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Any:
        """
        Async generation entrypoint called by DeepEval when `async_mode` is active.
        Delegates to synchronous `generate()` so all calls share the same rate-limit lock.
        """
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return self._model_name


# ---------------------------------------------------------------------------
# 2. Ground-Truth Evidence Matching Helpers for Retrieval Metrics
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Normalize bullets, whitespace, and smart quotes/dashes for reliable quote matching."""
    replaced = (
        text.replace("•", " ")
        .replace("–", "-")
        .replace("—", "-")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    return re.sub(r"\s+", " ", replaced).strip().lower()


def _chunk_matches_evidence(
    chunk_id: str,
    chunk_text: str,
    evidence_item: Dict[str, Any],
) -> bool:
    """
    Return True if a retrieved chunk matches a ground-truth `evidence_item`
    either by exact deterministic `chunk_id` OR by containing the ground-truth `quote`.
    """
    expected_id = str(evidence_item.get("id") or "").strip()
    if expected_id and chunk_id and chunk_id.strip() == expected_id:
        return True

    quote = str(evidence_item.get("quote") or "").strip()
    if quote and chunk_text:
        return _normalize_text(quote) in _normalize_text(chunk_text)

    return False


def _extract_retrieval_inputs(
    test_case: LLMTestCase,
    k: int,
) -> tuple[List[str], List[str], List[Dict[str, Any]], str]:
    """
    Extract `(top_k_contexts, top_k_doc_ids, evidence_list, category)` from an `LLMTestCase`.
    """
    contexts = list(test_case.retrieval_context or [])[:k]
    metadata = test_case.additional_metadata or {}
    doc_ids_raw = list(metadata.get("retrieved_doc_ids") or [])[:k]
    # Pad doc_ids with empty strings if shorter than contexts
    doc_ids = [
        str(doc_ids_raw[i]) if i < len(doc_ids_raw) else ""
        for i in range(len(contexts))
    ]
    evidence = list(metadata.get("evidence") or [])
    category = str(metadata.get("category") or "direct")
    return contexts, doc_ids, evidence, category


# ---------------------------------------------------------------------------
# 3. Deterministic Information Retrieval Metrics (`BaseMetric` Subclasses)
# ---------------------------------------------------------------------------

class PrecisionAtKMetric(BaseMetric):
    """
    Information Retrieval Precision@K Metric:
    Measures the proportion of retrieved top-K chunks that are relevant to the
    ground-truth `evidence` for the question.

    Also computes Normalized Precision@K (`relevant_chunks / min(K, len(evidence))`)
    in the reason string because when a question only has 1 relevant chunk in the
    entire document and K=4, raw `relevant_chunks / K` has a mathematical ceiling of 0.25.
    """

    def __init__(self, k: int = 4, threshold: float = 0.25) -> None:
        self.k = k
        self.threshold = threshold
        self.score: float = 0.0
        self.reason: str = ""
        self.success: bool = False
        self.evaluation_cost: float = 0.0

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        contexts, doc_ids, evidence, category = _extract_retrieval_inputs(test_case, self.k)

        if not evidence or category == "unanswerable":
            self.score = 1.0
            self.success = True
            self.reason = "Unanswerable question (no ground-truth evidence expected)."
            return self.score

        if not contexts:
            self.score = 0.0
            self.success = False
            self.reason = "No chunks were retrieved."
            return self.score

        relevant_chunk_count = 0
        for chunk_id, chunk_text in zip(doc_ids, contexts):
            if any(_chunk_matches_evidence(chunk_id, chunk_text, ev) for ev in evidence):
                relevant_chunk_count += 1

        effective_k = len(contexts)
        raw_precision = relevant_chunk_count / effective_k
        max_possible = min(effective_k, len(evidence))
        normalized_precision = min(1.0, relevant_chunk_count / max_possible) if max_possible > 0 else 0.0

        self.score = round(raw_precision, 4)
        self.success = self.score >= self.threshold
        self.reason = (
            f"{relevant_chunk_count}/{effective_k} retrieved chunks matched ground-truth evidence "
            f"(Raw Precision@{self.k}={raw_precision:.2f}, Normalized Precision@{self.k}={normalized_precision:.2f})."
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return f"Precision@{self.k}"


class RecallAtKMetric(BaseMetric):
    """
    Information Retrieval Recall@K Metric:
    Measures the fraction of ground-truth `evidence` items that were successfully
    retrieved within the top-K chunks:
        Recall@K = (matched distinct evidence items) / (total ground-truth evidence items)
    """

    def __init__(self, k: int = 4, threshold: float = 0.8) -> None:
        self.k = k
        self.threshold = threshold
        self.score: float = 0.0
        self.reason: str = ""
        self.success: bool = False
        self.evaluation_cost: float = 0.0

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        contexts, doc_ids, evidence, category = _extract_retrieval_inputs(test_case, self.k)

        if not evidence or category == "unanswerable":
            self.score = 1.0
            self.success = True
            self.reason = "Unanswerable question (no ground-truth evidence expected)."
            return self.score

        matched_evidence = 0
        missing_quotes: List[str] = []
        for ev in evidence:
            found = any(
                _chunk_matches_evidence(chunk_id, chunk_text, ev)
                for chunk_id, chunk_text in zip(doc_ids, contexts)
            )
            if found:
                matched_evidence += 1
            else:
                missing_quotes.append(str(ev.get("quote") or ev.get("id")))

        self.score = round(matched_evidence / len(evidence), 4)
        self.success = self.score >= self.threshold
        if missing_quotes:
            self.reason = (
                f"Found {matched_evidence}/{len(evidence)} ground-truth evidence items in top-{self.k}. "
                f"Missed: {missing_quotes}"
            )
        else:
            self.reason = f"Found all {matched_evidence}/{len(evidence)} ground-truth evidence items in top-{self.k}."
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return f"Recall@{self.k}"


class MRRMetric(BaseMetric):
    """
    Mean Reciprocal Rank (MRR@K) Metric:
    Measures how high up the FIRST relevant chunk appears in the retrieved list:
        RR = 1 / rank_of_first_relevant_chunk  (1-indexed; 0.0 if none found)
    """

    def __init__(self, k: int = 4, threshold: float = 0.5) -> None:
        self.k = k
        self.threshold = threshold
        self.score: float = 0.0
        self.reason: str = ""
        self.success: bool = False
        self.evaluation_cost: float = 0.0

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        contexts, doc_ids, evidence, category = _extract_retrieval_inputs(test_case, self.k)

        if not evidence or category == "unanswerable":
            self.score = 1.0
            self.success = True
            self.reason = "Unanswerable question (no ground-truth evidence expected)."
            return self.score

        for rank_idx, (chunk_id, chunk_text) in enumerate(zip(doc_ids, contexts), start=1):
            if any(_chunk_matches_evidence(chunk_id, chunk_text, ev) for ev in evidence):
                self.score = round(1.0 / rank_idx, 4)
                self.success = self.score >= self.threshold
                self.reason = f"First relevant evidence chunk appeared at rank #{rank_idx} (RR={self.score:.2f})."
                return self.score

        self.score = 0.0
        self.success = False
        self.reason = f"No relevant evidence chunk appeared in the top-{self.k} retrieved chunks."
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return f"MRR@{self.k}"


# ---------------------------------------------------------------------------
# 4. Metric Suite Factory
# ---------------------------------------------------------------------------

def build_rag_metrics(
    judge_llm: RateLimitedGeminiJudge,
    top_k: int = 4,
) -> List[BaseMetric]:
    """
    Build the complete 5-metric DeepEval RAG suite:
    1. Precision@K (Retrieval)
    2. Recall@K (Retrieval)
    3. MRR@K (Retrieval)
    4. FaithfulnessMetric (Generation — judged by `gemini-3.1-flash-lite`)
    5. AnswerRelevancyMetric (Generation — judged by `gemini-3.1-flash-lite`)
    """
    return [
        PrecisionAtKMetric(k=top_k, threshold=0.25),
        RecallAtKMetric(k=top_k, threshold=0.80),
        MRRMetric(k=top_k, threshold=0.50),
        FaithfulnessMetric(
            threshold=0.80,
            model=judge_llm,
            include_reason=True,
            async_mode=False,
        ),
        AnswerRelevancyMetric(
            threshold=0.80,
            model=judge_llm,
            include_reason=True,
            async_mode=False,
        ),
    ]
