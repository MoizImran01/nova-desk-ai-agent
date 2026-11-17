"""
Phase 2: Knowledge Base Ingestion & RAG Prediction Generator (`predictions.jsonl`).

What this script does:
1. Ensures `📋 Med Spa FAQs.pdf` is ingested into the hosted Pinecone index using
   `Gemini Embedding 1` (`models/gemini-embedding-001`) and deterministic chunk IDs.
2. Reads the 30 ground-truth test cases from `back-end/evals/rag/data/goldens.jsonl`.
3. For each test case (skipping any already saved in `predictions.jsonl`):
   - Retrieves the top-K chunks from Pinecone via `RAGService.retrieve()`.
   - Generates an answer using `gemini-3.5-flash-lite` with the production `FAQ_SYSTEM_PROMPT`.
   - Enforces a 5.0-second pacing interval (~12 RPM) to stay safely under the 15 RPM limit.
   - Immediately appends and flushes the prediction to `back-end/evals/rag/data/predictions.jsonl`
     so `gemini-3.5-flash-lite` quota (387/500 RPD) is never wasted if interrupted.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure `back-end` is on sys.path so `app.*` imports resolve cleanly
BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agent.prompts import FAQ_SYSTEM_PROMPT
from app.core.config import settings
from app.services.rag_service import RAGService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evals.rag.generate")

DATA_DIR = Path(__file__).resolve().parent / "data"
GOLDENS_PATH = DATA_DIR / "goldens.jsonl"
PREDICTIONS_PATH = DATA_DIR / "predictions.jsonl"
DEFAULT_PDF_PATH = REPO_ROOT / "📋 Med Spa FAQs.pdf"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def ingest_faq_pdf(rag_service: RAGService, pdf_path: Path) -> None:
    """
    Ingest `📋 Med Spa FAQs.pdf` into Pinecone with deterministic SHA-256 chunk IDs.
    Because `RAGService.compute_chunk_id` uses `filename`, we pass `pdf_path.name`
    so the generated chunk IDs match the ground-truth IDs in `goldens.jsonl`.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"FAQ PDF not found at: {pdf_path}")

    file_bytes = pdf_path.read_bytes()
    logger.info("Ingesting '%s' (%d bytes) into Pinecone index '%s'...", pdf_path.name, len(file_bytes), settings.PINECONE_INDEX_NAME)
    result = rag_service.ingest_pdf(file_bytes=file_bytes, filename=pdf_path.name)
    logger.info(
        "Ingestion complete: %d chunks upserted into Pinecone (first chunk_id=%s).",
        result.chunks_ingested,
        result.chunk_ids[0] if result.chunk_ids else "none",
    )


def build_generator_chain(model_name: str = "gemini-3.5-flash-lite"):
    """
    Build the answer generation chain using `gemini-3.5-flash-lite` and the
    production `FAQ_SYSTEM_PROMPT`.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", FAQ_SYSTEM_PROMPT),
        ("human", "{query}"),
    ])
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.GOOGLE_GEMINI_API_KEY,
        temperature=0.0,
        max_retries=0,
        timeout=30.0,
    )
    return prompt | llm | StrOutputParser()


def invoke_with_rate_limit(
    chain,
    payload: Dict[str, Any],
    min_interval_seconds: float = 5.0,
    last_call_ts: float = 0.0,
    max_retries: int = 4,
) -> tuple[str, float]:
    """
    Invoke the generator chain with a strict minimum interval (~12 RPM) and
    automatic retry backoff if HTTP 429 (`RESOURCE_EXHAUSTED`) occurs.
    """
    for attempt in range(max_retries + 1):
        now = time.monotonic()
        elapsed = now - last_call_ts
        if last_call_ts > 0 and elapsed < min_interval_seconds:
            time.sleep(min_interval_seconds - elapsed)

        call_ts = time.monotonic()
        try:
            answer = chain.invoke(payload)
            return str(answer).strip(), call_ts
        except Exception as exc:
            err_str = str(exc)
            if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries:
                backoff = 15.0 * (attempt + 1)
                logger.warning(
                    "Generator hit 429 rate limit (attempt %d/%d). Sleeping %.1fs...",
                    attempt + 1,
                    max_retries,
                    backoff,
                )
                time.sleep(backoff)
                last_call_ts = time.monotonic()
                continue
            raise
    raise RuntimeError("Unreachable")


def generate_predictions(
    *,
    top_k: int = 4,
    generator_model: str = "gemini-3.5-flash-lite",
    skip_ingest: bool = False,
    force_regenerate: bool = False,
    min_interval_seconds: float = 5.0,
) -> List[Dict[str, Any]]:
    """
    Run retrieval (`Gemini Embedding 1` + Pinecone) and generation (`gemini-3.5-flash-lite`)
    for all 30 golden questions, saving each result incrementally to `predictions.jsonl`.
    """
    rag_service = RAGService(default_top_k=top_k)

    if not skip_ingest:
        probe = rag_service.retrieve("What are the clinic's Saturday hours?", top_k=1)
        if probe.chunks:
            logger.info(
                "Pinecone index '%s' already contains FAQ vectors (probe doc_id=%s). Skipping re-ingestion.",
                settings.PINECONE_INDEX_NAME,
                probe.chunks[0].doc_id,
            )
        else:
            ingest_faq_pdf(rag_service, DEFAULT_PDF_PATH)
            time.sleep(2.0)

    goldens = load_jsonl(GOLDENS_PATH)
    if not goldens:
        raise RuntimeError(f"No golden test cases found at {GOLDENS_PATH}")

    existing_predictions = {} if force_regenerate else {
        row["id"]: row for row in load_jsonl(PREDICTIONS_PATH)
    }

    if force_regenerate and PREDICTIONS_PATH.exists():
        PREDICTIONS_PATH.unlink()

    generator_chain = build_generator_chain(model_name=generator_model)
    last_call_ts = 0.0
    all_predictions: List[Dict[str, Any]] = []

    logger.info(
        "Starting prediction pass across %d golden questions (already cached: %d, top_k=%d, model=%s)...",
        len(goldens),
        len(existing_predictions),
        top_k,
        generator_model,
    )

    with PREDICTIONS_PATH.open("a", encoding="utf-8") as out_file:
        for idx, golden in enumerate(goldens, start=1):
            qid = golden["id"]
            if qid in existing_predictions:
                logger.info("[%02d/%02d] %s — already cached in predictions.jsonl, skipping.", idx, len(goldens), qid)
                all_predictions.append(existing_predictions[qid])
                continue

            question = golden["question"]
            logger.info("[%02d/%02d] %s (%s) — retrieving top-%d chunks & generating answer...", idx, len(goldens), qid, golden["category"], top_k)

            # 1. Retrieve top-K chunks directly from Pinecone using Gemini Embedding 1
            retrieval = rag_service.retrieve(question, top_k=top_k)

            # 2. Generate answer using Gemini 3.5 Flash Lite + FAQ_SYSTEM_PROMPT
            actual_output, last_call_ts = invoke_with_rate_limit(
                generator_chain,
                {
                    "retrieved_faqs": retrieval.formatted_context,
                    "query": question,
                },
                min_interval_seconds=min_interval_seconds,
                last_call_ts=last_call_ts,
            )

            record = {
                "id": qid,
                "split": golden.get("split", "development"),
                "category": golden.get("category", "direct"),
                "question": question,
                "expected": golden.get("expected", ""),
                "evidence": golden.get("evidence", []),
                "required_facts": golden.get("required_facts", []),
                "top_k": top_k,
                "generator_model": generator_model,
                "retrieved_doc_ids": retrieval.retrieved_doc_ids,
                "retrieval_context": retrieval.raw_contexts,
                "retrieved_chunks_meta": [
                    {
                        "doc_id": chunk.doc_id,
                        "page": chunk.page,
                        "source": chunk.source,
                    }
                    for chunk in retrieval.chunks
                ],
                "actual_output": actual_output,
            }

            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()
            all_predictions.append(record)

    logger.info("All %d predictions saved to %s", len(all_predictions), PREDICTIONS_PATH)
    return all_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest FAQ PDF and generate RAG predictions for DeepEval.")
    parser.add_argument("--top-k", type=int, default=settings.RAG_TOP_K, help="Number of chunks to retrieve per query.")
    parser.add_argument("--model", type=str, default="gemini-3.5-flash-lite", help="Generator model name.")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip re-ingesting the FAQ PDF into Pinecone.")
    parser.add_argument("--force", action="store_true", help="Regenerate all predictions even if cached.")
    parser.add_argument("--interval", type=float, default=5.0, help="Minimum seconds between generator LLM calls.")
    args = parser.parse_args()

    generate_predictions(
        top_k=args.top_k,
        generator_model=args.model,
        skip_ingest=args.skip_ingest,
        force_regenerate=args.force,
        min_interval_seconds=args.interval,
    )


if __name__ == "__main__":
    main()
