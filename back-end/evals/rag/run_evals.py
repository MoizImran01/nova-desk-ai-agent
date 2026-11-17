"""
Phase 3: DeepEval RAG Evaluation Runner & Aggregator (`run_evals.py`).

What this script does:
1. Loads the generated predictions from `back-end/evals/rag/data/predictions.jsonl`
   (or runs `generate_predictions()` automatically if `predictions.jsonl` is missing).
2. Builds a `deepeval.test_case.LLMTestCase` for each of the 30 golden questions.
3. Evaluates each test case across all 5 metrics using `RateLimitedGeminiJudge`
   (`gemini-3.1-flash-lite`):
   - `Precision@K` (Retrieval)
   - `Recall@K` (Retrieval)
   - `MRR@K` (Retrieval)
   - `Faithfulness` (Generation — DeepEval `FaithfulnessMetric`)
   - `Answer Relevancy` (Generation — DeepEval `AnswerRelevancyMetric`)
4. Checkpoints each evaluated question incrementally to `eval_results_cache.jsonl`
   and writes the complete aggregated report to `back-end/evals/rag/data/eval_results.json`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

# Ensure `back-end` is on sys.path so `app.*` and `evals.*` imports resolve cleanly
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from deepeval.test_case import LLMTestCase

from app.core.config import settings
from evals.rag.generate_predictions import (
    PREDICTIONS_PATH,
    generate_predictions,
    load_jsonl,
)
from evals.rag.metrics import (
    RateLimitedGeminiJudge,
    build_rag_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evals.rag.runner")

DATA_DIR = Path(__file__).resolve().parent / "data"
EVAL_CACHE_PATH = DATA_DIR / "eval_results_cache.jsonl"
EVAL_REPORT_PATH = DATA_DIR / "eval_results.json"


def build_llm_test_case(prediction: Dict[str, Any]) -> LLMTestCase:
    """Convert a prediction record into a DeepEval `LLMTestCase`."""
    return LLMTestCase(
        input=prediction["question"],
        actual_output=prediction["actual_output"],
        expected_output=prediction["expected"],
        retrieval_context=prediction["retrieval_context"],
        additional_metadata={
            "id": prediction["id"],
            "split": prediction.get("split", "development"),
            "category": prediction.get("category", "direct"),
            "evidence": prediction.get("evidence", []),
            "required_facts": prediction.get("required_facts", []),
            "retrieved_doc_ids": prediction.get("retrieved_doc_ids", []),
            "retrieved_chunks_meta": prediction.get("retrieved_chunks_meta", []),
        },
    )


def evaluate_single_test_case(
    prediction: Dict[str, Any],
    judge_llm: RateLimitedGeminiJudge,
    top_k: int,
) -> Dict[str, Any]:
    """
    Run all 5 DeepEval metrics on a single `LLMTestCase` and return a structured
    dictionary of scores, pass/fail statuses, and judge explanations.
    """
    test_case = build_llm_test_case(prediction)
    metrics = build_rag_metrics(judge_llm=judge_llm, top_k=top_k)

    metric_results: Dict[str, Dict[str, Any]] = {}
    for metric in metrics:
        metric_name = getattr(metric, "__name__", metric.__class__.__name__)
        try:
            score = metric.measure(test_case)
            metric_results[metric_name] = {
                "score": round(float(score), 4) if score is not None else 0.0,
                "passed": bool(metric.is_successful()),
                "reason": str(getattr(metric, "reason", "") or ""),
                "threshold": float(getattr(metric, "threshold", 0.0)),
            }
        except Exception as exc:
            logger.error(
                "Metric '%s' failed on question %s: %s",
                metric_name,
                prediction["id"],
                exc,
                exc_info=True,
            )
            metric_results[metric_name] = {
                "score": 0.0,
                "passed": False,
                "reason": f"Evaluation error: {exc}",
                "threshold": float(getattr(metric, "threshold", 0.0)),
            }

    return {
        "id": prediction["id"],
        "split": prediction.get("split", "development"),
        "category": prediction.get("category", "direct"),
        "question": prediction["question"],
        "expected": prediction["expected"],
        "actual_output": prediction["actual_output"],
        "evidence": prediction.get("evidence", []),
        "required_facts": prediction.get("required_facts", []),
        "retrieved_doc_ids": prediction.get("retrieved_doc_ids", []),
        "retrieved_chunks_meta": prediction.get("retrieved_chunks_meta", []),
        "retrieval_context": prediction.get("retrieval_context", []),
        "metrics": metric_results,
    }


def _aggregate_group(rows: List[Dict[str, Any]], metric_names: List[str]) -> Dict[str, Any]:
    """Compute average score and pass rate for each metric across `rows`."""
    if not rows:
        return {"count": 0, "metrics": {}}

    summary: Dict[str, Any] = {}
    for m_name in metric_names:
        scores = [r["metrics"][m_name]["score"] for r in rows if m_name in r.get("metrics", {})]
        passes = [1 if r["metrics"][m_name]["passed"] else 0 for r in rows if m_name in r.get("metrics", {})]
        if scores:
            summary[m_name] = {
                "mean_score": round(sum(scores) / len(scores), 4),
                "pass_rate": round(sum(passes) / len(passes), 4),
                "passed_count": sum(passes),
                "total_count": len(scores),
            }
    return {
        "count": len(rows),
        "metrics": summary,
    }


def build_aggregated_report(
    evaluated_rows: List[Dict[str, Any]],
    top_k: int,
    generator_model: str,
    judge_model: str,
    total_judge_calls: int,
) -> Dict[str, Any]:
    """
    Build the complete evaluation report with overall, answerable-only, per-split,
    and per-category aggregations.
    """
    metric_names = [
        f"Precision@{top_k}",
        f"Recall@{top_k}",
        f"MRR@{top_k}",
        "Faithfulness",
        "Answer Relevancy",
    ]

    answerable_rows = [r for r in evaluated_rows if r.get("category") != "unanswerable"]

    by_split_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_category_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in evaluated_rows:
        by_split_groups[row.get("split", "development")].append(row)
        by_category_groups[row.get("category", "direct")].append(row)

    by_split = {
        split: _aggregate_group(rows, metric_names)
        for split, rows in sorted(by_split_groups.items())
    }
    by_category = {
        cat: _aggregate_group(rows, metric_names)
        for cat, rows in sorted(by_category_groups.items())
    }

    return {
        "config": {
            "top_k": top_k,
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "pinecone_index": settings.PINECONE_INDEX_NAME,
            "generator_model": generator_model,
            "judge_model": judge_model,
            "total_questions": len(evaluated_rows),
            "answerable_questions": len(answerable_rows),
            "unanswerable_questions": len(evaluated_rows) - len(answerable_rows),
            "total_judge_llm_calls": total_judge_calls,
        },
        "overall_all_30": _aggregate_group(evaluated_rows, metric_names),
        "overall_answerable_24": _aggregate_group(answerable_rows, metric_names),
        "by_split": by_split,
        "by_category": by_category,
        "results": evaluated_rows,
    }


def print_summary_table(report: Dict[str, Any]) -> None:
    """Print a clean terminal summary of the evaluation results."""
    top_k = report["config"]["top_k"]
    print("\n" + "=" * 78)
    print(f" NOVA DESK RAG EVALUATION REPORT (Top-K={top_k})")
    print(f" Generator: {report['config']['generator_model']} | Judge: {report['config']['judge_model']}")
    print("=" * 78)

    print("\n1. OVERALL METRICS (All 30 Questions):")
    for m_name, stats in report["overall_all_30"]["metrics"].items():
        print(
            f"   - {m_name:<18}: Mean = {stats['mean_score']:.4f} | "
            f"Pass Rate = {stats['pass_rate'] * 100:5.1f}% ({stats['passed_count']}/{stats['total_count']})"
        )

    print("\n2. ANSWERABLE QUESTIONS ONLY (24 Questions with Ground-Truth Evidence):")
    for m_name, stats in report["overall_answerable_24"]["metrics"].items():
        print(
            f"   - {m_name:<18}: Mean = {stats['mean_score']:.4f} | "
            f"Pass Rate = {stats['pass_rate'] * 100:5.1f}% ({stats['passed_count']}/{stats['total_count']})"
        )

    print("\n3. BREAKDOWN BY QUESTION CATEGORY:")
    for cat, group in report["by_category"].items():
        metrics_str = " | ".join(
            f"{m}: {s['mean_score']:.2f}" for m, s in group["metrics"].items()
        )
        print(f"   - {cat:<13} (n={group['count']:2d}): {metrics_str}")
    print("=" * 78 + "\n")


def run_evaluation(
    *,
    top_k: int = 4,
    generator_model: str = "gemini-3.5-flash-lite",
    judge_model: str = "gemini-3.1-flash-lite",
    judge_interval_seconds: float = 4.8,
    force_reeval: bool = False,
) -> Dict[str, Any]:
    """
    Main evaluation pipeline:
    1. Ensures `predictions.jsonl` exists.
    2. Runs DeepEval metrics across all 30 questions with incremental disk caching.
    3. Saves `eval_results.json` and prints the summary table.
    """
    predictions = load_jsonl(PREDICTIONS_PATH)
    if not predictions:
        logger.info("No predictions.jsonl found — running generate_predictions() first...")
        predictions = generate_predictions(top_k=top_k, generator_model=generator_model)

    cached_evals = {} if force_reeval else {
        row["id"]: row for row in load_jsonl(EVAL_CACHE_PATH)
    }
    if force_reeval and EVAL_CACHE_PATH.exists():
        EVAL_CACHE_PATH.unlink()

    judge_llm = RateLimitedGeminiJudge(
        model_name=judge_model,
        google_api_key=settings.GOOGLE_GEMINI_API_KEY,
        min_interval_seconds=judge_interval_seconds,
    )

    evaluated_rows: List[Dict[str, Any]] = []
    logger.info(
        "Evaluating %d predictions with DeepEval (already cached: %d, judge=%s)...",
        len(predictions),
        len(cached_evals),
        judge_model,
    )

    with EVAL_CACHE_PATH.open("a", encoding="utf-8") as cache_file:
        for idx, pred in enumerate(predictions, start=1):
            qid = pred["id"]
            if qid in cached_evals:
                logger.info("[%02d/%02d] %s — loaded from eval_results_cache.jsonl", idx, len(predictions), qid)
                evaluated_rows.append(cached_evals[qid])
                continue

            logger.info("[%02d/%02d] Evaluating %s (%s): '%s'...", idx, len(predictions), qid, pred.get("category"), pred["question"])
            eval_row = evaluate_single_test_case(
                prediction=pred,
                judge_llm=judge_llm,
                top_k=top_k,
            )

            cache_file.write(json.dumps(eval_row, ensure_ascii=False) + "\n")
            cache_file.flush()
            evaluated_rows.append(eval_row)

            m = eval_row["metrics"]
            logger.info(
                "   -> %s scores: P@%d=%.2f | R@%d=%.2f | MRR@%d=%.2f | Faithfulness=%.2f | Relevancy=%.2f",
                qid,
                top_k,
                m[f"Precision@{top_k}"]["score"],
                top_k,
                m[f"Recall@{top_k}"]["score"],
                top_k,
                m[f"MRR@{top_k}"]["score"],
                m["Faithfulness"]["score"],
                m["Answer Relevancy"]["score"],
            )

    report = build_aggregated_report(
        evaluated_rows=evaluated_rows,
        top_k=top_k,
        generator_model=generator_model,
        judge_model=judge_model,
        total_judge_calls=judge_llm.total_calls,
    )

    EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved full evaluation report to %s", EVAL_REPORT_PATH)
    print_summary_table(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepEval RAG evaluation suite.")
    parser.add_argument("--top-k", type=int, default=settings.RAG_TOP_K, help="Top-K chunks evaluated.")
    parser.add_argument("--generator-model", type=str, default="gemini-3.5-flash-lite", help="Generator model name.")
    parser.add_argument("--judge-model", type=str, default="gemini-3.1-flash-lite", help="DeepEval judge model name.")
    parser.add_argument("--interval", type=float, default=4.8, help="Minimum seconds between judge LLM calls.")
    parser.add_argument("--force", action="store_true", help="Re-evaluate all questions even if cached.")
    args = parser.parse_args()

    run_evaluation(
        top_k=args.top_k,
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        judge_interval_seconds=args.interval,
        force_reeval=args.force,
    )


if __name__ == "__main__":
    main()
