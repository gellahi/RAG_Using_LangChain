from __future__ import annotations

import argparse
import json

from rag_apps.common import paths
from rag_apps.common.evaluation import run_batch_queries
from rag_apps.common.logging_utils import get_logger
from .pipeline import build_pipeline


LOGGER = get_logger(__name__)


def load_queries(limit: int | None = None) -> list[str]:
    with paths.MEDICAL_QUERY_FILE.open("r", encoding="utf-8") as handle:
        queries = json.load(handle)
    return queries[:limit] if limit else queries


def evaluate(limit: int | None = None) -> None:
    pipeline = build_pipeline()
    queries = load_queries(limit)
    LOGGER.info("Running evaluation on %d queries", len(queries))
    run_batch_queries(
        pipeline.answer,
        queries,
        paths.EVAL_OUTPUT_DIR,
        prefix="medical_eval",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate medical RAG answers")
    parser.add_argument("--limit", type=int, default=None, help="Restrict query count for smoke tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(limit=args.limit)


if __name__ == "__main__":
    main()
