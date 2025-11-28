from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List

from .logging_utils import get_logger


LOGGER = get_logger(__name__)


def run_batch_queries(
    query_runner: Callable[[str], dict],
    queries: Iterable[str],
    output_dir: Path,
    prefix: str,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prefix}_{timestamp}.json"

    results: List[dict] = []
    for question in queries:
        LOGGER.info("Evaluating query: %s", question)
        result = query_runner(question)
        results.append({"question": question, **result})

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    LOGGER.info("Wrote %d evaluation rows to %s", len(results), output_path)
    return output_path
