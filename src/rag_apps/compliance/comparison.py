from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from rag_apps.common import paths
from rag_apps.common.logging_utils import get_logger
from .agent import build_agent


LOGGER = get_logger(__name__)


def summarize_sources(sources: list[dict]) -> str:
    if not sources:
        return ""
    return "; ".join(f"{item.get('doc_name')} ({item.get('file_type')})" for item in sources)


def generate_table(question: str) -> tuple[Path, Path]:
    agent = build_agent()
    results = agent.run_assessment(question)
    df = pd.DataFrame(results)
    df["source_summary"] = df["sources"].apply(summarize_sources)
    df["evidence"] = df["evidence"].apply(lambda ev: "; ".join(ev) if isinstance(ev, list) else ev)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = paths.EVAL_OUTPUT_DIR / f"compliance_comparison_{timestamp}.csv"
    md_path = paths.EVAL_OUTPUT_DIR / f"compliance_comparison_{timestamp}.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["sources"], errors="ignore").to_csv(csv_path, index=False)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(df.drop(columns=["sources"], errors="ignore").to_markdown(index=False))
    LOGGER.info("Saved comparison table to %s and %s", csv_path, md_path)
    return csv_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate compliance comparison report")
    parser.add_argument("question", help="Business question to evaluate, e.g. 'Do contracts meet security policies?'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_table(args.question)


if __name__ == "__main__":
    main()
