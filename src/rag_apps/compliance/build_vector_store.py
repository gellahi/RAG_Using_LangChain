from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from langchain.schema import Document

from rag_apps.common.key_manager import GeminiKeyManager
from rag_apps.common.llm import RotatingGeminiEmbeddings
from rag_apps.common.logging_utils import get_logger
from rag_apps.common.vectorstores import build_chroma_store
from .config import ComplianceConfig
from .ingest import build_chunks


LOGGER = get_logger(__name__)


def load_cached_chunks(cache_path: Path) -> List[Document]:
    with cache_path.open("r", encoding="utf-8") as handle:
        return [
            Document(page_content=payload["page_content"], metadata=payload["metadata"])
            for payload in (json.loads(line) for line in handle)
        ]


def ensure_chunks(config: ComplianceConfig, force: bool, limit: int | None = None) -> List[Document]:
    if config.cache_path.exists() and not force:
        LOGGER.info("Loading cached compliance chunks from %s", config.cache_path)
        return load_cached_chunks(config.cache_path)
    LOGGER.info("Creating compliance chunks (force=%s)", force)
    return build_chunks(config, limit=limit)


def build_store(force_chunks: bool = False, force_store: bool = False, limit: int | None = None) -> None:
    config = ComplianceConfig()
    chunks = ensure_chunks(config, force_chunks, limit=limit)
    manager = GeminiKeyManager.from_defaults()
    embeddings = RotatingGeminiEmbeddings(manager)
    build_chroma_store(chunks, embeddings, config.persist_directory, force_recreate=force_store)
    LOGGER.info("Compliance vector store ready at %s", config.persist_directory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the compliance vector store")
    parser.add_argument("--force-chunks", action="store_true", help="Recreate chunk cache")
    parser.add_argument("--force-store", action="store_true", help="Recreate Chroma store")
    parser.add_argument("--limit", type=int, default=None, help="Limit files for testing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_store(force_chunks=args.force_chunks, force_store=args.force_store, limit=args.limit)


if __name__ == "__main__":
    main()
