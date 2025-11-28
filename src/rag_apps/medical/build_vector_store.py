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
from .config import MedicalRAGConfig
from .prepare_dataset import prepare_chunks


LOGGER = get_logger(__name__)


def load_cached_chunks(cache_path: Path) -> List[Document]:
    with cache_path.open("r", encoding="utf-8") as handle:
        return [
            Document(page_content=line_obj["page_content"], metadata=line_obj["metadata"])
            for line_obj in (json.loads(line) for line in handle)
        ]


def ensure_chunks(config: MedicalRAGConfig, force: bool) -> List[Document]:
    if config.cache_path.exists() and not force:
        LOGGER.info("Loading chunks from %s", config.cache_path)
        return load_cached_chunks(config.cache_path)
    LOGGER.info("Cache missing or force rebuild requested; creating fresh chunks")
    return prepare_chunks(config)


def build_store(force_chunks: bool = False, force_store: bool = False) -> None:
    config = MedicalRAGConfig()
    chunks = ensure_chunks(config, force_chunks)
    manager = GeminiKeyManager.from_defaults()
    embeddings = RotatingGeminiEmbeddings(manager)
    build_chroma_store(chunks, embeddings, config.persist_directory, force_recreate=force_store)
    LOGGER.info("Medical vector store ready at %s", config.persist_directory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or refresh the medical vector store")
    parser.add_argument("--force-chunks", action="store_true", help="Regenerate chunks even if cache exists")
    parser.add_argument("--force-store", action="store_true", help="Rebuild vector store from scratch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_store(force_chunks=args.force_chunks, force_store=args.force_store)


if __name__ == "__main__":
    main()
