from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
from langchain.schema import Document

from rag_apps.common.chunking import chunk_documents
from rag_apps.common.logging_utils import get_logger
from .config import MedicalRAGConfig


LOGGER = get_logger(__name__)


def load_medical_documents(config: MedicalRAGConfig, sample_size: int | None = None) -> List[Document]:
    LOGGER.info("Loading medical dataset from %s", config.dataset_path)
    df = pd.read_csv(config.dataset_path)
    if sample_size:
        df = df.head(sample_size)

    field = config.transcription_field
    df = df[df[field].notna()]

    documents: List[Document] = []
    for idx, row in df.iterrows():
        text = str(row[field]).strip()
        if not text:
            continue
        metadata = {name: str(row.get(name, "")).strip() for name in config.metadata_fields}
        metadata["source_id"] = int(idx)
        metadata["description"] = str(row.get("description", "")).strip()
        documents.append(Document(page_content=text, metadata=metadata))
    LOGGER.info("Loaded %d raw documents", len(documents))
    return documents


def persist_chunks(chunks: List[Document], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for doc in chunks:
            payload = {"page_content": doc.page_content, "metadata": doc.metadata}
            handle.write(json.dumps(payload) + "\n")
    LOGGER.info("Persisted %d chunks to %s", len(chunks), output_path)


def prepare_chunks(config: MedicalRAGConfig, sample_size: int | None = None) -> List[Document]:
    documents = load_medical_documents(config, sample_size)
    chunks = chunk_documents(documents, config.chunk_size, config.chunk_overlap)
    persist_chunks(chunks, config.cache_path)
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare medical RAG dataset chunks")
    parser.add_argument("--sample-size", type=int, default=None, help="Limit rows for quick tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MedicalRAGConfig()
    prepare_chunks(config, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
