from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from langchain.schema import Document
from pypdf import PdfReader

from rag_apps.common.chunking import chunk_documents
from rag_apps.common.logging_utils import get_logger
from .config import ComplianceConfig


LOGGER = get_logger(__name__)


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def iter_contract_files(config: ComplianceConfig) -> Iterable[Path]:
    for base_dir in (config.txt_dir, config.pdf_dir):
        if not base_dir.exists():
            continue
        for path in sorted(base_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in config.allowed_extensions:
                yield path


def load_documents(config: ComplianceConfig, limit: int | None = None) -> List[Document]:
    documents: List[Document] = []
    for idx, path in enumerate(iter_contract_files(config)):
        if limit and idx >= limit:
            break
        LOGGER.info("Reading %s", path)
        text = extract_pdf_text(path) if path.suffix.lower() == ".pdf" else read_text_file(path)
        if not text.strip():
            continue
        base_root = config.pdf_dir.parent  # CUAD_v1 root
        try:
            relative_path = path.relative_to(base_root)
        except ValueError:
            relative_path = path.name
        metadata = {
            "doc_name": path.stem,
            "source_path": str(relative_path),
            "file_type": path.suffix.lower(),
        }
        documents.append(Document(page_content=text, metadata=metadata))
    LOGGER.info("Loaded %d compliance documents", len(documents))
    return documents


def persist_chunks(chunks: List[Document], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for doc in chunks:
            payload = {"page_content": doc.page_content, "metadata": doc.metadata}
            handle.write(json.dumps(payload) + "\n")
    LOGGER.info("Persisted %d chunks to %s", len(chunks), output_path)


def build_chunks(config: ComplianceConfig, limit: int | None = None) -> List[Document]:
    docs = load_documents(config, limit)
    chunks = chunk_documents(docs, config.chunk_size, config.chunk_overlap)
    persist_chunks(chunks, config.cache_path)
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest CUAD contracts and create cached chunks")
    parser.add_argument("--limit", type=int, default=None, help="Restrict number of files for quick runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ComplianceConfig()
    build_chunks(config, limit=args.limit)


if __name__ == "__main__":
    main()
