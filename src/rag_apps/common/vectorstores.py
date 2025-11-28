from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_chroma_store(
    documents: Iterable[Document],
    embeddings: Embeddings,
    persist_directory: Path,
    force_recreate: bool = False,
) -> Chroma:
    path = Path(persist_directory)
    if force_recreate and path.exists():
        shutil.rmtree(path)
    _ensure_dir(path)
    return Chroma.from_documents(
        documents=list(documents),
        embedding=embeddings,
        persist_directory=str(path),
    )


def load_chroma_store(embeddings: Embeddings, persist_directory: Path) -> Chroma:
    path = Path(persist_directory)
    if not path.exists():
        raise FileNotFoundError(f"No vector store found at {path}")
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(path),
    )
