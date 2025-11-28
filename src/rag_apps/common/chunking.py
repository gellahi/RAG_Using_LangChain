from __future__ import annotations

from typing import Iterable, List

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_text_splitter(chunk_size: int = 1200, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " "],
    )


def chunk_documents(documents: Iterable[Document], chunk_size: int = 1200, chunk_overlap: int = 200) -> List[Document]:
    splitter = create_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(list(documents))
