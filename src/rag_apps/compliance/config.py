from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_apps.common import paths


@dataclass(slots=True)
class ComplianceConfig:
    pdf_dir: Path = paths.CUAD_PDF_DIR
    txt_dir: Path = paths.CUAD_TXT_DIR
    chunk_size: int = 1500
    chunk_overlap: int = 250
    persist_directory: Path = paths.COMPLIANCE_VECTOR_DIR
    cache_path: Path = paths.COMPLIANCE_CHUNK_CACHE
    rules_path: Path = paths.RULES_FILE
    allowed_extensions: tuple[str, ...] = (".pdf", ".txt")
