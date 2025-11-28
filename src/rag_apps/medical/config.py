from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_apps.common import paths


@dataclass(slots=True)
class MedicalRAGConfig:
    dataset_path: Path = paths.MEDICAL_DATASET
    chunk_size: int = 1200
    chunk_overlap: int = 200
    persist_directory: Path = paths.MEDICAL_VECTOR_DIR
    cache_path: Path = paths.MEDICAL_CHUNK_CACHE
    specialty_field: str = "medical_specialty"
    transcription_field: str = "transcription"
    metadata_fields: tuple[str, ...] = (
        "medical_specialty",
        "sample_name",
        "keywords",
    )
