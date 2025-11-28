from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "datasets"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MEDICAL_DATASET = DATA_DIR / "MedicalTranscriptions" / "mtsamples.csv"
CUAD_DIR = DATA_DIR / "CUAD_v1"
CUAD_PDF_DIR = CUAD_DIR / "full_contract_pdf"
CUAD_TXT_DIR = CUAD_DIR / "full_contract_txt"

MEDICAL_VECTOR_DIR = ARTIFACTS_DIR / "medical_chroma"
COMPLIANCE_VECTOR_DIR = ARTIFACTS_DIR / "compliance_chroma"

MEDICAL_CHUNK_CACHE = ARTIFACTS_DIR / "medical_chunks.jsonl"
COMPLIANCE_CHUNK_CACHE = ARTIFACTS_DIR / "compliance_chunks.jsonl"

EVAL_OUTPUT_DIR = ARTIFACTS_DIR / "evaluation"
EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RULES_FILE = PROJECT_ROOT / "src" / "rag_apps" / "assets" / "compliance_rules.json"
MEDICAL_QUERY_FILE = PROJECT_ROOT / "src" / "rag_apps" / "assets" / "medical_eval_queries.json"
