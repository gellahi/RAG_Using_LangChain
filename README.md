# GenAI RAG Project

Two Retrieval-Augmented Generation pipelines built with LangChain + Gemini:

1. **Medical QA Assistant** – answers questions over the mtsamples medical transcription dataset.
2. **Policy Compliance Checker** – audits CUAD contracts against 15 custom governance rules.

Both pipelines share reusable utilities (key rotation, chunking, vector store helpers) under `src/rag_apps/common`.

## Setup

1. **Python environment**: Activate the provided Python 3.14 virtual environment or any compatible interpreter.
2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
3. **Expose the source tree**: either run commands from `src/` or set `PYTHONPATH`.
   ```powershell
   Set-Location src
   # or from project root
   $env:PYTHONPATH = "${PWD}\src"
   ```
4. **Gemini API keys**:
   - Keys are stored in `config/gemini_keys.json` (user-provided). Update/replace as needed.
   - Alternatively set an env var: `setx GEMINI_API_KEYS "key1,key2,..."`
   - Keys rotate automatically to avoid quota stalls.

Artifacts (chunks, vector stores, evaluation files) land in `artifacts/`.

## Task 1 – Medical RAG QA System

| Deliverable | Command |
| --- | --- |
| Preprocess + chunk dataset | `python -m rag_apps.medical.prepare_dataset` |
| Build/refresh Chroma vector store | `python -m rag_apps.medical.build_vector_store --force-store` |
| Run evaluation on 30 queries | `python -m rag_apps.medical.evaluate` |
| Launch Streamlit app | `streamlit run rag_apps/medical/streamlit_app.py` |

- Chunks cached at `artifacts/medical_chunks.jsonl`.
- Vector store persisted at `artifacts/medical_chroma`.
- Evaluation outputs saved under `artifacts/evaluation/medical_eval_*.json`.

## Task 2 – Policy Compliance Checker RAG System

| Deliverable | Command |
| --- | --- |
| Ingest PDF/TXT contracts & chunk | `python -m rag_apps.compliance.ingest --limit 50` (omit `--limit` for full run) |
| Build/refresh vector store | `python -m rag_apps.compliance.build_vector_store --force-store` |
| Generate compliant vs non-compliant table | `python -m rag_apps.compliance.comparison "Do agreements meet internal security policies?"` |
| Launch Streamlit compliance agent | `streamlit run rag_apps/compliance/streamlit_app.py` |

- Rules defined in `src/rag_apps/assets/compliance_rules.json` (15 rules, editable).
- Comparison reports saved to `artifacts/evaluation/compliance_comparison_*.csv|.md`.

## Streamlit Apps

- **Medical QA**: interactive question box, streaming answers with citations.
- **Compliance Checker**: filter rules by severity/category, inspect verdict, evidence, remediation, and sources per rule.

Ensure corresponding vector stores exist before launching Streamlit.

## Evaluation & Outputs

- Use `rag_apps.common.evaluation.run_batch_queries` helpers for reproducible runs.
- All evaluations stored under `artifacts/evaluation/` for auditability.

## Key Rotation & Safety

- `GeminiKeyManager` cycles through multiple Gemini keys automatically.
- The same rotation logic powers both chat completions and embeddings, minimizing manual recovery when quotas exhaust.

## Next Steps

- Extend rules or prompts as needed.
- Add automated regression tests or notebooks if deeper evaluation is required.
