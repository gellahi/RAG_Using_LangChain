from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from rag_apps.common.key_manager import GeminiKeyManager
from rag_apps.common.llm import RotatingGeminiChat, RotatingGeminiEmbeddings
from rag_apps.common.logging_utils import get_logger
from rag_apps.common.vectorstores import load_chroma_store
from .config import ComplianceConfig
from .rules import Rule, load_rules


LOGGER = get_logger(__name__)


PROMPT = """
You are a strict enterprise compliance auditor. Review the provided contract excerpts and rule details.
Return a JSON object with keys: verdict (Compliant/Non-Compliant/NotFound), evidence (list of citations), and remediation (string).
Be concise but specific.

Rule ID: {rule_id}
Rule Description: {rule_description}
Severity: {severity}

Business Question: {question}

Context:
{context}
""".strip()


def _format_context(docs: List[Document]) -> str:
    if not docs:
        return "No supporting passages found."
    formatted = []
    for doc in docs:
        formatted.append(
            f"[{doc.metadata.get('doc_name', 'unknown')}]\n{doc.page_content[:1200]}"
        )
    return "\n\n".join(formatted)


def _summaries(docs: List[Document]) -> List[dict]:
    seen = set()
    entries: List[dict] = []
    for doc in docs:
        name = doc.metadata.get("doc_name", "unknown")
        if name in seen:
            continue
        seen.add(name)
        entries.append(
            {
                "doc_name": name,
                "path": doc.metadata.get("source_path"),
                "file_type": doc.metadata.get("file_type"),
            }
        )
    return entries


@dataclass
class ComplianceAgent:
    chain: LLMChain
    retriever: any
    rules: List[Rule]

    def assess_rule(self, rule: Rule, question: str) -> dict:
        compound_query = f"{question}\nRule: {rule.description}"
        docs = self.retriever.get_relevant_documents(compound_query)
        context = _format_context(docs)
        response = self.chain.invoke(
            {
                "rule_id": rule.id,
                "rule_description": rule.description,
                "severity": rule.severity,
                "question": question,
                "context": context,
            }
        )
        payload = response["text"] if isinstance(response, dict) else response
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            parsed = {
                "verdict": "NotFound",
                "evidence": [payload],
                "remediation": "Could not parse structured output.",
            }
        return {
            "rule_id": rule.id,
            "category": rule.category,
            "severity": rule.severity,
            "verdict": parsed.get("verdict", "NotFound"),
            "evidence": parsed.get("evidence", []),
            "remediation": parsed.get("remediation", ""),
            "sources": _summaries(docs),
        }

    def run_assessment(self, question: str) -> List[dict]:
        results = []
        for rule in self.rules:
            LOGGER.info("Assessing %s", rule.id)
            results.append(self.assess_rule(rule, question))
        return results


def build_agent(config: ComplianceConfig | None = None) -> ComplianceAgent:
    config = config or ComplianceConfig()
    manager = GeminiKeyManager.from_defaults()
    chat = RotatingGeminiChat(manager)
    embeddings = RotatingGeminiEmbeddings(manager)
    store = load_chroma_store(embeddings, config.persist_directory)
    retriever = store.as_retriever(search_kwargs={"k": 8})
    prompt = PromptTemplate(
        template=PROMPT,
        input_variables=["rule_id", "rule_description", "severity", "question", "context"],
    )
    chain = LLMChain(llm=chat, prompt=prompt)
    rules = load_rules(config.rules_path)
    return ComplianceAgent(chain=chain, retriever=retriever, rules=rules)
