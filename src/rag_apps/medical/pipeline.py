from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from rag_apps.common.key_manager import GeminiKeyManager
from rag_apps.common.llm import RotatingGeminiChat, RotatingGeminiEmbeddings
from rag_apps.common.logging_utils import get_logger
from rag_apps.common.vectorstores import load_chroma_store
from .config import MedicalRAGConfig


LOGGER = get_logger(__name__)


PROMPT_TEMPLATE = """
You are a cautious medical analyst. Use the provided context to answer the user question.
Return concise, evidence-based guidance with bullet citations using [Source:sample_name].
If the answer is not contained in the context, reply with "I could not find that information.".

Context:
{context}

Question: {question}
""".strip()


def _format_context(docs: List[Document]) -> str:
    formatted = []
    for doc in docs:
        ref = doc.metadata.get("sample_name") or f"Record-{doc.metadata.get('source_id')}"
        specialty = doc.metadata.get("medical_specialty", "")
        formatted.append(f"[{ref} | {specialty}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def _summarize_sources(docs: List[Document]) -> List[dict]:
    unique = []
    seen = set()
    for doc in docs:
        ref = doc.metadata.get("sample_name") or str(doc.metadata.get("source_id"))
        if ref in seen:
            continue
        seen.add(ref)
        unique.append(
            {
                "sample_name": ref,
                "specialty": doc.metadata.get("medical_specialty"),
                "keywords": doc.metadata.get("keywords"),
            }
        )
    return unique


@dataclass
class MedicalRAGPipeline:
    chain: LLMChain
    retriever: any

    def answer(self, question: str) -> dict:
        docs = self.retriever.get_relevant_documents(question)
        if not docs:
            return {"answer": "No relevant context found.", "sources": []}
        context = _format_context(docs)
        response = self.chain.invoke({"question": question, "context": context})
        return {"answer": response["text"] if isinstance(response, dict) else response, "sources": _summarize_sources(docs)}


def build_pipeline(config: MedicalRAGConfig | None = None) -> MedicalRAGPipeline:
    config = config or MedicalRAGConfig()
    manager = GeminiKeyManager.from_defaults()
    chat = RotatingGeminiChat(manager)
    embeddings = RotatingGeminiEmbeddings(manager)
    vector_store = load_chroma_store(embeddings, config.persist_directory)
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    chain = LLMChain(llm=chat, prompt=prompt)
    LOGGER.info("Medical pipeline ready (vector dir: %s)", config.persist_directory)
    return MedicalRAGPipeline(chain=chain, retriever=retriever)
