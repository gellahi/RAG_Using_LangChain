from __future__ import annotations

from typing import Any, Iterable, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from .key_manager import GeminiKeyManager
from .logging_utils import get_logger


LOGGER = get_logger(__name__)


class RotatingGeminiChat(BaseChatModel):
    """Wraps ChatGoogleGenerativeAI with API key rotation."""

    model_name: str = "gemini-1.5-pro"

    def __init__(self, key_manager: GeminiKeyManager, model_name: str = "gemini-1.5-pro", **client_kwargs: Any):
        super().__init__()
        self.key_manager = key_manager
        self.model_name = model_name
        self.client_kwargs = client_kwargs

    @property
    def _llm_type(self) -> str:
        return "rotating-gemini-chat"

    def _build_client(self, api_key: str) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=api_key,
            **self.client_kwargs,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        attempts = len(self.key_manager.all_keys)
        last_error: Optional[Exception] = None
        for _ in range(attempts):
            api_key = self.key_manager.current
            client = self._build_client(api_key)
            try:
                return client._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                LOGGER.warning("Gemini call failed with key ****%s: %s", api_key[-4:], exc)
                self.key_manager.advance()
        if last_error:
            raise last_error
        raise RuntimeError("Gemini key rotation exhausted without success")


class RotatingGeminiEmbeddings(Embeddings):
    """Embedding wrapper that rotates Gemini keys."""

    def __init__(self, key_manager: GeminiKeyManager, model_name: str = "models/embedding-001", **client_kwargs: Any):
        self.key_manager = key_manager
        self.model_name = model_name
        self.client_kwargs = client_kwargs

    def _build_client(self, api_key: str) -> GoogleGenerativeAIEmbeddings:
        return GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=api_key,
            **self.client_kwargs,
        )

    def _call_with_rotation(self, func_name: str, *args: Any, **kwargs: Any) -> Any:
        attempts = len(self.key_manager.all_keys)
        last_error: Optional[Exception] = None
        for _ in range(attempts):
            api_key = self.key_manager.current
            client = self._build_client(api_key)
            try:
                return getattr(client, func_name)(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                LOGGER.warning("Gemini embedding failed with key ****%s: %s", api_key[-4:], exc)
                self.key_manager.advance()
        if last_error:
            raise last_error
        raise RuntimeError("Gemini embedding call failed for all keys")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call_with_rotation("embed_documents", texts)

    def embed_query(self, text: str) -> List[float]:
        return self._call_with_rotation("embed_query", text)


def build_rotating_resources(
    *,
    chat_model: str = "gemini-1.5-pro",
    embedding_model: str = "models/embedding-001",
    chat_kwargs: Optional[dict[str, Any]] = None,
    embedding_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[RotatingGeminiChat, RotatingGeminiEmbeddings]:
    manager = GeminiKeyManager.from_defaults()
    chat = RotatingGeminiChat(manager, model_name=chat_model, **(chat_kwargs or {}))
    embeddings = RotatingGeminiEmbeddings(manager, model_name=embedding_model, **(embedding_kwargs or {}))
    return chat, embeddings
