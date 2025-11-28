from __future__ import annotations

import itertools
import os
from typing import Iterable, List

from .logging_utils import get_logger


LOGGER = get_logger(__name__)


ENV_KEY_PREFIX = "GEMINI_API_KEY_"


class GeminiKeyManager:
    """Cycles through Gemini API keys so we can retry on quota errors."""

    def __init__(self, keys: Iterable[str]):
        cleaned = [k.strip() for k in keys if k and k.strip()]
        if not cleaned:
            raise ValueError("No Gemini API keys available")
        self._keys: List[str] = list(dict.fromkeys(cleaned))
        self._cycle = itertools.cycle(self._keys)
        self._current: str | None = None
        LOGGER.debug("Loaded %d Gemini API keys", len(self._keys))

    @classmethod
    def from_defaults(cls) -> "GeminiKeyManager":
        keys = _collect_env_keys()
        if not keys:
            raise EnvironmentError(
                "Set numbered GEMINI_API_KEY_<n> environment variables."
            )
        LOGGER.info("Loaded %d Gemini keys from environment", len(keys))
        return cls(keys)

    @property
    def current(self) -> str:
        if self._current is None:
            self._current = next(self._cycle)
        return self._current

    def advance(self) -> str:
        self._current = next(self._cycle)
        LOGGER.warning("Switching to next Gemini key")
        return self._current

    @property
    def all_keys(self) -> List[str]:
        return list(self._keys)


def _collect_env_keys() -> List[str]:
    keys: List[str] = []
    index = 1
    while True:
        value = os.getenv(f"{ENV_KEY_PREFIX}{index}")
        if not value:
            break
        keys.append(value.strip())
        index += 1
    return [k for k in keys if k]
