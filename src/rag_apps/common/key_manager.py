from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
from typing import Iterable, List

from . import paths
from .logging_utils import get_logger


LOGGER = get_logger(__name__)


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
        env_keys = os.getenv("GEMINI_API_KEYS")
        if env_keys:
            LOGGER.info("Loading Gemini keys from GEMINI_API_KEYS")
            return cls(env_keys.split(","))

        file_path = paths.GEMINI_KEY_FILE
        if file_path.exists():
            LOGGER.info("Loading Gemini keys from %s", file_path)
            with Path(file_path).open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return cls(data.get("keys", []))

        raise FileNotFoundError(
            "Set GEMINI_API_KEYS env var or populate config/gemini_keys.json"
        )

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
