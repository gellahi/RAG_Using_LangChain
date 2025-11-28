from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from rag_apps.common.logging_utils import get_logger
from .config import ComplianceConfig


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class Rule:
    id: str
    category: str
    description: str
    severity: str


def load_rules(path: Path | None = None) -> List[Rule]:
    config = ComplianceConfig()
    rule_path = path or config.rules_path
    with rule_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    rules = [Rule(**item) for item in raw]
    if len(rules) < 15:
        raise ValueError("At least 15 compliance rules are required")
    LOGGER.info("Loaded %d compliance rules", len(rules))
    return rules
