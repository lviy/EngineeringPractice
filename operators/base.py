from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OperatorCase:
    name: str
    summary: str
    params: dict[str, Any]
