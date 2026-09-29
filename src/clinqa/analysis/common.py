"""Shared structures for the analysis package."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

SEVERITIES = ("error", "warn", "info")


@dataclass
class Flag:
    check: str
    split: str
    id: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"check": self.check, "split": self.split, "id": self.id, "detail": self.detail}


@dataclass
class CheckResult:
    check_id: str
    title: str
    severity: str  # severity when flags are present
    description: str
    handling: str
    flags: list[Flag] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def counts_by_split(self, splits: list[str]) -> dict[str, int]:
        c = Counter(f.split for f in self.flags)
        return {s: c.get(s, 0) for s in splits}

    @property
    def status(self) -> str:
        return self.severity if self.flags else "pass"


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (deterministic, no interpolation)."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(1, math.ceil(q / 100 * len(s)))
    return s[k - 1]


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    n = len(values)
    mean = sum(values) / n
    return {
        "n": n,
        "mean": round(mean, 1),
        "median": percentile(values, 50),
        "min": min(values),
        "p5": percentile(values, 5),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def word_count(text: str) -> int:
    return len(text.split())


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def sentence_count(text: str) -> int:
    text = text.strip()
    return len([s for s in _SENT_RE.split(text) if s.strip()]) if text else 0


def mask_numbers(text: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", "#", text)


def truncate(text: str, n: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "\u2026"
