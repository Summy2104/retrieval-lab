"""Retrieval metrics, and guards that keep the evaluation set itself honest.

The metrics here are the easy part. The hard part — and the reason this module
carries as much validation as measurement — is that an evaluation set can look
perfect while measuring nothing. A set whose queries share wording with their
targets is answerable by word overlap alone, so it scores 1.000 under every
configuration, real or stubbed. That is a green light wired to nothing.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("name", "documents", "queries")
SUPPORTED_REQUIREMENTS = ("semantic_embedding",)
_CJK = re.compile(r"^[一-鿿]{2}$")


class FixtureError(ValueError):
    """A fixture is structurally invalid."""


@dataclass(frozen=True)
class QueryResult:
    query: str
    hit_rank: int | None
    returned: list[str]


@dataclass(frozen=True)
class Metrics:
    query_count: int
    hit_count: int
    recall_at_k: float
    mrr: float
    hit_rate: float
    avg_hit_rank: float | None


def load_fixtures(directory: str | Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(Path(directory).glob("*.json")):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        validate(fixture)
        out.append(fixture)
    return out


def validate(fixture: dict[str, Any]) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in fixture]
    if missing:
        raise FixtureError(f"missing fields: {', '.join(missing)}")
    requirement = fixture.get("requires")
    if requirement is not None and requirement not in SUPPORTED_REQUIREMENTS:
        raise FixtureError(f"unknown requires: {requirement!r}")
    for i, q in enumerate(fixture["queries"]):
        if not q.get("text"):
            raise FixtureError(f"queries[{i}].text is required")
        if not q.get("expects"):
            raise FixtureError(f"queries[{i}].expects is required")


def needs_semantic_embedding(fixture: dict[str, Any]) -> bool:
    """Whether this fixture only produces a meaningful score under a real model."""
    return fixture.get("requires") == "semantic_embedding"


def shared_wording(query: str, target: str) -> set[str]:
    """Wording a keyword matcher could latch onto: shared words and CJK bigrams.

    Deliberately not built from the production tokenizer — this checks a property
    of the fixture, and reusing the tokenizer would only prove the two agree with
    each other. CJK pairs are restricted to two CJK characters so an English
    query cannot collide with an English target on incidental letter pairs.
    """
    q, t = query.casefold(), target.casefold()
    shared = {w for w in q.split() if w and w in t}
    shared |= {q[i:i + 2] for i in range(len(q) - 1)
               if _CJK.match(q[i:i + 2]) and q[i:i + 2] in t}
    return shared


def score(results: Sequence[QueryResult]) -> Metrics:
    n = len(results)
    hits = [r for r in results if r.hit_rank is not None]
    ranks = [r.hit_rank for r in hits if r.hit_rank is not None]
    return Metrics(
        query_count=n,
        hit_count=len(hits),
        recall_at_k=len(hits) / n if n else 0.0,
        mrr=sum(1.0 / r for r in ranks) / n if n else 0.0,
        hit_rate=len(hits) / n if n else 0.0,
        avg_hit_rank=sum(ranks) / len(ranks) if ranks else None,
    )


def first_hit_rank(returned: Sequence[str], expects: str) -> int | None:
    """1-based rank of the first returned document containing ``expects``."""
    for rank, content in enumerate(returned, start=1):
        if expects in content:
            return rank
    return None
