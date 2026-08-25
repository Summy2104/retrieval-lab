"""Reciprocal Rank Fusion, plus the candidate-depth decision it depends on.

RRF's constant k=60 (Cormack et al., 2009) deliberately flattens the gap between
top ranks — rank 1 scores 1/61 and rank 2 scores 1/62, a 1.6% difference — so no
single retrieval channel's top hit can dominate the fused order.

That flattening has a cost this module exists to document: fusion needs *depth*
to fuse. Feeding each leg only ``limit`` candidates makes small-``limit`` queries
degenerate into a coin flip, because each leg contributes exactly one rank-1
entry and the two tie at 1/61. See ``candidate_depth``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

RRF_K = 60
_DEPTH_MULTIPLIER = 5
_DEPTH_MIN = 20
_DEPTH_MAX = 100


def candidate_depth(limit: int) -> int:
    """How many candidates each leg must supply for fusion to mean anything.

    Deliberately NOT ``limit``. With ``limit=1`` each leg hands over its own top
    hit, both score 1/(k+1), and the winner is decided by whatever tie-break the
    storage layer happens to apply — insertion order on one backend, primary-key
    order on another. Measured: the same single-query fixture failed 8 of 12 runs
    until the depth was decoupled from the limit.
    """
    return min(max(limit * _DEPTH_MULTIPLIER, _DEPTH_MIN), _DEPTH_MAX)


def rrf_score(rank: int) -> float:
    """Contribution of one 1-based rank. Guard against a 0-based caller."""
    if rank < 1:
        raise ValueError("rank is 1-based; got %d" % rank)
    return 1.0 / (RRF_K + rank)


def fuse(
    ranked_lists: Iterable[Sequence[str]],
    *,
    limit: int,
    tie_break: dict[str, tuple[int, str]] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into one ordered list of ``(id, score)``.

    ``tie_break`` supplies a deterministic secondary key per id (e.g. position
    then id). Without it, equal scores fall back to the id itself — never to
    iteration order, which is what makes tie outcomes reproducible.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score(rank)

    def sort_key(item: tuple[str, float]) -> tuple[float, int, str]:
        doc_id, score = item
        secondary = tie_break.get(doc_id, (0, doc_id)) if tie_break else (0, doc_id)
        return (-score, secondary[0], secondary[1])

    return sorted(scores.items(), key=sort_key)[:limit]
