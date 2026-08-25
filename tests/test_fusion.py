"""RRF behaviour, and the candidate-depth degeneracy that made a fixture flaky."""

import pytest

from rlab.fusion import RRF_K, candidate_depth, fuse, rrf_score


def test_rrf_flattens_the_top_of_the_ranking() -> None:
    """k=60 is the whole point: rank 1 and rank 2 differ by under 2%."""
    gap = rrf_score(1) / rrf_score(2) - 1

    assert gap < 0.02
    assert rrf_score(1) == pytest.approx(1 / (RRF_K + 1))


def test_a_document_ranked_first_by_both_legs_tops_out_at_two_over_k_plus_one() -> None:
    fused = fuse([["A", "B"], ["A", "C"]], limit=1)

    assert fused[0][0] == "A"
    assert fused[0][1] == pytest.approx(2 / (RRF_K + 1))


def test_candidate_depth_is_decoupled_from_limit() -> None:
    """The bug: depth == limit makes small-limit fusion a coin flip.

    With one candidate per leg, each leg contributes a single rank-1 entry, the
    two tie at 1/(k+1), and the winner is whatever the storage layer's tie-break
    happens to be. Measured 8 failures in 12 runs before decoupling.
    """
    assert candidate_depth(1) > 1
    assert candidate_depth(1) >= 20
    assert candidate_depth(1000) <= 100, "depth must stay bounded"


def test_ties_break_deterministically_not_by_iteration_order() -> None:
    """Equal scores must resolve the same way every run, or tests go flaky."""
    lists = [["A", "B"], ["B", "A"]]
    first = fuse(lists, limit=2)
    reversed_input = fuse([lists[1], lists[0]], limit=2)

    assert [d for d, _ in first] == [d for d, _ in reversed_input]


def test_explicit_tie_break_key_wins_over_id() -> None:
    fused = fuse([["A", "B"], ["B", "A"]], limit=2, tie_break={"B": (0, "B"), "A": (1, "A")})

    assert [d for d, _ in fused] == ["B", "A"]


def test_zero_based_rank_is_rejected() -> None:
    """A 0-based caller would silently inflate every score; fail loudly instead."""
    with pytest.raises(ValueError, match="1-based"):
        rrf_score(0)
