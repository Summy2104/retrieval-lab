"""Guards on the evaluation set itself.

Two real failure modes, each of which silently turns a fixture into a green light
wired to nothing:

1. A query drifts into reusing its target's wording, so word overlap answers it
   and the fixture stops measuring semantics. Observed: an evaluation set that
   scored 1.000 under four different configurations, including one with a
   provably zero-signal embedding.
2. A target sits first in the document list. With no usable signal every score
   ties, and the winner is decided by the storage layer's tie-break — insertion
   order on some backends. Measured: moving one target to the front lifted a
   zero-signal retriever from 0.00 to 0.50 without changing a word.
"""

from pathlib import Path

import pytest

from rlab.evaluation import (
    load_fixtures,
    needs_semantic_embedding,
    shared_wording,
    validate,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _semantic() -> list[dict]:
    return [f for f in load_fixtures(FIXTURES) if needs_semantic_embedding(f)]


def test_the_suite_contains_a_semantic_fixture() -> None:
    """Without this, the guards below iterate an empty list and prove nothing."""
    assert _semantic(), "expected at least one fixture requiring semantic embedding"


@pytest.mark.parametrize("fixture", _semantic(), ids=lambda f: f["name"])
def test_semantic_queries_share_no_wording_with_their_target(fixture: dict) -> None:
    contents = {d["content"] for d in fixture["documents"]}
    for query in fixture["queries"]:
        target = next(c for c in contents if query["expects"] in c)
        shared = shared_wording(query["text"], target)
        assert not shared, (
            f"query {query['text']!r} shares wording {sorted(shared)} with its target; "
            "word overlap could answer it without any semantic matching"
        )


@pytest.mark.parametrize("fixture", _semantic(), ids=lambda f: f["name"])
def test_semantic_targets_are_never_the_first_document(fixture: dict) -> None:
    first = fixture["documents"][0]["content"]
    for query in fixture["queries"]:
        assert query["expects"] not in first, (
            f"query {query['text']!r} targets the first document, which wins every tie; "
            "move a distractor to the front"
        )


def test_every_shipped_fixture_validates() -> None:
    for fixture in load_fixtures(FIXTURES):
        validate(fixture)


def test_unknown_requirement_is_rejected() -> None:
    with pytest.raises(Exception, match="requires"):
        validate({"name": "x", "documents": [], "queries": [], "requires": "a_gpu"})
