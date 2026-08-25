"""Index and query must derive terms from the same function, or CJK dies silently."""

import pytest

from rlab.keyword import KeywordIndex
from rlab.tokenize import index_text, query_expr, terms

_DOC = "数据库故障需要联系平台团队处理，升级路径为值班工程师。"


@pytest.mark.parametrize("query", ["故障", "数据库", "数据库故障", "平台团队", "值班工程师"])
def test_chinese_queries_match(query: str) -> None:
    """The regression this whole module exists for: Chinese keyword search was dead.

    FTS5's unicode61 tokenizer keeps a whole CJK run as one token, so a bigram
    query could never match Chinese stored verbatim. Every one of these queries
    returned zero rows before index and query shared a tokenizer.
    """
    index = KeywordIndex()
    index.add("d", _DOC)

    assert [h.id for h in index.search(query)] == ["d"]


def test_caller_gets_the_original_text_not_the_index_form() -> None:
    index = KeywordIndex()
    index.add("d", _DOC)

    assert index.search("数据库")[0].content == _DOC
    assert index.raw_index_text("d") != _DOC, "the shadow table must hold terms"


def test_index_and_query_derive_from_one_function() -> None:
    """Structural guard: drift between the two sides is the original bug."""
    doc_terms = set(terms(_DOC))
    query_terms = set(terms("数据库故障"))

    assert query_terms <= doc_terms
    assert all(t in index_text(_DOC).split() for t in query_terms)
    assert all(f'"{t}"' in query_expr("数据库故障") for t in query_terms)


def test_ascii_is_byte_identical_to_storing_raw_text() -> None:
    """Latin text must be unaffected — unicode61 already splits on punctuation."""
    assert index_text("Restart service when ERR42 appears.") == "Restart service when ERR42 appears"


def test_two_char_cjk_query_survives() -> None:
    """A trigram tokenizer would fail here; two-character queries are the common case."""
    assert terms("故障") == ["故障"]


def test_mixed_language_content() -> None:
    index = KeywordIndex()
    index.add("d", "VPN 接入问题由基础设施支持团队处理，错误码 ERR-42。")

    for query in ("VPN", "接入", "基础设施", "ERR"):
        assert [h.id for h in index.search(query)] == ["d"], query
