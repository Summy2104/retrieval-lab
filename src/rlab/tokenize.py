"""Term extraction shared by the index and the query side.

The whole point of this module is that there is exactly one function producing
terms. SQLite FTS5's ``unicode61`` tokenizer keeps an entire CJK run as a single
token, so a Chinese query built from bigrams can never match Chinese stored
verbatim — index and query must agree by construction, not by convention.
"""

from __future__ import annotations

import re

# Latin/digit runs stay whole; each CJK character is emitted separately so a run
# can be re-joined into overlapping bigrams below.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[一-鿿]")
_CJK_RE = re.compile(r"^[一-鿿]$")


def _is_cjk(token: str) -> bool:
    return bool(_CJK_RE.match(token))


def _cjk_bigrams(chars: list[str]) -> list[str]:
    """Overlapping bigrams for a CJK run; a lone character is kept as-is.

    Bigrams (not unigrams) because single Chinese characters are far too common
    to discriminate, and not trigrams because real queries are often two
    characters long (故障 / 薪资) and would then match nothing.
    """
    if not chars:
        return []
    if len(chars) == 1:
        return [chars[0]]
    return ["".join(pair) for pair in zip(chars, chars[1:], strict=False)]


def terms(text: str) -> list[str]:
    """The single source of truth. Both index and query MUST derive from this."""
    out: list[str] = []
    pending: list[str] = []
    for token in _TOKEN_RE.findall(text):
        if _is_cjk(token):
            pending.append(token)
            continue
        out.extend(_cjk_bigrams(pending))
        pending = []
        out.append(token)
    out.extend(_cjk_bigrams(pending))
    return out


def index_text(content: str) -> str:
    """What goes into the FTS shadow table — never the original text."""
    return " ".join(terms(content))


def query_expr(query: str) -> str:
    """The same terms, each quoted so FTS5 operators cannot be injected."""
    return " ".join(f'"{t}"' for t in terms(query))
