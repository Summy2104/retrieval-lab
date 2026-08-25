"""BM25 keyword retrieval over SQLite FTS5.

The index stores the tokenized form from ``tokenize.index_text``, not the
original text. The original is served from the content table, which the search
join reads — so callers always get real text back while matching happens on
terms both sides agree on.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from rlab.tokenize import index_text, query_expr

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    id        TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    position  INTEGER NOT NULL DEFAULT 0,
    content   TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
    doc_id UNINDEXED, namespace UNINDEXED, content, tokenize='unicode61'
);
"""


@dataclass(frozen=True)
class Hit:
    id: str
    content: str
    score: float


class KeywordIndex:
    def __init__(self, path: str = ":memory:") -> None:
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)

    def add(self, doc_id: str, content: str, *, namespace: str = "default",
            position: int = 0) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO docs(id, namespace, position, content)"
            " VALUES(:id,:ns,:pos,:c)",
            {"id": doc_id, "ns": namespace, "pos": position, "c": content},
        )
        self._db.execute("DELETE FROM docs_fts WHERE doc_id = :id", {"id": doc_id})
        self._db.execute(
            "INSERT INTO docs_fts(doc_id, namespace, content) VALUES(:id,:ns,:c)",
            # Tokenized, not raw: must match what query_expr emits.
            {"id": doc_id, "ns": namespace, "c": index_text(content)},
        )
        self._db.commit()

    def search(self, query: str, *, namespace: str = "default", limit: int = 10) -> list[Hit]:
        expr = query_expr(query)
        if not expr:
            return []
        rows = self._db.execute(
            """
            SELECT d.id, d.content, d.position, bm25(docs_fts) AS bm25
            FROM docs_fts JOIN docs d ON d.id = docs_fts.doc_id
            WHERE docs_fts MATCH :expr AND docs_fts.namespace = :ns
            ORDER BY bm25 ASC, d.position ASC, d.id ASC
            LIMIT :limit
            """,
            {"expr": expr, "ns": namespace, "limit": limit},
        ).fetchall()
        # bm25() is negative (more negative = more relevant); negate to normalize.
        return [Hit(id=r["id"], content=r["content"], score=-r["bm25"]) for r in rows]

    def raw_index_text(self, doc_id: str) -> str | None:
        """Exposed for tests that assert the index stores terms, not the original."""
        row = self._db.execute(
            "SELECT content FROM docs_fts WHERE doc_id = :id", {"id": doc_id}
        ).fetchone()
        return row["content"] if row else None
