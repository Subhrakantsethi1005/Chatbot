"""A small, dependency-free TF-IDF retriever.

Each entry gets two vectors: one for its question and one for question +
answer. A query is scored mostly against the question (FAQ-style matching)
and partly against the full text. Scores are cosine similarities in [0, 1]
and double as the answer's confidence.

The index is serialised to JSON so every knowledge-base version carries its
own immutable index. Swapping in an embedding model later only requires an
object with the same ``search``/``to_dict``/``from_dict`` methods.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .documents import KBEntry
from .textutil import content_tokens, normalize

QUESTION_WEIGHT = 0.75


@dataclass
class SearchHit:
    entry_id: str
    score: float


def _features(text: str) -> list[str]:
    tokens = content_tokens(text)
    return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


def _vector(terms: list[str], idf: dict[str, float], default_idf: float) -> dict[str, float]:
    counts = Counter(terms)
    vec = {t: (1.0 + math.log(c)) * idf.get(t, default_idf) for t, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return {t: v / norm for t, v in vec.items()} if norm else {}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())


class TfidfIndex:
    backend = "tfidf"

    def __init__(self, idf: dict[str, float], vectors: dict[str, tuple[dict[str, float], dict[str, float]]],
                 default_idf: float, exact: dict[str, str] | None = None) -> None:
        self.idf = idf
        self.vectors = vectors
        self.default_idf = default_idf
        self.exact = exact or {}   # normalised question -> entry id

    @classmethod
    def build(cls, entries: list[KBEntry]) -> "TfidfIndex":
        docs = {e.id: (_features(e.question), _features(e.answer)) for e in entries}
        df: Counter[str] = Counter()
        for q_terms, a_terms in docs.values():
            df.update(set(q_terms) | set(a_terms))
        n = max(len(docs), 1)
        idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        default_idf = math.log(1 + n) + 1.0
        vectors = {
            eid: (_vector(q_terms, idf, default_idf), _vector(q_terms * 2 + a_terms, idf, default_idf))
            for eid, (q_terms, a_terms) in docs.items()
        }
        exact: dict[str, str] = {}
        for entry in entries:
            exact.setdefault(normalize(entry.question), entry.id)
        return cls(idf, vectors, default_idf, exact)

    def __len__(self) -> int:
        return len(self.vectors)

    def search(self, query: str, k: int = 3) -> list[SearchHit]:
        # An exact question match wins outright (this also covers questions
        # made only of stop-words, such as "who are you?").
        exact_id = self.exact.get(normalize(query))
        qvec = _vector(_features(query), self.idf, self.default_idf)
        if not qvec:
            return [SearchHit(exact_id, 1.0)] if exact_id else []
        scored = [SearchHit(exact_id, 1.0)] if exact_id else []
        for eid, (q_vec, full_vec) in self.vectors.items():
            if eid == exact_id:
                continue
            score = QUESTION_WEIGHT * _cosine(qvec, q_vec) + (1 - QUESTION_WEIGHT) * _cosine(qvec, full_vec)
            if score > 0:
                scored.append(SearchHit(eid, round(min(score, 1.0), 4)))
        scored.sort(key=lambda h: (-h.score, h.entry_id))
        return scored[:k]

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "default_idf": self.default_idf,
            "idf": self.idf,
            "vectors": {eid: [q, f] for eid, (q, f) in self.vectors.items()},
            "exact": self.exact,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TfidfIndex":
        if data.get("backend") != cls.backend:
            raise ValueError(f"unsupported index backend: {data.get('backend')!r}")
        vectors = {eid: (pair[0], pair[1]) for eid, pair in data["vectors"].items()}
        return cls(data["idf"], vectors, float(data["default_idf"]), data.get("exact", {}))
