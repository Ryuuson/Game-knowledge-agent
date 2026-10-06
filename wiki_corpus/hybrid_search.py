"""Dependency-free Chinese/English BM25 and reciprocal rank fusion.

Chinese characters and adjacent bigrams preserve terms without a tokenizer
download. RRF combines ranks, never treats BM25 or fusion scores as cosine.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence


def tokenize(text: str) -> list[str]:
    tokens = []
    for part in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text.casefold()):
        if "\u4e00" <= part[0] <= "\u9fff":
            tokens.extend(part)
            tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


class KeywordIndex:
    """Build BM25 postings once and visit only documents matching query tokens."""

    def __init__(self, chunks: Sequence[dict], *, k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for index, chunk in enumerate(self.chunks):
            title = str(chunk.get("title", ""))
            terms = tokenize(" ".join([title, title, str(chunk.get("section_path") or ""),
                                       str(chunk.get("text", ""))]))
            self.lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                self.postings[term].append((index, frequency))
        self.average_length = sum(self.lengths) / max(1, len(self.lengths)) or 1.0

    def search(self, query: str, *, top_k: int = 5) -> list[dict]:
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        scores: dict[int, float] = defaultdict(float)
        count = len(self.chunks)
        for term in set(tokenize(query)):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            idf = math.log(1 + (count - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings:
                normalization = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                scores[index] += idf * frequency * (self.k1 + 1) / (frequency + normalization)
        indexes = sorted(scores, key=lambda i: (-scores[i], i))[:top_k]
        return [{**self.chunks[i], "score": 0.0, "keyword_score": scores[i]} for i in indexes]


def reciprocal_rank_fusion(dense_hits: Sequence[dict], keyword_hits: Sequence[dict],
                           *, top_k: int = 5, rrf_k: int = 60) -> list[dict]:
    if top_k < 1 or rrf_k < 1:
        raise ValueError("top_k and rrf_k must be positive")
    fused: dict[str, dict] = {}
    for channel, hits in (("dense", dense_hits), ("keyword", keyword_hits)):
        seen = set()
        for rank, hit in enumerate(hits, 1):
            chunk_id = str(hit["chunk_id"])
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            if chunk_id not in fused:
                fused[chunk_id] = {**hit, "score": 0.0, "fusion_score": 0.0}
            item = fused[chunk_id]
            item["fusion_score"] += 1 / (rrf_k + rank)
            item[f"{channel}_rank"] = rank
            if channel == "dense":
                item["score"] = float(hit.get("score", 0.0))
            else:
                item["keyword_score"] = float(hit.get("keyword_score", 0.0))
    return sorted(fused.values(), key=lambda h: -h["fusion_score"])[:top_k]
