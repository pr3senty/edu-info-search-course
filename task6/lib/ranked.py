import math
import heapq
from collections import Counter, defaultdict
from typing import Union

from .index import PositionalIndex


def term_idf(N: int, df: int) -> float:
    return math.log((N + 1) / (df + 1)) + 1.0


def tf_weight(tf: int) -> float:
    return 1.0 + math.log(tf)


def term_doc_weight(tf: int, idf: float) -> float:
    return (1.0 + math.log(tf)) * idf


def compute_doc_norms(index: PositionalIndex) -> dict[int, float]:
    N = index.doc_count
    norms_sq = defaultdict(float)

    for term in index.iter_terms():
        postings = index.postings(term)
        if not postings:
            continue

        df = len(postings)
        idf = term_idf(N, df)

        for doc_id, positions in postings:
            tf = len(positions)
            w_td = tf_weight(tf) * idf
            norms_sq[doc_id] += w_td * w_td
    
    return {doc_id: math.sqrt(v) for doc_id, v in norms_sq.items()}


def score_candidates_cosine(
    index: PositionalIndex, 
    query_terms: list[str],
    candidates: set[int],
    doc_norms: dict[int, float]
) -> list[tuple[int, float]]:
    N = index.doc_count
    q_tf = Counter(query_terms)

    scores = defaultdict(float)
    q_weights = {}

    for term, qtf in q_tf.items():
        postings = index.postings(term)
        if not postings:
            continue

        df = len(postings)
        idf = term_idf(N, df)

        q_weights[term] = tf_weight(qtf) * idf
    
    q_norm = math.sqrt(sum(w * w for w in q_weights.values()))
    if q_norm == 0:
        return []
    
    for term, w_tq in q_weights.items():
        postings = index.postings(term)
        df = len(postings)
        idf = term_idf(N, df)

        for doc_id, positions in postings:
            if doc_id not in candidates:
                continue

            tf = len(positions)
            w_td = tf_weight(tf) * idf
            scores[doc_id] += w_td * w_tq

    results = []
    for doc_id, dot in scores.items():
        d_norm = doc_norms.get(doc_id)
        if not d_norm:
            continue

        score = dot / (d_norm * q_norm)
        results.append((doc_id, score))

    return results


def build_champion_list(
    postings: list[tuple[int, list[int]]],
    N: int, r: int
) -> list[int]:
    df = len(postings)
    if df == 0:
        return []
    
    idf = term_idf(N, df)

    scored = []
    for doc_id, positions in postings:
        tf = len(positions)
        scored.append((term_doc_weight(tf, idf), doc_id))
    scored.sort(reverse=True)

    return [doc_id for _, doc_id in scored[:r]]


def build_tiers(
    postings: list[tuple[int, list[int]]],
    tier_sizes: list[int], N: int
) -> list[list[tuple[int, list[int]]]]:
    df = len(postings)
    if df == 0:
        return []
    
    idf = term_idf(N, df)

    scored = []
    for doc_id, positions in postings:
        tf = len(positions)
        score = term_doc_weight(tf, idf)
        scored.append((score, doc_id, positions))
    scored.sort(reverse=True)

    tiers: list[list[tuple[int, list[int]]]] = []
    start = 0
    for size in tier_sizes:
        chunk = scored[start: start + size]
        tiers.append([(doc_id, positions) for _, doc_id, positions in chunk])
        start += size

    if start < len(scored):
        chunk = scored[start:]
        tiers.append([(doc_id, positions) for _, doc_id, positions in chunk])

    # tiers postings must stay sorted by doc_id
    for i in range(len(tiers)):
        tiers[i].sort(key=lambda x: x[0])
    
    return tiers


class RankedRetriever:

    def __init__(
            self, 
            index: PositionalIndex,
            champion_size: int = 32,
            tier_sizes: Union[list[int], None] = None
        ):
        self.index = index
        self.champion_size = champion_size
        self.tier_sizes = tier_sizes or [16, 64]

        self.champions: dict[str, list[int]] = {}
        self.tiers: dict[str, list[list[tuple[int, list[int]]]]] = {}
        self.doc_norms: dict[int, float] = {}
    
    def build(self) -> None:
        self.champions = {}
        self.tiers = {}
        self.doc_norms = compute_doc_norms(self.index)
        N = self.index.doc_count

        for term in self.index.iter_terms():
            postings = self.index.postings(term)
            self.champions[term] = build_champion_list(
                postings, N, self.champion_size
            )
            self.tiers[term] = build_tiers(
                postings, self.tier_sizes, N
            )

    def rank_query_exact(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        query_terms = self.index.analyzer.terms(query)
        if not query_terms:
            return []
        
        candidates = set()
        for term in set(query_terms):
            for doc_id, _ in self.index.postings(term):
                candidates.add(doc_id)
        
        scored = score_candidates_cosine(
            self.index,
            query_terms,
            candidates,
            self.doc_norms
        )
        return heapq.nlargest(k, scored, key=lambda x: x[1])
    

    def rank_query_inexact(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        query_terms = self.index.analyzer.terms(query)
        if not query_terms:
            return []
        
        candidates: set[int] = set()

        for term in set(query_terms):
            candidates.update(self.champions.get(term, []))
        
        if len(candidates) < k:
            max_tiers = 0
            for term in set(query_terms):
                max_tiers = max(max_tiers, len(self.tiers.get(term, [])))

            for tier_idx in range(max_tiers):
                for term in set(query_terms):
                    term_tiers = self.tiers.get(term, [])
                    if tier_idx >= len(term_tiers):
                        continue
                    for doc_id, _ in term_tiers[tier_idx]:
                        candidates.add(doc_id)
                
                if len(candidates) >= 2 * k:
                    break
        
        scored = score_candidates_cosine(
            self.index,
            query_terms,
            candidates,
            self.doc_norms
        )
        return heapq.nlargest(k, scored, lambda x: x[1])