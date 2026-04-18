from collections import defaultdict
from typing import Iterable

from .lsm import MergeLSMTree
from .analyzer import TextAnalyzer
from .utils import encode_postings


def intersect_docs(
    a: list[tuple[int, list[int]]],
    b: list[tuple[int, list[int]]]
) -> list[tuple[int, list[int]]]:
    i = j = 0
    out: list[tuple[int, list[int]]] = []

    while i < len(a) and j < len(b):
        doc_a, _ = a[i]
        doc_b, _ = b[j]

        if doc_a == doc_b:
            out.append((doc_a, []))
            i += 1
            j += 1
        elif doc_a < doc_b:
            i += 1
        else:
            j += 1
    
    return out

def union_docs(
    a: list[tuple[int, list[int]]],
    b: list[tuple[int, list[int]]]
) -> list[tuple[int, list[int]]]:
    i = j = 0
    out: list[tuple[int, list[int]]] = []

    while i < len(a) and j < len(b):
        doc_a, _ = a[i]
        doc_b, _ = b[j]

        if doc_a == doc_b:
            out.append((doc_a, []))
            i += 1
            j += 1
        elif doc_a < doc_b:
            out.append((doc_a, []))
            i += 1
        else:
            out.append((doc_b, []))
            j += 1

    while i < len(a):
        out.append((a[i][0], []))
        i += 1
    
    while j < len(b):
        out.append((b[j][0], []))
        j += 1
    
    return out

def subtract_docs(
    a: list[tuple[int, list[int]]],
    b: list[tuple[int, list[int]]]
) -> list[tuple[int, list[int]]]:
    i = j = 0
    out: list[tuple[int, list[int]]] = []

    while i < len(a) and j < len(b):
        doc_a, _ = a[i]
        doc_b, _ = b[j]

        if doc_a == doc_b:
            i += 1
            j += 1
        elif doc_a < doc_b:
            out.append((doc_a, []))
            i += 1
        else:
            j += 1

    while i < len(a):
        out.append((a[i][0], []))
        i += 1

    return out

def intersect_positions(
    a: list[int],
    b: list[int],
    k: int = 1
) -> list[int]:
    i = j = 0
    out: list[int] = []

    while i < len(a) and j < len(b):
        diff = b[j] - a[i]

        if diff == k:
            out.append(b[j])
            i += 1
            j += 1
        elif diff < k:
            j += 1
        else:
            i += 1
    
    return out

def positional_intersect(
    a: list[tuple[int, list[int]]],
    b: list[tuple[int, list[int]]],
    k: int = 1
) -> list[tuple[int, list[int]]]:
    i = j = 0
    out: list[tuple[int, list[int]]] = []

    while i < len(a) and j < len(b):
        doc_a, pos_a = a[i]
        doc_b, pos_b = b[j]

        if doc_a == doc_b:
            matched = intersect_positions(pos_a, pos_b, k=k)
            if matched:
                out.append((doc_a, matched))
            i += 1
            j += 1
        elif doc_a < doc_b:
            i += 1
        else:
            j += 1

    return out 


class PositionalIndex:

    def __init__( 
        self,
        lsm: MergeLSMTree,
        analyzer: TextAnalyzer,
        documents_count: int = 0
    ):
        self.lsm = lsm
        self.analyzer = analyzer
        self.doc_count = documents_count

    def add_document(self, doc_text: str):
        doc_id = self.doc_count
        self.doc_count += 1

        positions_by_term: dict[str, list[int]] = {}
        for pos, term in enumerate(self.analyzer.terms(doc_text)):
            positions_by_term.setdefault(term, []).append(pos)

        for term, positions in positions_by_term.items():
            payload = encode_postings([(doc_id, positions)])
            self.lsm.put(term, payload)
        
        return doc_id
    
    def postings(self, term: str) -> list[tuple[int, list[int]]]:
        from .utils import decode_postings

        data = self.lsm.get(term)
        if data is None:
            return []
        
        return decode_postings(data)
    
    def search_term(self, term: str) -> list[int]:
        normalized = self.analyzer.terms(term)
        if not normalized:
            return []
        
        postings = self.postings(normalized[0])
        return [doc_id for doc_id, _ in postings]
    
    def and_query(self, terms: Iterable[str]) -> list[int]:
        normalized_terms: list[str] = []
        for term in terms:
            xs = self.analyzer.terms(term)
            if not xs:
                return []
            normalized_terms.append(xs[0])
        
        if not normalized_terms:
            return []
        
        cur = self.postings(normalized_terms[0])
        for term in terms[1:]:
            cur = intersect_docs(cur, self.postings(term))
            if not cur:
                return []

        return [doc_id for doc_id, _ in cur]
    
    def or_query(self, terms: Iterable[str]) -> list[int]:
        normalized_terms: list[str] = []
        for term in terms:
            xs = self.analyzer.terms(term)
            if not xs:
                return []
            normalized_terms.append(xs[0])
        
        if not normalized_terms:
            return []
        
        cur: list[tuple[int, list[int]]] = []
        for term in normalized_terms:
            cur = union_docs(cur, self.postings(term))
        
        return [doc_id for doc_id, _ in cur]

    def phrase_query(self, phrase: str) -> list[int]:
        terms = self.analyzer.terms(phrase)
        if not terms:
            return []
        
        cur = self.postings(terms[0])
        for term in terms[1:]:
            cur = positional_intersect(cur, self.postings(term))
            if not cur:
                return []
            
        return [doc_id for doc_id, _ in cur]

    
        