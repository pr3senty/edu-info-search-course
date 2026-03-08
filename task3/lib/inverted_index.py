from pyroaring import BitMap
from typing import Union, Iterator

from lib.lsm import MergeLSMTree
from .analyzer import TextAnalyzer
from .kgram_index import KGramIndex
from . import utils

class InvertedIndex:
    def __init__(
            self, 
            lsm: MergeLSMTree, 
            analyzer: TextAnalyzer,
            kgram_index: Union[KGramIndex, None] = None,
            documents_count: int = 0
        ):
        self.lsm = lsm
        self.analyzer = analyzer
        self.kgram = kgram_index
        self.doc_count = documents_count

    def rebuild_kgram(self) -> None:
        if self.kgram is None:
            return

        self.kgram.clear()
        for term, _ in self.lsm.scan("", "\uffff", limit=None):
            self.kgram.add_term(term)

    def add_document(self, doc_text: str) -> int:
        doc_id = self.doc_count
        self.doc_count += 1

        terms = set(self.analyzer.terms(doc_text))  # уникальные термины
        for term in terms:
            self.lsm.put(term, doc_id)
            if self.kgram is not None:
                self.kgram.add_term(term)

        return doc_id

    def postings(self, term: str) -> BitMap:
        bm = BitMap()

        b = self.lsm.get(term)
        if b is not None:
            bm |= BitMap.deserialize(b)

        return bm
    
    def prefix_query(self, prefix: str, limit: Union[int, None] = None) -> BitMap:
        prefix = prefix.lower()

        lo = prefix
        hi = prefix + "\uffff"

        bm = BitMap()
        for _, postings_bytes in self.lsm.scan(lo, hi, limit=limit):
            bm |= BitMap.deserialize(postings_bytes)

        return bm
    
    def prefix_terms(self, prefix: str, limit: Union[int, None] = None) -> Iterator[str]:
        prefix = prefix.lower()

        lo = prefix
        hi = prefix + "\uffff"

        for term, _ in self.lsm.scan(lo, hi, limit=limit):
            yield term
    
    def wildcard_terms(self, pattern: str, limit: Union[int, None] = None) -> Iterator[str]:
        if self.kgram is None:
            raise ValueError("KGramIndex is not configured")
        
        pattern = pattern.lower()
        if pattern.count("*") == 1 and pattern.endswith("*"):
            prefix = pattern[:-1]
            yield from self.prefix_terms(prefix, limit=limit)
            return
        
        grams = utils.wildcard_kgrams(pattern, self.kgram.k)
        regex = utils.wildcard_to_regex(pattern)

        if not grams:
            # полный поиск по всем terms
            for term, _ in self.lsm.scan("", "\uffff", limit=limit):
                if regex.fullmatch(term):
                    yield term
            return

        candidates = [self.kgram.get_terms(g) for g in grams]
        if not candidates:
            return
        
        candidates = set.intersection(*candidates)
        for i, term in enumerate(sorted(candidates)):
            if regex.fullmatch(term):
                yield term
            if limit is not None and i == limit:
                return
    
    def wildcard_query(self, pattern: str) -> BitMap:
        bm = BitMap()

        for term in self.wildcard_terms(pattern):
            bm |= self.postings(term)

        return bm

    def all_docs(self) -> BitMap:
        return BitMap(range(self.doc_count))