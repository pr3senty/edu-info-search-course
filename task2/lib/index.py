from pyroaring import BitMap

from lib.lsm import MergeLSMTree
from .analyzer import TextAnalyzer

class InvertedIndex:
    def __init__(
            self, 
            lsm: MergeLSMTree, 
            analyzer: TextAnalyzer, 
            documents_count: int = 0
        ):
        self.lsm = lsm
        self.analyzer = analyzer
        self.doc_count = documents_count

    def add_document(self, doc_text: str) -> int:
        doc_id = self.doc_count
        self.doc_count += 1

        terms = set(self.analyzer.terms(doc_text))  # уникальные термины
        for term in terms:
            self.lsm.put(term, doc_id)

        return doc_id

    def postings(self, term: str) -> BitMap:
        bm = BitMap()

        b = self.lsm.get(term)
        if b is not None:
            bm |= BitMap.deserialize(b)

        return bm

    def all_docs(self) -> BitMap:
        return BitMap(range(self.doc_count))