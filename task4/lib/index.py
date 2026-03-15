from typing import Union, Literal
from pyroaring import BitMap

from lib.lsm import MergeLSMTree
from .analyzer import TextAnalyzer


class IndexKey:

    @staticmethod
    def get(type: Literal["term", "date", "start_date", "end_date"], value: str) -> str:
        return f"{IndexKey.get_prefix(type)}:{value}"
    
    @staticmethod
    def get_prefix(type: Literal["term", "date", "start_date", "end_date"]):

        if type == "date":
            return "d"
        if type == "term":
            return "t"
        if type == "start_date":
            return "s"
        if type == "end_date":
            return "e"
        
        raise ValueError(f"Unexpected type for a prefix: type={type}")


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

    def add_document(
            self, 
            doc_text: str,
            date: Union[str, None] = None,
            start_date: Union[str, None] = None,
            end_date: Union[str, None] = None
        ) -> int:
        doc_id = self.doc_count
        self.doc_count += 1

        terms = set(self.analyzer.terms(doc_text))  # уникальные термины
        for term in terms:
            self.lsm.put(IndexKey.get("term", term), doc_id)

        if date is not None:
            self.lsm.put(IndexKey.get("date", date), doc_id)
        
        if start_date is not None:
            self.lsm.put(IndexKey.get("start_date", start_date), doc_id)
        
        if end_date is not None:
            self.lsm.put(IndexKey.get("end_date", end_date), doc_id)

        return doc_id

    def postings(self, term: str) -> BitMap:
        bm = BitMap()

        b = self.lsm.get(IndexKey.get("term", term))
        if b is not None:
            bm |= BitMap.deserialize(b)

        return bm
    
    def postings_exact_date(self, date: str) -> BitMap:
        bm = BitMap()

        b = self.lsm.get(IndexKey.get("date" ,date))
        if b is not None:
            bm |= BitMap.deserialize(b)

        return bm
    
    def postings_range(
            self, 
            kind: Literal["term", "date", "start_date", "end_date"], 
            lo: str, hi: str
        ) -> BitMap:
        bm = BitMap()

        for _, v in self.lsm.scan(IndexKey.get(kind, lo), IndexKey.get(kind, hi)):
            bm |= BitMap.deserialize(v)

        return bm

    def all_docs(self) -> BitMap:
        return BitMap(range(self.doc_count))