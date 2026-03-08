

class KGramIndex:

    def __init__(self, k: int = 3):
        self.k = k
        self.index: dict[str, set[str]] = {}
    
    def _grams(self, term: str) -> set[str]:
        padded = f"${term}$"
        if len(padded) < self.k:
            return {padded}

        return {
            padded[i: i + self.k]
            for i in range(len(padded) - self.k + 1)
        }
    
    def add_term(self, term: str):
        for gram in self._grams(term):
            self.index.setdefault(gram, set()).add(term)
    
    def add_terms(self, terms: list[str]):
        for term in terms:
            self.add_term(term)

    def get_terms(self, gram: str) -> set[str]:
        return self.index.get(gram, set())
    
    def clear(self) -> None:
        self.index.clear()