import re
from pyroaring import BitMap

from .index import InvertedIndex


def merge_roaring_bytes(vals: list[bytes]) -> bytes | None:
    bm = BitMap()
    for b in vals:
        bm |= BitMap.deserialize(b)
    if len(bm) == 0:
        return None
    return bm.serialize()

OPS = {"NOT": 3, "AND": 2, "OR": 1}

def tokenize_query(q: str) -> list[str]:
    # слова + операторы + скобки
    parts = re.findall(r"\(|\)|AND|OR|NOT|[A-Za-zА-Яа-яЁё]+", q.upper())
    return parts

def to_rpn(tokens: list[str]) -> list[str]:
    out = []
    st = []
    for t in tokens:
        if t == "(":
            st.append(t)
        elif t == ")":
            while st and st[-1] != "(":
                out.append(st.pop())
            st.pop()
        elif t in OPS:
            while st and st[-1] in OPS and OPS[st[-1]] >= OPS[t]:
                out.append(st.pop())
            st.append(t)
        else:
            out.append(t)
    while st:
        out.append(st.pop())
    return out

def eval_rpn(rpn: list[str], idx: InvertedIndex) -> BitMap:
    stack: list[BitMap] = []
    all_docs = idx.all_docs()

    for t in rpn:
        if t == "NOT":
            a = stack.pop()
            stack.append(all_docs - a)
        elif t == "AND":
            b = stack.pop(); a = stack.pop()
            stack.append(a & b)
        elif t == "OR":
            b = stack.pop(); a = stack.pop()
            stack.append(a | b)
        else:
            # надо прогнать через analyzer так же, как при индексации
            term_norm = idx.analyzer.terms(t.lower())

            # terms() может вернуть 0 (стоп-слово); возьмем первый
            if not term_norm:
                stack.append(BitMap())  # пусто
            else:
                stack.append(idx.postings(term_norm[0]))

    return stack[-1] if stack else BitMap()