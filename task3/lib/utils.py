import re
from pyroaring import BitMap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .inverted_index import InvertedIndex


def merge_roaring_bytes(vals: list[bytes]) -> bytes | None:
    bm = BitMap()
    for b in vals:
        bm |= BitMap.deserialize(b)
    if len(bm) == 0:
        return None
    return bm.serialize()

def wildcard_to_regex(pattern: str) -> re.Pattern:
    escaped = re.escape(pattern.lower())
    escaped = escaped.replace(r"\*", ".*")
    return re.compile(rf"^{escaped}$")

def wildcard_kgrams(pattern: str, k: int) -> set[str]:
    pattern = pattern.lower()
    parts = pattern.split("*")

    grams = set()
    # если шаблон не начинается с *, можно использовать $
    if not pattern.startswith("*") and parts[0]:
        left = "$" + parts[0]
        if len(left) >= k:
            for i in range(len(left) - k + 1):
                grams.add(left[i: i + k])

    for part in parts:
        if len(part) >= k:
            for i in range(len(part) - k + 1):
                grams.add(part[i: i + k])

    # если шаблон не заканчивается на *, можно использовать $
    if not pattern.endswith("*") and parts[-1]:
        right = parts[-1] + "$"
        if len(right) >= k:
            for i in range(len(right) - k + 1):
                grams.add(right[i: i + k])

    return grams

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

def eval_rpn(rpn: list[str], idx: "InvertedIndex") -> BitMap:
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