import re

from .index import (
    PositionalIndex,
    intersect_docs, union_docs, subtract_docs
)


OPS = {"NOT": 3, "AND": 2, "OR": 1}


def tokenize_query(q: str) -> list[str]:
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
            if not st:
                raise ValueError("mismatched parentheses")
            st.pop()
        elif t in OPS:
            while st and st[-1] in OPS and OPS[st[-1]] >= OPS[t]:
                out.append(st.pop())
            st.append(t)
        else:
            out.append(t)

    while st:
        top = st.pop()
        if top == "(":
            raise ValueError("mismatched parentheses")
        out.append(top)

    return out


def all_docs_postings(idx: PositionalIndex) -> list[tuple[int, list[int]]]:
    return [(doc_id, []) for doc_id in range(idx.doc_count)]


def eval_rpn_postings(rpn: list[str], idx: PositionalIndex) -> list[tuple[int, list[int]]]:
    stack: list[list[tuple[int, list[int]]]] = []
    universe = all_docs_postings(idx)

    for t in rpn:
        if t == "NOT":
            if not stack:
                raise ValueError("bad query: NOT without operand")
            a = stack.pop()
            stack.append(subtract_docs(universe, a))

        elif t == "AND":
            if len(stack) < 2:
                raise ValueError("bad query: AND without two operands")
            b = stack.pop()
            a = stack.pop()
            stack.append(intersect_docs(a, b))

        elif t == "OR":
            if len(stack) < 2:
                raise ValueError("bad query: OR without two operands")
            b = stack.pop()
            a = stack.pop()
            stack.append(union_docs(a, b))

        else:
            term_norm = idx.analyzer.terms(t.lower())
            if not term_norm:
                stack.append([])
            else:
                stack.append(idx.postings(term_norm[0]))

    if len(stack) != 1:
        raise ValueError("bad query")

    return stack[0]