import re
from datetime import datetime, timedelta
from pyroaring import BitMap

from .index import InvertedIndex


DATE_RE = re.compile(r"^DATE\[(\d{8}),(\d{8})\]$", re.IGNORECASE)
APPEARED_RE = re.compile(r"^APPEARED\[(\d{8}),(\d{8})\]$", re.IGNORECASE)
VALID_RE = re.compile(r"^VALID\[(\d{8}),(\d{8})\]$", re.IGNORECASE)

MIN_DATE = "00010101"


def merge_roaring_bytes(vals: list[bytes]) -> bytes | None:
    bm = BitMap()
    for b in vals:
        bm |= BitMap.deserialize(b)
    if len(bm) == 0:
        return None
    return bm.serialize()

OPS = {"NOT": 3, "AND": 2, "OR": 1}

def prev_day(date_str: str) -> str:
    d = datetime.strptime(date_str, "%Y%m%d").date()
    return (d - timedelta(days=1)).strftime("%Y%m%d")

def tokenize_query(q: str) -> list[str]:
    parts = re.findall(
        r"""
        DATE\[\d{8},\d{8}\]
        |APPEARED\[\d{8},\d{8}\]
        |VALID\[\d{8},\d{8}\]
        |\(
        |\)
        |AND
        |OR
        |NOT
        |[A-Za-zА-Яа-яЁё0-9_]+
        """,
        q,
        flags=re.IGNORECASE | re.VERBOSE
    )
    return parts

def to_rpn(tokens: list[str]) -> list[str]:
    out = []
    st = []

    for t in tokens:
        tu = t.upper()

        if t == "(":
            st.append(t)

        elif t == ")":
            while st and st[-1] != "(":
                out.append(st.pop())
            if not st:
                raise ValueError("Mismatched parentheses")
            st.pop()

        elif tu in OPS:
            while st and st[-1].upper() in OPS and OPS[st[-1].upper()] >= OPS[tu]:
                out.append(st.pop())
            st.append(t)

        else:
            out.append(t)

    while st:
        if st[-1] == "(":
            raise ValueError("Mismatched parentheses")
        out.append(st.pop())

    return out


def eval_rpn(rpn: list[str], idx: InvertedIndex) -> BitMap:
    stack: list[BitMap] = []
    all_docs = idx.all_docs()

    for t in rpn:
        tu = t.upper()

        if tu == "NOT":
            if not stack:
                raise ValueError("NOT requires one operand")
            a = stack.pop()
            stack.append(all_docs - a)

        elif tu == "AND":
            if len(stack) < 2:
                raise ValueError("AND requires two operands")
            b = stack.pop()
            a = stack.pop()
            stack.append(a & b)

        elif tu == "OR":
            if len(stack) < 2:
                raise ValueError("OR requires two operands")
            b = stack.pop()
            a = stack.pop()
            stack.append(a | b)

        else:
            m = DATE_RE.match(t)
            if m:
                lo, hi = m.groups()
                stack.append(idx.postings_range("date", lo, hi))
                continue

            m = APPEARED_RE.match(t)
            if m:
                lo, hi = m.groups()
                stack.append(idx.postings_range("start_date", lo, hi))
                continue

            m = VALID_RE.match(t)
            if m:
                lo, hi = m.groups()

                started = idx.postings_range("start_date", MIN_DATE, hi)
                ended_before = idx.postings_range("end_date", MIN_DATE, prev_day(lo))

                stack.append(started - ended_before)
                continue

            term_norm = idx.analyzer.terms(t.lower())

            if not term_norm:
                stack.append(BitMap())
            else:
                result = BitMap()
                for term in term_norm:
                    result |= idx.postings(term)
                stack.append(result)

    if len(stack) != 1:
        raise ValueError("Invalid RPN expression")

    return stack[0]