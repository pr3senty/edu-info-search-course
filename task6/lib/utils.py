from typing import Union


def encode_varint(x: int) -> bytes:
    if x < 0:
        raise ValueError("varint supports only non-negative integers")
    
    out = bytearray()
    while True:
        b = x & 0x7F
        x >>= 7
        if x:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)

def decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    shift = 0
    value = 0

    while True:
        if pos >= len(data):
            raise ValueError("unexpected end of varint")
        
        b = data[pos]
        pos += 1

        value |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return value, pos
        
        shift += 7
        if shift > 63:
            raise ValueError("varint is too long")
        
def encode_postings(postings: list[tuple[int, list[int]]]) -> bytes:

    out = bytearray()
    out += encode_varint(len(postings))

    prev_doc = 0
    for doc_id, positions in postings:
        if not positions:
            raise ValueError("positions list must not be empty")
        
        doc_gap = doc_id - prev_doc
        if doc_gap < 0:
            raise ValueError("postings must be sorting by doc_id")
        
        out += encode_varint(doc_gap)
        prev_doc = doc_id

        out += encode_varint(len(positions))

        prev_pos = 0
        for pos in positions:
            gap = pos - prev_pos
            if gap < 0:
                raise ValueError("positions must be sorted ascending")
            out += encode_varint(gap)
            prev_pos = pos

    return bytes(out)

def decode_postings(data: bytes) -> list[tuple[int, list[int]]]:
    pos = 0
    doc_freq, pos = decode_varint(data, pos)

    out: list[tuple[int, list[int]]] = []

    prev_doc = 0
    for _ in range(doc_freq):
        doc_gap, pos = decode_varint(data, pos)
        doc_id = prev_doc + doc_gap
        prev_doc = doc_id

        tf, pos = decode_varint(data, pos)
        if tf <= 0:
            raise ValueError("tf must be positive")
        
        positions = []
        prev_pos = 0
        for _ in range(tf):
            gap, pos = decode_varint(data, pos)
            cur = prev_pos + gap
            positions.append(cur)
            prev_pos = cur
        
        out.append((doc_id, positions))

    return out


def merge_sorted_positions(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    out: list[int] = []

    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    
    while i < len(a):
        out.append(a[i])
        i += 1

    while j < len(b):
        out.append(b[j])
        j += 1
    
    return out


def merge_postings_lists(
    a: list[tuple[int, list[int]]],
    b: list[tuple[int, list[int]]]
) -> list[tuple[int, list[int]]]:
    i = j = 0
    out: list[tuple[int, list[int]]] = []

    while i < len(a) and j < len(b):
        doc_a, postings_a = a[i]
        doc_b, postings_b = b[j]

        if doc_a == doc_b:
            out.append((doc_a, merge_sorted_positions(postings_a, postings_b)))
            i += 1
            j += 1
        elif doc_a < doc_b:
            out.append((doc_a, postings_a))
            i += 1
        else:
            out.append((doc_b, postings_b))
            j += 1
    
    while i < len(a):
        out.append(a[i])
        i += 1
    
    while j < len(b):
        out.append(b[j])
        j += 1

    return out

def merge_postings_bytes(vals: list[bytes]) -> Union[bytes, None]:
    if not vals:
        return None
    
    cur: list[tuple[int, list[int]]] = []
    for raw in vals:
        decoded = decode_postings(raw)
        cur = merge_postings_lists(cur, decoded)
    
    if not cur:
        return None
    
    return encode_postings(cur)

def mem_merge_postings(prev: Union[bytes, None], new: bytes) -> bytes:
    if prev is None:
        return new
    
    merged = merge_postings_bytes([prev, new])
    if merged is None:
        raise RuntimeError("unexpected None in mem_merge_postings")
    
    return merged
