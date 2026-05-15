import time
import struct

"""
Struct types:
 * < – little-endian
 * H – uint16
 * I – uint32
 * Q – uint64
"""
U32 = struct.Struct("<I")
U64 = struct.Struct("<Q")
FOOTER = struct.Struct("<QQ")  # index_offset, bloom_offset
MAGIC = b"SST1"
HEADER = struct.Struct("<4sI")  # magic, block_size

TOMBSTONE = b"\x00"

def now_ns() -> int:
    return time.perf_counter_ns()

def encode_kv(key: str, value: bytes) -> bytes:
    kb = key.encode("utf-8")

    # len(kb), kb, len(value), value
    return U32.pack(len(kb)) + kb + U32.pack(len(value)) + value

def decode_kv(buf: memoryview, pos: int) -> tuple[str, bytes, int]:
    klen = U32.unpack_from(buf, pos)[0]; pos += 4
    key = bytes(buf[pos: pos + klen]).decode("utf-8"); pos += klen

    vlen = U32.unpack_from(buf, pos)[0]; pos += 4
    val = bytes(buf[pos: pos + vlen]); pos += vlen

    return key, val, pos