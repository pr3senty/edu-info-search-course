import struct
import hashlib
from typing import Union

from .helpers import (
    U32, U64
)

class BloomFilter:
    """
    Simple Bloom filter.
    Stores a bitarray (bytes) and k hash functions derived from blake2b.
    """
    __slots__ = ("m_bits", "k", "bits")

    def __init__(self, m_bits: int, k: int, bits: Union[bytearray, None] = None):
        self.m_bits = m_bits
        self.k = k
        self.bits = (bits or bytearray((m_bits + 7) // 8))


    @staticmethod
    def for_n(n: int, bits_per_key: int = 10) -> "BloomFilter":
        n = max(1, n)

        m_bits = max(64, n * bits_per_key)
        k = max(2, min(12, int(0.693 * m_bits / n)))

        return BloomFilter(m_bits=m_bits, k=k)

    @staticmethod
    def for_keys(keys: list[str], bits_per_key: int = 10) -> "BloomFilter":
        n = max(1, len(keys))
        m_bits = max(64, n * bits_per_key)

        # k ≈ ln(2) * m/n
        k = max(2, min(12, int(0.693 * m_bits / n)))

        bf = BloomFilter(m_bits=m_bits, k=k)
        for key in keys:
            bf.add(key)

        return bf

    def _hashes(self, key: str) -> list[int]:
        kb = key.encode("utf-8")
        out = []

        # Produce 8 bytes from blake2b with different salts
        for i in range(self.k):
            h = hashlib.blake2b(kb, digest_size=8, person=b"LSM_BF__",
                                salt=U32.pack(i)).digest()
            x = U64.unpack(h)[0]
            out.append(x % self.m_bits)

        return out

    def add(self, key: str) -> None:
        for b in self._hashes(key):
            self.bits[b >> 3] |= (1 << (b & 7))

    def may_contain(self, key: str) -> bool:
        for b in self._hashes(key):
            if (self.bits[b >> 3] & (1 << (b & 7))) == 0:
                return False
    
        return True

    def serialize(self) -> bytes:
        # m_bits (u32), k (u32), len(bits) (u32), bits
        return struct.pack("<III", self.m_bits, self.k, len(self.bits)) + bytes(self.bits)

    @staticmethod
    def deserialize(data: bytes) -> "BloomFilter":
        m_bits, k, blen = struct.unpack_from("<III", data, 0)

        bits = bytearray(data[12:12+blen])
    
        return BloomFilter(m_bits=m_bits, k=k, bits=bits)