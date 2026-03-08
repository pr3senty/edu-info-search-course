import os
import bisect
from dataclasses import dataclass
from typing import Union, Iterator, Callable

from .helpers import (
    encode_kv, decode_kv,
    HEADER, MAGIC, U32, U64, FOOTER
)
from .bloom import BloomFilter

@dataclass(frozen=True)
class IndexEntry:
    first_key: str
    offset: int
    length: int

class SSTableWriter:
    """
    SSTable file layout:
      header (MAGIC, block_size)
      data blocks...
      bloom section (serialized BloomFilter)
      index section:
        u32 count
        repeated:
          u32 first_key_len + first_key_bytes
          u64 offset
          u32 length
      footer: (index_offset u64, bloom_offset u64)
    """

    def __init__(self, path: str, block_size: int = 32 * 1024, bits_per_key: int = 10):
        self.path = path
        self.block_size = block_size
        self.bits_per_key = bits_per_key

    def write(self, items: list[tuple[str, bytes]]) -> None:
        # items must be sorted by key
        keys = [k for k, _ in items]
        bf = BloomFilter.for_keys(keys, bits_per_key=self.bits_per_key)

        index: list[IndexEntry] = []
        with open(self.path, "wb") as f:
            f.write(HEADER.pack(MAGIC, self.block_size))

            block_buf = bytearray()
            block_first_key: Union[str, None] = None
            block_offset = f.tell()

            def flush_block():
                nonlocal block_buf, block_first_key, block_offset
                if not block_buf:
                    return
                
                f.write(U32.pack(len(block_buf)))
                f.write(block_buf)
                index.append(IndexEntry(
                    first_key=block_first_key if block_first_key is not None else "",
                    offset=block_offset,
                    length=4 + len(block_buf),
                ))

                block_buf = bytearray()
                block_first_key = None
                block_offset = f.tell()

            # data
            for k, v in items:
                rec = encode_kv(k, v)
                if block_first_key is None:
                    block_first_key = k

                if len(block_buf) + len(rec) > self.block_size and block_buf:
                    flush_block()
                    block_first_key = k

                block_buf.extend(rec)

            flush_block()

            # bloom
            bloom_offset = f.tell()
            bloom_data = bf.serialize()
            f.write(U32.pack(len(bloom_data)))
            f.write(bloom_data)

            # index
            index_offset = f.tell()
            f.write(U32.pack(len(index)))
            for e in index:
                kb = e.first_key.encode("utf-8")
                f.write(U32.pack(len(kb)))
                f.write(kb)
                f.write(U64.pack(e.offset))
                f.write(U32.pack(e.length))

            # footer
            f.write(FOOTER.pack(index_offset, bloom_offset))

    def write_from(self, make_iter: Callable[[], Iterator[tuple[str, bytes]]]):
        n = 0
        prev = None
        for k, _ in make_iter():
            if prev is not None and k < prev:
                raise ValueError("Items must be sorted by key")
            prev = k
            n += 1

        bf = BloomFilter.for_n(n, bits_per_key=self.bits_per_key)

        index: list[IndexEntry] = []
        with open(self.path, "wb") as f:
            f.write(HEADER.pack(MAGIC, self.block_size))

            block_buf = bytearray()
            block_first_key = None
            block_offset = f.tell()

            def flush_block():
                nonlocal block_buf, block_first_key, block_offset
                if not block_buf:
                    return
                f.write(U32.pack(len(block_buf)))
                f.write(block_buf)
                index.append(IndexEntry(
                    first_key=block_first_key,
                    offset=block_offset,
                    length=4 + len(block_buf)
                ))
                block_buf = bytearray()
                block_first_key = None
                block_offset = f.tell()

            # data
            prev = None
            for k, v in make_iter():
                if prev is not None and k < prev:
                    raise ValueError("Items must be sorted by key")
                prev = k

                bf.add(k)

                rec = encode_kv(k, v)
                if block_first_key is None:
                    block_first_key = k
                
                if len(block_buf) + len(rec) > self.block_size and block_buf:
                    flush_block()
                    block_first_key = k

                block_buf.extend(rec)
            
            flush_block()

            # bloom
            bloom_offset = f.tell()
            bloom_data = bf.serialize()
            f.write(U32.pack(len(bloom_data)))
            f.write(bloom_data)

            # index
            index_offset = f.tell()
            f.write(U32.pack(len(index)))
            for e in index:
                kb = e.first_key.encode("utf-8")
                f.write(U32.pack(len(kb)))
                f.write(kb)
                f.write(U64.pack(e.offset))
                f.write(U32.pack(e.length))

            f.write(FOOTER.pack(index_offset, bloom_offset))

class SSTableReader:
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "rb")
        self.block_size = None
        self.bloom: Union[BloomFilter, None] = None
        self.index: list[IndexEntry] = []
        self._block_cache: dict[int, list[tuple[str, bytes]]] = {}  # offset -> parsed records

        self._load_metadata()

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass

    def _load_metadata(self) -> None:
        self.f.seek(0)

        header = self.f.read(HEADER.size)
        magic, block_size = HEADER.unpack(header)
        if magic != MAGIC:
            raise ValueError(f"Bad SSTable magic in {self.path}")
        self.block_size = block_size

        # footer at end
        self.f.seek(-FOOTER.size, os.SEEK_END)
        index_offset, bloom_offset = FOOTER.unpack(self.f.read(FOOTER.size))

        # bloom
        self.f.seek(bloom_offset)
        bloom_len = U32.unpack(self.f.read(4))[0]
        bloom_data = self.f.read(bloom_len)
        self.bloom = BloomFilter.deserialize(bloom_data)

        # index
        self.f.seek(index_offset)
        count = U32.unpack(self.f.read(4))[0]
        idx: list[IndexEntry] = []
        for _ in range(count):
            klen = U32.unpack(self.f.read(4))[0]
            k = self.f.read(klen).decode("utf-8")
            off = U64.unpack(self.f.read(8))[0]
            ln = U32.unpack(self.f.read(4))[0]
            idx.append(IndexEntry(k, off, ln))
        self.index = idx

    def _index_first_keys(self) -> list[str]:
        return [e.first_key for e in self.index]

    def _read_block_records(self, entry: IndexEntry) -> list[tuple[str, bytes]]:
        # Small cache to avoid re-parsing for short range scans / repeated gets
        if entry.offset in self._block_cache:
            return self._block_cache[entry.offset]

        self.f.seek(entry.offset)
        blen = U32.unpack(self.f.read(4))[0]
        buf = self.f.read(blen)
        mv = memoryview(buf)
        pos = 0
        recs: list[tuple[str, bytes]] = []
        while pos < len(mv):
            k, v, pos = decode_kv(mv, pos)
            recs.append((k, v))

        # bounded cache
        if len(self._block_cache) >= 64:
            self._block_cache.pop(next(iter(self._block_cache)))

        self._block_cache[entry.offset] = recs
        return recs

    def get(self, key: str) -> Union[bytes, None]:
        assert self.bloom is not None
        if not self.bloom.may_contain(key):
            return None

        # find the last index entry with first_key <= key
        first_keys = self._index_first_keys()
        i = bisect.bisect_right(first_keys, key) - 1
        if i < 0:
            return None

        entry = self.index[i]
        recs = self._read_block_records(entry)
        keys = [k for k, _ in recs]

        j = bisect.bisect_left(keys, key)
        if j < len(recs) and recs[j][0] == key:
            return recs[j][1]
    
        return None
    
    def _iter_block_records(self, entry: IndexEntry) -> Iterator[tuple[str, bytes]]:
        self.f.seek(entry.offset)
        blen = U32.unpack(self.f.read(4))[0]

        buf = self.f.read(blen)
        mv = memoryview(buf)

        pos = 0
        while pos < len(mv):
            k, v, pos = decode_kv(mv, pos)
            yield k, v
    
    def iter_all(self) -> Iterator[tuple[str, bytes]]:
        """
        Yield sorted (key, value) over full keyspace
        """

        for entry in self.index:
            yield from self._iter_block_records(entry)

    def iter_range(self, lo: str, hi: str) -> Iterator[tuple[str, bytes]]:
        """
        Yield sorted (key, value) for lo <= key < hi from this SSTable.
        Reads only needed blocks.
        """
        first_keys = self._index_first_keys()
        i = bisect.bisect_right(first_keys, lo) - 1
        if i < 0:
            i = 0

        while i < len(self.index):
            entry = self.index[i]
            recs = self._read_block_records(entry)

            # fast skip if block entirely < lo or starts > hi
            for k, v in recs:
                if k < lo:
                    continue
                if k >= hi:
                    return

                yield (k, v)

            i += 1