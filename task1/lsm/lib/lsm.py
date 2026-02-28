import os
import time
import heapq
import random
from typing import Union, Iterator

from .helpers import TOMBSTONE
from .sstable import SSTableReader, SSTableWriter

class LSMTree:
    """
    Tiered LSM:
      - memtable (dict) for latest writes
      - when memtable reaches threshold, flush to new SSTable (run) in level 0 list
      - if too many runs, compact (merge all runs into one new run)
    """

    def __init__(
        self,
        dir_path: str,
        memtable_max_items: int = 50_000,
        run_max_count: int = 8,
        block_size: int = 32 * 1024,
        bits_per_key: int = 10,
    ):
        self.dir_path = dir_path
        os.makedirs(self.dir_path, exist_ok=True)

        self.memtable_max_items = memtable_max_items
        self.run_max_count = run_max_count
        self.block_size = block_size
        self.bits_per_key = bits_per_key

        self.mem: dict[str, bytes] = {}
        self.runs: list[SSTableReader] = []  # newest first

        self._load_existing_runs()

    def close(self) -> None:
        for r in self.runs:
            r.close()

    def _run_files(self) -> list[str]:
        files = [f for f in os.listdir(self.dir_path) if f.endswith(".sst")]
        files.sort()
        return [os.path.join(self.dir_path, f) for f in files]

    def _load_existing_runs(self) -> None:
        # Load existing runs (if any) oldest->newest by name; store newest first
        for p in self._run_files():
            self.runs.append(SSTableReader(p))
        self.runs.reverse()

    def put(self, key: str, value: bytes) -> None:
        self.mem[key] = value
        if len(self.mem) >= self.memtable_max_items:
            self.flush()

    def delete(self, key: str) -> None:
        self.mem[key] = TOMBSTONE
        if len(self.mem) >= self.memtable_max_items:
            self.flush()

    def flush(self) -> None:
        if not self.mem:
            return

        items = sorted(self.mem.items(), key=lambda kv: kv[0])
        ts = int(time.time() * 1e6)
        path = os.path.join(self.dir_path, f"run_{ts}_{random.randint(0, 1_000_000)}.sst")
        SSTableWriter(path, block_size=self.block_size, bits_per_key=self.bits_per_key).write(items)

        self.mem.clear()
        self.runs.insert(0, SSTableReader(path))  # newest first

        if len(self.runs) > self.run_max_count:
            self.compact_all()

    def get(self, key: str) -> Union[bytes, None]:
        if key in self.mem:
            v = self.mem[key]
            return None if v == TOMBSTONE else v

        # newest to oldest
        for run in self.runs:
            v = run.get(key)
            if v is not None:
                return None if v == TOMBSTONE else v
        return None

    def scan(self, lo: str, hi: str, limit: int = 64) -> list[tuple[str, bytes]]:
        """
        Short range scan. Returns up to 'limit' items in [lo, hi].
        Ensures newest-wins semantics across mem + runs.
        """
        # Prepare iterators (each yields sorted keys)
        iters: list[Iterator[tuple[str, bytes]]] = []

        mem_items = [(k, v) for k, v in self.mem.items() if lo <= k <= hi]
        mem_items.sort(key=lambda kv: kv[0])
        iters.append(iter(mem_items))
        
        # newest first
        for run in self.runs:
            iters.append(run.iter_range(lo, hi))

        # K-way merge with newest-wins
        heap: list[tuple[str, int, bytes, int]] = []  # (key, source_rank, val, iter_id)
        for iter_id, it in enumerate(iters):
            try:
                k, v = next(it)
                heapq.heappush(heap, (k, iter_id, v, iter_id))
            except StopIteration:
                pass

        out: list[tuple[str, bytes]] = []
        seen: set[str] = set()

        while heap and len(out) < limit:
            k, rank, v, iter_id = heapq.heappop(heap)
            if k not in seen:
                seen.add(k)
                if v != TOMBSTONE:
                    out.append((k, v))

            # advance that iterator
            try:
                nk, nv = next(iters[iter_id])
                heapq.heappush(heap, (nk, rank, nv, iter_id))
            except StopIteration:
                pass

            # skip remaining duplicates in heap by marking seen; (newest comes first due to smaller rank)

        return out

    def compact_all(self) -> None:
        """
        Tiered compaction: merge all runs into one run
        """
        if not self.runs:
            return

        sources = list(self.runs)  # newest first

        def make_merged_iter() -> Iterator[tuple[str, bytes]]:
            iters = [r.iter_all() for r in sources]

            heap: list[tuple[str, int, bytes, int]] = []
            for rank, it in enumerate(iters):
                try:
                    k, v = next(it)
                    heapq.heappush(heap, (k, rank, v, rank))
                except StopIteration:
                    pass

            while heap:
                k, rank, v, iter_id = heapq.heappop(heap)

                # Advance iterator for current pop
                try:
                    nk, nv = next(iters[iter_id])
                    heapq.heappush(heap, (nk, rank, nv, iter_id))
                except StopIteration:
                    pass

                while heap and heap[0][0] == k:
                    _, rank2, _, iter_id2 = heapq.heappop(heap)

                    try:
                        nk2, nv2 = next(iters[iter_id2])
                        heapq.heappush(heap, (nk2, rank2, nv2, iter_id2))
                    except StopIteration:
                        pass

                if v != TOMBSTONE:
                    yield (k, v)

        ts = int(time.time() * 1e6)
        new_path = os.path.join(self.dir_path, f"run_{ts}_compact.sst")

        SSTableWriter(
            new_path, 
            block_size=self.block_size,
            bits_per_key=self.bits_per_key
        ).write_from(make_merged_iter)

        # Swap in new run, delete old files
        old_runs = self.runs
        self.runs = [SSTableReader(new_path)]
        for r in old_runs:
            p = r.path
            r.close()
            try:
                os.remove(p)
            except OSError:
                pass
