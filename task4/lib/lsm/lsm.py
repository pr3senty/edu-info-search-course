import os, time, heapq, random
from typing import Callable, Iterator, Optional, Union
from pyroaring import BitMap

from .helpers import TOMBSTONE
from .sstable import SSTableReader, SSTableWriter


MergeFunc = Callable[[list[bytes]], Optional[bytes]]  # returns merged bytes or None


class MergeLSMTree:

    def __init__(
        self,
        dir_path: str,
        merge_func: MergeFunc,
        memtable_max_items: int = 50_000,
        run_max_count: int = 8,
        block_size: int = 32 * 1024,
        bits_per_key: int = 10,
    ):
        self.dir_path = dir_path
        os.makedirs(self.dir_path, exist_ok=True)

        self.merge_func = merge_func
        self.memtable_max_items = memtable_max_items
        self.run_max_count = run_max_count
        self.block_size = block_size
        self.bits_per_key = bits_per_key

        self.mem: dict[str, BitMap] = {}
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
        for p in self._run_files():
            self.runs.append(SSTableReader(p))
        self.runs.reverse()

    def put(self, key: str, value: int) -> None:
        self.mem.setdefault(key, BitMap()).add(value)
        if len(self.mem) >= self.memtable_max_items:
            self.flush()

    def flush(self) -> None:
        if not self.mem:
            return

        items = sorted(self.mem.items(), key=lambda kv: kv[0])
        ts = int(time.time() * 1e6)
        path = os.path.join(self.dir_path, f"run_{ts}_{random.randint(0, 1_000_000)}.sst")
        SSTableWriter(
            path, 
            block_size=self.block_size, 
            bits_per_key=self.bits_per_key
        ).write([(k, v.serialize()) for k, v in items])

        self.mem.clear()
        self.runs.insert(0, SSTableReader(path))

        if len(self.runs) > self.run_max_count:
            self.compact_all()

    def get(self, key: str) -> Optional[bytes]:
        vals: list[bytes] = []

        v = self.mem.get(key)
        if v is not None:
            vals.append(v.serialize())

        for run in self.runs:
            rv = run.get(key)
            if rv is not None:
                vals.append(rv)

        if not vals:
            return None

        return self.merge_func([v for v in vals if v != TOMBSTONE])

    def scan(self, lo: str, hi: str, limit: Union[int, None] = None) -> Iterator[tuple[str, bytes]]:
        """
        Range scan over [lo, hi]. Yields up to 'limit' merged items.
        If limit is None, yields all matching items
        """
        iters: list[Iterator[tuple[str, bytes]]] = []

        mem_items = [(k, v.serialize()) for k, v in self.mem.items() if lo <= k <= hi]
        mem_items.sort(key=lambda kv: kv[0])
        iters.append(iter(mem_items))
        
        for run in self.runs:
            iters.append(run.iter_range(lo, hi))

        # K-way merge
        heap: list[tuple[str, int, bytes, int]] = []  # (key, source_rank, val, iter_id)
        for iter_id, it in enumerate(iters):
            try:
                k, v = next(it)
                heapq.heappush(heap, (k, iter_id, v, iter_id))
            except StopIteration:
                pass
                
        count: int = 0
        while heap:
            k, rank, v, iter_id = heapq.heappop(heap)

            # advance that iterator
            try:
                nk, nv = next(iters[iter_id])
                heapq.heappush(heap, (nk, rank, nv, iter_id))
            except StopIteration:
                pass

            # collect all values for this key
            vals = [v]
            while heap and heap[0][0] == k:
                _, r2, v2, id2 = heapq.heappop(heap)
                vals.append(v2)
                try:
                    nk2, nv2 = next(iters[id2])
                    heapq.heappush(heap, (nk2, r2, nv2, id2))
                except StopIteration:
                    pass

            merged = self.merge_func([x for x in vals if x != TOMBSTONE])
            if merged is not None:
                yield (k, merged)
                count += 1
                if limit is not None and count == limit:
                    break

    def compact_all(self) -> None:
        if not self.runs:
            return

        sources = list(self.runs)  # snapshot (newest first)

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

                # advance iterator
                try:
                    nk, nv = next(iters[iter_id])
                    heapq.heappush(heap, (nk, rank, nv, iter_id))
                except StopIteration:
                    pass

                # collect all values for this key
                vals = [v]
                while heap and heap[0][0] == k:
                    _, r2, v2, id2 = heapq.heappop(heap)
                    vals.append(v2)
                    try:
                        nk2, nv2 = next(iters[id2])
                        heapq.heappush(heap, (nk2, r2, nv2, id2))
                    except StopIteration:
                        pass

                merged = self.merge_func([x for x in vals if x != TOMBSTONE])
                if merged is not None:
                    yield (k, merged)

        ts = int(time.time() * 1e6)
        new_path = os.path.join(self.dir_path, f"run_{ts}_compact.sst")
        SSTableWriter(
            new_path, 
            block_size=self.block_size, 
            bits_per_key=self.bits_per_key
        ).write_from(make_merged_iter)

        old_runs = self.runs
        self.runs = [SSTableReader(new_path)]
        for r in old_runs:
            p = r.path
            r.close()
            try:
                os.remove(p)
            except OSError:
                pass