import os, time, heapq, random
from typing import Callable, Iterator, Optional, Union

from .helpers import TOMBSTONE
from .sstable import SSTableReader, SSTableWriter


MergeFunc = Callable[[list[bytes]], Union[bytes, None]]  # returns merged bytes or None
MemMergeFunc = Callable[[Union[bytes, None], bytes], bytes]


class MergeLSMTree:

    def __init__(
        self,
        dir_path: str,
        merge_func: MergeFunc,
        mem_merge_func: MemMergeFunc,
        memtable_max_items: int = 50_000,
        run_max_count: int = 8,
        block_size: int = 32 * 1024,
        bits_per_key: int = 10,
    ):
        self.dir_path = dir_path
        os.makedirs(self.dir_path, exist_ok=True)

        self.merge_func = merge_func
        self.mem_merge_func = mem_merge_func
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
        for p in self._run_files():
            self.runs.append(SSTableReader(p))
        self.runs.reverse()


    def put(self, key: str, value: bytes) -> None:
        prev = self.mem.get(key)
        self.mem[key] = self.mem_merge_func(prev, value)
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
        ).write(items)

        self.mem.clear()
        self.runs.insert(0, SSTableReader(path))

        if len(self.runs) > self.run_max_count:
            self.compact_all()


    def get(self, key: str) -> Union[bytes, None]:
        vals: list[bytes] = []

        v = self.mem.get(key)
        if v is not None:
            vals.append(v)

        for run in self.runs:
            rv = run.get(key)
            if rv is not None:
                vals.append(rv)

        if not vals:
            return None

        return self.merge_func([v for v in vals if v != TOMBSTONE])
    

    def iter_keys(self) -> Iterator[str]:
        seen = set()

        for key in sorted(self.mem.keys()):
            if key not in seen:
                seen.add(key)
                yield key

        for run in self.runs:
            for key, _ in run.iter_all():
                if key not in seen:
                    seen.add(key)
                    yield key


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