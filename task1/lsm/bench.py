import os
import time
import random
import argparse
import shutil

from lsm.lib.lsm import LSMTree

def now_ns() -> int:
    return time.perf_counter_ns()

def rand_key(rng: random.Random, length: int = 16) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(rng.choice(alphabet) for _ in range(length))

def percentile(sorted_vals: list[int], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int((p / 100.0) * (len(sorted_vals) - 1))
    return sorted_vals[idx] / 1e6  # ns -> ms

def bench(args: argparse.Namespace) -> None:
    d = args.dir
    if args.reset and os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

    tree = LSMTree(
        d,
        memtable_max_items=args.mem,
        run_max_count=args.runs,
        block_size=args.block,
        bits_per_key=args.bpk,
    )

    rng = random.Random(args.seed)
    n = args.n

    # Generate keys up-front to make workloads reproducible
    keys = [rand_key(rng, args.keylen) for _ in range(n)]
    # Make keys unique-ish
    keys = list(dict.fromkeys(keys))
    n = len(keys)
    vals = [os.urandom(args.vallen) for _ in range(n)]

    print(f"[insert] n={n}, mem={args.mem}, run_max={args.runs}, block={args.block}, bpk={args.bpk}")
    t0 = now_ns()
    lat = []
    for k, v in zip(keys, vals):
        s = now_ns()
        tree.put(k, v)
        lat.append(now_ns() - s)
    tree.flush()
    t1 = now_ns()
    lat.sort()
    print(f"  total: {(t1-t0)/1e9:.3f}s  throughput: {n/((t1-t0)/1e9):.1f} ops/s")
    print(f"  latency ms: p50={percentile(lat,50):.3f} p95={percentile(lat,95):.3f} p99={percentile(lat,99):.3f}")

    # Point reads: mix of hits/misses
    reads = args.reads
    miss_keys = [rand_key(rng, args.keylen) for _ in range(reads)]
    read_keys = []
    for i in range(reads):
        if i % 2 == 0:
            read_keys.append(keys[rng.randrange(n)])  # hit
        else:
            read_keys.append(miss_keys[i])            # miss

    print(f"[point read] ops={reads} (50% hit / 50% miss)")
    t0 = now_ns()
    lat = []
    hit = 0
    for k in read_keys:
        s = now_ns()
        v = tree.get(k)
        lat.append(now_ns() - s)
        if v is not None:
            hit += 1
    t1 = now_ns()
    lat.sort()
    print(f"  total: {(t1-t0)/1e9:.3f}s  throughput: {reads/((t1-t0)/1e9):.1f} ops/s  hits={hit}")
    print(f"  latency ms: p50={percentile(lat,50):.3f} p95={percentile(lat,95):.3f} p99={percentile(lat,99):.3f}")

    # Short range reads
    ranges = args.ranges
    span = args.span
    # For a "short range", we take a random key and define hi by appending a suffix that keeps ordering nearby.
    # Since keys are random, "nearby" isn't perfect. For more realistic scans, you can use sequential keys.
    print(f"[short range scan] ops={ranges}, limit={span}")
    t0 = now_ns()
    lat = []
    total_items = 0
    for _ in range(ranges):
        base = keys[rng.randrange(n)]
        lo = base
        hi = base + "~~~~"  # likely to include some close keys if present
        s = now_ns()
        out = tree.scan(lo, hi, limit=span)
        lat.append(now_ns() - s)
        total_items += len(out)
    t1 = now_ns()
    lat.sort()
    print(f"  total: {(t1-t0)/1e9:.3f}s  throughput: {ranges/((t1-t0)/1e9):.1f} ops/s  avg_items={total_items/max(1,ranges):.2f}")
    print(f"  latency ms: p50={percentile(lat,50):.3f} p95={percentile(lat,95):.3f} p99={percentile(lat,99):.3f}")

    tree.close()

def demo(args: argparse.Namespace) -> None:
    if args.reset and os.path.exists(args.dir):
        shutil.rmtree(args.dir)
    tree = LSMTree(args.dir, memtable_max_items=5, run_max_count=3)

    tree.put("alice", b"1")
    tree.put("bob", b"2")
    tree.put("carol", b"3")
    tree.put("bob", b"22")
    tree.put("dave", b"4")
    tree.flush()

    print("bob =", tree.get("bob"))
    print("scan[b..d] =", tree.scan("b", "d", limit=10))

    tree.delete("carol")
    tree.flush()

    print("carol =", tree.get("carol"))
    print("scan[a..z] =", tree.scan("a", "z", limit=50))
    tree.close()