import argparse

from lsm.bench import bench, demo


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dir", default="./lsmdata", help="Storage directory")
    common.add_argument("--reset", action="store_true", help="Delete directory before run")

    b = sub.add_parser("bench", parents=[common])
    b.add_argument("--n", type=int, default=200_000, help="number of inserts")
    b.add_argument("--reads", type=int, default=200_000, help="number of point reads")
    b.add_argument("--ranges", type=int, default=50_000, help="number of short range scans")
    b.add_argument("--span", type=int, default=64, help="max items returned by each range scan")
    b.add_argument("--mem", type=int, default=50_000, help="memtable max items before flush")
    b.add_argument("--runs", type=int, default=8, help="max runs before tiered compaction")
    b.add_argument("--block", type=int, default=32 * 1024, help="data block size (bytes)")
    b.add_argument("--bpk", type=int, default=10, help="bloom bits per key")
    b.add_argument("--keylen", type=int, default=16, help="key length")
    b.add_argument("--vallen", type=int, default=32, help="value length bytes")
    b.add_argument("--seed", type=int, default=123, help="rng seed")
    b.set_defaults(func=bench)

    d = sub.add_parser("demo", parents=[common])
    d.set_defaults(func=demo)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
