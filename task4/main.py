import os
import re
import json
from pathlib import Path
from typing import Iterator, Any

from lib import (
    InvertedIndex, TextAnalyzer,
    MergeLSMTree, utils
)


DOCS_META = "docs.json"
DOCS_DIR = "./docs"
IDX_DATA_DIR = "./idx_data"
SUPPORTED_DOCUMENT_EXTENSIONS = (".txt", ".md", ".py")


HELP = """Commands:
  query <expr>        Run query
                      Examples:
                        query (привет AND мир)
                        query python AND DATE[20240101,20240131]
                        query APPEARED[20200101,20201231]
                        query VALID[20220101,20231231]

  add <path>          Add one file or all supported files from directory
  setmeta <docid> ... Set metadata for document
                      Examples:
                        setmeta 3 date=19990112
                        setmeta 3 start=19990120 end=19990124
                        setmeta 3 end=null

  show <docid>        Show document metadata and first lines
  list                List all indexed docs
  stats               Show docs/runs sizes
  flush               Flush memtable to a new SSTable
  compact             Run full compaction
  rebuild             Rebuild full index from docs.json
  help                Show this help
  exit / quit         Exit
"""


def load_docs_map(idx_dir: Path) -> dict[str, dict[str, Any]]:
    p = idx_dir / DOCS_META
    if not p.exists():
        return {}

    raw: dict = json.loads(p.read_text(encoding="utf-8"))

    fixed: dict[str, dict[str, Any]] = {}
    for docid, value in raw.items():
        fixed[str(docid)] = {
            "path": value.get("path"),
            "date": value.get("date"),
            "start_date": value.get("start_date"),
            "end_date": value.get("end_date"),
        }
    return fixed


def save_docs_map(idx_dir: Path, docs_map: dict[str, dict[str, Any]]) -> None:
    (idx_dir / DOCS_META).write_text(
        json.dumps(docs_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def iter_text_files(docs_dir: Path) -> Iterator[Path]:
    for root, _, files in os.walk(docs_dir):
        for fn in files:
            if fn.lower().endswith(SUPPORTED_DOCUMENT_EXTENSIONS):
                yield Path(root) / fn


def validate_date_or_none(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    if not re.fullmatch(r"\d{8}", value):
        raise ValueError(f"{field_name} must be YYYYMMDD or null")
    return value


def validate_meta(meta: dict[str, Any]) -> dict[str, Any]:
    date = validate_date_or_none(meta.get("date"), "date")
    start_date = validate_date_or_none(meta.get("start_date"), "start_date")
    end_date = validate_date_or_none(meta.get("end_date"), "end_date")

    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date cannot be greater than end_date")

    return {
        "path": str(Path(meta["path"]).resolve()),
        "date": date,
        "start_date": start_date,
        "end_date": end_date,
    }


def make_doc_meta(
    path: str,
    date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    return validate_meta(
        {
            "path": path,
            "date": date,
            "start_date": start_date,
            "end_date": end_date,
        }
    )


def next_doc_id(docs_map: dict[str, dict[str, Any]]) -> int:
    if not docs_map:
        return 0
    return max(int(k) for k in docs_map.keys()) + 1


def existing_paths(docs_map: dict[str, dict[str, Any]]) -> set[str]:
    return {meta["path"] for meta in docs_map.values()}


def build_runtime(index_dir: Path, docs_map: dict[str, dict[str, Any]]):
    runs_dir = index_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    lsm = MergeLSMTree(
        dir_path=str(runs_dir),
        merge_func=utils.merge_roaring_bytes,
        memtable_max_items=50_000,
        run_max_count=8,
    )
    analyzer = TextAnalyzer()
    index = InvertedIndex(lsm, analyzer, documents_count=next_doc_id(docs_map))
    return lsm, index


def rebuild_index_from_docs_map(
    index_dir: Path,
    docs_map: dict[str, dict[str, Any]],
    verbose: bool = True,
):
    runs_dir = index_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # close/remove old runs
    for p in runs_dir.glob("*.sst"):
        try:
            p.unlink()
        except OSError:
            pass

    lsm = MergeLSMTree(
        dir_path=str(runs_dir),
        merge_func=utils.merge_roaring_bytes,
        memtable_max_items=50_000,
        run_max_count=8,
    )
    analyzer = TextAnalyzer()
    index = InvertedIndex(lsm, analyzer, documents_count=next_doc_id(docs_map))

    total = 0
    for docid_str in sorted(docs_map.keys(), key=lambda x: int(x)):
        docid = int(docid_str)
        meta = validate_meta(docs_map[docid_str])
        path = Path(meta["path"])

        if not path.exists():
            if verbose:
                print(f"Skipping missing file for docid={docid}: {path}")
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            if verbose:
                print(f"Skipping unreadable file for docid={docid}: {e}")
            continue

        # enforce doc ids stability across rebuilds
        index.doc_count = docid
        assigned = index.add_document(
            text,
            date=meta.get("date"),
            start_date=meta.get("start_date"),
            end_date=meta.get("end_date"),
        )
        if assigned != docid:
            raise RuntimeError(f"Rebuild docid mismatch: expected {docid}, got {assigned}")

        total += 1
        if verbose and total % 100 == 0:
            print(f"Reindexed {total} docs...")

    index.doc_count = next_doc_id(docs_map)
    lsm.flush()

    if verbose:
        print(f"Rebuild done. Indexed {total} docs.")

    return lsm, index


def build_or_resume_index(index_dir: Path, docs_dir: Path, verbose: bool = True):
    index_dir.mkdir(parents=True, exist_ok=True)
    docs_map = load_docs_map(index_dir)

    # if no metadata exists yet, auto-import files from docs_dir with empty meta
    if not docs_map:
        added = 0
        for fp in iter_text_files(docs_dir):
            docid = str(added)
            docs_map[docid] = make_doc_meta(str(fp.resolve()))
            added += 1
        save_docs_map(index_dir, docs_map)

    lsm, index = rebuild_index_from_docs_map(index_dir, docs_map, verbose=verbose)

    if verbose:
        print(f"Index ready. Total docs in docs.json: {len(docs_map)}")
        print(f"Index dir: {index_dir}")
        print("Type 'help' for commands.")

    return lsm, index, docs_map


def cmd_stats(index_dir: Path, docs_map: dict[str, dict[str, Any]]):
    runs = sorted((index_dir / "runs").glob("*.sst"))
    total = sum(p.stat().st_size for p in runs) if runs else 0
    print(f"Docs: {len(docs_map)}")
    print(f"Runs: {len(runs)}")
    print(f"Runs size: {total / (1024 * 1024):.2f} MiB")


def cmd_show(docs_map: dict[str, dict[str, Any]], docid: int, head_lines: int = 20):
    meta = docs_map.get(str(docid))
    if not meta:
        print("No such docid in docs.json")
        return

    path = meta["path"]
    print(f"docid={docid}")
    print(f"path={path}")
    print(f"date={meta.get('date')}")
    print(f"start_date={meta.get('start_date')}")
    print(f"end_date={meta.get('end_date')}")

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(head_lines):
                line = f.readline()
                if not line:
                    break
                print(line.rstrip("\n"))
    except OSError as e:
        print(f"Cannot open file: {e}")


def cmd_list(docs_map: dict[str, dict[str, Any]]):
    for docid in sorted(docs_map.keys(), key=lambda x: int(x)):
        meta = docs_map[docid]
        print(
            f"{docid}: {meta['path']} | "
            f"date={meta.get('date')} | "
            f"start={meta.get('start_date')} | "
            f"end={meta.get('end_date')}"
        )


def cmd_add_path(
    index_dir: Path,
    docs_map: dict[str, dict[str, Any]],
    path: str,
    verbose: bool = True,
) -> bool:
    p = Path(path).resolve()
    if not p.exists():
        print("Path does not exist:", p)
        return False

    paths = existing_paths(docs_map)
    added = 0
    skipped = 0

    def add_file(fp: Path):
        nonlocal added, skipped
        sp = str(fp.resolve())
        if sp in paths:
            skipped += 1
            return

        docid = str(next_doc_id(docs_map))
        docs_map[docid] = make_doc_meta(sp)
        paths.add(sp)
        added += 1
        if verbose:
            print(f"registered docid={docid}: {sp}")

    if p.is_file():
        if not p.name.lower().endswith(SUPPORTED_DOCUMENT_EXTENSIONS):
            print("Unsupported file extension:", p)
            return False
        add_file(p)
    elif p.is_dir():
        for fp in iter_text_files(p):
            add_file(fp)
    else:
        print("Unsupported path type:", p)
        return False

    save_docs_map(index_dir, docs_map)
    print(f"Done. Added={added}, skipped={skipped}")
    print("Run 'rebuild' to reindex newly added files.")
    return added > 0


def parse_meta_args(args: list[str]) -> dict[str, Any]:
    updates: dict[str, Any] = {}

    for item in args:
        if "=" not in item:
            raise ValueError(f"Invalid argument: {item}")

        key, value = item.split("=", 1)
        key = key.lower().strip()
        value = value.strip()

        if key not in {"date", "start", "end"}:
            raise ValueError(f"Unknown meta field: {key}")

        mapped = {
            "date": "date",
            "start": "start_date",
            "end": "end_date",
        }[key]

        if value.lower() == "null":
            updates[mapped] = None
        else:
            if not re.fullmatch(r"\d{8}", value):
                raise ValueError(f"Invalid date format: {value}")
            updates[mapped] = value

    start_date = updates.get("start_date")
    end_date = updates.get("end_date")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date cannot be greater than end_date")

    return updates


def cmd_setmeta(index_dir: Path, docs_map: dict[str, dict[str, Any]], args: str) -> bool:
    parts = args.split()
    if len(parts) < 2:
        print("Usage: setmeta <docid> date=YYYYMMDD start=YYYYMMDD end=YYYYMMDD")
        return False

    docid = parts[0]
    meta = docs_map.get(docid)
    if meta is None:
        print("No such docid")
        return False

    updates = parse_meta_args(parts[1:])
    new_meta = dict(meta)
    new_meta.update(updates)
    docs_map[docid] = validate_meta(new_meta)
    save_docs_map(index_dir, docs_map)

    print(f"Updated metadata for docid={docid}")
    print("Run 'rebuild' to apply metadata changes to the index.")
    return True


def print_query_debug(q: str, tokens: list[str], rpn: list[str]):
    print(f"Query: {q}")
    print("Tokens:", tokens)
    print("RPN:", rpn)


def run_repl(lsm, index, docs_map, index_dir: Path):
    print(HELP)

    while True:
        try:
            s = input("invindex> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not s:
            continue

        if s in ("exit", "quit"):
            break

        if s == "help":
            print(HELP)
            continue

        if s == "list":
            cmd_list(docs_map)
            continue

        if s == "stats":
            cmd_stats(index_dir, docs_map)
            continue

        if s == "flush":
            lsm.flush()
            print("Flushed.")
            continue

        if s == "compact":
            lsm.compact_all()
            print("Compacted.")
            continue

        if s == "rebuild":
            lsm.close()
            lsm, index = rebuild_index_from_docs_map(index_dir, docs_map, verbose=True)
            print("Rebuilt index from docs.json")
            continue

        if s.startswith("show "):
            try:
                docid = int(s.split(maxsplit=1)[1])
                cmd_show(docs_map, docid)
            except Exception:
                print("Usage: show <docid>")
            continue

        if s.startswith("query "):
            q = s[len("query "):].strip()
            try:
                tokens = utils.tokenize_query(q)
                rpn = utils.to_rpn(tokens)
                res = utils.eval_rpn(rpn, index)
                ids = list(res)
                print(f"Matched {len(ids)} docs")
                print(ids)
            except Exception as e:
                print(f"Query error: {e}")
            continue

        if s.startswith("add "):
            path = s[len("add "):].strip()
            if not path:
                print("Usage: add <file_or_dir_path>")
                continue
            try:
                changed = cmd_add_path(index_dir, docs_map, path, verbose=True)
                if changed:
                    print("Tip: run 'rebuild' now, or restart the program.")
            except Exception as e:
                print(f"Add error: {e}")
            continue

        if s.startswith("setmeta "):
            args = s[len("setmeta "):].strip()
            try:
                changed = cmd_setmeta(index_dir, docs_map, args)
                if changed:
                    print("Tip: run 'rebuild' now, or restart the program.")
            except Exception as e:
                print(f"setmeta error: {e}")
            continue

        print("Unknown command. Type 'help'.")


def main():
    docs_dir = Path(DOCS_DIR).resolve()
    index_dir = Path(IDX_DATA_DIR).resolve()

    lsm = None
    try:
        lsm, index, docs_map = build_or_resume_index(
            index_dir=index_dir,
            docs_dir=docs_dir,
            verbose=True,
        )
        run_repl(lsm, index, docs_map, index_dir)
    finally:
        if lsm is not None:
            lsm.close()
        print("Bye.")


if __name__ == "__main__":
    main()