import os
import json
from pathlib import Path

from lib import (
    MergeLSMTree,
    PositionalIndex,
    TextAnalyzer, 
    utils
)
import lib.rpn as rpn_module


DOCS_META = "docs.json"
DOCS_DIR = "./docs"
IDX_DATA_DIR = "./idx_data"
SUPPORTED_DOCUMENT_EXTENSIONS = (".txt", ".md", ".py")


def load_docs_map(idx_dir: Path) -> dict[str, str]:
    p = idx_dir / DOCS_META
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_docs_map(idx_dir: Path, m: dict[str, str]) -> None:
    (idx_dir / DOCS_META).write_text(
        json.dumps(m, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def iter_text_files(docs_dir: Path):
    for root, _, files in os.walk(docs_dir):
        for fn in files:
            if fn.lower().endswith(SUPPORTED_DOCUMENT_EXTENSIONS):
                yield Path(root) / fn


def build_or_resume_index(index_dir: Path, docs_dir: Path, verbose: bool = True):
    index_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = index_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    docs_map = load_docs_map(index_dir)

    lsm = MergeLSMTree(
        dir_path=str(runs_dir),
        merge_func=utils.merge_postings_bytes,
        mem_merge_func=utils.mem_merge_postings,
        memtable_max_items=50_000,
        run_max_count=8,
    )
    analyzer = TextAnalyzer()

    documents_count = 0
    if docs_map:
        documents_count = max(int(k) for k in docs_map.keys()) + 1

    index = PositionalIndex(lsm, analyzer, documents_count=documents_count)

    existing_paths = set(docs_map.values())
    added = 0

    for fp in iter_text_files(docs_dir):
        sp = str(fp.resolve())
        if sp in existing_paths:
            continue

        text = fp.read_text(encoding="utf-8", errors="ignore")
        doc_id = index.add_document(text)
        docs_map[str(doc_id)] = sp
        added += 1

        if verbose and added % 100 == 0:
            print(f"Indexed {added} new docs...")

    lsm.flush()
    save_docs_map(index_dir, docs_map)

    if verbose:
        print(f"Index ready. Total docs: {len(docs_map)} (added {added}).")
        print(f"Index dir: {index_dir}")
        print("Type 'help' for commands.")

    return lsm, index, docs_map


HELP = """Commands:
  query <expr>        Boolean query. Example: query (привет AND мир) OR python
  phrase <text>       Exact phrase query. Example: phrase hello world
  term <word>         Search one term
  add <path>          Add a file OR index all files under a directory
  show <docid>        Show file path and first lines of the document
  stats               Show docs/runs sizes
  flush               Flush memtable to a new run
  compact             Run full compaction
  help                Show this help
  exit / quit         Exit program
"""


def cmd_stats(index_dir: Path, docs_map: dict[str, str]):
    runs = sorted((index_dir / "runs").glob("*.sst"))
    total = sum(p.stat().st_size for p in runs) if runs else 0

    print(f"Docs: {len(docs_map)}")
    print(f"Runs: {len(runs)}")
    print(f"Runs size: {total / (1024 * 1024):.2f} MiB")


def cmd_show(docs_map: dict[str, str], docid: int, head_lines: int = 20):
    p = docs_map.get(str(docid))
    if not p:
        print("No such docid in docs.json")
        return

    print(f"docid={docid}")
    print(p)

    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(head_lines):
                line = f.readline()
                if not line:
                    break
                print(line.rstrip("\n"))
    except OSError as e:
        print(f"Cannot open file: {e}")


def cmd_add_path(index: PositionalIndex, docs_map: dict[str, str], path: str, verbose: bool = True) -> None:
    p = Path(path).resolve()
    if not p.exists():
        print("Path does not exist:", p)
        return

    existing_paths = set(docs_map.values())
    added = 0
    skipped = 0
    failed = 0

    def add_file(fp: Path):
        nonlocal added, skipped, failed

        sp = str(fp.resolve())
        if sp in existing_paths:
            skipped += 1
            return

        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            failed += 1
            return

        doc_id = index.add_document(text)
        docs_map[str(doc_id)] = sp
        existing_paths.add(sp)
        added += 1

        if verbose:
            print(f"added docid={doc_id}: {sp}")

    if p.is_file():
        add_file(p)
    elif p.is_dir():
        for fp in iter_text_files(p):
            add_file(fp)
    else:
        print("Unsupported path type:", p)
        return

    print(f"Done. Added={added}, skipped={skipped}, failed={failed}")


def run_repl(lsm, index: PositionalIndex, docs_map, index_dir: Path):
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

        if s == "stats":
            cmd_stats(index_dir, docs_map)
            continue

        if s == "flush":
            lsm.flush()
            save_docs_map(index_dir, docs_map)
            print("Flushed.")
            continue

        if s == "compact":
            lsm.compact_all()
            save_docs_map(index_dir, docs_map)
            print("Compacted.")
            continue

        if s.startswith("show "):
            try:
                docid = int(s.split(maxsplit=1)[1])
                cmd_show(docs_map, docid)
            except Exception:
                print("Usage: show <docid>")
            continue

        if s.startswith("term "):
            q = s[len("term "):].strip()
            if not q:
                print("Usage: term <word>")
                continue

            try:
                ids = index.search_term(q)
                print(f"Matched {len(ids)} docs")
                print(ids)
            except Exception as e:
                print(f"Term query error: {e}")
            continue

        if s.startswith("phrase "):
            q = s[len("phrase "):].strip()
            if not q:
                print("Usage: phrase <text>")
                continue

            try:
                ids = index.phrase_query(q)
                print(f"Matched {len(ids)} docs")
                print(ids)
            except Exception as e:
                print(f"Phrase query error: {e}")
            continue

        if s.startswith("query "):
            q = s[len("query "):].strip()
            if not q:
                print("Usage: query <expr>")
                continue

            try:
                rpn = rpn_module.to_rpn(rpn_module.tokenize_query(q))
                postings = rpn_module.eval_rpn_postings(rpn, index)
                ids = [doc_id for doc_id, _ in postings]
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
                cmd_add_path(index, docs_map, path, verbose=True)
            except Exception as e:
                print(f"Add error: {e}")
            continue

        print("Unknown command. Type 'help'.")


def main():
    docs_dir = Path(DOCS_DIR).resolve()
    index_dir = Path(IDX_DATA_DIR).resolve()

    if not docs_dir.exists():
        print(f"Docs directory does not exist: {docs_dir}")
        return

    lsm = None
    try:
        lsm, index, docs_map = build_or_resume_index(index_dir, docs_dir, verbose=True)
        run_repl(lsm, index, docs_map, index_dir)
    finally:
        if lsm is not None:
            lsm.flush()
            lsm.close()


if __name__ == "__main__":
    main()