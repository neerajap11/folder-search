#!/usr/bin/env python3
"""Local hybrid search over a folder. Nothing leaves this machine.

  python folder_search.py index ~/Downloads
  python folder_search.py search "which resume mentions SQL"
  python folder_search.py stats
"""
import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).parent / "index.db"
MODEL = "BAAI/bge-small-en-v1.5"
EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md"}
SKIP_DIRS = {"node_modules", ".git", "__pycache__", "venv", ".venv", "site-packages", "__MACOSX"}
MAX_FILE_BYTES = 100 * 1024 * 1024
CHUNK_CHARS = 1000
OVERLAP_CHARS = 150
RRF_K = 60

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY, mtime REAL, size INTEGER, status TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY, path TEXT, location TEXT, text TEXT, embedding BLOB
);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, filename);
"""


def connect():
    db = sqlite3.connect(DB_PATH)
    db.executescript(SCHEMA)
    return db


def get_embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(MODEL)


def read_sections(path):
    """Yield (location, text) pairs, such as ("page 3", "...")."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        import pymupdf
        with pymupdf.open(path) as doc:
            for i, page in enumerate(doc, 1):
                yield f"page {i}", page.get_text()
    elif ext == ".docx":
        import docx
        d = docx.Document(path)
        parts = [p.text for p in d.paragraphs]
        for table in d.tables:
            parts += [" | ".join(c.text for c in row.cells) for row in table.rows]
        yield "document", "\n".join(parts)
    elif ext == ".pptx":
        import pptx
        for i, slide in enumerate(pptx.Presentation(path).slides, 1):
            parts = [s.text_frame.text for s in slide.shapes if s.has_text_frame]
            yield f"slide {i}", "\n".join(parts)
    else:
        yield "document", path.read_text(errors="ignore")


def chunk(text):
    text = " ".join(text.split())
    start = 0
    while start < len(text):
        end = start + CHUNK_CHARS
        if end < len(text):
            # Prefer breaking at a space so words stay whole.
            space = text.rfind(" ", start + CHUNK_CHARS // 2, end)
            end = space if space != -1 else end
        yield text[start:end]
        if end >= len(text):
            break
        start = end - OVERLAP_CHARS


def find_files(root):
    for folder, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            path = Path(folder) / name
            if path.suffix.lower() in EXTENSIONS and not name.startswith("~$"):
                yield path


def remove_file(db, path):
    ids = [r[0] for r in db.execute("SELECT id FROM chunks WHERE path = ?", (path,))]
    db.executemany("DELETE FROM chunks_fts WHERE rowid = ?", [(i,) for i in ids])
    db.execute("DELETE FROM chunks WHERE path = ?", (path,))
    db.execute("DELETE FROM files WHERE path = ?", (path,))


def index(root):
    root = Path(root).expanduser().resolve()
    db = connect()
    embedder = None
    known ={r[0]: (r[1], r[2]) for r in db.execute("SELECT path, mtime, size FROM files")}
    paths = list(find_files(root))
    seen = {str(p) for p in paths}
    for stale in [p for p in known if p.startswith(str(root)) and p not in seen]:
        remove_file(db, stale)

    for n, path in enumerate(paths, 1):
        key = str(path)
        stat = path.stat()
        if known.get(key) == (stat.st_mtime, stat.st_size):
            continue
        remove_file(db, key)
        status, detail, rows = "indexed", "", []
        if stat.st_size > MAX_FILE_BYTES:
            status, detail = "skipped", "over size limit"
        else:
            try:
                for location, text in read_sections(path):
                    rows += [(location, c) for c in chunk(text)]
                if not rows:
                    status, detail = "no_text", "no extractable text, may need OCR"
            except Exception as e:
                status, detail = "error", f"{type(e).__name__}: {e}"[:200]
        if rows:
            embedder = embedder or get_embedder()
            vectors = embedder.embed([f"{path.name}: {c}" for _, c in rows], batch_size=32)
            for (location, text), vector in zip(rows, vectors):
                blob = np.asarray(vector, dtype=np.float32).tobytes()
                cur = db.execute(
                    "INSERT INTO chunks (path, location, text, embedding) VALUES (?, ?, ?, ?)",
                    (key, location, text, blob),
                )
                db.execute(
                    "INSERT INTO chunks_fts (rowid, text, filename) VALUES (?, ?, ?)",
                    (cur.lastrowid, text, path.name),
                )
        db.execute(
            "INSERT INTO files VALUES (?, ?, ?, ?, ?)",
            (key, stat.st_mtime, stat.st_size, status, detail),
        )
        db.commit()
        print(f"[{n}/{len(paths)}] {status}: {path.name} ({len(rows)} chunks)", flush=True)
    db.commit()
    stats()


def watch(root, every):
    """Re-index on a timer. Only new or changed files do any real work."""
    while True:
        index(root)
        print(f"Next check in {every} seconds. Press Ctrl+C to stop.", flush=True)
        time.sleep(every)


def fts_query(query):
    # Quote each term so characters like "-" or ":" can't break FTS5 syntax.
    terms = ['"' + t.replace('"', '""') + '"' for t in query.split()]
    return " OR ".join(terms)


def search(query, limit=8):
    db = connect()
    rows = db.execute("SELECT id, embedding FROM chunks").fetchall()
    if not rows:
        sys.exit("The index is empty. Run the index command first.")
    ids = np.array([r[0] for r in rows])
    matrix = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
    query_vector = next(iter(get_embedder().query_embed(query)))
    scores = matrix @ query_vector
    semantic = ids[np.argsort(-scores)[:50]].tolist()
    keyword = [r[0] for r in db.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT 50",
        (fts_query(query),),
    )]

    # Reciprocal rank fusion: reward chunks ranked highly by either search.
    fused = {}
    for ranking in (semantic, keyword):
        for rank, chunk_id in enumerate(ranking):
            fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (RRF_K + rank + 1)

    results = []
    for chunk_id in sorted(fused, key=fused.get, reverse=True)[:limit]:
        path, location, text = db.execute(
            "SELECT path, location, text FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        results.append({
            "path": path, "location": location, "text": text,
            "score": round(fused[chunk_id], 4),
            "matched": "both" if chunk_id in semantic and chunk_id in keyword
            else "meaning" if chunk_id in semantic else "keyword",
        })
    return results


def print_results(results):
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {Path(r['path']).name} ({r['location']}), matched: {r['matched']}")
        print(f"   {r['path']}")
        print(f"   {r['text'][:280]}")


def stats():
    db = connect()
    print("\nFiles by status:")
    for status, count in db.execute("SELECT status, COUNT(*) FROM files GROUP BY status"):
        print(f"  {status}: {count}")
    print(f"Chunks: {db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index").add_argument("folder")
    watch_parser = sub.add_parser("watch")
    watch_parser.add_argument("folder")
    watch_parser.add_argument("--every", type=int, default=600, help="Seconds between checks.")
    search_parser = sub.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=8)
    sub.add_parser("stats")
    args = parser.parse_args()
    if args.command == "index":
        index(args.folder)
    elif args.command == "watch":
        watch(args.folder, args.every)
    elif args.command == "search":
        print_results(search(args.query, args.limit))
    else:
        stats()
