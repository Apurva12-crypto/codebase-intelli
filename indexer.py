"""
indexer.py
Takes chunks from chunker.py, vectorizes them with TF-IDF (fully local,
no network calls, no model downloads -- important for demo reliability),
and saves:
  - index.pkl        (TF-IDF vectorizer + matrix + chunk metadata)
  - call_graph.json  (who calls whom, for structural lookup)
"""

import os
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from chunker import chunk_repo

INDEX_FILE = "index.pkl"
CALL_GRAPH_FILE = "call_graph.json"


def build_call_graph(chunks):
    """
    Build a bidirectional call graph:
      calls[name] = [names it calls]
      called_by[name] = [names that call it]
    Matched by function/method short name (best-effort, not fully qualified).
    """
    calls_map = {}
    called_by_map = {}

    short_name_to_full = {}
    for c in chunks:
        short = c["name"].split(".")[-1]
        short_name_to_full.setdefault(short, []).append(c["name"])

    for c in chunks:
        caller = c["name"]
        calls_map[caller] = c["calls"]
        for callee_short in c["calls"]:
            if callee_short in short_name_to_full:
                for callee_full in short_name_to_full[callee_short]:
                    called_by_map.setdefault(callee_full, [])
                    if caller not in called_by_map[callee_full]:
                        called_by_map[callee_full].append(caller)

    return {"calls": calls_map, "called_by": called_by_map}


def index_repo(repo_path, index_file=INDEX_FILE, call_graph_file=CALL_GRAPH_FILE):
    """Index a repo. Optionally write to custom paths (used by the API server
    so multiple repos/sessions don't clobber each other's index files)."""
    print(f"Chunking repo at {repo_path} ...")
    chunks = chunk_repo(repo_path)
    print(f"Got {len(chunks)} chunks.")

    if not chunks:
        print("No chunks found. Is this a Python repo? Exiting.")
        return 0

    print("Building call graph ...")
    call_graph = build_call_graph(chunks)
    with open(call_graph_file, "w") as f:
        json.dump(call_graph, f, indent=2)
    print(f"Call graph saved to {call_graph_file}")

    print("Vectorizing chunks with TF-IDF ...")
    documents = [f"{c['type']} {c['name']} in {c['file']}:\n{c['source']}" for c in chunks]

    vectorizer = TfidfVectorizer(
        max_features=20000,
        stop_words="english",
        token_pattern=r"(?u)\b\w[\w_]+\b",  # keep underscores, matters for code identifiers
    )
    matrix = vectorizer.fit_transform(documents)

    with open(index_file, "wb") as f:
        pickle.dump({
            "vectorizer": vectorizer,
            "matrix": matrix,
            "chunks": chunks,  # full chunk dicts, so query.py can return source directly
        }, f)

    print(f"Done. Indexed {len(chunks)} chunks into {index_file}.")
    return len(chunks)


if __name__ == "__main__":
    import sys
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    index_repo(repo)