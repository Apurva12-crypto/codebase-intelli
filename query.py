"""
query.py
Hybrid retrieval over the indexed repo:
  1. semantic_search(query)  -> TF-IDF cosine similarity over chunk text
  2. find_callers(name)      -> structural lookup: who calls this function/class
  3. find_callees(name)      -> structural lookup: what does this function call
  4. get_chunk_by_name(name) -> exact lookup of a chunk's full source
  5. grep_repo(pattern)      -> raw text search fallback (regex over files)

These are the "tools" the agent in main.py will call.
"""

import os
import re
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity

INDEX_FILE = "index.pkl"
CALL_GRAPH_FILE = "call_graph.json"

_index_cache = None
_call_graph_cache = None


def _load_index(index_file=None):
    """If index_file is given, load fresh from that path (no caching --
    used by the API server where different sessions have different indexes).
    Otherwise fall back to the cached default INDEX_FILE (used by the CLI)."""
    if index_file is not None:
        with open(index_file, "rb") as f:
            return pickle.load(f)
    global _index_cache
    if _index_cache is None:
        with open(INDEX_FILE, "rb") as f:
            _index_cache = pickle.load(f)
    return _index_cache


def _load_call_graph(call_graph_file=None):
    if call_graph_file is not None:
        with open(call_graph_file) as f:
            return json.load(f)
    global _call_graph_cache
    if _call_graph_cache is None:
        with open(CALL_GRAPH_FILE) as f:
            _call_graph_cache = json.load(f)
    return _call_graph_cache


def semantic_search(query, top_k=5, index_file=None):
    """Vector (TF-IDF) search over all chunks. Returns list of chunk summaries."""
    idx = _load_index(index_file)
    vectorizer = idx["vectorizer"]
    matrix = idx["matrix"]
    chunks = idx["chunks"]

    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix)[0]
    top_indices = sims.argsort()[::-1][:top_k]

    results = []
    for i in top_indices:
        if sims[i] <= 0:
            continue
        c = chunks[i]
        results.append({
            "name": c["name"],
            "file": c["file"],
            "type": c["type"],
            "start_line": c["start_line"],
            "end_line": c["end_line"],
            "source": c["source"],
            "score": float(sims[i]),
        })
    return results


def get_chunk_by_name(name, index_file=None):
    """Exact lookup: return full chunk (source, file, lines) for a given function/class name."""
    idx = _load_index(index_file)
    matches = [c for c in idx["chunks"] if c["name"] == name or c["name"].endswith("." + name)]
    return matches


def find_callers(name, call_graph_file=None):
    """Who calls this function/class? Returns list of caller names."""
    cg = _load_call_graph(call_graph_file)
    return cg["called_by"].get(name, [])


def find_callees(name, call_graph_file=None):
    """What does this function call? Returns list of callee names (raw, unresolved short names)."""
    cg = _load_call_graph(call_graph_file)
    return cg["calls"].get(name, [])


def grep_repo(pattern, repo_path, max_results=20):
    """Raw regex search across all files in the repo. Fallback for anything
    the structured index doesn't capture (imports, comments, config files, etc)."""
    results = []
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return [{"error": f"invalid regex: {e}"}]

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in
                   {".git", "venv", "env", "__pycache__", "node_modules", ".venv", "chroma_db"}
                   and not d.startswith(".")]
        for fname in files:
            if len(results) >= max_results:
                return results
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", errors="ignore") as f:
                    for lineno, line in enumerate(f, start=1):
                        if regex.search(line):
                            results.append({
                                "file": fpath,
                                "line": lineno,
                                "text": line.strip()[:200],
                            })
                            if len(results) >= max_results:
                                break
            except Exception:
                continue
    return results


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "how does session send a request"
    print(f"Semantic search for: {q}\n")
    for r in semantic_search(q, top_k=5):
        print(f"- [{r['score']:.3f}] {r['type']} {r['name']} ({r['file']}:{r['start_line']})")