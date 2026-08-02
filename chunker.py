"""
chunker.py
Walks a Python repo, parses each file with tree-sitter, and extracts
function/class-level chunks (not arbitrary text splits).

Each chunk = one function or class, with metadata:
  - file path
  - name
  - type (function/class/method)
  - start_line, end_line
  - source code text
  - calls: list of function names called inside this chunk (for call graph)
"""

import os
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

IGNORE_DIRS = {".git", "venv", "env", "__pycache__", "node_modules", ".venv", "chroma_db", ".idea", ".pytest_cache"}


def find_python_files(repo_path):
    py_files = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
    return py_files


def _extract_calls(node, source_bytes):
    """Walk a node's subtree and collect names of functions called inside it."""
    calls = []

    def walk(n):
        if n.type == "call":
            func_node = n.child_by_field_name("function")
            if func_node:
                if func_node.type == "identifier":
                    calls.append(source_bytes[func_node.start_byte:func_node.end_byte].decode("utf8"))
                elif func_node.type == "attribute":
                    attr = func_node.child_by_field_name("attribute")
                    if attr:
                        calls.append(source_bytes[attr.start_byte:attr.end_byte].decode("utf8"))
        for child in n.children:
            walk(child)

    walk(node)
    return list(set(calls))


def _node_name(node, source_bytes):
    name_node = node.child_by_field_name("name")
    if name_node:
        return source_bytes[name_node.start_byte:name_node.end_byte].decode("utf8")
    return "anonymous"


def chunk_file(file_path):
    """Return a list of chunk dicts for one file: top-level functions and classes
    (and methods within classes)."""
    with open(file_path, "rb") as f:
        source_bytes = f.read()

    tree = parser.parse(source_bytes)
    root = tree.root_node
    chunks = []

    def handle_node(node, parent_class=None):
        if node.type in ("function_definition", "class_definition"):
            name = _node_name(node, source_bytes)
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            text = source_bytes[node.start_byte:node.end_byte].decode("utf8")
            calls = _extract_calls(node, source_bytes)

            chunk_type = "method" if (node.type == "function_definition" and parent_class) else \
                         ("function" if node.type == "function_definition" else "class")

            full_name = f"{parent_class}.{name}" if parent_class else name

            chunks.append({
                "file": file_path,
                "name": full_name,
                "type": chunk_type,
                "start_line": start_line,
                "end_line": end_line,
                "source": text,
                "calls": calls,
            })

            # recurse into class body to grab methods, but not into function bodies
            # (we don't want to double-chunk nested functions separately for v1 simplicity)
            if node.type == "class_definition":
                body = node.child_by_field_name("body")
                if body:
                    for child in body.children:
                        handle_node(child, parent_class=name)
        else:
            for child in node.children:
                handle_node(child, parent_class=parent_class)

    handle_node(root)
    return chunks


def chunk_repo(repo_path):
    """Return all chunks across the whole repo."""
    all_chunks = []
    for file_path in find_python_files(repo_path):
        try:
            all_chunks.extend(chunk_file(file_path))
        except Exception as e:
            print(f"  [skip] {file_path}: {e}")
    return all_chunks


if __name__ == "__main__":
    import sys
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    chunks = chunk_repo(repo)
    print(f"Extracted {len(chunks)} chunks from {repo}")
    for c in chunks[:5]:
        print(f"- {c['type']} {c['name']} ({c['file']}:{c['start_line']}-{c['end_line']}) calls={c['calls']}")