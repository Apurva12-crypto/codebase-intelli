"""
server.py
Minimal hosted version of codebase-intelli.

Flow:
  1. POST /index    {"github_url": "..."}   -> clones + indexes the repo,
                                               returns a session_id
  2. POST /ask      {"session_id": "...", "question": "..."} -> runs the
                                               agent loop against that
                                               session's index, returns
                                               the cited answer

Everything runs server-side. The user only ever sends a GitHub URL and a
question -- your source code and their code both stay on your machine.

Run with:
    uvicorn server:app --host 0.0.0.0 --port 8000

Requires ANTHROPIC_API_KEY in .env (same as main.py).
"""

import os
import re
import json
import shutil
import subprocess
import tempfile
import uuid
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import anthropic

import indexer
import query

load_dotenv()

app = FastAPI(title="codebase-intelli API")

# Allow a simple frontend (or curl / anyone) to call this from a browser.
# Tighten this to specific origins before you actually ship this publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- basic abuse / cost controls ----------
# This app calls the Anthropic API on your key and clones arbitrary repos.
# Without limits, one person could rack up a large bill or fill your disk.
MAX_SESSIONS = 30                 # oldest sessions get evicted past this
MAX_QUESTIONS_PER_SESSION = 15    # per-session question cap
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 10      # per-IP requests per window, across /index and /ask

_rate_limit_log = {}  # ip -> list of timestamps


def _check_rate_limit(ip: str):
    now = time.time()
    hits = _rate_limit_log.setdefault(ip, [])
    hits[:] = [t for t in hits if now - t < RATE_LIMIT_WINDOW_SECONDS]
    if len(hits) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(429, "Rate limit exceeded. Please wait a minute and try again.")
    hits.append(now)


def _evict_oldest_session_if_needed():
    if len(SESSIONS) < MAX_SESSIONS:
        return
    oldest_id = min(SESSIONS, key=lambda k: SESSIONS[k]["created_at"])
    session = SESSIONS.pop(oldest_id, None)
    if session:
        shutil.rmtree(os.path.dirname(session["repo_path"]), ignore_errors=True)

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

# Where cloned repos + per-session index files live
WORKDIR = os.path.join(tempfile.gettempdir(), "codebase_intelli_sessions")
os.makedirs(WORKDIR, exist_ok=True)

# In-memory session store: session_id -> {repo_path, index_file, call_graph_file, num_chunks, created_at}
SESSIONS = {}

GITHUB_URL_RE = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+/?$")


# ---------- request/response models ----------

class IndexRequest(BaseModel):
    github_url: str


class IndexResponse(BaseModel):
    session_id: str
    num_chunks: int
    repo: str


class AskRequest(BaseModel):
    session_id: str
    question: str


class AskResponse(BaseModel):
    answer: str
    tool_calls: list


# ---------- helpers ----------

def clone_repo(github_url: str, dest: str):
    """Shallow clone for speed. Raises on failure."""
    result = subprocess.run(
        ["git", "clone", "--depth", "1", github_url, dest],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()[:500]}")


TOOLS = [
    {
        "name": "semantic_search",
        "description": "Search the codebase for functions/classes semantically related to a natural-language query. Use this first to find relevant starting points for any question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_callers",
        "description": "Find all functions/methods that call a given function or class by name.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "find_callees",
        "description": "Find all functions that a given function/class calls internally.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "get_chunk_by_name",
        "description": "Get the full source code of a specific function/class by its exact name, with file path and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "grep_repo",
        "description": "Regex search across all files in the repo. Use when semantic search or the call graph don't have what you need.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
]

SYSTEM_PROMPT = """You are a codebase intelligence agent. You help engineers understand an
unfamiliar codebase by answering their questions with grounded, accurate, cited information.

Rules:
- ALWAYS use tools to look up real information before answering. Never guess.
- Use semantic_search first to find relevant code for the question.
- Use find_callers / find_callees for usage / blast-radius questions.
- Use get_chunk_by_name to read full implementations before explaining them.
- Use grep_repo as a fallback.
- Cite exact file paths and line numbers for every claim, like: (src/app.py:132-155)
- Structure your final answer for someone NEW to this codebase.
- If you can't find something after searching, say so honestly.
"""


def call_tool(name, tool_input, session):
    try:
        if name == "semantic_search":
            return json.dumps(
                query.semantic_search(
                    tool_input["query"],
                    top_k=tool_input.get("top_k", 5),
                    index_file=session["index_file"],
                ), indent=2)
        elif name == "find_callers":
            return json.dumps(
                query.find_callers(tool_input["name"], call_graph_file=session["call_graph_file"]),
                indent=2)
        elif name == "find_callees":
            return json.dumps(
                query.find_callees(tool_input["name"], call_graph_file=session["call_graph_file"]),
                indent=2)
        elif name == "get_chunk_by_name":
            return json.dumps(
                query.get_chunk_by_name(tool_input["name"], index_file=session["index_file"]),
                indent=2)
        elif name == "grep_repo":
            return json.dumps(
                query.grep_repo(tool_input["pattern"], session["repo_path"]),
                indent=2)
        else:
            return json.dumps({"error": f"unknown tool {name}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def run_agent(question: str, session: dict, max_turns=8):
    messages = [{"role": "user", "content": question}]
    tool_call_log = []

    for _ in range(max_turns):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if response.stop_reason != "tool_use":
            text_blocks = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_blocks), tool_call_log

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tu in tool_uses:
            tool_call_log.append({"tool": tu.name, "input": tu.input})
            result = call_tool(tu.name, tu.input, session)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result[:8000],
            })

        messages.append({"role": "user", "content": tool_results})

    return "Hit max turns without a final answer. Try a narrower question.", tool_call_log


# ---------- routes ----------

@app.get("/")
def health():
    return {"status": "ok", "active_sessions": len(SESSIONS)}


@app.post("/index", response_model=IndexResponse)
def index_endpoint(req: IndexRequest, request: Request):
    _check_rate_limit(request.client.host)

    url = req.github_url.strip().rstrip("/")
    if not GITHUB_URL_RE.match(url):
        raise HTTPException(400, "Please provide a valid public GitHub repo URL, e.g. https://github.com/owner/repo")

    _evict_oldest_session_if_needed()

    session_id = uuid.uuid4().hex[:12]
    repo_path = os.path.join(WORKDIR, session_id, "repo")
    index_file = os.path.join(WORKDIR, session_id, "index.pkl")
    call_graph_file = os.path.join(WORKDIR, session_id, "call_graph.json")
    os.makedirs(os.path.dirname(repo_path), exist_ok=True)

    try:
        clone_repo(url, repo_path)
    except Exception as e:
        shutil.rmtree(os.path.dirname(repo_path), ignore_errors=True)
        raise HTTPException(400, f"Could not clone repo: {e}")

    try:
        num_chunks = indexer.index_repo(repo_path, index_file=index_file, call_graph_file=call_graph_file)
    except Exception as e:
        shutil.rmtree(os.path.dirname(repo_path), ignore_errors=True)
        raise HTTPException(500, f"Indexing failed: {e}")

    if not num_chunks:
        shutil.rmtree(os.path.dirname(repo_path), ignore_errors=True)
        raise HTTPException(400, "No Python files found to index in this repo.")

    SESSIONS[session_id] = {
        "repo_path": repo_path,
        "index_file": index_file,
        "call_graph_file": call_graph_file,
        "num_chunks": num_chunks,
        "created_at": time.time(),
        "repo_url": url,
    }

    return IndexResponse(session_id=session_id, num_chunks=num_chunks, repo=url)


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest, request: Request):
    _check_rate_limit(request.client.host)

    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(404, "Unknown session_id. Call /index first.")

    session["question_count"] = session.get("question_count", 0) + 1
    if session["question_count"] > MAX_QUESTIONS_PER_SESSION:
        raise HTTPException(429, f"This session hit its {MAX_QUESTIONS_PER_SESSION}-question limit. Re-index to start a new session.")

    answer, tool_calls = run_agent(req.question, session)
    return AskResponse(answer=answer, tool_calls=tool_calls)


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    session = SESSIONS.pop(session_id, None)
    if session:
        shutil.rmtree(os.path.dirname(session["repo_path"]), ignore_errors=True)
    return {"deleted": bool(session)}