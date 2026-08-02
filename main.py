"""
main.py
The agent loop: Claude answers questions about a codebase by calling tools
(semantic search, call graph lookup, grep, exact chunk lookup) in a loop
until it has enough grounded information to give a cited answer.

Usage:
    python main.py "how does authentication work in this repo?"

Requires ANTHROPIC_API_KEY in .env or environment.
Requires you've already run: python indexer.py /path/to/repo
"""

import os
import sys
import json
from dotenv import load_dotenv
import anthropic

import query

load_dotenv()

REPO_PATH = os.environ.get("REPO_PATH", ".")  # set this to the repo you indexed
MODEL = "claude-sonnet-4-6"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

TOOLS = [
    {
        "name": "semantic_search",
        "description": "Search the codebase for functions/classes semantically related to a natural-language query. Use this first to find relevant starting points for any question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language description of what you're looking for"},
                "top_k": {"type": "integer", "description": "Number of results to return (default 5)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_callers",
        "description": "Find all functions/methods that call a given function or class by name. Use this to understand blast radius or how something is used across the codebase.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Exact function/class name, e.g. 'Session.send' or 'parse_url'"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "find_callees",
        "description": "Find all functions that a given function/class calls internally. Use this to understand what a function depends on.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Exact function/class name"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_chunk_by_name",
        "description": "Get the full source code of a specific function/class by its exact name, with file path and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Exact or partial function/class name"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "grep_repo",
        "description": "Regex search across all files in the repo (including non-code files like configs, imports, comments). Use this when semantic search or the call graph don't have what you need.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
            },
            "required": ["pattern"],
        },
    },
]


def call_tool(name, tool_input):
    """Dispatch a tool call to the corresponding function in query.py."""
    try:
        if name == "semantic_search":
            results = query.semantic_search(tool_input["query"], top_k=tool_input.get("top_k", 5))
            return json.dumps(results, indent=2)
        elif name == "find_callers":
            return json.dumps(query.find_callers(tool_input["name"]), indent=2)
        elif name == "find_callees":
            return json.dumps(query.find_callees(tool_input["name"]), indent=2)
        elif name == "get_chunk_by_name":
            return json.dumps(query.get_chunk_by_name(tool_input["name"]), indent=2)
        elif name == "grep_repo":
            return json.dumps(query.grep_repo(tool_input["pattern"], REPO_PATH), indent=2)
        else:
            return json.dumps({"error": f"unknown tool {name}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


SYSTEM_PROMPT = """You are a codebase intelligence agent. You help engineers, especially
new team members, understand an unfamiliar codebase by answering their questions
with grounded, accurate, cited information.

Rules:
- ALWAYS use tools to look up real information before answering. Never guess or
  make up file paths, function names, or behavior.
- Use semantic_search first to find relevant code for the question.
- Use find_callers / find_callees when the question is about how something is used,
  what depends on it, or "blast radius" of changing it.
- Use get_chunk_by_name to pull full source when you need to read the actual
  implementation to explain it correctly.
- Use grep_repo as a fallback for things the structured tools don't cover
  (imports, config, string literals, etc).
- You may call multiple tools, and call them multiple times, before answering.
- When you give your final answer, ALWAYS cite exact file paths and line numbers
  for every claim, like: (src/flask/app.py:132-155)
- Structure your final answer for someone NEW to this codebase: explain the flow
  step by step, don't assume prior context.
- If you can't find something after searching, say so honestly instead of guessing.
"""


def run_agent(user_question, max_turns=8):
    messages = [{"role": "user", "content": user_question}]

    for turn in range(max_turns):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b.text for b in response.content if b.type == "text"]

        if text_blocks:
            for t in text_blocks:
                print(f"\n[agent reasoning]: {t}\n")

        if response.stop_reason != "tool_use":
            final_text = "\n".join(text_blocks)
            return final_text

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tu in tool_uses:
            print(f"[calling tool] {tu.name}({tu.input})")
            result = call_tool(tu.name, tu.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result[:8000],
            })

        messages.append({"role": "user", "content": tool_results})

    return "Hit max turns without a final answer. Try a narrower question."


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How does this codebase work at a high level?"
    print(f"Question: {question}\n{'='*60}")
    answer = run_agent(question)
    print("\n" + "=" * 60)
    print("FINAL ANSWER:\n")
    print(answer)