"""
Coding agent — identical shape to ../../langgraph-coding-agent/agent.py.

Lives as a package submodule (coding_agent.agent) instead of a flat script
so it has a distinct import path from research_agent.agent — that's what
lets multi-agent-supervisor-dry import both without a filename collision,
even though both files are named agent.py.

`app` (the compiled graph) is defined at module level, so it's usable two
ways: run this file directly for a standalone REPL, or `from coding_agent.agent
import app` to reuse it as a subgraph elsewhere — the `if __name__ ==
"__main__"` guard at the bottom is what keeps the import path from also
starting a REPL.
"""

import os
import subprocess
from typing import Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

@tool
def read_file(path: str) -> str:
    """Read the contents of a file at the given path."""
    try:
        return open(path).read()
    except FileNotFoundError:
        return f"Error: file not found: {path}"


@tool
def list_files(path: str) -> str:
    """List all files in a directory."""
    try:
        return "\n".join(os.listdir(path))
    except FileNotFoundError:
        return f"Error: path not found: {path}"


@tool
def bash(command: str) -> str:
    """Execute a bash command and return its output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout
    if result.stderr:
        output += "\nSTDERR: " + result.stderr
    return output or "(no output)"


@tool
def edit_file(path: str, content: str) -> str:
    """Write content to a file, creating it if it doesn't exist."""
    try:
        open(path, "w").write(content)
        return f"Successfully wrote {path}"
    except PermissionError:
        return f"Error: can't write to the path: {path}"


@tool
def search_code(pattern: str, path: str = ".") -> str:
    """Search for a pattern in files recursively using ripgrep."""
    result = subprocess.run(
        f"rg -n {pattern} {path}", shell=True, capture_output=True, text=True
    )
    output = result.stdout
    if result.stderr:
        output += "\nSTDERR: " + result.stderr
    return output or "(no output)"


tools = [read_file, list_files, bash, edit_file, search_code]

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

model = ChatOpenAI(model="gpt-4o", streaming=True).bind_tools(tools)

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────

class State(dict):
    messages: Annotated[list[BaseMessage], add_messages]

# ─────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────

def call_model(state: State) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

# ─────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END

# ─────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────

graph = StateGraph(State)

graph.add_node("call_model", call_model)
graph.add_node("tools", tool_node)

graph.set_entry_point("call_model")

graph.add_conditional_edges(
    "call_model",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)

graph.add_edge("tools", "call_model")

app = graph.compile()

# ─────────────────────────────────────────────
# REPL — only runs when this file is executed directly
# ─────────────────────────────────────────────

def run():
    messages = []
    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            break
        if user_input.strip().lower() == "exit":
            break
        if not user_input.strip():
            continue

        messages.append(HumanMessage(content=user_input))

        print("Agent: ", end="", flush=True)

        for chunk, _ in app.stream(
            {"messages": messages},
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                print(chunk.content, end="", flush=True)

        print()

        result = app.invoke({"messages": messages})
        messages = result["messages"]


if __name__ == "__main__":
    run()
