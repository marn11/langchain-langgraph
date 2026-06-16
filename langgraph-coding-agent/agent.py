"""
LangGraph coding agent — same capabilities as coding-agent/agent.py,
rebuilt with LangChain + LangGraph.

Original:  raw OpenAI SDK, manual tool-call loop, dict-based messages
This file: @tool decorators, typed State, explicit graph with nodes + edges
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
#
# Original: big JSON dicts describing each tool schema
# LangGraph: plain Python functions with @tool — the schema is derived
#            from the function signature + docstring automatically
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
#
# Original: openai.OpenAI() client, model string passed to each call
# LangGraph: ChatOpenAI instance with tools bound once up front.
#            .bind_tools() tells the model about the tools — equivalent
#            to passing `tools=available_tools` in every API call.
# ─────────────────────────────────────────────

model = ChatOpenAI(model="gpt-4o", streaming=True).bind_tools(tools)

# ─────────────────────────────────────────────
# STATE
#
# Original: a plain list called `conversation` that you appended to manually
# LangGraph: a typed dict the graph carries between every node.
#
# `Annotated[list[BaseMessage], add_messages]` means:
#   - the field holds a list of messages
#   - when a node returns new messages, add_messages *appends* them
#     rather than replacing the whole list (reducer pattern)
# ─────────────────────────────────────────────

class State(dict):
    messages: Annotated[list[BaseMessage], add_messages]

# ─────────────────────────────────────────────
# NODES
#
# Original: the body of your `while True` loop
# LangGraph: each logical step is a node — a function that receives
#            State and returns a dict of State updates.
#
# Node 1 — call_model: invoke the LLM with the current message history
# Node 2 — tool_node:  run whichever tools the model requested.
#           ToolNode is a prebuilt that replaces the big if/elif
#           dispatch block you wrote by hand in the original agent.
# ─────────────────────────────────────────────

def call_model(state: State) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

# ─────────────────────────────────────────────
# ROUTING
#
# Original: `if final.choices[0].finish_reason == "tool_calls": ...`
# LangGraph: a conditional edge function — return the name of the
#            next node, or END to stop the graph.
# ─────────────────────────────────────────────

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END

# ─────────────────────────────────────────────
# GRAPH
#
# Original: implicit — the loop in your script IS the graph
# LangGraph: explicit declaration of nodes + edges, then .compile()
# ─────────────────────────────────────────────

graph = StateGraph(State)

graph.add_node("call_model", call_model)
graph.add_node("tools", tool_node)

graph.set_entry_point("call_model")

graph.add_conditional_edges(
    "call_model",     # from this node...
    should_continue,  # ...call this to decide where to go next
    {
        "tools": "tools",
        END: END,
    },
)

# After tools run, always go back to the model (the loop)
graph.add_edge("tools", "call_model")

app = graph.compile()

# ─────────────────────────────────────────────
# REPL
#
# Original: while True: input() → stream deltas → append to conversation
# LangGraph: same loop, but app.invoke() runs the full graph turn.
#            app.stream() with stream_mode="messages" lets us print
#            token-by-token just like the original streaming loop.
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

        # Run a clean invoke to get the final state with all messages
        # (tool calls + results included) to carry into the next turn.
        result = app.invoke({"messages": messages})
        messages = result["messages"]


if __name__ == "__main__":
    run()
