"""
LangGraph agent with checkpointing, persistence, and human-in-the-loop (HITL).

New concepts vs the other agents in this repo:
  - checkpointer: graph.compile(checkpointer=...) saves the full State after
    every node. Without one, interrupt() below has nowhere to save state to
    and can't work at all.
  - persistence: SqliteSaver writes those checkpoints to a real file, so a
    conversation survives killing and restarting this script — not just
    InMemorySaver's RAM-only checkpointing, which dies with the process.
  - thread_id: once a checkpointer is attached, you stop manually carrying
    `messages` between turns (compare langgraph-coding-agent's `messages =
    result["messages"]`). Each turn you pass only the NEW HumanMessage; the
    checkpointer supplies the rest of the history for that thread_id.
  - interrupt(): pauses a node mid-graph and durably waits for a human
    response, instead of always running tools automatically like ToolNode
    did in the coding agent. Resumed with Command(resume=...).

Verified against the actual installed langgraph package before writing this
(see conversation) rather than trusting docs summaries at face value —
result["__interrupt__"] is a list of Interrupt objects; .value is the
payload passed to interrupt(); SqliteSaver.from_conn_string() is a context
manager.
"""

import subprocess
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# ─────────────────────────────────────────────
# TOOLS — same idea as langgraph-coding-agent, trimmed to two.
# Both go through the same approval gate below; nothing runs unapproved.
# ─────────────────────────────────────────────

@tool
def read_file(path: str) -> str:
    """Read the contents of a file at the given path."""
    try:
        return open(path).read()
    except FileNotFoundError:
        return f"Error: file not found: {path}"


@tool
def bash(command: str) -> str:
    """Execute a bash command and return its output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout
    if result.stderr:
        output += "\nSTDERR: " + result.stderr
    return output or "(no output)"


tools = [read_file, bash]
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)
tool_node = ToolNode(tools)


# ─────────────────────────────────────────────
# STATE
#
# `approved` is read by the conditional edge right after `approve` runs —
# it doesn't need to persist beyond that one routing decision, but it has
# to live in State because conditional-edge functions only ever see State.
# ─────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    approved: bool


# ─────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────

def call_model(state: State) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: State) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "approve"
    return END


def approve(state: State) -> dict:
    """Pauses the graph and waits for a human y/n before any tool runs.

    On resume, this whole function re-runs from the top (LangGraph replays
    the node up to the interrupt point) — everything above `interrupt(...)`
    must be safe to repeat, which it is here since it's just reading state.
    """
    calls = state["messages"][-1].tool_calls
    decision = interrupt(
        {
            "question": "Approve these tool calls?",
            "calls": [{"name": c["name"], "args": c["args"]} for c in calls],
        }
    )
    if decision:
        return {"approved": True}

    # Rejected: every tool_call_id still needs a matching ToolMessage or the
    # next call_model turn is an invalid message sequence for the API.
    rejections = [
        ToolMessage(content="Rejected by user.", tool_call_id=c["id"]) for c in calls
    ]
    return {"messages": rejections, "approved": False}


def after_approve(state: State) -> str:
    return "tools" if state["approved"] else "call_model"


# ─────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────

graph = StateGraph(State)

graph.add_node("call_model", call_model)
graph.add_node("approve", approve)
graph.add_node("tools", tool_node)

graph.set_entry_point("call_model")

graph.add_conditional_edges(
    "call_model", should_continue, {"approve": "approve", END: END}
)
graph.add_conditional_edges(
    "approve", after_approve, {"tools": "tools", "call_model": "call_model"}
)
graph.add_edge("tools", "call_model")


# ─────────────────────────────────────────────
# REPL
#
# No module-level `app` — SqliteSaver.from_conn_string() is a context
# manager, so the graph is compiled inside it and only lives for the
# duration of `run()`.
# ─────────────────────────────────────────────

def run_until_done(app, payload, config) -> dict:
    result = app.invoke(payload, config=config)
    while "__interrupt__" in result:
        info = result["__interrupt__"][0].value
        print(f"\n[approval needed] {info['calls']}")
        answer = input("Approve? (y/n): ").strip().lower().startswith("y")
        result = app.invoke(Command(resume=answer), config=config)
    return result


def run():
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        thread_id = input("Thread id (reuse one to resume a past conversation): ").strip() or "default"
        config = {"configurable": {"thread_id": thread_id}}

        while True:
            try:
                user_input = input("You: ")
            except EOFError:
                break
            if user_input.strip().lower() == "exit":
                break
            if not user_input.strip():
                continue

            # Only the new message is passed — the checkpointer supplies
            # the rest of this thread's history automatically.
            result = run_until_done(
                app, {"messages": [HumanMessage(content=user_input)], "approved": False}, config
            )
            print("Agent:", result["messages"][-1].content)


if __name__ == "__main__":
    run()
