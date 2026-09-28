"""
LangGraph multi-agent supervisor.

New concept vs the other two agents in this repo:
  - a supervisor node routes each turn to ONE of several worker subgraphs
  - workers are full compiled StateGraphs, added as nodes in a parent graph
  - the graph is a hub-and-spoke cycle: supervisor -> worker -> supervisor -> ...
    (same 2-node-cycle idea as langgraph-coding-agent's call_model/tools loop,
    but with more than one possible target)

Self-contained on purpose (own venv/.env, like the other two folders) —
rebuilds slimmed-down coder/researcher subgraphs instead of importing the
sibling projects, so it doesn't depend on their separate virtualenvs.
"""

import os
import subprocess
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

# ─────────────────────────────────────────────
# PARENT STATE
#
# Same `messages` reducer pattern as langgraph-coding-agent, plus `next` —
# the supervisor's routing decision, read by the conditional edge below.
# ─────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str


# ─────────────────────────────────────────────
# WORKER 1 — coder subgraph
#
# Same shape as langgraph-coding-agent/agent.py, trimmed to two tools.
# Its State only declares `messages`, which the parent State also has —
# shared key, so the compiled graph can be added directly via add_node,
# no wrapper function needed.
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


coder_tools = [read_file, bash]
coder_model = ChatOpenAI(model="gpt-4o").bind_tools(coder_tools)


class CoderState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def coder_call_model(state: CoderState) -> dict:
    return {"messages": [coder_model.invoke(state["messages"])]}


def coder_should_continue(state: CoderState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


coder_graph = StateGraph(CoderState)
coder_graph.add_node("call_model", coder_call_model)
coder_graph.add_node("tools", ToolNode(coder_tools))
coder_graph.set_entry_point("call_model")
coder_graph.add_conditional_edges(
    "call_model", coder_should_continue, {"tools": "tools", END: END}
)
coder_graph.add_edge("tools", "call_model")
coder_app = coder_graph.compile()


# ─────────────────────────────────────────────
# WORKER 2 — researcher subgraph
#
# Same shape as research-agent/agent.py. Its State (query/search_queries/
# search_results/report) shares NO keys with the parent State, so it can't
# be added directly — `call_researcher` below is the wrapper node that
# translates the parent's `messages` into the subgraph's `query`, and the
# subgraph's `report` back into a `messages` entry.
# ─────────────────────────────────────────────

class SearchPlan(BaseModel):
    queries: list[str]


class ResearchState(TypedDict):
    query: str
    search_queries: list[str]
    search_results: list[str]
    report: str


planner = model.with_structured_output(SearchPlan)
tavily = TavilySearch(max_results=3)


def plan(state: ResearchState) -> dict:
    result: SearchPlan = planner.invoke(
        f"Generate 3 to 5 specific web search queries to thoroughly answer this question:\n\n{state['query']}"
    )
    return {"search_queries": result.queries}


def run_searches(state: ResearchState) -> dict:
    results = []
    for query in state["search_queries"]:
        response = tavily.invoke(query)
        for hit in response.get("results", []):
            results.append(f"SOURCE: {hit['url']}\n{hit['content']}")
    return {"search_results": results}


def synthesize(state: ResearchState) -> dict:
    joined_results = "\n\n---\n\n".join(state["search_results"])
    prompt = (
        f"You are a research assistant. Using the search results below, write a "
        f"thorough and well-structured answer to this question:\n\n"
        f"QUESTION: {state['query']}\n\n"
        f"SEARCH RESULTS:\n{joined_results}"
    )
    return {"report": model.invoke(prompt).content}


researcher_graph = StateGraph(ResearchState)
researcher_graph.add_node("plan", plan)
researcher_graph.add_node("search", run_searches)
researcher_graph.add_node("synthesize", synthesize)
researcher_graph.set_entry_point("plan")
researcher_graph.add_edge("plan", "search")
researcher_graph.add_edge("search", "synthesize")
researcher_graph.add_edge("synthesize", END)
researcher_app = researcher_graph.compile()


def call_researcher(state: State) -> dict:
    query = state["messages"][-1].content
    result = researcher_app.invoke({"query": query})
    return {"messages": [AIMessage(content=result["report"])]}


# ─────────────────────────────────────────────
# SUPERVISOR
#
# model.with_structured_output(Route) forces a pick from a closed set of
# options — same mechanism research-agent used for SearchPlan, applied
# here to routing instead of query generation.
# ─────────────────────────────────────────────

class Route(BaseModel):
    next: Literal["coder", "researcher", "FINISH"]


router = model.with_structured_output(Route)

SYSTEM_PROMPT = (
    "You are a supervisor coordinating two workers:\n"
    "- 'coder': reads/edits files and runs bash commands on the local repo.\n"
    "- 'researcher': searches the web and writes a report.\n"
    "Given the conversation so far, choose which worker should act next, "
    "or FINISH if the last worker's output already answers the request."
)


def supervisor(state: State) -> dict:
    messages = [HumanMessage(content=SYSTEM_PROMPT)] + state["messages"]
    result: Route = router.invoke(messages)
    return {"next": result.next}


def route(state: State) -> str:
    return state["next"] if state["next"] != "FINISH" else END


# ─────────────────────────────────────────────
# GRAPH
#
# supervisor -> {coder, researcher, END}, and both workers loop back to
# supervisor unconditionally — a hub-and-spoke cycle, not a straight line.
# ─────────────────────────────────────────────

graph = StateGraph(State)

graph.add_node("supervisor", supervisor)
graph.add_node("coder", coder_app)              # shared "messages" key -> no wrapper
graph.add_node("researcher", call_researcher)   # different schema -> wrapper

graph.set_entry_point("supervisor")
graph.add_conditional_edges(
    "supervisor",
    route,
    {"coder": "coder", "researcher": "researcher", END: END},
)
graph.add_edge("coder", "supervisor")
graph.add_edge("researcher", "supervisor")

app = graph.compile()


# ─────────────────────────────────────────────
# REPL
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
        result = app.invoke({"messages": messages, "next": ""})
        messages = result["messages"]
        print("Agent:", messages[-1].content)


if __name__ == "__main__":
    run()
