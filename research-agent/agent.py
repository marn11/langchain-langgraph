"""
LangGraph research agent.

Workflow:
  plan → search → synthesize

New concepts vs the coding agent:
  - TypedDict State with custom fields (not just messages)
  - model.with_structured_output() to get typed objects back from the LLM
  - Linear graph (no cycle) — shows LangGraph works for pipelines too
  - Tavily web search as an external tool
"""

from typing import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END

load_dotenv()

# ─────────────────────────────────────────────
# STATE
#
# In the coding agent, State only held `messages` — a single list that
# grew as the agent worked.
#
# Here State has four distinct fields, each owned by a different node:
#   query          → set once by the caller, never changed
#   search_queries → written by `plan`, read by `search`
#   search_results → written by `search`, read by `synthesize`
#   report         → written by `synthesize`, returned to the user
#
# TypedDict (vs the dict subclass in the coding agent) is the idiomatic
# way to define State in LangGraph — it gives you type checking without
# any extra machinery.
# ─────────────────────────────────────────────

class State(TypedDict):
    query: str
    search_queries: list[str]
    search_results: list[str]
    report: str


# ─────────────────────────────────────────────
# STRUCTURED OUTPUT
#
# model.with_structured_output(SearchPlan) tells the LLM to return JSON
# that matches this Pydantic schema — instead of free text you get a
# proper Python object with a `.queries` list you can iterate over.
#
# Under the hood LangChain uses OpenAI's function-calling to enforce
# the schema — same mechanism as tool calls, but for output shaping.
# ─────────────────────────────────────────────

class SearchPlan(BaseModel):
    queries: list[str]  # 3-5 specific search queries to answer the question


model = ChatOpenAI(model="gpt-4o")
planner = model.with_structured_output(SearchPlan)
search = TavilySearch(max_results=3)


# ─────────────────────────────────────────────
# NODE 1 — plan
#
# Reads:  state["query"]
# Writes: state["search_queries"]
#
# The LLM's job here is only to decompose the user's question into
# concrete search queries. It doesn't search or write — single responsibility.
# ─────────────────────────────────────────────

def plan(state: State) -> dict:
    print(f"\n[plan] Breaking down: {state['query']}")
    result: SearchPlan = planner.invoke(
        f"Generate 3 to 5 specific web search queries to thoroughly answer this question:\n\n{state['query']}"
    )
    print(f"[plan] Queries: {result.queries}")
    return {"search_queries": result.queries}


# ─────────────────────────────────────────────
# NODE 2 — search
#
# Reads:  state["search_queries"]
# Writes: state["search_results"]
#
# No LLM here — this node is a pure data transformer. It calls Tavily
# for each query and collects the results. This is a key LangGraph idea:
# nodes don't have to involve an LLM. They're just functions.
# ─────────────────────────────────────────────

def run_searches(state: State) -> dict:
    results = []
    for query in state["search_queries"]:
        print(f"[search] {query}")
        response = search.invoke(query)
        # TavilySearch returns {"results": [...], "query": "..."}
        for hit in response.get("results", []):
            results.append(f"SOURCE: {hit['url']}\n{hit['content']}")
    return {"search_results": results}


# ─────────────────────────────────────────────
# NODE 3 — synthesize
#
# Reads:  state["query"], state["search_results"]
# Writes: state["report"]
#
# The LLM reads all collected results and writes a coherent final report.
# ─────────────────────────────────────────────

def synthesize(state: State) -> dict:
    print(f"\n[synthesize] Writing report...")
    joined_results = "\n\n---\n\n".join(state["search_results"])
    prompt = (
        f"You are a research assistant. Using the search results below, write a "
        f"thorough and well-structured answer to this question:\n\n"
        f"QUESTION: {state['query']}\n\n"
        f"SEARCH RESULTS:\n{joined_results}"
    )
    response = model.invoke(prompt)
    return {"report": response.content}


# ─────────────────────────────────────────────
# GRAPH
#
# This graph is LINEAR — no cycles, no conditional edges.
# Compare with the coding agent where add_edge("tools", "call_model")
# created a loop. Here each node runs exactly once, in order.
#
# plan → run_searches → synthesize → END
# ─────────────────────────────────────────────

graph = StateGraph(State)

graph.add_node("plan", plan)
graph.add_node("search", run_searches)
graph.add_node("synthesize", synthesize)

graph.set_entry_point("plan")
graph.add_edge("plan", "search")
graph.add_edge("search", "synthesize")
graph.add_edge("synthesize", END)

app = graph.compile()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def run():
    while True:
        try:
            query = input("\nResearch question (or 'exit'): ").strip()
        except EOFError:
            break
        if query.lower() == "exit":
            break
        if not query:
            continue

        result = app.invoke({"query": query})

        print("\n" + "═" * 60)
        print("REPORT")
        print("═" * 60)
        print(result["report"])
        print("═" * 60)


if __name__ == "__main__":
    run()
