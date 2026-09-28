"""
Research agent — identical shape to ../../research-agent/agent.py.

Lives as a package submodule (research_agent.agent) for the same reason as
coding_agent.agent: a distinct import path, so multi-agent-supervisor-dry
can import both without a filename collision. `app` is the reusable part;
`run()` behind the __main__ guard is standalone-only.
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
# ─────────────────────────────────────────────

class State(TypedDict):
    query: str
    search_queries: list[str]
    search_results: list[str]
    report: str


# ─────────────────────────────────────────────
# STRUCTURED OUTPUT
# ─────────────────────────────────────────────

class SearchPlan(BaseModel):
    queries: list[str]  # 3-5 specific search queries to answer the question


model = ChatOpenAI(model="gpt-4o")
planner = model.with_structured_output(SearchPlan)
search = TavilySearch(max_results=3)


# ─────────────────────────────────────────────
# NODE 1 — plan
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
# ─────────────────────────────────────────────

def run_searches(state: State) -> dict:
    results = []
    for query in state["search_queries"]:
        print(f"[search] {query}")
        response = search.invoke(query)
        for hit in response.get("results", []):
            results.append(f"SOURCE: {hit['url']}\n{hit['content']}")
    return {"search_results": results}


# ─────────────────────────────────────────────
# NODE 3 — synthesize
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
# ENTRY POINT — only runs when this file is executed directly
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
