"""
Multi-agent supervisor — DRY version of ../multi-agent-supervisor/agent.py.

There, coder_app and researcher_app were rebuilt inline because the three
projects live in separate venvs with colliding agent.py filenames. Here,
coding_agent/ and research_agent/ are real subpackages with distinct import
paths (coding_agent.agent vs research_agent.agent), so their `app`s are
imported directly — the tool/node/pipeline code exists in exactly one place.

Everything below this point is genuinely new code: the supervisor routing
logic and the wrapper needed for researcher_app's disjoint State schema.
Same reasoning as before governs which worker needs a wrapper — see
https://docs.langchain.com/oss/python/langgraph/use-subgraphs
"""

from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from coding_agent.agent import app as coder_app
from research_agent.agent import app as researcher_app

load_dotenv()

model = ChatOpenAI(model="gpt-4o")


# ─────────────────────────────────────────────
# PARENT STATE
# ─────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str


# ─────────────────────────────────────────────
# researcher_app's State (query/search_queries/search_results/report)
# shares no keys with State above, so it needs a wrapper node — same
# translate-in/translate-out pattern as the non-DRY version.
#
# coder_app's State only declares `messages`, which State above also has,
# so it's added directly below with no wrapper.
# ─────────────────────────────────────────────

def call_researcher(state: State) -> dict:
    query = state["messages"][-1].content
    result = researcher_app.invoke({"query": query})
    return {"messages": [AIMessage(content=result["report"])]}


# ─────────────────────────────────────────────
# SUPERVISOR
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
