"""
deepagents example — the same capabilities as langgraph-coding-agent/,
multi-agent-supervisor/, and hitl-agent/, built by CONFIG instead of by
hand-wiring a StateGraph.

  concept              | your hand-rolled version                     | here
  ----------------------|-----------------------------------------------|--------------------------------
  coder tools           | @tool read_file/bash + StateGraph loop        | LocalShellBackend's built-in
  (read/write/shell)    | (langgraph-coding-agent)                      | read_file/write_file/execute/...
  sub-agent delegation  | supervisor node + coder_app/researcher_app    | subagents=[...]
                        | subgraphs (multi-agent-supervisor)             |
  HITL approval          | approve node + interrupt()/                   | interrupt_on={"execute": True}
                        | Command(resume=True/False) (hitl-agent)        |
  persistence            | SqliteSaver + compile(checkpointer=...)        | checkpointer=SqliteSaver(...)
                        | (hitl-agent)                                    | — same class, passed as a param

Verified against the installed deepagents==0.7.19 / langchain==1.4.2 source
directly (grepped the actual package, not just docs) — two things worth
flagging that a docs summary got wrong or left vague:

1. The default backend (when `backend=` is omitted) is `StateBackend()` — an
   in-memory *virtual* filesystem with no shell tool at all; `execute` only
   exists if the backend implements it. `LocalShellBackend` is opt-in
   specifically because, per its own docstring, it runs "unrestricted shell
   execution on your local machine... Use with extreme caution." Your
   hand-rolled `bash` tool in langgraph-coding-agent had that exact risk
   from line one, with no equivalent opt-in gate — deepagents defaults safe.

2. `interrupt_on={"execute": True}` does NOT resume with a plain bool like
   hitl-agent's `Command(resume=True)`. It resumes with a structured
   decision list: `Command(resume={"decisions": [{"type": "approve"}, ...]})`
   — one decision per pending tool call, and four possible decision types
   (approve/edit/reject/respond), richer than the boolean gate built by hand.
"""

from dotenv import load_dotenv
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_tavily import TavilySearch
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

researcher_subagent = {
    "name": "researcher",
    "description": "Searches the web and writes a report on a topic.",
    "system_prompt": (
        "You are a research assistant. Search the web and write a "
        "thorough, well-structured report answering the question."
    ),
    "tools": [TavilySearch(max_results=3)],
}

# Opt-in real shell/filesystem access, scoped to this folder — compare the
# unconditional `bash`/`read_file` @tool functions in langgraph-coding-agent,
# which had no equivalent scoping or opt-in step.
backend = LocalShellBackend(root_dir=".")


def run_until_done(agent, payload, config) -> dict:
    result = agent.invoke(payload, config=config)
    while "__interrupt__" in result:
        request = result["__interrupt__"][0].value
        decisions = []
        for action in request["action_requests"]:
            print(f"\n[approval needed] {action['name']}({action['args']})")
            approved = input("Approve? (y/n): ").strip().lower().startswith("y")
            decisions.append({"type": "approve"} if approved else {"type": "reject"})
        result = agent.invoke(Command(resume={"decisions": decisions}), config=config)
    return result


def run():
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
        agent = create_deep_agent(
            model="openai:gpt-4o",
            backend=backend,
            subagents=[researcher_subagent],
            interrupt_on={"execute": True},
            checkpointer=checkpointer,
            system_prompt=(
                "You are a coding assistant with shell access to the local "
                "repo. Delegate research questions to the researcher subagent."
            ),
        )

        thread_id = input("Thread id (reuse to resume a past conversation): ").strip() or "default"
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

            result = run_until_done(
                agent, {"messages": [{"role": "user", "content": user_input}]}, config
            )
            print("Agent:", result["messages"][-1].content)


if __name__ == "__main__":
    run()
