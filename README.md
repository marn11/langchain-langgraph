# LangGraph Agent Patterns

A series of progressively more advanced agent architectures built with **LangChain + LangGraph** — each one adds exactly one new concept on top of the previous, from a single hand-rolled tool loop up to a full multi-agent supervisor, and finally the same capabilities assembled by *configuration* instead of by hand.

## The progression

| # | Pattern | Folder | New concept |
|---|---|---|---|
| 1 | **Coding agent** | [`langgraph-coding-agent/`](langgraph-coding-agent) | A raw-SDK coding agent ([`marn11/coding-agent`](https://github.com/marn11/coding-agent)) rebuilt with `@tool` decorators, a typed `State`, and an explicit graph of nodes + edges |
| 2 | **Research agent** | [`research-agent/`](research-agent) | `TypedDict` state with custom fields beyond messages — a plan → search → synthesize workflow |
| 3 | **Multi-agent supervisor** | [`multi-agent-supervisor/`](multi-agent-supervisor) | A supervisor node routing each turn to one of several worker *subgraphs* (hub-and-spoke cycle) |
| 4 | **Supervisor, DRY** | [`multi-agent-supervisor-dry/`](multi-agent-supervisor-dry) | Same architecture, but with workers as real importable subpackages so the tool/node/pipeline code exists in exactly one place |
| 5 | **Human-in-the-loop** | [`hitl-agent/`](hitl-agent) | Checkpointing + persistence (`SqliteSaver`) so `interrupt()` can pause the graph and wait for a human decision |
| 6 | **DeepAgents** | [`deepagents-example/`](deepagents-example) | All of the above capabilities — tools, sub-agents, HITL — assembled by **configuration** rather than hand-wiring a `StateGraph` |

## Why this exists

Agent frameworks are easy to *use* and hard to *understand*. This repo deliberately builds each capability by hand first, then shows the framework's built-in equivalent — so every abstraction has a concrete "what it replaces" anchor.

## Setup

Each folder is a self-contained project. Typical requirements:

- Python 3.12+ with [uv](https://docs.astral.sh/uv/)
- An `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in `.env` (see each folder's `.env.example`)
- `uv sync` inside the folder, then `uv run agent.py`

## Stack

Python · LangChain · LangGraph · SQLite checkpointing · OpenAI / Anthropic
