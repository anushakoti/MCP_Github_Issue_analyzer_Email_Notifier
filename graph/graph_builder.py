"""
LangGraph Multi-Agent Workflow Builder.

Defines the AgentState TypedDict and assembles the directed graph:

  START
    │
    ▼
  github_analyzer_node   ← fetches + analyzes GitHub issues via MCP
    │
    ▼
  should_email_router    ← conditional edge: send email or end?
    │
    ├─ "send_email" ──► email_notifier_node   ← sends report via MCP
    │                          │
    └─ "end"                   ▼
                              END

State fields:
  - messages:      full conversation history (LangGraph convention)
  - issues_report: AI-generated analysis string populated by github_analyzer_node
  - next_agent:    routing signal set by github_analyzer_node
  - email_sent:    boolean flag set to True by email_notifier_node
"""

import asyncio
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from agents.github_issue_analyzer_agent import build_github_analyzer_agent
from agents.email_notifier_agent import build_email_notifier_agent


# ── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    issues_report: str          # populated after GitHub analysis
    next_agent: str             # "send_email" | "end"
    email_sent: bool            # True once email is dispatched


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def github_analyzer_node(state: AgentState) -> dict:
    """
    Node 1: GitHub Issue Analyzer.

    - Builds the analyzer react agent (connects to GitHub MCP).
    - Invokes it with the user's message.
    - Extracts the issues_report from the final AIMessage.
    - Sets next_agent to 'send_email' if the user requested notification,
      otherwise 'end'.
    """
    agent = await build_github_analyzer_agent()

    result = await agent.ainvoke({"messages": state["messages"]})

    # The final message from the agent contains the full analysis report
    last_msg: AIMessage = result["messages"][-1]
    report_content = last_msg.content

    # Decide routing: send email if user mentioned email/notify/send
    user_text = " ".join(
        m.content for m in state["messages"] if isinstance(m, HumanMessage)
    ).lower()

    route = "send_email" if any(
        kw in user_text for kw in ("email", "notify", "send", "notification")
    ) else "end"

    return {
        "messages": result["messages"],
        "issues_report": report_content,
        "next_agent": route,
    }


async def email_notifier_node(state: AgentState) -> dict:
    """
    Node 2: Email Notifier.

    - Builds the email notifier react agent (connects to Gmail MCP).
    - Passes the issues_report as the task.
    - Sets email_sent = True on success.
    """
    agent = await build_email_notifier_agent()

    # Give the agent the report to email
    task_message = HumanMessage(
        content=f"Please send the following GitHub issues report via email:\n\n{state['issues_report']}"
    )

    result = await agent.ainvoke({"messages": [task_message]})

    last_msg: AIMessage = result["messages"][-1]

    return {
        "messages": result["messages"],
        "email_sent": True,
        "issues_report": state["issues_report"],
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def should_send_email(state: AgentState) -> str:
    """
    Conditional edge: route to email_notifier if user requested it,
    otherwise go straight to END.
    """
    return state.get("next_agent", "end")


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_graph():
    """
    Assemble and compile the LangGraph multi-agent workflow.

    Returns:
        Compiled StateGraph ready for ainvoke() / astream().
    """
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("github_analyzer", github_analyzer_node)
    workflow.add_node("email_notifier", email_notifier_node)

    # Entry point
    workflow.add_edge(START, "github_analyzer")

    # Conditional routing after analysis
    workflow.add_conditional_edges(
        "github_analyzer",
        should_send_email,
        {
            "send_email": "email_notifier",
            "end": END,
        },
    )

    # Email notifier always terminates
    workflow.add_edge("email_notifier", END)

    return workflow.compile()
