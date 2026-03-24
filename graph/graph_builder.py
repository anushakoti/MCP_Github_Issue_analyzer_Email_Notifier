from typing import Literal

from langgraph.graph import END, START, StateGraph

from graph.nodes import email_agent_node, github_agent_node, supervisor_node
from graph.state import AgentState


def route_from_supervisor(
    state: AgentState,
) -> Literal["github_agent", "email_agent", "__end__"]:
    """Conditional edge: supervisor's decision → next node."""
    decision = state.get("next_agent", "FINISH")
    if decision == "github_agent":
        return "github_agent"
    elif decision == "email_agent":
        return "email_agent"
    return END


def build_graph():
    """Construct and compile the multi-agent graph."""
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("github_agent", github_agent_node)
    builder.add_node("email_agent", email_agent_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route_from_supervisor)
    builder.add_edge("github_agent", "supervisor")
    builder.add_edge("email_agent", "supervisor")

    graph = builder.compile()
    print("Multi-agent graph compiled")
    return graph
