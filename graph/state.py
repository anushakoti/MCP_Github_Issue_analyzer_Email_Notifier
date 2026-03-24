from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State passed between every node in the graph."""

    messages: Annotated[list[BaseMessage], add_messages]

    issues_report: str

    next_agent: str

    email_sent: bool
