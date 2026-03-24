import json
import os

from langchain_aws import ChatBedrock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode

from graph.state import AgentState
from tools.github_tool import get_github_tools
from tools.gmail_tool import get_email_tools

llm = ChatBedrock(model="global.anthropic.claude-sonnet-4-5-20250929-v1:0")

SUPERVISOR_SYSTEM = (
    "You are a Supervisor that manages two specialist agents:\n"
    "- github_agent : fetches and formats GitHub issues\n"
    "- email_agent  : sends the formatted issues report by email\n\n"
    "Rules:\n"
    "1. Always call github_agent first to get the issues.\n"
    "2. After github_agent finishes, call email_agent to send the report.\n"
    "3. After email_agent confirms the email is sent, respond with FINISH.\n\n"
    'Respond ONLY with a JSON object, no markdown, no extra text:\n'
    '{"next": "github_agent" | "email_agent" | "FINISH", "reason": "<short reason>"}'
)

def supervisor_node(state: AgentState) -> AgentState:
    """Decides which agent runs next based on current state."""
    print("\n[Supervisor] Evaluating state...")

    last_msg = state["messages"][-1].content[:200] if state["messages"] else "none"

    supervisor_messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM),
        HumanMessage(
            content=(
                "Current state:\n"
                f"  issues_report available: {bool(state.get('issues_report'))}\n"
                f"  email_sent: {state.get('email_sent', False)}\n"
                f"  last message: {last_msg}\n"
                "\nWhat should happen next?"
            )
        ),
    ]

    response = llm.invoke(supervisor_messages)

    try:
        raw = response.content.strip().strip("```json").strip("```").strip()
        decision = json.loads(raw)
        next_agent = decision.get("next", "FINISH")
        reason     = decision.get("reason", "")
    except Exception:
        if not state.get("issues_report"):
            next_agent, reason = "github_agent", "No issues fetched yet"
        elif not state.get("email_sent"):
            next_agent, reason = "email_agent", "Issues ready, email not sent"
        else:
            next_agent, reason = "FINISH", "All done"

    print(f"[Supervisor] -> {next_agent}  ({reason})")

    return {
        "messages":      state["messages"],
        "issues_report": state.get("issues_report", ""),
        "next_agent":    next_agent,
        "email_sent":    state.get("email_sent", False),
    }


async def github_agent_node(state: AgentState) -> AgentState:
    """Fetches and formats GitHub issues using the GitHub MCP tool."""
    print("\n[GithubAgent] Fetching issues...")

    github_repo = os.environ.get("GITHUB_REPO", "")

    github_agent_system = (
        "You are the GitHub Agent.\n\n"
        "Your ONLY task:\n"
        "1. Call the GitHub MCP tool to list ALL open issues in repo: " + github_repo + "\n"
        "2. Return them formatted EXACTLY as:\n\n"
        "Issue #<number>:\n"
        "- type     : bug | feature\n"
        "- priority : low | medium | high\n"
        "- summary  : <one-line description>\n"
        "- detail   : <concise explanation>\n\n"
        "Do not send emails. Do not do anything else. Return only the formatted issue list."
    )

    github_tools     = await get_github_tools()
    llm_with_tools   = llm.bind_tools(github_tools)
    github_tool_node = ToolNode(github_tools)

    loop_messages = [SystemMessage(content=github_agent_system)] + list(state["messages"])

    while True:
        response = llm_with_tools.invoke(loop_messages)
        loop_messages.append(response)
        if not response.tool_calls:
            break
        tool_result = await github_tool_node.ainvoke({"messages": loop_messages})
        loop_messages.extend(tool_result["messages"])

    formatted_report = response.content
    print(f"[GithubAgent] Done. Report preview:\n{formatted_report[:300]}...")

    return {
        "messages":      [AIMessage(content="[GithubAgent] " + formatted_report)],
        "issues_report": formatted_report,
        "next_agent":    "",
        "email_sent":    state.get("email_sent", False),
    }


async def email_agent_node(state: AgentState) -> AgentState:
    """Sends the formatted issue report via Composio Gmail MCP tool."""
    print("\n[EmailAgent] Sending email...")

    issues_report = state.get("issues_report", "No issues report found.")

    email_recipient = os.environ.get("EMAIL_RECIPIENT", "")
    github_repo     = os.environ.get("GITHUB_REPO", "")

    email_agent_system = (
        "You are the Email Agent.\n\n"
        "You will receive a pre-formatted GitHub issues report.\n"
        "Your ONLY task is to send ONE email using the Composio Gmail tool:\n\n"
        "  To      : " + email_recipient + "\n"
        "  Subject : Issues Report - " + github_repo + "\n"
        "  Body    :\n"
        "    Hi,\n\n"
        "    Here is the current issue report for " + github_repo + ":\n\n"
        "    <paste the issues report here>\n\n"
        "    Please review and prioritise accordingly.\n\n"
        "    Regards\n\n"
        "Send the email immediately. Confirm once sent."
    )

    email_tools     = await get_email_tools()
    llm_with_tools  = llm.bind_tools(email_tools)
    email_tool_node = ToolNode(email_tools)

    loop_messages = [
        SystemMessage(content=email_agent_system),
        HumanMessage(content="Send the email now with this issues report:\n\n" + issues_report),
    ]

    while True:
        response = llm_with_tools.invoke(loop_messages)
        loop_messages.append(response)
        if not response.tool_calls:
            break
        tool_result = await email_tool_node.ainvoke({"messages": loop_messages})
        loop_messages.extend(tool_result["messages"])

    confirmation = response.content
    print(f"[EmailAgent] Done: {confirmation}")

    return {
        "messages":      [AIMessage(content="[EmailAgent] " + confirmation)],
        "issues_report": issues_report,
        "next_agent":    "FINISH",
        "email_sent":    True,
    }
