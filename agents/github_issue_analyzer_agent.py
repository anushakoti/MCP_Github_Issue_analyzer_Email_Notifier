"""
GitHub Issue Analyzer Agent.

Responsibilities:
  1. Connect to the GitHub MCP server to fetch open issues for a repo.
  2. Pass all issue data to AWS Bedrock Claude for AI analysis.
  3. Return a structured issues_report covering: severity classification,
     recurring patterns, top blockers, and recommended next steps.

The agent is created as a LangGraph react agent so it can autonomously
decide which MCP tools to call and how many issues to retrieve.
"""


import os
from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage
from langchain.agent import create_agent
from tools.github_tools import get_github_mcp_tools
from dotenv import load_dotenv

load_dotenv()
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")

SYSTEM_PROMPT = """You are a senior software engineering analyst specializing in GitHub issue triage.

When given a repository name GITHUB_REPO, you must:
1. Use the GitHub MCP tools to list ALL open issues for that repository.
2. For each issue, retrieve its title, body, labels, comments count, and creation date.
3. Analyze the full set of issues and produce a structured report with these sections:

## GitHub Issues Analysis Report

### Summary
- Total open issues, date range, most active labels

### Severity Classification
-  High Priority (bugs, security, blockers) — list issue numbers + titles
-  Medium Priority (enhancements, performance) — list issue numbers + titles
-  Low Priority (documentation, minor features) — list issue numbers + titles

### Patterns & Themes
- Recurring complaints or error types (group similar issues)
- Areas of the codebase most affected

### Top 3 Issues to Fix First
- Ranked by impact with brief justification

### Recommendations
- Suggested next steps for the engineering team

Be precise and structured. Do not skip any section.
"""


async def build_github_analyzer_agent():
    """
    Build and return a react agent that uses GitHub MCP tools + Bedrock LLM.

    Returns:
        A compiled LangGraph react agent ready for ainvoke().
    """
    llm = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name=os.environ.get("AWS_DEFAULT_REGION", ""),
        model_kwargs={"max_tokens": 4096, "temperature": 0.1},
    )

    github_tools = await get_github_mcp_tools()

    agent = create_agent(
        model=llm,
        tools=github_tools,
        state_modifier=SystemMessage(content=SYSTEM_PROMPT),
    )

    return agent
