"""
GitHub MCP tool wrappers using langchain-mcp-adapters.
Provides async tools for fetching issues from any GitHub repository.
"""

import os
from typing import Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

load_dotenv()

async def get_github_mcp_tools() -> list[Any]:
    """
    Connect to the GitHub MCP server via Composio and return bound tools.

    The GitHub MCP server exposes tools like:
      - list_issues(owner, repo, state, labels)
      - get_issue(owner, repo, issue_number)
      - list_issue_comments(owner, repo, issue_number)

    Returns:
        List of LangChain-compatible MCP tools.
    """
    github_token = os.environ.get("GITHUB_TOKEN", "")

    client = MultiServerMCPClient(
        {
            "github": {
                "transport": "http",
                "url": "https://api.githubcopilot.com/mcp/",
                "headers": {"Authorization": f"Bearer {github_token}"},
            }
        }
    )

    tools = await client.get_tools()
    # Filter to only GitHub-related tools
    github_tools = [t for t in tools if "github" in t.name.lower() or "issue" in t.name.lower()]
    return github_tools


def parse_repo_from_input(user_input: str) -> tuple[str, str]:
    """
    Extract owner and repo name from a user query string.

    Supports patterns like:
      - 'GITHUB_REPO'
      - 'analyze issues in GITHUB_REPO'

    Args:
        user_input: Raw user message string.

    Returns:
        (owner, repo) tuple or ("", "") if not found.
    """
    import re

    match = re.search(r"([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)", user_input)
    if match:
        return match.group(1), match.group(2)
    return "", ""
