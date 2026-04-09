"""
Email MCP tool wrappers using langchain-mcp-adapters.
Provides async tools for dispatching email notifications via Composio Gmail MCP.
"""

import os
from typing import Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

load_dotenv()

async def get_email_mcp_tools() -> list[Any]:
    """
    Connect to the Gmail MCP server via Composio and return bound tools.

    The Gmail MCP server exposes tools like:
      - send_email(to, subject, body)
      - list_emails(query, max_results)

    Returns:
        List of LangChain-compatible MCP tools.
    """
    composio_token   = os.environ.get("COMPOSIO_TOKEN", "")
    composio_mcp_url = os.environ.get("COMPOSIO_MCP_URL", "")

    client = MultiServerMCPClient(
        {
            "composio": {
                "transport": "http",
                "url": composio_mcp_url,
                "headers": {"x-api-key": composio_token},
            }
        }
    )

    tools = await client.get_tools()
    # Filter to only email-related tools
    email_tools = [t for t in tools if "email" in t.name.lower() or "gmail" in t.name.lower() or "send" in t.name.lower()]
    return email_tools
