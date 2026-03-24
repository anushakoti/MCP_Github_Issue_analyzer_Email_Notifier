import os
from langchain_mcp_adapters.client import MultiServerMCPClient


async def get_email_tools():
    """Connect to the Composio Gmail MCP server and return its tools."""
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
