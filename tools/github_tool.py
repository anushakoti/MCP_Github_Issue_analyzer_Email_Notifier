import os
from langchain_mcp_adapters.client import MultiServerMCPClient


async def get_github_tools():
    """Connect to the GitHub MCP server and return its tools."""
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
    print(f"[github_tool] {len(tools)} tools loaded: {[t.name for t in tools]}")
    return tools
