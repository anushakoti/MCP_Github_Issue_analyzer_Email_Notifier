"""
GitHub Issue Analyzer & Email Notifier — Interactive CLI.

Usage:
    python cli.py

Example prompts:
    Analyze open issues in anushakoti/RAG_HealthCare and email me a summary
    What are the top issues in octocat/Hello-World?
    Analyze facebook/react and send a report to my email
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from graph.graph_builder import build_graph


async def main():
    graph = build_graph()

    print("=" * 60)
    print("  🔍 GitHub Issue Analyzer & Email Notifier")
    print("=" * 60)
    print("  Powered by: LangGraph + AWS Bedrock + MCP")
    print("  Type 'quit' or 'exit' to stop.\n")
    print("  Example: 'Analyze issues in owner/repo and email me'")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye! 👋")
            break

        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "issues_report": "",
            "next_agent": "",
            "email_sent": False,
        }

        print("\n⏳ Analyzing repository issues...\n")

        try:
            result = await graph.ainvoke(initial_state)
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue

        messages = result["messages"]
        final_response = messages[-1].content

        print(f"Agent:\n{final_response}\n")

        if result.get("email_sent"):
            print("📧 Email notification dispatched successfully.\n")

        print("-" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
