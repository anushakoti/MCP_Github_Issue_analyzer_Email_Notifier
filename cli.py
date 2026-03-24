import asyncio
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from graph.graph_builder import build_graph


async def main():
    graph = build_graph()
    messages = []
    print("Github issue analyzer and email notifier CLI (type 'quit' to exit)\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit"):
            break

        initial_state = {
            "messages":      [HumanMessage(content=user_input)],
            "issues_report": "",
            "next_agent":    "",
            "email_sent":    False,
        }

        result = await graph.ainvoke(initial_state)
        messages = result["messages"]

        print(f"\nAgent: {messages[-1].content}\n")


if __name__ == "__main__":
    asyncio.run(main())
