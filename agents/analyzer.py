from langchain_aws import ChatBedrock

_llm = ChatBedrock(model="global.anthropic.claude-sonnet-4-5-20250929-v1:0")

def bind_tools(tools):
    """Return the shared LLM bound to the given tool list."""
    return _llm.bind_tools(tools)
