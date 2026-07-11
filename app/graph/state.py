from typing_extensions import TypedDict


class AgentState(TypedDict):
    user_query: str
    llm_response: str
    intent: str
