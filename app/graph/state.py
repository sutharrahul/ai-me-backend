from typing_extensions import TypedDict
from app.services.vector_store import ScoredChunk


class AgentState(TypedDict):
    user_query: str
    llm_response: str
    intent: str
    chunks: list[ScoredChunk]
    top_k:int
    session_id : str
