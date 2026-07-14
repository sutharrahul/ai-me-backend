from typing_extensions import TypedDict
from app.services.vector_store import ScoredChunk


class AgentState(TypedDict):
    """State passed between the intent-classification/retrieval steps
    reused directly (without the LangGraph runtime) by `POST /chat/query`
    in `app/api/routes/chat.py` - generation and persistence happen there,
    streamed, rather than as further steps on this state."""

    user_query: str
    intent: str
    chunks: list[ScoredChunk]
    top_k: int
    # Recent prior messages ({"role": ..., "message": ...} dicts, oldest
    # first) for the same conversation, if any - see `HISTORY_MESSAGE_LIMIT`
    # in `app/api/routes/chat.py`. Lets classify_intent/retrieve_chunks
    # interpret short follow-ups ("yes", "tell me more") in context instead
    # of treating every message as a standalone, memoryless query.
    history: list[dict]
