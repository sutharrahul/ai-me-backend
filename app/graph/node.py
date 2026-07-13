from .state import AgentState
from app.services.llm import GeminiLLMClient
from .system_prompt import INTENT_SYSTEM_PROMPT
from app.config import Settings
from app.services.embeddings import GeminiEmbeddingProvider
from app.services.vector_store import QdrantVectorStore

settings = Settings()
llm_bot = GeminiLLMClient(settings=settings)

query_embed = GeminiEmbeddingProvider(settings)
query_chunks = QdrantVectorStore(settings)


def retrieve_chunks(state: AgentState):
    query_embedding = query_embed.embed_query(state["user_query"])
    state["chunks"] = query_chunks.query(query_embedding, top_k=state["top_k"])

    return state


def classify_intent(state: AgentState):
    intent = llm_bot.generate_response(
        system_prompt=INTENT_SYSTEM_PROMPT, user_query=state["user_query"]
    )
    # The model reliably returns the label plus a trailing newline (e.g.
    # "GREETING\n") - route_intent below does an exact match, so this must
    # be stripped or every message falls through to its "unknown" branch.
    state["intent"] = intent.strip()
    return state


def route_intent(state: AgentState):
    intent = state["intent"]

    if intent == "GREETING":
        return "greeting"
    elif intent == "SMALL TALK":
        return "small_talks"
    elif intent == "PORTFOLIO":
        return "retrieve_chunks"
    else:
        return "unknown"

