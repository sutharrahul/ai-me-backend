from .state import AgentState
from services.llm import GeminiLLMClient
from system_prompt import (
    INTENT_SYSTEM_PROMPT,
    GREETING_SYSTEM_PROMPT,
    SMALL_TAKS_SYSTEM_PROMPT,
    UNKWON_SYSTEM_PROMPT,
)
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
    state["intent"] = intent
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


def greeting(state: AgentState):
    greet = llm_bot.generate_response(
        system_prompt=GREETING_SYSTEM_PROMPT, user_query=state["user_query"]
    )
    state["llm_response"] = greet

    return state


def small_talks(state: AgentState):
    greet = llm_bot.generate_response(
        system_prompt=SMALL_TAKS_SYSTEM_PROMPT, user_query=state["user_query"]
    )
    state["llm_response"] = greet

    return state


def unknown(state: AgentState):
    renponse = llm_bot.generate_response(
        system_prompt=UNKWON_SYSTEM_PROMPT, user_query=state["user_query"]
    )
    state["llm_response"] = renponse

    return state


def portfolio(state: AgentState):
    response = llm_bot.generate_answer(
        question=state["user_query"], 
        chunks=state["chunks"]
    )

    state["llm_response"] = response

    return state


def store_chat():
    pass
