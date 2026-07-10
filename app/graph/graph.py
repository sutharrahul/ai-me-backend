from typing_extensions import TypedDict
from services.llm import GeminiLLMClient
from system_prompt import (
    INTENT_SYSTEM_PROMPT,
    GREETING_SYSTEM_PROMPT,
    SMALL_TAKS_SYSTEM_PROMPT,
    UNKWON_SYSTEM_PROMPT,
)
from app.config import Settings
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    user_query: str
    llm_response: str
    intent: str


settings = Settings()
llm_bot = GeminiLLMClient(settings=settings)


chunks = None

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
        return "portfolio"
    else:
        return "unknown"


def greeting(state: AgentState):
    greet = llm_bot.generate_response(system_prompt=GREETING_SYSTEM_PROMPT,user_query=state["user_query"])
    state["llm_response"] = greet

    return state

def small_talks(state: AgentState):
    greet = llm_bot.generate_response(system_prompt=SMALL_TAKS_SYSTEM_PROMPT,user_query=state["user_query"])
    state["llm_response"] = greet

    return state



def unknown(state: AgentState):
    renponse = llm_bot.generate_response(system_prompt=UNKWON_SYSTEM_PROMPT,user_query=state["user_query"])
    state["llm_response"] = renponse

    return state

def portfolio(state:AgentState):
    response = llm_bot.generate_answer(question=state["user_query"], chunks=chunks)

    state["llm_response"] = response

    return state

def store_chat():
    pass

graph_builder = StateGraph(AgentState)

graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("greeting",greeting)
graph_builder.add_node("small_talks",small_talks)
graph_builder.add_node("unknown",unknown)
graph_builder.add_node("portfolio",portfolio)
graph_builder.add_node("store_chat", store_chat)

graph_builder.add_edge(START, "classify_intent")
graph_builder.add_conditional_edges("classify_intent", route_intent)
graph_builder.add_edge("greeting", "store_chat")
graph_builder.add_edge("small_talks", "store_chat")
graph_builder.add_edge("unknown", "store_chat")
graph_builder.add_edge("portfolio", "store_chat")
graph_builder.add_edge("store_chat", END)

app  = graph_builder.compile()
