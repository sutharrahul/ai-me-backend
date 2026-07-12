from langgraph.graph import StateGraph, START, END
from .state import AgentState
from .node import (
    classify_intent,
    greeting,
    small_talks,
    unknown,
    portfolio,
    store_chat,
    route_intent,
    retrieve_chunks,
)

graph_builder = StateGraph(AgentState)

graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("retrieve_chunks", retrieve_chunks)
graph_builder.add_node("greeting", greeting)
graph_builder.add_node("small_talks", small_talks)
graph_builder.add_node("unknown", unknown)
graph_builder.add_node("portfolio", portfolio)
graph_builder.add_node("store_chat", store_chat)

graph_builder.add_edge(START, "classify_intent")
graph_builder.add_conditional_edges("classify_intent", route_intent)
graph_builder.add_edge("greeting", "store_chat")
graph_builder.add_edge("small_talks", "store_chat")
graph_builder.add_edge("unknown", "store_chat")
graph_builder.add_edge("retrieve_chunks", "portfolio")
graph_builder.add_edge("portfolio", "store_chat")
graph_builder.add_edge("store_chat", END)

graph = graph_builder.compile()
