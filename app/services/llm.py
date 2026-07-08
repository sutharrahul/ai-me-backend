"""LLM client for the portfolio chatbot.

Wraps Google Gemini's chat model (via LangChain) and generates grounded
answers: given a user's question and the chunks retrieved from the vector
store, it produces a natural-language answer backed only by that retrieved
context.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import Settings
from app.services.vector_store import ScoredChunk


class GeminiLLMClient:
    """Chat completion backed by Google's Gemini model via LangChain."""

    def __init__(self, settings: Settings):
        client_kwargs: dict[str, object] = {
            "model": settings.gemini_chat_model,
            # Low temperature keeps answers factual and consistent rather than
            # creative — important for a grounded Q&A assistant.
            "temperature": 0.2,
        }
        api_key = settings.resolved_google_api_key()
        if api_key:
            client_kwargs["google_api_key"] = api_key
        self._client = ChatGoogleGenerativeAI(**client_kwargs)
        self._system_prompt = settings.system_prompt

    def generate_answer(self, question: str, chunks: list[ScoredChunk]) -> str:
        """Generate a grounded answer for `question` using `chunks` as context.
        Called once per user message after the vector store returns the top-k
        most relevant chunks."""
        context = _build_context(chunks)
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(
                content=(
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer the question using only the context above."
                )
            ),
        ]
        response = self._client.invoke(messages)
        content = response.content
        # LangChain's `.content` is typed as `str | list`, so normalise to a
        # plain string before returning it through the API.
        return content if isinstance(content, str) else str(content)


def get_llm_client(settings: Settings) -> GeminiLLMClient:
    """Returns the LLM client. Used by `dependencies.py` for DI."""
    return GeminiLLMClient(settings)


def _build_context(chunks: list[ScoredChunk]) -> str:
    """Formats retrieved chunks into a single text block for the prompt,
    each labelled with its source filename and chunk index. Returns a
    placeholder string if no chunks were retrieved so the LLM is explicitly
    told there's no context rather than getting an empty prompt section."""
    if not chunks:
        return "(no relevant context found)"

    return "\n\n".join(
        f"[Source: {chunk.filename} | chunk #{chunk.chunk_index}]\n{chunk.content}"
        for chunk in chunks
    )
