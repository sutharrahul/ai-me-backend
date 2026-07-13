"""LLM client for the portfolio chatbot.

Wraps a chat model (Google Gemini, or a local Ollama model for dev — see
`settings.llm_provider`) and generates grounded answers: given a user's
question and the chunks retrieved from the vector store, it produces a
natural-language answer backed only by that retrieved context.
"""

from __future__ import annotations

from typing import Iterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from app.config import Settings
from app.services.vector_store import ScoredChunk


class GeminiLLMClient:
    """Chat completion backed by Google's Gemini model via LangChain, or by a
    local Ollama model when `settings.llm_provider == "ollama"`."""

    def __init__(self, settings: Settings):
        if settings.llm_provider == "ollama":
            self._client = ChatOllama(
                model=settings.ollama_chat_model,
                base_url=settings.ollama_base_url,
                temperature=0.2,
            )
        else:
            client_kwargs: dict[str, object] = {
                "model": settings.gemini_chat_model,
                # Low temperature keeps answers factual and consistent rather
                # than creative — important for a grounded Q&A assistant.
                "temperature": 0.2,
            }
            api_key = settings.resolved_google_api_key()
            if api_key:
                client_kwargs["google_api_key"] = api_key
            self._client = ChatGoogleGenerativeAI(**client_kwargs)
        self._system_prompt = settings.system_prompt

    def invoke(self, system_prompt: str, user_query: str) -> str:
        message = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ]

        response = self._client.invoke(message)
        content = response.content
        return content if isinstance(content, str) else str(content)

  
    def generate_answer(self, question: str, chunks: list[ScoredChunk]) -> str:
        """Generate a grounded answer for `question` using `chunks` as context.
        Called once per user message after the vector store returns the top-k

        most relevant chunks."""
        context = _build_context(chunks=chunks)
        return self.invoke(
            system_prompt=self._system_prompt,
            user_query=(
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer the question using only the context above."
            ),
        )

    def generate_response(self, user_query: str, system_prompt: str) -> str:
        return self.invoke(
            system_prompt=system_prompt,
            user_query=user_query,
        )

    def stream(self, system_prompt: str, user_query: str) -> Iterator[str]:
        message = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ]

        for chunk in self._client.stream(message):
            content = chunk.content
            if content:
                yield content if isinstance(content, str) else str(content)

    def stream_answer(self, question: str, chunks: list[ScoredChunk]) -> Iterator[str]:
        """Streaming counterpart to `generate_answer` - same prompt, yielded
        token-by-token as the model generates them."""
        context = _build_context(chunks=chunks)
        yield from self.stream(
            system_prompt=self._system_prompt,
            user_query=(
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer the question using only the context above."
            ),
        )

    def stream_response(self, user_query: str, system_prompt: str) -> Iterator[str]:
        yield from self.stream(system_prompt=system_prompt, user_query=user_query)


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
