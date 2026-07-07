"""Chat completion wrapper used to generate grounded answers.

This is the last step of the RAG pipeline: given a user's question and the
chunks retrieved from the vector store, ask an LLM to produce a natural-
language answer that's grounded in that retrieved context (rather than the
model's own possibly-outdated or made-up knowledge). `LLMClient` is an
abstraction over the actual model provider (Gemini vs OpenAI vs Ollama),
so `RagPipeline` never needs to know which one is active.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from app.config import Settings
from app.services.vector_store import ScoredChunk


class LLMClient(ABC):
    """Interface every chat/completion backend must implement. Add a new
    provider (e.g. Anthropic) by subclassing this and wiring it into
    `get_llm_client` below."""

    @abstractmethod
    def generate_answer(self, question: str, chunks: list[ScoredChunk]) -> str:
        """Generate a grounded answer for `question` using `chunks` as
        context. Use case: called once per user message, after the vector
        store has already returned the top-k most relevant chunks."""


class GeminiLLMClient(LLMClient):
    """Chat completion backed by Google's Gemini models via LangChain.
    This is the default provider (`llm_provider="gemini"` in config)."""

    def __init__(self, settings: Settings):
        self._client = ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=settings.google_api_key.get_secret_value(),
            # Low temperature keeps answers factual/consistent rather than
            # creative, which matters more for a grounded Q&A assistant
            # than for open-ended writing.
            temperature=0.2,
        )
        self._system_prompt = settings.system_prompt

    def generate_answer(self, question: str, chunks: list[ScoredChunk]) -> str:
        context = _build_context(chunks)
        messages = [
            # The system prompt sets the assistant's persona/rules (see
            # `settings.system_prompt`); the human message carries the
            # actual retrieved context + question for this turn.
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
        # LangChain's `.content` is typed as `str | list`, so normalize to
        # a plain string before returning it through the API.
        return content if isinstance(content, str) else str(content)


class OpenAILLMClient(LLMClient):
    """Chat completion backed by OpenAI's models. Used when
    `llm_provider="openai"` in config."""

    def __init__(self, settings: Settings):
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.chat_model
        self._system_prompt = settings.system_prompt

    def generate_answer(self, question: str, chunks: list[ScoredChunk]) -> str:
        context = _build_context(chunks)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer the question using only the context above."
                ),
            },
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""


class OllamaLLMClient(LLMClient):
    """Chat completion backed by a local Ollama model via LangChain. Used
    when `llm_provider="ollama"` in config - handy for testing without an
    API key/cost. Requires `ollama serve` running locally with the model
    already pulled (e.g. `ollama pull llama3.2`)."""

    def __init__(self, settings: Settings):
        self._client = ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            temperature=0.2,
        )
        self._system_prompt = settings.system_prompt

    def generate_answer(self, question: str, chunks: list[ScoredChunk]) -> str:
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
        return content if isinstance(content, str) else str(content)


def get_llm_client(settings: Settings) -> LLMClient:
    """Factory that returns the configured LLM client. This is the single
    place that decides Gemini vs OpenAI vs Ollama based on
    `settings.llm_provider`, mirroring `get_embedding_provider` in
    `embeddings.py` - see `dependencies.py` for where it's called."""
    if settings.llm_provider == "openai":
        return OpenAILLMClient(settings)
    if settings.llm_provider == "ollama":
        return OllamaLLMClient(settings)
    return GeminiLLMClient(settings)


def _build_context(chunks: list[ScoredChunk]) -> str:
    """Formats retrieved chunks into a single text block to embed in the
    prompt, each one labeled with its source filename and chunk index so
    the model (and a human reading the prompt while debugging) can tell
    where each piece of context came from. Returns a placeholder string if
    no chunks were retrieved, so the LLM is explicitly told there's no
    context rather than silently getting an empty prompt section."""
    if not chunks:
        return "(no relevant context found)"

    return "\n\n".join(
        f"[Source: {chunk.filename} | chunk #{chunk.chunk_index}]\n{chunk.content}"
        for chunk in chunks
    )
