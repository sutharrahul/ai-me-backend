"""LLM client for the portfolio chatbot.

Wraps a chat model (Google Gemini, or a local Ollama model for dev — see
`settings.llm_provider`) and generates grounded answers: given a user's
question and the chunks retrieved from the vector store, it produces a
natural-language answer backed only by that retrieved context.
"""

from __future__ import annotations

import re
from typing import Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from app.config import Settings
from app.services.vector_store import ScoredChunk

# One entry per prior message, e.g. {"role": "user", "message": "..."} -
# the same shape already persisted in `Chat.messages` (see
# `app/db/schema_modal.py`), so callers can pass a DB-fetched slice
# straight through.
ConversationHistory = list[dict]


def _build_messages(
    system_prompt: str,
    user_query: str,
    history: ConversationHistory | None = None,
) -> list[BaseMessage]:
    """Builds a proper multi-turn message list (not a string-concatenated
    transcript) so the model sees prior turns as actual conversation
    history. Without this, every request is answered with zero memory of
    what was just discussed - a short follow-up like "yes" or "tell me
    more" has no way to inherit the prior turn's meaning."""
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    for turn in history or []:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["message"]))
        else:
            messages.append(AIMessage(content=turn["message"]))
    messages.append(HumanMessage(content=user_query))
    return messages


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

    def invoke(
        self,
        system_prompt: str,
        user_query: str,
        history: ConversationHistory | None = None,
    ) -> str:
        message = _build_messages(system_prompt, user_query, history)

        response = self._client.invoke(message)
        content = response.content
        return content if isinstance(content, str) else str(content)

    def generate_answer(
        self,
        question: str,
        chunks: list[ScoredChunk],
        history: ConversationHistory | None = None,
    ) -> str:
        """Generate a grounded answer for `question` using `chunks` as context.
        Called once per user message after the vector store returns the top-k

        most relevant chunks."""
        context = _build_context(chunks=chunks)
        return self.invoke(
            system_prompt=self._system_prompt,
            user_query=(
                f"My own notes about myself (already in my voice):\n{context}\n\n"
                f"Visitor's question: {question}\n\n"
                "Answer as myself, using only the notes above."
            ),
            history=history,
        )

    def generate_response(
        self,
        user_query: str,
        system_prompt: str,
        history: ConversationHistory | None = None,
    ) -> str:
        return self.invoke(
            system_prompt=system_prompt,
            user_query=user_query,
            history=history,
        )

    def stream(
        self,
        system_prompt: str,
        user_query: str,
        history: ConversationHistory | None = None,
    ) -> Iterator[str]:
        message = _build_messages(system_prompt, user_query, history)

        for chunk in self._client.stream(message):
            content = chunk.content
            if content:
                yield content if isinstance(content, str) else str(content)

    def stream_answer(
        self,
        question: str,
        chunks: list[ScoredChunk],
        history: ConversationHistory | None = None,
    ) -> Iterator[str]:
        """Streaming counterpart to `generate_answer` - same prompt, yielded
        token-by-token as the model generates them."""
        context = _build_context(chunks=chunks)
        yield from self.stream(
            system_prompt=self._system_prompt,
            user_query=(
                f"My own notes about myself (already in my voice):\n{context}\n\n"
                f"Visitor's question: {question}\n\n"
                "Answer as myself, using only the notes above."
            ),
            history=history,
        )

    def stream_response(
        self,
        user_query: str,
        system_prompt: str,
        history: ConversationHistory | None = None,
    ) -> Iterator[str]:
        yield from self.stream(system_prompt=system_prompt, user_query=user_query, history=history)


def _build_context(chunks: list[ScoredChunk]) -> str:
    """Formats retrieved chunks into a single text block for the prompt.
    Deliberately has NO "[Source: filename]"-style citation labels - that
    framing reads like "here are reference documents to summarize", which
    pushes the model toward third-person reporting ("according to the
    document...") and fights the first-person persona in
    PORTFOLIO_SYSTEM_PROMPT. `filename`/`chunk_index` are still used
    elsewhere (the `sources` field returned to the frontend) - just not
    shown to the model. Returns a placeholder string if no chunks were
    retrieved so the LLM is explicitly told there's nothing rather than
    getting an empty prompt section."""
    if not chunks:
        return "(no relevant information found)"

    return "\n\n---\n\n".join(chunk.content for chunk in chunks)


# Order matters: specific verb patterns must run before the bare "Rahul"
# fallback, or e.g. "Rahul has" would become "I has" (wrong conjugation)
# instead of "I have". This is a last-resort safety net, not the primary
# defense - PORTFOLIO_SYSTEM_PROMPT already instructs first person
# extensively, but the model still occasionally drifts into third person
# on some topics (observed repeatedly for skills-related questions even
# after multiple rounds of prompt tuning), so this catches what the
# prompt alone doesn't. Deliberately narrow (only the patterns actually
# observed) rather than a general-purpose pronoun flipper, which would
# risk mangling grammar it hasn't been tested against.
_FIRST_PERSON_FIXUPS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bRahul(?:'s|’s)\b"), "my"),
    # Verb-conjugation-aware, covering both "Rahul <verb>" and "he/He
    # <verb>" (the pronoun used to keep referring to Rahul mid-paragraph)
    # since both need the same first-person conjugation.
    (re.compile(r"\b(?:Rahul|[Hh]e) (?:has|have)\b"), "I have"),
    (re.compile(r"\b(?:Rahul|[Hh]e) (?:is|was)\b"), "I'm"),
    (re.compile(r"\b(?:Rahul|[Hh]e) (?:maintains|maintained)\b"), "I maintain"),
    (re.compile(r"\b(?:Rahul|[Hh]e) (?:specializes|specialized)\b"), "I specialize"),
    (re.compile(r"\b(?:Rahul|[Hh]e) (?:works|worked)\b"), "I work"),
    (re.compile(r"\b(?:Rahul|[Hh]e) (?:builds|built)\b"), "I've built"),
    (re.compile(r"\b(?:Rahul|[Hh]e) (?:drives|drove)\b"), "I drive"),
    (re.compile(r"\b(?:Rahul|[Hh]e) (?:enjoys|enjoyed)\b"), "I enjoy"),
    (re.compile(r"\b(?:Rahul|[Hh]e) (?:believes|believed)\b"), "I believe"),
    (re.compile(r"\b(?:Rahul|[Hh]e) (?:contributes|contributed)\b"), "I contribute"),
    # Bare fallback for any verb not covered above - imperfect conjugation
    # (e.g. an unanticipated verb could come out as "I drives") is an
    # acceptable trade-off against leaving "Rahul"/"he" visible.
    (re.compile(r"\bRahul\b"), "I"),
    (re.compile(r"\bHe\b"), "I"),
    (re.compile(r"\bhe\b"), "I"),
    (re.compile(r"\bhis\b"), "my"),
    (re.compile(r"\bHis\b"), "My"),
    (re.compile(r"\bhim\b"), "me"),
]


def fix_first_person(text: str) -> str:
    """Rewrites any residual third-person self-reference ("Rahul", "he",
    "his"...) into first person. See `_FIRST_PERSON_FIXUPS` above for why
    this exists and its limits."""
    for pattern, replacement in _FIRST_PERSON_FIXUPS:
        text = pattern.sub(replacement, text)
    return text
