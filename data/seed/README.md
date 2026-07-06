# Project: Portfolio RAG Chatbot

This is one of Rahul Suthar's own projects: the very AI assistant answering
questions on his portfolio site right now.

## What it does

Visitors to Rahul's portfolio can chat with an AI assistant that answers
questions about his background, skills, experience, and projects, grounded
in real content rather than guesses. This is a Retrieval-Augmented
Generation (RAG) system: instead of relying purely on a language model's
built-in knowledge, it retrieves relevant snippets from Rahul's own
"about me" content and feeds them to the model as context before it answers.

## Architecture

- **Frontend**: Next.js (App Router, TypeScript, Tailwind CSS) chat UI with
  a light/dark/system theme toggle.
- **Backend**: Python FastAPI service built with LangChain.
- **Embeddings + chat model**: pluggable providers — Google Gemini or
  OpenAI for hosted use, or a local Ollama model for free/offline testing.
- **Vector store**: pluggable — PostgreSQL with the pgvector extension, or
  Qdrant, both runnable locally via Docker.
- **Ingestion**: `.txt`/`.md`/`.pdf` files dropped into `backend/data/seed/`
  are chunked, embedded, and stored via a one-off CLI script rather than an
  upload API, since the content (Rahul's bio) rarely changes.

## Why Rahul built it

Rahul is particularly interested in real-time, AI-powered product
experiences, and wanted his own portfolio to demonstrate that interest
directly: instead of a static "About" page, visitors can ask whatever
they're curious about and get a grounded, conversational answer.
