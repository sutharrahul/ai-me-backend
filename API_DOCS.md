# RAG Backend API Documentation

## Overview
This is the backend service for the AI-Me RAG (Retrieval-Augmented Generation) application.

## Endpoints

### Document Management
- `POST /api/documents/upload` - Upload and ingest documents
- `GET /api/documents` - List ingested documents
- `DELETE /api/documents/{id}` - Remove a document

### Query
- `POST /api/query` - Execute RAG query with embeddings
- `GET /api/query/{id}` - Get query results

## Authentication
Bearer token required in Authorization header.

