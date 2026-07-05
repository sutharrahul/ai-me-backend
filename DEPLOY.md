# Deployment Guide

## Building Docker Image
\`\`\`bash
docker build -t ai-me-backend:latest .
docker tag ai-me-backend:latest ai-me-backend:$(date +%Y.%m.%d)
\`\`\`

## Running Locally
\`\`\`bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
\`\`\`

## Environment Variables
See `.env.local` for required configuration.

