#!/bin/bash

# Start Ollama in background
ollama serve &

# Wait for Ollama to fully start
sleep 10

# Pull Mistral model if not already
ollama pull mistral:7b-instruct-v0.2-q4_K_M

# Start FastAPI app
uvicorn app:app --host 0.0.0.0 --port 8000
