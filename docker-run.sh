#!/bin/bash
# Convenience script to run the Docker container

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set. LLM features will not work."
    echo "Set it with: export OPENAI_API_KEY='your-key-here'"
    echo ""
fi

# Run with docker-compose
docker-compose up --build

