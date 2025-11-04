@echo off
REM Convenience script to run the Docker container on Windows

REM Check if OPENAI_API_KEY is set
if "%OPENAI_API_KEY%"=="" (
    echo Warning: OPENAI_API_KEY not set. LLM features will not work.
    echo Set it with: set OPENAI_API_KEY=your-key-here
    echo.
)

REM Run with docker-compose
docker-compose up --build

