@echo off
echo Starting Multimodal Medical RAG...

:: Start the FastAPI backend in a new window
echo Starting FastAPI Backend...
start "API Backend" cmd /c ".\venv\Scripts\activate && uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"


:: Wait a moment to ensure API starts
timeout /t 3 > nul

:: Start the Gradio UI in a new window
echo Starting Gradio UI...
start "Gradio UI" cmd /c ".\venv\Scripts\activate && python -m ui.app"

echo Both services have been started in separate windows!
echo - API is running at http://localhost:8000
echo - UI is running at http://localhost:7860
