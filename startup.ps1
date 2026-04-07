# Strip any ghost Windows registry overrides on startup so .env is respected!
$env:OPENAI_API_KEY=""

Write-Host "Clearing previously used ports (8000, 7860)..." -ForegroundColor Yellow
Get-NetTCPConnection -LocalPort 8000, 7860 -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    Where-Object { $_ -ne $PID } |
    ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }

Write-Host "Starting Multimodal Medical RAG..." -ForegroundColor Cyan

# Start the FastAPI backend
Write-Host "Starting FastAPI Backend..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", ".\venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"


# Wait a moment to ensure API starts cleanly
Start-Sleep -Seconds 3

# Start the Gradio UI
Write-Host "Starting Gradio UI..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", ".\venv\Scripts\python.exe -m ui.app"

Write-Host "Both services have been started in separate PowerShell windows!" -ForegroundColor Cyan
Write-Host " - API is running at http://localhost:8000"
Write-Host " - UI is running at http://localhost:7860"
