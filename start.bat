@echo off
setlocal
cd /d C:\Users\24592\rag-workbench

rem If port 8000 already has a running service, just open the page.
curl -s -m 2 -o nul http://127.0.0.1:8000/api/health
if %errorlevel%==0 goto open

rem Start the server in its own window (close that window to stop it).
start "RAG Workbench server - close this window to stop" cmd /k ".venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000"

rem Wait until it is ready (up to ~20 seconds).
for /l %%i in (1,1,20) do (
  curl -s -m 1 -o nul http://127.0.0.1:8000/api/health
  if not errorlevel 1 goto open
  timeout /t 1 /nobreak >nul
)

echo.
echo [RAG Workbench] Startup timed out. Check .venv and port 8000.
pause
goto :eof

:open
start "" http://127.0.0.1:8000
endlocal
