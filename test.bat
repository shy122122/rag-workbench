@echo off
setlocal
cd /d C:\Users\24592\rag-workbench

rem Force UTF-8 so Chinese assertion messages are readable in the console.
rem (Default console encoding here is GBK, which turns them into mojibake.)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

.venv\Scripts\python.exe -m pytest -q %*
pause
endlocal
