@echo off
setlocal
cd /d "%~dp0"
start "Vtube-Studio-Bridge" "..\..\.venv\Scripts\pythonw.exe" "main.py" --input 0 --send-fps 30 --landmarks
