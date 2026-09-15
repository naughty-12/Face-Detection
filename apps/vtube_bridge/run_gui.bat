@echo off
setlocal
cd /d "%~dp0"
rem Previously pointed at ..\..\.venv\Scripts\pythonw.exe, but this project has no
rem .venv, so the launcher failed silently. Use pythonw from PATH instead.
start "vtube_bridge" pythonw "main.py" --input 0 --send-fps 30 --landmarks
