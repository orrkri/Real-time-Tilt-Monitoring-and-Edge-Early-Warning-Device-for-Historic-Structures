@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    set PY=py
) else (
    set PY=python
)
%PY% -m pip install -r requirements.txt
start "" http://127.0.0.1:5000/
%PY% server.py
