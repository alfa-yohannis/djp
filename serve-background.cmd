@echo off
rem Runs the web server without a console window. Started by the scheduled
rem task "DJP pajak server" so it outlives terminals and Claude sessions; the
rem task is stopped with:  Stop-ScheduledTask "DJP pajak server"
rem
rem While it runs, new crawl files are indexed every AUTO_INGEST_SECONDS.
rem
rem Watch:  Get-Content D:\ALFA\djp\data\server.log -Wait -Tail 20
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
.venv\Scripts\python.exe main.py serve >> data\server.log 2>&1
