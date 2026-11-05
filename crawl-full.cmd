@echo off
rem Full reference-set crawl of pajak.go.id: every section except the optional
rem Pengumuman and Berita. Run by the scheduled task "DJP pajak crawl" so it
rem outlives terminals and Claude sessions.
rem
rem Two crawl passes: the second re-reads every listing and picks up pages a
rem dropped connection made the first pass miss. Safe to start again after an
rem interruption: pages already saved are skipped.
rem
rem No indexing here: the running djp server indexes new files by itself every
rem AUTO_INGEST_SECONDS (default 60). A second indexer in this process would
rem overwrite the server's vectors. If the server is down, it catches up on start.
rem
rem Watch:  Get-Content D:\ALFA\djp\data\crawl-full.log -Wait -Tail 20
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set LOG=data\crawl-full.log

echo ===== %date% %time%  crawl pass 1 ===== >> %LOG%
.venv\Scripts\python.exe main.py crawl --full --no-ingest >> %LOG% 2>&1
echo ===== %date% %time%  crawl pass 2 ===== >> %LOG%
.venv\Scripts\python.exe main.py crawl --full --no-ingest >> %LOG% 2>&1
echo ===== %date% %time%  done, exit %errorlevel% ===== >> %LOG%
