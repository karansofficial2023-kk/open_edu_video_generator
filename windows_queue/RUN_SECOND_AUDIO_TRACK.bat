@echo off
setlocal
set "SCRIPT=%~dp0video_dubbing_queue.py"
where py >nul 2>nul
if not errorlevel 1 (
  py -3 "%SCRIPT%" --mode dual
) else (
  python "%SCRIPT%" --mode dual
)
set "RC=%ERRORLEVEL%"
echo.
echo Queue exit code: %RC%
pause
exit /b %RC%
