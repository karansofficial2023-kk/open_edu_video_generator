@echo off
setlocal
if "%~1"=="" (
  echo Usage: %~nx0 V000001
  echo Find video IDs in video_ids.csv.
  exit /b 2
)
set "SCRIPT=%~dp0video_dubbing_queue.py"
where py >nul 2>nul
if not errorlevel 1 (
  py -3 "%SCRIPT%" --mode teacher-rerun --video-id "%~1"
) else (
  python "%SCRIPT%" --mode teacher-rerun --video-id "%~1"
)
set "RC=%ERRORLEVEL%"
echo.
echo Rerun exit code: %RC%
pause
exit /b %RC%
