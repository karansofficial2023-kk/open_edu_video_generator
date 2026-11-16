@echo off
rem ===========================================================================
rem  Hindi dubbing V2.2  -  TEACHER CORRECTION RERUN for ONE video ID (e.g. V000001)
rem
rem  Applies the reviewed Hindi corrections from D:\Python\video_dubbing\teacher_corrections\<ID>.json by
rem  shot ID, then re-dubs that single video with every v2.2 check (formula voice rule, contract
rem  precheck, voice matching, end-of-video check). Find IDs in D:\Python\video_dubbing\video_ids.csv.
rem  Output goes to D:\Python\video_dubbing_output_v2 (set DUB_OUTPUT_ROOT to change it).
rem ===========================================================================
setlocal
if "%~1"=="" (
  echo Usage: %~nx0 V000001
  echo Find video IDs in D:\Python\video_dubbing\video_ids.csv.
  exit /b 2
)
title Hindi Dubbing V2.2 - teacher rerun %~1
call "%~dp0_v2_preflight.bat"
if errorlevel 1 ( echo. & echo V2.2 pre-flight failed - rerun not started. & pause & exit /b 1 )
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
