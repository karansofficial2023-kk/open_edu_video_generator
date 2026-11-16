@echo off
rem ===========================================================================
rem  Hindi dubbing V2.2  -  SECOND AUDIO TRACK   (English stays the default, Hindi is selectable)
rem
rem  Identical code and checks to RUN_FIX_AND_RERUN.bat; the only difference is the output:
rem    track 1 = original English (default), track 2 = Hindi dub.
rem    - new videos           : fresh ASR + translation + contract repair + dub
rem    - dual-track dubs done with v2.2: skipped
rem    - replacement-audio dubs, older contracts, dubs below QA v22: rebuilt as dual-track
rem  Output: D:\Python\video_dubbing_output_v2   (set DUB_OUTPUT_ROOT to change it)
rem  Fresh transcription needs Ollama running (Qwen audit).
rem ===========================================================================
setlocal
title Hindi Dubbing V2.2 - second audio track
call "%~dp0_v2_preflight.bat"
if errorlevel 1 ( echo. & echo V2.2 pre-flight failed - queue not started. & pause & exit /b 1 )
set "WHISPER_PREPROCESS=1"
set "WHISPER_ALTERNATE=1"
set "WHISPERX_EVIDENCE=1"
set "FFMPEG_PATH=ffmpeg"
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
if defined DUB_OUTPUT_ROOT ( echo Results: %DUB_OUTPUT_ROOT%\reports\queue_results.csv ) else ( echo Results: D:\Python\video_dubbing_output_v2\reports\queue_results.csv )
echo QA per video: ...\dubbed\^<video^>\translation_qa_report.json  ^(strict_formula_rule, repairs, advisory_notes, issues^)
pause
exit /b %RC%
