@echo off
rem ===========================================================================
rem  Hindi dubbing V2.2  -  NEW VIDEOS / RESUME   (Hindi audio replaces the original)
rem
rem  Put videos in D:\Python\video_dubbing, then run this file.
rem    - new videos           : fresh ASR + translation + contract repair + dub
rem    - videos done with v2.2: skipped
rem    - unfinished videos, contracts from an older recipe, dubs below QA v22: redone
rem  Output: D:\Python\video_dubbing_output_v2   (set DUB_OUTPUT_ROOT to change it)
rem
rem  Everything in v2.2 is active here:
rem    formulas/equations/bonds/labels spoken by an English voice (male: hfc_male, female: lessac),
rem    pitch/level-matched to the Hindi narrator; STEM-vocabulary ASR without cross-window repeats;
rem    closing-line (tail) rescue; formula masking during translation; NCERT glossary; contract
rem    precheck/repair before dubbing; stricter post-TTS and end-of-video checks.
rem  Fresh transcription needs Ollama running (Qwen audit).
rem ===========================================================================
setlocal
title Hindi Dubbing V2.2 - new videos / resume
call "%~dp0_v2_preflight.bat"
if errorlevel 1 ( echo. & echo V2.2 pre-flight failed - queue not started. & pause & exit /b 1 )
set "WHISPER_PREPROCESS=1"
set "WHISPER_ALTERNATE=1"
set "WHISPERX_EVIDENCE=1"
set "FFMPEG_PATH=ffmpeg"
set "SCRIPT=%~dp0video_dubbing_queue.py"
where py >nul 2>nul
if not errorlevel 1 (
  py -3 "%SCRIPT%" --mode fix
) else (
  python "%SCRIPT%" --mode fix
)
set "RC=%ERRORLEVEL%"
echo.
echo Queue exit code: %RC%
if defined DUB_OUTPUT_ROOT ( echo Results: %DUB_OUTPUT_ROOT%\reports\queue_results.csv ) else ( echo Results: D:\Python\video_dubbing_output_v2\reports\queue_results.csv )
echo QA per video: ...\dubbed\^<video^>\translation_qa_report.json  ^(strict_formula_rule, repairs, advisory_notes, issues^)
pause
exit /b %RC%
