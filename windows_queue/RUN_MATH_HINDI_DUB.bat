@echo off
rem ===========================================================================
rem Mathematics Hindi dubbing - Hindi-only audio, resumable production queue.
rem Mathematical variables, operators, fractions and powers use the Hindi voice.
rem Physics and chemistry queues are unaffected because this setting is process-local.
rem ===========================================================================
setlocal
title Mathematics Hindi Dubbing - Resume Queue
set "FORMULA_SPEECH_MODE=hindi_math"
set "DUB_OUTPUT_ROOT=D:\Python\video_dubbing_output_math"
echo Formula speech mode: %FORMULA_SPEECH_MODE%
echo Input: D:\Python\video_dubbing
echo Output: %DUB_OUTPUT_ROOT%
echo Existing completed outputs will be skipped; unfinished work will resume.
echo.
call "%~dp0RUN_FIX_AND_RERUN.bat"
exit /b %ERRORLEVEL%
