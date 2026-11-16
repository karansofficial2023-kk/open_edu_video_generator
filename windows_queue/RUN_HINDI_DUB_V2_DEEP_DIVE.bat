@echo off
rem ===========================================================================
rem  Hindi dubbing  -  V2 DEEP DIVE
rem
rem  What v2 does
rem   * STRICT RULE: chemical formulas, equations, bonds (C=O = "C double bond O"), groups
rem     (-COOH), IUPAC suffixes (-one), atom labels (AB, B1) and acronyms (AC) are spoken by an
rem     ENGLISH voice. They are never read in Hindi. Everything else is spoken in Hindi.
rem   * Better ASR: English locked, STEM vocabulary prompt (Buckminsterfullerene, hexagonal,
rem     slip ring ...), CJK debris removed, zero-length digits kept, "1." / "2." not split off.
rem   * Formulas / labels are masked as [1] [2] before MADLAD translates, then restored.
rem   * The Qwen repair prompt carries the NCERT glossary (भुजा AB, मध्यमा उंगली, स्लिप रिंग ...).
rem   * The translated contract is inspected and repaired BEFORE dubbing.
rem   * v2.1: Whisper no longer repeats text across windows (no phantom words / unpunctuated run-ons),
rem     long run-on cues are cut at pauses, split formulas (R -C -O -N -H2) are rejoined,
rem     ambiguous voice pitch picks the nearer voice and is flagged, QA cue windows are tolerant.
rem   * v2.2: the CLOSING LINE ("Keep imbibing, we believe in you") is rescued by a tail re-decode and the
rem     QA flags any spoken audio left after the last cue; the English formula voice is pitch/level-matched
rem     to the Hindi narrator (male English voice = hfc_male) so it sounds less like two speakers.
rem   * Contracts from an older ASR recipe and dubs below QA v22 are rebuilt automatically (QUICK reuses seeds).
rem   * Stricter post-TTS speech check; timing never aborts a video.
rem
rem  Usage (options can be combined)
rem     RUN_HINDI_DUB_V2_DEEP_DIVE.bat              3 test videos, fresh ASR + translation (best quality)
rem     RUN_HINDI_DUB_V2_DEEP_DIVE.bat QUICK        reuse the translated contracts of the last run, only
rem                                                 repair them and re-dub (fast)
rem     RUN_HINDI_DUB_V2_DEEP_DIVE.bat ALL          every video in D:\Python\video_dubbing
rem     RUN_HINDI_DUB_V2_DEEP_DIVE.bat NEW          only NEW videos in the input folder (finished v2 dubs are skipped)
rem     RUN_HINDI_DUB_V2_DEEP_DIVE.bat KEEP         keep the original English track as a 2nd audio track
rem  Fresh mode needs Ollama running (Qwen audit), exactly like the previous run.
rem ===========================================================================
setlocal EnableExtensions EnableDelayedExpansion
title Hindi Dubbing V2 Deep Dive

set "GEN=%~dp0.."
set "INPUT_DIR=D:\Python\video_dubbing"
set "OUT=D:\Python\video_dubbing_output_qwen_v2_deep_dive"
set "SEED=D:\Python\video_dubbing_output_qwen_improved\transcribe"
set "OPT_ALL=" & set "OPT_QUICK=" & set "OPT_KEEP=" & set "OPT_NEW="
for %%A in (%*) do (
  if /I "%%~A"=="ALL" set "OPT_ALL=1"
  if /I "%%~A"=="QUICK" set "OPT_QUICK=1"
  if /I "%%~A"=="KEEP" set "OPT_KEEP=1"
  if /I "%%~A"=="NEW" set "OPT_NEW=1"
)
set "EXTRA="
if defined OPT_NEW set "OPT_ALL=1"
if defined OPT_KEEP set "EXTRA=!EXTRA! -KeepOriginalAudio"
rem (all of these paths contain no spaces, so they are passed unquoted)
if defined OPT_NEW (
  rem NEW: only videos without a finished v2 dub are processed; finished ones are skipped
  set "RERUN="
) else if defined OPT_QUICK (
  set "RERUN=-RerunDubbing" & set "EXTRA=!EXTRA! -SeedContractsFrom %SEED%"
) else (
  set "RERUN=-RerunDubbing -RerunTranscribe"
)

pushd "%GEN%" || (echo Cannot open %GEN% & pause & exit /b 1)

echo.
echo [1/3] Pre-flight (voices, Ollama/Java warnings, tests)...
call "%~dp0_v2_preflight.bat"
if errorlevel 1 (
  echo.
  echo V2.2 pre-flight failed - queue not started.
  popd & pause & exit /b 3
)

echo.
echo [2/3] Preparing the input list...
if not exist "%OUT%\reports" mkdir "%OUT%\reports"
set "MANIFEST=%OUT%\reports\v2_inputs.txt"
if defined OPT_ALL (
  set "INPUT_ARG=-InputPath %INPUT_DIR%"
) else (
  > "%MANIFEST%" echo %INPUT_DIR%\AC Motor.mp4
  >> "%MANIFEST%" echo %INPUT_DIR%\Allotropes of Carbon.mp4
  >> "%MANIFEST%" echo %INPUT_DIR%\Carbonyl and Nitrogen Containing Functional Groups.mp4
  set "INPUT_ARG=-InputManifest %MANIFEST%"
)

echo.
echo [3/3] Transcribe + contract precheck + Hindi dubbing queue (v2 deep dive)
echo      Output: %OUT%
if defined OPT_NEW ( echo      Mode  : NEW - only videos without a finished v2 dub ) else if defined OPT_QUICK ( echo      Mode  : QUICK - reusing contracts from %SEED% ) else ( echo      Mode  : FRESH - new ASR and translation )
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%GEN%\run_audio_dub_queue.ps1" !INPUT_ARG! -OutputRoot "%OUT%" !RERUN! !EXTRA!
set "RC=%ERRORLEVEL%"

echo.
echo Queue exit code: %RC%
echo Results : %OUT%\reports\queue_results.csv
echo QA files: %OUT%\dubbed\^<video^>\translation_qa_report.json   ^(strict_formula_rule, repairs, advisory_notes^)
echo Contract repairs: %OUT%\transcribe\^<video^>_hi_contract.precheck.json
popd
pause
exit /b %RC%
