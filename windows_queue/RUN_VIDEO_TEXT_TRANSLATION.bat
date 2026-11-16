@echo off
rem ===========================================================================
rem Hindi dubbing plus automatic ENGLISH -> HINDI replacement of on-screen text (pipeline v3).
rem
rem  THIS FILE ONLY: the feature is switched on here (VIDEO_TEXT_TRANSLATION=1, process-local) and nowhere else. The standard
rem  launchers (RUN_FIX_AND_RERUN, RUN_SECOND_AUDIO_TRACK, RUN_MATH_HINDI_DUB, TEACHER_RERUN_BY_ID, DEEP_DIVE) never turn it on, never
rem  run its tests and are not affected by it. The YAML key production.video_text_translation stays false.
rem  Rule: an explicit VIDEO_TEXT_TRANSLATION=1/0 in the environment wins; if it is not set, the YAML key decides.
rem
rem  - Whole text blocks (labels, bullets, sentences) are found by OCR, translated as one sentence by the local MADLAD model
rem    (Ollama is NOT needed for this), erased and redrawn in Hindi with proper Devanagari shaping, in the original colour.
rem  - Formulae, equations, numbers, units and chemical symbols are never altered; a block that cannot keep them intact stays English.
rem  - A Hindi shaping engine (PySide6) is required: this launcher checks for it and runs the job with a Python that has it.
rem  - Only ONE video file is delivered per lesson (the dub with audio). The silent edited picture is an intermediate and is removed
rem    after muxing. Set KEEP_VISUAL_INTERMEDIATE=1 to keep it under dubbed\<video>\_work.
rem  - Reports: dubbed\<video>\video_text_translation_report.json (blocks found, translated, rejected and why) and the plan
rem    <video>.visual_translations.auto.json beside the video (delete it to re-plan; a hand-edited copy named
rem    <video>.visual_translations.json always wins).
rem  YAML keys (production section) you may tune: video_text_translation_engine (madlad | qwen | glossary),
rem    video_text_translation_ocr_every, video_text_translation_min_confidence.
rem  Output is separate and resumable.
rem ===========================================================================
setlocal
title Hindi Dubbing with Video Text Translation (v3)
set "VIDEO_TEXT_TRANSLATION=1"
set "VIDEO_TEXT_TRANSLATION_AUTO=1"
set "FORMULA_SPEECH_MODE=english_stem"
set "DUB_OUTPUT_ROOT=D:\Python\video_dubbing_output_visual_hi_v2"
echo Video text translation v3: ENABLED  (whole-block translation, MADLAD, shaped Hindi)
echo Input: D:\Python\video_dubbing
echo Output: %DUB_OUTPUT_ROOT%
echo.

rem ---- checks that belong to THIS launcher only ----
set "VOK="
py -3 -c "import PySide6, cv2, rapidocr_onnxruntime, numpy, PIL" >nul 2>&1 && set "VOK=py -3"
if not defined VOK python -c "import PySide6, cv2, rapidocr_onnxruntime, numpy, PIL" >nul 2>&1 && set "VOK=python"
if not defined VOK (
  echo MISSING: a Python with PySide6 + OpenCV + RapidOCR ^(needed to draw Hindi correctly^).
  echo Install:  pip install PySide6 opencv-python rapidocr-onnxruntime
  pause
  exit /b 2
)
if not exist "D:\AI\models\madlad400-3b-mt" echo WARNING: MADLAD model folder not found - translation will fall back to Qwen ^(needs Ollama^) or reviewed wording only.
set "PYTHONIOENCODING=utf-8"
pushd "%~dp0.."
%VOK% -m unittest tests.test_video_text_blocks tests.test_video_text_generic >nul 2>&1
set "VT=%ERRORLEVEL%"
popd
if not "%VT%"=="0" (
  echo The video-text translation tests FAILED - not starting. Run: python -m unittest tests.test_video_text_blocks tests.test_video_text_generic
  pause
  exit /b 3
)
echo Visual-translation tools and tests: OK
echo.
call "%~dp0RUN_FIX_AND_RERUN.bat"
exit /b %ERRORLEVEL%
