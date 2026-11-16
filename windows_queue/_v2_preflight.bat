@echo off
rem ===========================================================================
rem  Shared V2.2 pre-flight used by every launcher in this folder.
rem   FATAL  (queue is not started):
rem     1. Piper program and the Hindi voices exist
rem     2. the English voices for the strict formula rule exist
rem     3. the v2 tests pass (dubbing + Transcribe Python)
rem   WARNING (queue still starts):
rem     4. Ollama (Qwen audit) is not reachable   -> translation audits are skipped
rem     5. JDK 17 is not at the usual place        -> Transcribe may not start
rem  Returns errorlevel 0 when the queue may start.
rem ===========================================================================
setlocal EnableExtensions EnableDelayedExpansion
set "GEN=%~dp0.."

echo [v2.2 preflight] checking Piper and voices...
for %%F in ("C:\piper\piper.exe"
            "D:\AI\models\piper-hi\hi\hi_IN\pratham\medium\hi_IN-pratham-medium.onnx"
            "D:\AI\models\piper-hi\hi\hi_IN\pratham\medium\hi_IN-pratham-medium.onnx.json"
            "D:\AI\models\piper-hi\hi\hi_IN\priyamvada\medium\hi_IN-priyamvada-medium.onnx"
            "D:\AI\models\piper-hi\hi\hi_IN\priyamvada\medium\hi_IN-priyamvada-medium.onnx.json"
            "D:\AI\models\piper-en\en_US-lessac-medium.onnx"
            "D:\AI\models\piper-en\en_US-lessac-medium.onnx.json"
            "D:\AI\models\piper-en\en_US-hfc_male-medium.onnx"
            "D:\AI\models\piper-en\en_US-hfc_male-medium.onnx.json") do (
  if not exist %%F (
    echo [v2.2 preflight] MISSING: %%F
    echo   English voices: https://huggingface.co/rhasspy/piper-voices  ^(en/en_US/lessac/medium and en/en_US/hfc_male/medium^)
    exit /b 2
  )
)

echo [v2.2 preflight] checking Ollama and Java...
curl -s -m 4 -o nul http://localhost:11434/api/tags
if errorlevel 1 (
  echo [v2.2 preflight] WARNING: Ollama is not reachable at http://localhost:11434 - Qwen audits will be skipped. Start Ollama for best quality.
)
if not exist "C:\Program Files\Java\jdk-17\bin\java.exe" (
  echo [v2.2 preflight] WARNING: JDK 17 not found at C:\Program Files\Java\jdk-17 - Transcribe may fail to start.
)

where py >nul 2>nul
if not errorlevel 1 ( set "PY=py -3" ) else ( set "PY=python" )
set "PYTHONIOENCODING=utf-8"
echo [v2.2 preflight] running v2 tests...
pushd "%GEN%"
%PY% -m unittest tests.test_dub_v2_rules tests.test_dub_repair_regressions tests.test_translation_qa_dubbing >nul 2>&1
set "T1=!ERRORLEVEL!"
popd
pushd "%GEN%\..\transcribe"
%PY% -m unittest discover -s tests_py >nul 2>&1
set "T2=!ERRORLEVEL!"
popd
if not "!T1!"=="0" (
  echo [v2.2 preflight] dubbing tests FAILED - run: python -m unittest tests.test_dub_v2_rules ^(in open_edu_video_generator^)
  exit /b 3
)
if not "!T2!"=="0" (
  echo [v2.2 preflight] Transcribe tests FAILED - run: python -m unittest discover -s tests_py ^(in transcribe^)
  exit /b 3
)
echo [v2.2 preflight] OK - voices present, tests pass.
exit /b 0
