param([Parameter(ValueFromRemainingArguments=$true)][string[]]$VideoArgs)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
# On-screen text translation needs Hindi text shaping (PySide6), OpenCV and RapidOCR. The ComfyUI embedded Python has none of them,
# and without shaping the Hindi is drawn misspelt (vowel signs and half-letters break). Use a Python that has all of them.
if ($env:VIDEO_TEXT_TRANSLATION -eq '1' -or $env:VIDEO_TEXT_TRANSLATION_AUTO -eq '1') {
    $candidates = @()
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) { $candidates += ,@($pyLauncher.Source, '-3') }
    $systemPython = Get-Command python -ErrorAction SilentlyContinue
    if ($systemPython) { $candidates += ,@($systemPython.Source) }
    foreach ($candidate in $candidates) {
        $exe = $candidate[0]
        $preArgs = @($candidate | Select-Object -Skip 1)
        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $exe @preArgs -c 'import PySide6, cv2, rapidocr_onnxruntime, numpy, PIL' 2>$null
        $hasVisualTools = $LASTEXITCODE -eq 0
        $ErrorActionPreference = $previousErrorActionPreference
        if ($hasVisualTools) {
            Write-Host "VIDEO TEXT TRANSLATION: using $exe $($preArgs -join ' ') (Hindi shaping available)"
            & $exe @preArgs -m app.main @VideoArgs
            exit $LASTEXITCODE
        }
    }
    Write-Warning 'VIDEO TEXT TRANSLATION: no Python with PySide6 + OpenCV + RapidOCR was found; on-screen translation will refuse to draw unshaped Hindi. Install: pip install PySide6 opencv-python rapidocr-onnxruntime'
}
$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$portablePython = 'D:/AI/ComfyUI_windows_portable/python_embeded/python.exe'
if (Test-Path -LiteralPath $projectPython) {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $projectPython -c 'import sys' 2>$null
    $venvOk = $LASTEXITCODE -eq 0
    $ErrorActionPreference = $previousErrorActionPreference
    if ($venvOk) {
        & $projectPython -m app.main @VideoArgs
        exit $LASTEXITCODE
    }
}
if ((Test-Path -LiteralPath $portablePython) -and (Test-Path -LiteralPath '.runtime-deps')) {
    & $portablePython scripts/local_entry.py @VideoArgs
    exit $LASTEXITCODE
}
throw 'Install Python 3.12, create .venv, and install requirements.txt. See docs/ANIMATION_WORKFLOW.md.'
