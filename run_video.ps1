param([Parameter(ValueFromRemainingArguments=$true)][string[]]$VideoArgs)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
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
