[CmdletBinding()]
param(
    [ValidateScript({ Test-Path -LiteralPath $_ })]
    [string]$InputPath,

    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$InputManifest,

    [string]$OutputRoot = ".\outputs\hindi_audio_queue",
    [string]$TargetLanguage = 'hi',
    [string]$GradeLevel = '10th standard',
    [string]$AcademicLevel = 'secondary',
    [string]$TranscribeRoot = 'D:\Python\claude version\transcribe',
    [string]$BaseConfig = '.\config.realistic.12gb.yaml',
    [string]$FfprobePath = 'C:\Users\User\Documents\Codex\Tools\Transcribe\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffprobe.exe',
    [string]$PiperPath = 'C:\piper\piper.exe',
    [string]$PiperMaleModel = 'D:\AI\models\piper-hi\hi\hi_IN\pratham\medium\hi_IN-pratham-medium.onnx',
    [string]$PiperMaleConfig = 'D:\AI\models\piper-hi\hi\hi_IN\pratham\medium\hi_IN-pratham-medium.onnx.json',
    [string]$PiperFemaleModel = 'D:\AI\models\piper-hi\hi\hi_IN\priyamvada\medium\hi_IN-priyamvada-medium.onnx',
    [string]$PiperFemaleConfig = 'D:\AI\models\piper-hi\hi\hi_IN\priyamvada\medium\hi_IN-priyamvada-medium.onnx.json',
    # v2 strict rule: chemical formulas/equations/labels are spoken by these ENGLISH voices, never by the Hindi voice.
    [string]$PiperEnglishFemaleModel = 'D:\AI\models\piper-en\en_US-lessac-medium.onnx',
    [string]$PiperEnglishFemaleConfig = 'D:\AI\models\piper-en\en_US-lessac-medium.onnx.json',
    [string]$PiperEnglishMaleModel = 'D:\AI\models\piper-en\en_US-hfc_male-medium.onnx',
    [string]$PiperEnglishMaleConfig = 'D:\AI\models\piper-en\en_US-hfc_male-medium.onnx.json',
    # Reuse already-translated contracts (and ASR transcripts) from an earlier output root instead of re-running Transcribe.
    [string]$SeedContractsFrom = '',
    [switch]$RerunTranscribe,
    [switch]$RerunDubbing,
    [switch]$KeepOriginalAudio,
    [switch]$RequireOllamaAudit
)

$ErrorActionPreference = 'Stop'
# Enable the production ASR quality passes automatically for every queue launch.
# These are process-scoped so the launcher does not alter Windows policy or other apps.
$env:WHISPER_PREPROCESS = '1'
$env:WHISPER_ALTERNATE = '1'
$env:WHISPERX_EVIDENCE = '1'
$env:FFMPEG_PATH = 'ffmpeg'
$script:RequireAsrVersion = [string]::IsNullOrWhiteSpace($SeedContractsFrom)
$generatorRoot = $PSScriptRoot
Set-Location -LiteralPath $generatorRoot
# Ollama (Qwen) is REQUIRED: the semantic audit/repair, the re-translation of drifted lines, the on-screen-text second opinion and the
# vision cross-check all need it. Checked before anything runs; if it is down this tries to start it, and if it still does not answer
# (or a needed model is missing) the queue STOPS instead of producing unaudited dubs. Set ALLOW_NO_OLLAMA=1 to run without it anyway.
$ollamaUrl = if ($env:OLLAMA_URL) { $env:OLLAMA_URL } else { 'http://localhost:11434' }
$requiredModels = @('qwen3:14b', 'qwen3:8b', 'qwen3-vl:8b-instruct')
function Get-OllamaModels {
    try {
        $reply = Invoke-WebRequest -Uri "$ollamaUrl/api/tags" -UseBasicParsing -TimeoutSec 4
        return @((($reply.Content | ConvertFrom-Json).models) | ForEach-Object { $_.name })
    } catch { return $null }
}
$installed = Get-OllamaModels
if ($null -eq $installed) {
    Write-Host "OLLAMA: not answering at $ollamaUrl - trying to start it..."
    $exe = (Get-Command ollama -ErrorAction SilentlyContinue).Source
    if (-not $exe) { $exe = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe' }
    if (Test-Path -LiteralPath $exe) { Start-Process -FilePath $exe -ArgumentList 'serve' -WindowStyle Hidden }
    for ($i = 0; $i -lt 30 -and $null -eq $installed; $i++) { Start-Sleep -Seconds 2; $installed = Get-OllamaModels }
}
$missing = @()
if ($null -ne $installed) { $missing = @($requiredModels | Where-Object { $installed -notcontains $_ }) }
if ($null -ne $installed -and $missing.Count -eq 0) {
    $env:OLLAMA_AVAILABLE = '1'
    Write-Host "OLLAMA: running at $ollamaUrl, models ready ($($requiredModels -join ', ')) - Qwen audits, repairs and vision checks WILL be used."
} elseif ($env:ALLOW_NO_OLLAMA -eq '1') {
    $env:OLLAMA_AVAILABLE = '0'
    Write-Warning "OLLAMA unavailable or incomplete (missing: $($missing -join ', ')) - continuing WITHOUT Qwen because ALLOW_NO_OLLAMA=1. QA reports will say so."
} else {
    $reason = if ($null -eq $installed) { "Ollama is not running and could not be started" } else { "Ollama is missing models: $($missing -join ', ')  (run: ollama pull <name>)" }
    Write-Host ""
    Write-Host "STOPPED: $reason." -ForegroundColor Red
    Write-Host "Qwen is needed for QA. Start Ollama (or pull the models) and run this bat again. To run without it: set ALLOW_NO_OLLAMA=1" -ForegroundColor Red
    exit 4
}
# Vocal separation (demucs) cleans the music/effects under the narrator before ASR. It lives in the Python the Transcribe stage uses.
$asrPython = if ($env:DEMUCS_PYTHON) { $env:DEMUCS_PYTHON } else { 'D:\AI\ComfyUI_windows_portable\python_embeded\python.exe' }
if ($env:WHISPER_STEM_SEPARATION -ne '0' -and (Test-Path -LiteralPath $asrPython)) {
    & $asrPython -c "import demucs.separate" 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "DEMUCS: installed - vocals are separated from the music before speech recognition." }
    else { Write-Warning "DEMUCS: not installed in $asrPython - ASR will hear the background music. Install: `"$asrPython`" -m pip install --no-deps demucs dora-search openunmix lameenc soundfile retrying submitit treetable sphn" }
}
if ([string]::IsNullOrWhiteSpace($InputPath) -eq [string]::IsNullOrWhiteSpace($InputManifest)) {
    throw 'Specify exactly one of -InputPath or -InputManifest.'
}
function Test-AudioTranslationContract([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    try {
        $contract = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        $shots = @($contract.scenes | ForEach-Object { $_.shots })
        # v2.5 (closing-words rescue, re-translation audit decides, Devanagari letter count; v2.4 split-word and decimal repair in ASR; v2.3 re-translation of drifted lines, dropped-sentence check): a contract made by an older ASR/translation recipe (before the tail rescue) is rebuilt (reused seeded contracts are exempt).
        if ($script:RequireAsrVersion -and [string]$contract.asr_pipeline -ne 'v2.5') { return $false }
        return ($contract.audio_translation_only -eq $true -and
            -not [string]::IsNullOrWhiteSpace([string]$contract.language.bcp47) -and $shots.Count -gt 0 -and
            @($shots | Where-Object { [string]::IsNullOrWhiteSpace([string]$_.source_narration) }).Count -eq 0)
    } catch { return $false }
}
function Get-ActionableTranslationReviewCount([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return 0 }
    $review = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    return @($review.review_items | Where-Object {
        $_.where -ne 'title' -and $_.where -ne 'auto_repair' -and $_.resolution -ne 'verified_repair'
    }).Count
}
$inputFullPath = if ($InputPath) { (Resolve-Path -LiteralPath $InputPath).Path } else { '' }
$outputFullPath = [System.IO.Path]::GetFullPath($OutputRoot)
$reportDir = Join-Path $outputFullPath 'reports'
$transcribeOutput = Join-Path $outputFullPath 'transcribe'
$dubOutputRoot = Join-Path $outputFullPath 'dubbed'
$desiredAudioMode = if ($KeepOriginalAudio.IsPresent) { 'dual_audio' } else { 'replacement_audio' }
$null = New-Item -ItemType Directory -Path $reportDir, $transcribeOutput, $dubOutputRoot -Force

foreach ($required in @($FfprobePath, $PiperPath, $PiperMaleModel, $PiperMaleConfig,
                         $PiperFemaleModel, $PiperFemaleConfig, $BaseConfig,
                         $PiperEnglishFemaleModel, $PiperEnglishFemaleConfig,
                         $PiperEnglishMaleModel, $PiperEnglishMaleConfig,
                         (Join-Path $TranscribeRoot 'run_translation_queue.ps1'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file is missing: $required"
    }
}

$extensions = @('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.webm', '.flv', '.m4v', '.mpeg', '.mpg')
if ($InputManifest) {
    $manifestFullPath = (Resolve-Path -LiteralPath $InputManifest).Path
    $manifestEntries = @(Get-Content -LiteralPath $manifestFullPath | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    $videos = @($manifestEntries | ForEach-Object {
        if (-not (Test-Path -LiteralPath $_ -PathType Leaf)) { throw "Manifest video is missing: $_" }
        Get-Item -LiteralPath $_
    })
} elseif (Test-Path -LiteralPath $inputFullPath -PathType Container) {
    $videos = @(Get-ChildItem -LiteralPath $inputFullPath -File -Recurse |
        Where-Object { $extensions -contains $_.Extension.ToLowerInvariant() } |
        Sort-Object FullName)
} else {
    $videos = @(Get-Item -LiteralPath $inputFullPath)
    if ($extensions -notcontains $videos[0].Extension.ToLowerInvariant()) {
        throw "Unsupported video file extension: $($videos[0].Extension)"
    }
}
if ($videos.Count -eq 0) { throw 'No supported video files were found.' }
foreach ($video in $videos) {
    if ($extensions -notcontains $video.Extension.ToLowerInvariant()) {
        throw "Unsupported video file extension: $($video.Extension)"
    }
}

$inventory = [System.Collections.Generic.List[object]]::new()
$audioVideos = [System.Collections.Generic.List[object]]::new()
$noAudioVideos = [System.Collections.Generic.List[object]]::new()
$probeFailures = [System.Collections.Generic.List[object]]::new()
foreach ($video in $videos) {
    $probeText = (& $FfprobePath -v error -select_streams 'a:0' -show_entries stream=index -of 'csv=p=0' $video.FullName 2>&1 | Out-String).Trim()
    $probeCode = $LASTEXITCODE
    if ($probeCode -ne 0) {
        $row = [pscustomobject]@{ source_path = $video.FullName; status = 'probe_error'; detail = $probeText }
        $probeFailures.Add($row); $inventory.Add($row)
    } elseif ([string]::IsNullOrWhiteSpace($probeText)) {
        $row = [pscustomobject]@{ source_path = $video.FullName; status = 'no_audio_stream'; detail = '' }
        $noAudioVideos.Add($row); $inventory.Add($row)
    } else {
        $row = [pscustomobject]@{ source_path = $video.FullName; status = 'audio_found'; detail = $probeText }
        $audioVideos.Add($video); $inventory.Add($row)
    }
}

if ($noAudioVideos.Count) { $noAudioVideos | Export-Csv -LiteralPath (Join-Path $reportDir 'no_audio.csv') -NoTypeInformation -Encoding UTF8 }
else { [System.IO.File]::WriteAllText((Join-Path $reportDir 'no_audio.csv'), '"source_path","status","detail"' + "`n", [System.Text.UTF8Encoding]::new($false)) }
if ($probeFailures.Count) { $probeFailures | Export-Csv -LiteralPath (Join-Path $reportDir 'probe_failures.csv') -NoTypeInformation -Encoding UTF8 }
else { [System.IO.File]::WriteAllText((Join-Path $reportDir 'probe_failures.csv'), '"source_path","status","detail"' + "`n", [System.Text.UTF8Encoding]::new($false)) }
$inventory | Export-Csv -LiteralPath (Join-Path $reportDir 'inventory.csv') -NoTypeInformation -Encoding UTF8
Write-Host "Inventory scanned: $($videos.Count) total | $($audioVideos.Count) with audio | $($noAudioVideos.Count) no audio | $($probeFailures.Count) probe errors"
foreach ($video in $noAudioVideos) { Write-Host "NO_AUDIO (not a failure): $($video.source_path)" }
foreach ($video in $probeFailures) { Write-Host "PROBE_ERROR: $($video.source_path) - $($video.detail)" }

# Transcribe writes contracts into one flat folder, so duplicate stems would collide. Report and exclude those safely.
$duplicateStems = @($audioVideos | Group-Object { [System.IO.Path]::GetFileNameWithoutExtension($_.Name) } |
    Where-Object { $_.Count -gt 1 })
$duplicateRows = [System.Collections.Generic.List[object]]::new()
foreach ($group in $duplicateStems) {
    foreach ($video in $group.Group) {
        $duplicateRows.Add([pscustomobject]@{ source_path = $video.FullName; stem = $group.Name; status = 'duplicate_stem' })
    }
}
if ($duplicateRows.Count) { $duplicateRows | Export-Csv -LiteralPath (Join-Path $reportDir 'duplicate_names.csv') -NoTypeInformation -Encoding UTF8 }
else { [System.IO.File]::WriteAllText((Join-Path $reportDir 'duplicate_names.csv'), '"source_path","stem","status"' + "`n", [System.Text.UTF8Encoding]::new($false)) }
$duplicatePaths = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($row in $duplicateRows) { [void]$duplicatePaths.Add($row.source_path) }
$eligible = @($audioVideos | Where-Object { -not $duplicatePaths.Contains($_.FullName) })
if ($eligible.Count -eq 0) {
    Write-Host "Inventory complete: no unique audio-bearing videos to process. Reports: $reportDir"
    exit 0
}

$results = [System.Collections.Generic.List[object]]::new()
foreach ($row in $noAudioVideos) { $results.Add([pscustomobject]@{ source_path = $row.source_path; status = 'no_audio_stream'; detail = '' }) }
foreach ($row in $probeFailures) { $results.Add([pscustomobject]@{ source_path = $row.source_path; status = 'probe_error'; detail = $row.detail }) }
foreach ($row in $duplicateRows) { $results.Add([pscustomobject]@{ source_path = $row.source_path; status = 'duplicate_stem'; detail = $row.stem }) }
foreach ($video in $eligible) { $results.Add([pscustomobject]@{ source_path = $video.FullName; status = 'queued'; detail = '' }) }
function Set-QueueResult([object]$row) {
    $index = -1
    for ($i = 0; $i -lt $script:results.Count; $i++) {
        if ($script:results[$i].source_path -eq $row.source_path) { $index = $i; break }
    }
    if ($index -ge 0) { $script:results[$index] = $row } else { $script:results.Add($row) }
    $script:results | Export-Csv -LiteralPath (Join-Path $reportDir 'queue_results.csv') -NoTypeInformation -Encoding UTF8
}
$results | Export-Csv -LiteralPath (Join-Path $reportDir 'queue_results.csv') -NoTypeInformation -Encoding UTF8

if ($SeedContractsFrom) {
    $seedDir = (Resolve-Path -LiteralPath $SeedContractsFrom).Path
    foreach ($video in $eligible) {
        foreach ($suffix in @("_$TargetLanguage`_contract.json", "_$TargetLanguage`_translation_review.json",
                              '_transcript.json', '_transcript.txt', '_manifest.json', '_glossary_candidates.json')) {
            $from = Join-Path $seedDir ($video.BaseName + $suffix)
            $to = Join-Path $transcribeOutput ($video.BaseName + $suffix)
            if ((Test-Path -LiteralPath $from -PathType Leaf) -and -not (Test-Path -LiteralPath $to -PathType Leaf)) {
                Copy-Item -LiteralPath $from -Destination $to
                Write-Host "SEEDED: $($video.BaseName)$suffix"
            }
        }
    }
}

# A single JVM/Transcribe queue handles every missing contract, avoiding a full model reload for each video.
$missingContracts = @($eligible | Where-Object {
    $contract = Join-Path $transcribeOutput ($_.BaseName + "_$TargetLanguage`_contract.json")
    $RerunTranscribe.IsPresent -or -not (Test-AudioTranslationContract $contract)
})
$transcribedThisRun = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($video in $missingContracts) { [void]$transcribedThisRun.Add($video.FullName) }
if ($missingContracts.Count -gt 0) {
    Write-Host "TRANSCRIBE_START: $($missingContracts.Count) video(s) need a translation contract."
    $manifest = Join-Path $reportDir 'transcribe_inputs.txt'
    $manifestText = (($missingContracts | ForEach-Object { $_.FullName }) -join "`n") + "`n"
    [System.IO.File]::WriteAllText($manifest, $manifestText, [System.Text.UTF8Encoding]::new($false))
    foreach ($video in $missingContracts) {
        Set-QueueResult ([pscustomobject]@{ source_path = $video.FullName; status = 'transcribing'; detail = '' })
    }
    $queueRunner = Join-Path $TranscribeRoot 'run_translation_queue.ps1'
    try {
        if ($RequireOllamaAudit.IsPresent) {
            & $queueRunner -InputManifest $manifest -OutputDir $transcribeOutput `
                -GradeLevel $GradeLevel -AcademicLevel $AcademicLevel -TargetLanguages $TargetLanguage -RequireOllamaAudit
        } else {
            & $queueRunner -InputManifest $manifest -OutputDir $transcribeOutput `
                -GradeLevel $GradeLevel -AcademicLevel $AcademicLevel -TargetLanguages $TargetLanguage
        }
        Write-Host "TRANSCRIBE_FINISHED: translation queue returned successfully."
    } catch {
        Write-Warning "Transcribe queue returned an error; continuing with any contracts it completed. $($_.Exception.Message)"
    }
} else {
    Write-Host "TRANSCRIBE_SKIP: all eligible videos already have valid translation contracts."
}

$contractIndex = 0
foreach ($video in $eligible) {
    $contractIndex++
    $contract = Join-Path $transcribeOutput ($video.BaseName + "_$TargetLanguage`_contract.json")
    if (Test-AudioTranslationContract $contract) {
        Set-QueueResult ([pscustomobject]@{ source_path = $video.FullName; status = 'ready_to_dub'; detail = '' })
        Write-Host "[$contractIndex/$($eligible.Count)] CONTRACT_READY: $($video.FullName)"
    } else {
        Set-QueueResult ([pscustomobject]@{ source_path = $video.FullName; status = 'transcribe_failed'; detail = "Missing contract: $contract" })
        Write-Host "[$contractIndex/$($eligible.Count)] TRANSCRIBE_FAILED: $($video.FullName) | expected $contract"
    }
}

# Generate a local Hindi Piper config with both voice models. Unknown/ambiguous pitch falls back to the female base model.
$baseText = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $BaseConfig).Path)
$baseText = $baseText -replace '(?m)^tts_provider:.*$', 'tts_provider: "piper"'
$baseText = $baseText -replace '(?m)^piper_path:.*$', "piper_path: `"$($PiperPath.Replace('\','/'))`""
$baseText = $baseText -replace '(?m)^piper_model:.*$', "piper_model: `"$($PiperFemaleModel.Replace('\','/'))`""
$baseText = $baseText -replace '(?m)^piper_config:.*$', "piper_config: `"$($PiperFemaleConfig.Replace('\','/'))`"`npiper_gender_models:`n  male: `"$($PiperMaleModel.Replace('\','/'))`"`n  female: `"$($PiperFemaleModel.Replace('\','/'))`"`npiper_gender_configs:`n  male: `"$($PiperMaleConfig.Replace('\','/'))`"`n  female: `"$($PiperFemaleConfig.Replace('\','/'))`""
$baseText = [regex]::Replace($baseText, '(?m)^piper_english\w*:[^\r\n]*\r?\n(?:[ \t]+[^\r\n]*\r?\n)*', '')
function ConvertTo-YamlPath([string]$Path) { return $Path.Replace('\', '/') }
$baseText = $baseText.TrimEnd() + "`n`n# Dubbing v2 deep dive: formulas, equations and labels are spoken by English voices.`n" +
    "piper_english_model: `"$(ConvertTo-YamlPath $PiperEnglishFemaleModel)`"`n" +
    "piper_english_config: `"$(ConvertTo-YamlPath $PiperEnglishFemaleConfig)`"`n" +
    "piper_english_gender_models:`n  male: `"$(ConvertTo-YamlPath $PiperEnglishMaleModel)`"`n  female: `"$(ConvertTo-YamlPath $PiperEnglishFemaleModel)`"`n" +
    "piper_english_gender_configs:`n  male: `"$(ConvertTo-YamlPath $PiperEnglishMaleConfig)`"`n  female: `"$(ConvertTo-YamlPath $PiperEnglishFemaleConfig)`"`n"
$dubConfig = Join-Path $reportDir 'config.hindi.piper.queue.yaml'
[System.IO.File]::WriteAllText($dubConfig, $baseText, [System.Text.UTF8Encoding]::new($false))
# On-screen text translation is OFF unless the visual launcher (RUN_VIDEO_TEXT_TRANSLATION.bat) sets VIDEO_TEXT_TRANSLATION=1. If the
# variable is not set at all, the YAML key production.video_text_translation decides, and the queue is told so here, so skip logic,
# review flags and the Python choice agree with what the generator will do. An explicit 0 or 1 in the environment always wins.
if ([string]::IsNullOrWhiteSpace($env:VIDEO_TEXT_TRANSLATION) -and $baseText -match '(?m)^\s+video_text_translation:\s*true\b') { $env:VIDEO_TEXT_TRANSLATION = '1' }
if ([string]::IsNullOrWhiteSpace($env:VIDEO_TEXT_TRANSLATION_AUTO) -and $baseText -match '(?m)^\s+video_text_translation_auto:\s*true\b') { $env:VIDEO_TEXT_TRANSLATION_AUTO = '1' }

$runVideo = Join-Path $generatorRoot 'run_video.ps1'
$sourceOcr = Join-Path $generatorRoot 'scripts/audit_source_video.py'
$projectPython = Join-Path $generatorRoot '.venv/Scripts/python.exe'
$portablePython = 'D:/AI/ComfyUI_windows_portable/python_embeded/python.exe'
$ocrPython = if (Test-Path -LiteralPath $projectPython -PathType Leaf) {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $projectPython -c 'import sys' 2>$null
    $ErrorActionPreference = $previousErrorActionPreference
    if ($LASTEXITCODE -eq 0) { $projectPython } else { $portablePython }
} else { $portablePython }
if (-not (Test-Path -LiteralPath $ocrPython -PathType Leaf)) { throw "OCR Python is missing: $ocrPython" }
$videoIndex = 0
foreach ($video in $eligible) {
    $videoIndex++
    $itemLabel = "[$videoIndex/$($eligible.Count)]"
    $contract = Join-Path $transcribeOutput ($video.BaseName + "_$TargetLanguage`_contract.json")
    if (-not (Test-AudioTranslationContract $contract)) {
        continue
    }
    $itemOutput = Join-Path $dubOutputRoot $video.BaseName
    $manifestPath = Join-Path $itemOutput 'audio_replacement_manifest.json'
    if (-not $RerunTranscribe -and -not $RerunDubbing -and -not $transcribedThisRun.Contains($video.FullName) -and
        (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        try {
            $previous = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
            $previousMode = if ($previous.audio_track_mode) { [string]$previous.audio_track_mode } else { 'replacement_audio' }
            $currentContractHash = (Get-FileHash -LiteralPath $contract -Algorithm SHA256).Hash.ToLowerInvariant()
            $reviewPath = Join-Path $transcribeOutput ($video.BaseName + "_$TargetLanguage`_translation_review.json")
            $currentReviewHash = if (Test-Path -LiteralPath $reviewPath -PathType Leaf) {
                (Get-FileHash -LiteralPath $reviewPath -Algorithm SHA256).Hash.ToLowerInvariant()
            } else { '' }
            $visualRequested = $env:VIDEO_TEXT_TRANSLATION -eq '1'
            $visualCurrent = $previous.video_text_translation.enabled -eq $true -and
                $previous.video_text_translation.pipeline_version -ge 9 -and
                @('completed', 'completed_needs_review') -contains $previous.video_text_translation.status
            if ((-not $visualRequested -or $visualCurrent) -and
                $previous.source_video -eq $video.FullName -and $previousMode -eq $desiredAudioMode -and
                $previous.qa_version -ge 23 -and $previous.pipeline_version -eq 'v2-deep-dive' -and @('english_stem','hindi_math','english_voice_strict') -contains $previous.formula_rule -and $previous.source_size -eq $video.Length -and
                $previous.source_mtime_utc_ticks -eq $video.LastWriteTimeUtc.Ticks -and
                $previous.visual_source_signature.review_sha256 -eq $currentReviewHash -and
                $previous.visual_source_qa_report_path -and
                (Test-Path -LiteralPath $previous.visual_source_qa_report_path -PathType Leaf) -and
                $previous.contract_sha256 -eq $currentContractHash -and $previous.output_video -and
                (Test-Path -LiteralPath $previous.output_video -PathType Leaf)) {
                $reviewCount = 0
                if (Test-Path -LiteralPath $reviewPath -PathType Leaf) {
                    $reviewCount = Get-ActionableTranslationReviewCount $reviewPath
                }
                $qaPath = Join-Path $itemOutput 'translation_qa_report.json'
                if (Test-Path -LiteralPath $qaPath -PathType Leaf) {
                    $qa = Get-Content -LiteralPath $qaPath -Raw | ConvertFrom-Json
                    if ($qa.translation_review_integrated) { $reviewCount = [int]$qa.issue_count }
                    else { $reviewCount += [int]$qa.issue_count }
                }
                if ($previous.source_voice_profile.gender -eq 'unknown' -or [double]$previous.source_voice_profile.confidence -lt 0.55) {
                    $reviewCount++
                }
                if ($visualRequested -and $previous.video_text_translation.status -ne 'completed') { $reviewCount++ }
                $finalStatus = if ($reviewCount -gt 0) { 'completed_needs_teacher_review' } else { 'completed_existing' }
                Set-QueueResult ([pscustomobject]@{ source_path = $video.FullName; status = $finalStatus; output_video = $previous.output_video; voice_gender = $previous.source_voice_profile.gender; voice_confidence = $previous.source_voice_profile.confidence; translation_review_flags = $reviewCount; detail = '' })
                Write-Host "$itemLabel SKIPPED: already completed in $previousMode mode | status=$finalStatus | output=$($previous.output_video)"
                continue
            }
        } catch { }
    }
    try {
        Write-Host "$itemLabel CONTRACT_PRECHECK_START: inspecting and repairing the contract before dubbing | $contract"
        Set-QueueResult ([pscustomobject]@{ source_path = $video.FullName; status = 'contract_precheck'; detail = '' })
        & $ocrPython (Join-Path $generatorRoot 'scripts/precheck_contract.py') $contract
        if ($LASTEXITCODE -ne 0) { throw "Contract precheck rejected the contract (exit code $LASTEXITCODE)" }
        Write-Host "$itemLabel SOURCE_OCR_START: checking printed formulas and arithmetic before dubbing | $($video.FullName)"
        Set-QueueResult ([pscustomobject]@{ source_path = $video.FullName; status = 'source_ocr'; detail = '' })
        & $ocrPython $sourceOcr --video $video.FullName --contract $contract --config $dubConfig
        if ($LASTEXITCODE -ne 0) { throw "Source OCR preflight exited with code $LASTEXITCODE" }
        $ocrEvidenceRoot = Join-Path $transcribeOutput ($video.BaseName + "_$TargetLanguage`_source_qa")
        $ocrReportPath = Join-Path $ocrEvidenceRoot 'visual_source_qa/report.json'
        if (-not (Test-Path -LiteralPath $ocrReportPath -PathType Leaf)) { throw "Source OCR report is missing: $ocrReportPath" }
        $ocrReport = Get-Content -LiteralPath $ocrReportPath -Raw | ConvertFrom-Json
        $ocrState = if (@($ocrReport.issues).Count -gt 0) { 'SOURCE_OCR_REVIEW' } else { 'SOURCE_OCR_READY' }
        Write-Host "$itemLabel ${ocrState}: $(@($ocrReport.issues).Count) visual finding(s) | report=$ocrReportPath"
        Write-Host "$itemLabel DUBBING_START: $($video.FullName) | mode=$desiredAudioMode | contract=$contract | output=$itemOutput"
        Set-QueueResult ([pscustomobject]@{ source_path = $video.FullName; status = 'dubbing'; detail = '' })
        $runArgs = @('--config', $dubConfig, '--input', $contract, '--output', $itemOutput,
            '--replace-audio', '--source-video', $video.FullName)
        if ($KeepOriginalAudio.IsPresent) { $runArgs += '--keep-original-audio' }
        & $runVideo @runArgs
        $runCode = $LASTEXITCODE
        if ($runCode -ne 0) { throw "Generator exited with code $runCode" }
        if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw 'Generator did not write audio_replacement_manifest.json' }
        $dubManifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        $reviewPath = Join-Path $transcribeOutput ($video.BaseName + "_$TargetLanguage`_translation_review.json")
        $reviewCount = 0
        if (Test-Path -LiteralPath $reviewPath -PathType Leaf) {
            $reviewCount = Get-ActionableTranslationReviewCount $reviewPath
        }
        $qaPath = Join-Path $itemOutput 'translation_qa_report.json'
        if (Test-Path -LiteralPath $qaPath -PathType Leaf) {
            $qa = Get-Content -LiteralPath $qaPath -Raw | ConvertFrom-Json
            if ($qa.translation_review_integrated) { $reviewCount = [int]$qa.issue_count }
            else { $reviewCount += [int]$qa.issue_count }
        }
        if ($env:VIDEO_TEXT_TRANSLATION -eq '1' -and $dubManifest.video_text_translation.status -ne 'completed') {
            $reviewCount++
            Write-Warning "$itemLabel Visual text translation is incomplete; inspect video_text_translation_report.json"
        }
        if ($dubManifest.source_voice_profile.gender -eq 'unknown' -or [double]$dubManifest.source_voice_profile.confidence -lt 0.55) {
            $reviewCount++
        }
        $finalStatus = if ($reviewCount -gt 0) { 'completed_needs_teacher_review' } else { 'completed' }
        Set-QueueResult ([pscustomobject]@{ source_path = $video.FullName; status = $finalStatus; output_video = $dubManifest.output_video; voice_gender = $dubManifest.source_voice_profile.gender; voice_confidence = $dubManifest.source_voice_profile.confidence; translation_review_flags = $reviewCount; detail = '' })
        Write-Host "$itemLabel DUBBING_FINISHED: status=$finalStatus | output=$($dubManifest.output_video) | QA/teacher flags=$reviewCount"
    } catch {
        $qaReportPath = Join-Path $itemOutput 'translation_qa_report.json'
        $qaBlocked = $false
        if (Test-Path -LiteralPath $qaReportPath -PathType Leaf) {
            $qaBlocked = (Get-Content -LiteralPath $qaReportPath -Raw | ConvertFrom-Json).status -eq 'blocked'
        }
        if ($qaBlocked) {
            $qaReport = Get-Content -LiteralPath $qaReportPath -Raw | ConvertFrom-Json
            $detail = "Critical translation QA blocked dubbing ($($qaReport.issue_count) issue(s)); see $qaReportPath"
            Set-QueueResult ([pscustomobject]@{ source_path = $video.FullName; status = 'qa_blocked'; output_video = ''; voice_gender = ''; voice_confidence = ''; translation_review_flags = ''; detail = $detail })
            Write-Warning "$itemLabel $detail"
        } else {
            Set-QueueResult ([pscustomobject]@{ source_path = $video.FullName; status = 'dub_failed'; output_video = ''; voice_gender = ''; voice_confidence = ''; translation_review_flags = ''; detail = $_.Exception.Message })
            Write-Warning "$itemLabel DUBBING_FAILED: $($video.FullName) | $($_.Exception.Message)"
        }
    }
}
Write-Host "Queue finished. Inventory: $(Join-Path $reportDir 'inventory.csv')"
Write-Host "No audio: $(Join-Path $reportDir 'no_audio.csv')"
Write-Host "Results: $(Join-Path $reportDir 'queue_results.csv')"
$failedRows = @($results | Where-Object { $_.status -in @('probe_error', 'duplicate_stem', 'transcribe_failed', 'dub_failed', 'qa_blocked') })
if ($failedRows.Count -gt 0) {
    Write-Error "Queue completed with $($failedRows.Count) failed or skipped item(s). Review $(Join-Path $reportDir 'queue_results.csv')"
    exit 1
}
