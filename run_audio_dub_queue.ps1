[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ })]
    [string]$InputPath,

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
    [switch]$RerunTranscribe,
    [switch]$RerunDubbing,
    [switch]$KeepOriginalAudio
)

$ErrorActionPreference = 'Stop'
$generatorRoot = $PSScriptRoot
Set-Location -LiteralPath $generatorRoot
function Test-AudioTranslationContract([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    try {
        $contract = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        $shots = @($contract.scenes | ForEach-Object { $_.shots })
        return ($contract.audio_translation_only -eq $true -and
            -not [string]::IsNullOrWhiteSpace([string]$contract.language.bcp47) -and $shots.Count -gt 0 -and
            @($shots | Where-Object { [string]::IsNullOrWhiteSpace([string]$_.source_narration) }).Count -eq 0)
    } catch { return $false }
}
$inputFullPath = (Resolve-Path -LiteralPath $InputPath).Path
$outputFullPath = [System.IO.Path]::GetFullPath($OutputRoot)
$reportDir = Join-Path $outputFullPath 'reports'
$transcribeOutput = Join-Path $outputFullPath 'transcribe'
$dubOutputRoot = Join-Path $outputFullPath 'dubbed'
$desiredAudioMode = if ($KeepOriginalAudio.IsPresent) { 'dual_audio' } else { 'replacement_audio' }
$null = New-Item -ItemType Directory -Path $reportDir, $transcribeOutput, $dubOutputRoot -Force

foreach ($required in @($FfprobePath, $PiperPath, $PiperMaleModel, $PiperMaleConfig,
                         $PiperFemaleModel, $PiperFemaleConfig, $BaseConfig,
                         (Join-Path $TranscribeRoot 'run_translation_queue.ps1'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file is missing: $required"
    }
}

$extensions = @('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.webm', '.flv', '.m4v', '.mpeg', '.mpg')
if (Test-Path -LiteralPath $inputFullPath -PathType Container) {
    $videos = @(Get-ChildItem -LiteralPath $inputFullPath -File -Recurse |
        Where-Object { $extensions -contains $_.Extension.ToLowerInvariant() } |
        Sort-Object FullName)
} else {
    $videos = @(Get-Item -LiteralPath $inputFullPath)
    if ($extensions -notcontains $videos[0].Extension.ToLowerInvariant()) {
        throw "Unsupported video file extension: $($videos[0].Extension)"
    }
}
if ($videos.Count -eq 0) { throw "No supported video files were found in $inputFullPath" }

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
        & $queueRunner -InputManifest $manifest -OutputDir $transcribeOutput `
            -GradeLevel $GradeLevel -AcademicLevel $AcademicLevel -TargetLanguages $TargetLanguage
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
$dubConfig = Join-Path $reportDir 'config.hindi.piper.queue.yaml'
[System.IO.File]::WriteAllText($dubConfig, $baseText, [System.Text.UTF8Encoding]::new($false))

$runVideo = Join-Path $generatorRoot 'run_video.ps1'
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
            if ($previous.source_video -eq $video.FullName -and $previousMode -eq $desiredAudioMode -and $previous.output_video -and
                (Test-Path -LiteralPath $previous.output_video -PathType Leaf)) {
                $reviewPath = Join-Path $transcribeOutput ($video.BaseName + "_$TargetLanguage`_translation_review.json")
                $reviewCount = 0
                if (Test-Path -LiteralPath $reviewPath -PathType Leaf) {
                    $review = Get-Content -LiteralPath $reviewPath -Raw | ConvertFrom-Json
                    $reviewCount = @($review.review_items).Count
                }
                $qaPath = Join-Path $itemOutput 'translation_qa_report.json'
                if (Test-Path -LiteralPath $qaPath -PathType Leaf) {
                    $qa = Get-Content -LiteralPath $qaPath -Raw | ConvertFrom-Json
                    $reviewCount += [int]$qa.issue_count
                }
                if ($previous.source_voice_profile.gender -eq 'unknown' -or [double]$previous.source_voice_profile.confidence -lt 0.55) {
                    $reviewCount++
                }
                $finalStatus = if ($reviewCount -gt 0) { 'completed_needs_teacher_review' } else { 'completed_existing' }
                Set-QueueResult ([pscustomobject]@{ source_path = $video.FullName; status = $finalStatus; output_video = $previous.output_video; voice_gender = $previous.source_voice_profile.gender; voice_confidence = $previous.source_voice_profile.confidence; translation_review_flags = $reviewCount; detail = '' })
                Write-Host "$itemLabel SKIPPED: already completed in $previousMode mode | status=$finalStatus | output=$($previous.output_video)"
                continue
            }
        } catch { }
    }
    try {
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
            $review = Get-Content -LiteralPath $reviewPath -Raw | ConvertFrom-Json
            $reviewCount = @($review.review_items).Count
        }
        $qaPath = Join-Path $itemOutput 'translation_qa_report.json'
        if (Test-Path -LiteralPath $qaPath -PathType Leaf) {
            $qa = Get-Content -LiteralPath $qaPath -Raw | ConvertFrom-Json
            $reviewCount += [int]$qa.issue_count
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
