# Re-renders the 14 lessons that were last rendered BEFORE the final-frame safety check existed (see the October 2026 scorecard).
# They are plant, maths, physics and chemistry lessons, so the risk is low, but each one should pass the same check as the rest.
#
#   powershell -File tools\rerender_old_lessons.ps1                (all 14, one after another, into outputs\rerender_old)
#   powershell -File tools\rerender_old_lessons.ps1 -Only "Parts of a Flower","Morphology of Flower"
#
# Expect 40-60 minutes per lesson (about 10-14 hours for all). Progress: outputs\rerender_old.log ; each finished lesson has
# outputs\rerender_old\<lesson>\final_qa.json (passed + critical = 0 is the result to look for).
param([string[]]$Only = @(), [string]$Contracts = "C:\Users\User\output", [string]$Out = "outputs\rerender_old")

$Lessons = @(
    "5913833-Types of Pollination",
    "A Simple Question Based on Minima of a Function",
    "All About Electroplating",
    "Conceptual Problem on Resistance and Resistivity",
    "Modes of Reproduction in Plants",
    "Morphology of Flower",
    "Parts of a Flower",
    "Pollination Meaning and Types",
    "Pollination and Fertilisation",
    "Regeneration in Hydra",
    "Seed Germination",
    "Seed Germination and Structure of the Seedling",
    "Sexual Reproduction in Flowering Plants",
    "Vegetative Propagation"
)
if ($Only.Count -gt 0) { $Lessons = $Lessons | Where-Object { $Only -contains $_ } }

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
& powershell -NoProfile -File "$root\tools\start_comfyui.ps1"
if ($LASTEXITCODE -ne 0) { Write-Host "ComfyUI is not running; stopping."; exit 1 }

$inputs = @()
foreach ($name in $Lessons) {
    $file = Join-Path $Contracts "$($name)_contract.json"
    if (Test-Path -LiteralPath $file) { $inputs += $file } else { Write-Host "MISSING contract: $file" }
}
Write-Host "Rendering $($inputs.Count) lesson(s) into $Out"
$py = "D:\AI\ComfyUI_windows_portable\python_embeded\python.exe"
& $py -u scripts/local_entry.py --config config.realistic.12gb.yaml --input @inputs --output $Out *> "outputs\rerender_old.log"
Write-Host "Done. See outputs\rerender_old.log and $Out\queue_summary.json"
