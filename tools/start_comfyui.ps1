# Starts ComfyUI for the generator with the memory flag the long renders need.
# Without --disable-smart-memory ComfyUI keeps ~18 GB of RAM resident and the render can be killed for low memory.
# Usage:  powershell -File tools\start_comfyui.ps1            (starts it if port 8188 is not answering)
param([string]$Root = "D:\AI\ComfyUI_windows_portable", [int]$Port = 8188)

function Test-Comfy { try { (Invoke-WebRequest "http://127.0.0.1:$Port/system_stats" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 } catch { $false } }

if (Test-Comfy) { Write-Host "ComfyUI already answers on port $Port. If it was started without --disable-smart-memory, stop it and run this script again."; exit 0 }

$p = Start-Process -FilePath "$Root\python_embeded\python.exe" `
    -ArgumentList "-s", "ComfyUI\main.py", "--windows-standalone-build", "--disable-smart-memory", "--port", "$Port" `
    -WorkingDirectory $Root -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput "$Root\comfy_stdout.log" -RedirectStandardError "$Root\comfy_stderr.log"
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep 3
    if (Test-Comfy) { Write-Host "ComfyUI is up (launcher pid $($p.Id)) with --disable-smart-memory"; exit 0 }
}
Write-Host "ComfyUI did not answer within 2 minutes; see $Root\comfy_stderr.log"
exit 1
