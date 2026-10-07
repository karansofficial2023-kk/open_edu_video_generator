# Stops a long render ONLY when the machine is truly out of memory (commit charge near the limit, i.e. the pagefile is nearly exhausted),
# not when free RAM merely dips while a model loads.
#   powershell -ExecutionPolicy Bypass -File tools\memory_watchdog.ps1 -ProcessId <render pid> [-CommitPercent 92]
param([Parameter(Mandatory)][int]$ProcessId, [int]$CommitPercent = 92, [int]$Strikes = 3, [string]$Log = "")
if (-not $Log) { $Log = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\outputs\memory_watchdog.log" }
$Log = [IO.Path]::GetFullPath($Log)
New-Item -ItemType Directory -Force (Split-Path -Parent $Log) | Out-Null
$hits = 0       # NOT $strikes: PowerShell variable names are case-insensitive and would overwrite the $Strikes parameter
while (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
    $os = Get-CimInstance Win32_OperatingSystem
    $used = $os.TotalVirtualMemorySize - $os.FreeVirtualMemory
    $percent = [math]::Round(100 * $used / $os.TotalVirtualMemorySize, 1)
    $freeGb = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
    "$(Get-Date -Format HH:mm:ss) commit=$percent% freeRAM=${freeGb}GB" | Add-Content $Log
    if ($percent -ge $CommitPercent) { $hits++ } else { $hits = 0 }
    if ($hits -ge $Strikes) {
        "$(Get-Date -Format HH:mm:ss) STOPPING render ${ProcessId} - commit $percent% for $Strikes samples" | Add-Content $Log
        cmd /c "taskkill /T /F /PID $ProcessId >nul 2>&1"
        break
    }
    Start-Sleep -Seconds 5
}
"$(Get-Date -Format HH:mm:ss) watchdog finished" | Add-Content $Log
