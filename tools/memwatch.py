"""Logs memory every few seconds, to diagnose out-of-memory kills of long renders.

    python tools/memwatch.py outputs/memwatch.log [seconds]

Each line: free RAM, Windows commit charge (the number that limits total memory including the pagefile), the image server's
(ComfyUI) working set, and the three largest processes.
"""
import subprocess
import sys
import time

PS = (
    "$o=Get-CimInstance Win32_OperatingSystem;"
    "$c=($o.TotalVirtualMemorySize-$o.FreeVirtualMemory)/1MB;"
    "$comfy=Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -like '*ComfyUI*main.py*' };"
    "$r=0; if($comfy){$r=(Get-Process -Id $comfy.ProcessId).WorkingSet64/1GB};"
    "$t=(Get-Process|Sort-Object WorkingSet64 -Descending|Select-Object -First 3|ForEach-Object{'{0}={1:N1}' -f $_.Name,($_.WorkingSet64/1GB)}) -join ' ';"
    "'free={0:N1}GB commit={1:N1}GB/{2:N1}GB comfy={3:N1}GB top: {4}' -f ($o.FreePhysicalMemory/1MB),$c,($o.TotalVirtualMemorySize/1MB),$r,$t"
)

if __name__ == "__main__":
    log = open(sys.argv[1], "a", buffering=1, encoding="utf-8")
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 5
    while True:
        line = subprocess.run(["powershell", "-NoProfile", "-Command", PS], capture_output=True, text=True).stdout.strip()
        log.write(f"{time.strftime('%H:%M:%S')}  {line}\n")
        time.sleep(interval)
