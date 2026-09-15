param(
    [string]$Python = "py -3.12"
)

Write-Host "Creating virtual environment..."
Invoke-Expression "$Python -m venv .venv"

Write-Host "Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host "Installing Python packages..."
python -m pip install --upgrade pip
pip install -r requirements.txt

if (!(Test-Path .\config.yaml)) {
    Copy-Item .\config.example.yaml .\config.yaml
    Write-Host "Created config.yaml. Edit tool paths before running."
}

Write-Host "Setup complete."

