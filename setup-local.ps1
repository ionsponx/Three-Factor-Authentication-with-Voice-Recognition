$ErrorActionPreference = 'Stop'

$ProjectRoot = $PSScriptRoot
$ToolsDir = Join-Path $ProjectRoot '.tools'
$UvDir = Join-Path $ToolsDir 'uv'
$UvExe = Join-Path $UvDir 'uv.exe'
$UvZip = Join-Path $ToolsDir 'uv.zip'

New-Item -ItemType Directory -Force -Path $UvDir | Out-Null
Push-Location $ProjectRoot

try {

if (-not (Test-Path $UvExe)) {
    Write-Host 'Downloading uv...'
    Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile $UvZip
    Expand-Archive -LiteralPath $UvZip -DestinationPath $UvDir -Force
}

$env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot '.python'

Write-Host 'Installing Python 3.11...'
& $UvExe python install 3.11

Write-Host 'Creating virtual environment...'
& $UvExe venv --python 3.11 '.venv'

Write-Host 'Installing Python dependencies...'
& $UvExe pip install -r 'requirements.txt'

$EnvFile = Join-Path $ProjectRoot 'email.env'
$EnvExample = Join-Path $ProjectRoot 'email.env.example'
if (-not (Test-Path $EnvFile)) {
    Copy-Item -LiteralPath $EnvExample -Destination $EnvFile
    Write-Host 'Created email.env from email.env.example'
}

Write-Host ''
Write-Host 'Setup complete. Run the app with:'
Write-Host '  powershell -NoProfile -ExecutionPolicy Bypass -File .\run-local.ps1'
}
finally {
    Pop-Location
}
