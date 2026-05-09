$ErrorActionPreference = 'Stop'

$ProjectRoot = $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$EnvFile = Join-Path $ProjectRoot 'email.env'
$EnvExample = Join-Path $ProjectRoot 'email.env.example'

if (-not (Test-Path $PythonExe)) {
    throw 'Virtual environment is missing. Run .\setup-local.ps1 first.'
}

if (-not (Test-Path $EnvFile)) {
    Copy-Item -LiteralPath $EnvExample -Destination $EnvFile
}

Write-Host 'Starting Flask app on http://127.0.0.1:5000'
Push-Location $ProjectRoot
try {
    & $PythonExe 'main.py'
}
finally {
    Pop-Location
}
