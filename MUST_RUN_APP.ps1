$ErrorActionPreference = 'Stop'

$ProjectRoot = $PSScriptRoot
$RunScript = Join-Path $ProjectRoot 'run-local.ps1'

& powershell -NoProfile -ExecutionPolicy Bypass -File $RunScript
