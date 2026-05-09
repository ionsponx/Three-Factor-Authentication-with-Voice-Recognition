$ErrorActionPreference = 'Stop'

$ProjectRoot = $PSScriptRoot
$SetupScript = Join-Path $ProjectRoot 'setup-local.ps1'

Write-Host 'Installing everything required for this project...'
Write-Host ''

& powershell -NoProfile -ExecutionPolicy Bypass -File $SetupScript

Write-Host ''
Write-Host 'All required dependencies are installed.'
Write-Host 'Start the app with:'
Write-Host '  powershell -NoProfile -ExecutionPolicy Bypass -File .\MUST_RUN_APP.ps1'
