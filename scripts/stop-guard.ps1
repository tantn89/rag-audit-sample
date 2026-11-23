<#
.SYNOPSIS
  Stop the llm-api-guard process started by start-guard.ps1 and delete its local token.
#>
$ErrorActionPreference = 'Stop'
$runDir = Join-Path (Split-Path $PSScriptRoot -Parent) '.run'
$pidFile = Join-Path $runDir 'guard.pid'
if (Test-Path $pidFile) {
  $guardPid = [int](Get-Content -LiteralPath $pidFile -Raw)
  $proc = Get-Process -Id $guardPid -ErrorAction SilentlyContinue
  # Only stop the PID we recorded, and only if it is still a python process.
  if ($proc -and $proc.ProcessName -like 'python*') {
    Stop-Process -Id $guardPid
    Write-Host "Stopped llm-api-guard PID $guardPid"
  } else {
    Write-Host "Guard PID $guardPid is not running."
  }
  Remove-Item -LiteralPath $pidFile
} else {
  Write-Host 'No guard.pid found; nothing to stop.'
}
Remove-Item -LiteralPath (Join-Path $runDir 'guard-token') -ErrorAction SilentlyContinue
