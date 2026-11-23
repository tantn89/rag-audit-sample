<#
.SYNOPSIS
  Start llm-api-guard in the background for this project and save its per-launch local token.

.DESCRIPTION
  The guard listens on 127.0.0.1:<listen_port>, injects the real ShopAIKey key from the vault and
  forwards only to the fixed upstream. This script:
    1. refuses to start a second guard when the port is already taken (stop the owner instead);
    2. generates a random local token that the RAG app uses as its "OpenAI API key";
    3. writes the token to .run/guard-token and the PID to .run/guard.pid (both git-ignored).
  The real upstream key is never seen by this script or the app.
#>
[CmdletBinding()]
param(
  [string] $GuardDir = "$HOME\Documents\Claude\Projects\llm-api-guard",
  [int] $TimeoutSeconds = 60
)
$ErrorActionPreference = 'Stop'
$runDir = Join-Path (Split-Path $PSScriptRoot -Parent) '.run'
New-Item -ItemType Directory -Force $runDir | Out-Null

$cfg = Get-Content -LiteralPath (Join-Path $GuardDir 'guard-config.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$port = [int]$cfg.listen_port
$busy = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
  throw "Port $port is held by PID $($busy[0].OwningProcess). Stop it first (scripts\stop-guard.ps1 or Stop-Process)."
}

$bytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
$token = 'guard-' + [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')

$env:GUARD_LOCAL_TOKEN = $token
$env:PYTHONIOENCODING = 'utf-8'
try {
  $proxy = Start-Process -FilePath python -ArgumentList '-m', 'llm_api_guard', 'serve' `
    -WorkingDirectory $GuardDir -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $runDir 'guard.out.log') `
    -RedirectStandardError (Join-Path $runDir 'guard.err.log')
} finally {
  Remove-Item Env:GUARD_LOCAL_TOKEN -ErrorAction SilentlyContinue
}

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while (-not (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) {
  if ($proxy.HasExited) { throw "llm-api-guard exited early (exit $($proxy.ExitCode)); see .run\guard.err.log" }
  if ((Get-Date) -gt $deadline) {
    Stop-Process -Id $proxy.Id -ErrorAction SilentlyContinue
    throw "llm-api-guard did not open port $port within $TimeoutSeconds s."
  }
  Start-Sleep -Milliseconds 300
}

Set-Content -LiteralPath (Join-Path $runDir 'guard-token') -Value $token -NoNewline -Encoding ascii
Set-Content -LiteralPath (Join-Path $runDir 'guard.pid') -Value $proxy.Id -NoNewline -Encoding ascii
Write-Host "llm-api-guard running: PID $($proxy.Id), http://127.0.0.1:$port (token saved to .run\guard-token)"
