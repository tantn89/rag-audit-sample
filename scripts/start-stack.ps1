<#
.SYNOPSIS
  Start the audit stack: llm-api-guard on the host, then AnythingLLM in Docker pointed at it.

.DESCRIPTION
  Reuses a guard already started by start-guard.ps1 for this project (token in .run\guard-token);
  otherwise starts one. Waits until the app answers /api/ping. Stop everything with
  scripts\stop-stack.ps1.

  -Embedder openai adds compose.embedder-openai.yaml (text-embedding-3-small through the guard).
  After switching embedders in either direction, run `python -m rag_audit reembed`.
#>
[CmdletBinding()]
param(
  [int] $TimeoutSeconds = 180,
  [ValidateSet('native', 'openai')] [string] $Embedder = 'native'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$tokenFile = Join-Path $repo '.run\guard-token'
$composeFiles = @('-f', (Join-Path $repo 'compose.yaml'))
if ($Embedder -eq 'openai') { $composeFiles += @('-f', (Join-Path $repo 'compose.embedder-openai.yaml')) }

docker info --format '{{.ServerVersion}}' *> $null
if ($LASTEXITCODE -ne 0) { throw 'Docker engine is not running. Start Docker Desktop first.' }

if (-not (Test-Path $tokenFile)) { & (Join-Path $PSScriptRoot 'start-guard.ps1') }

$env:GUARD_TOKEN = (Get-Content -LiteralPath $tokenFile -Raw).Trim()
try {
  # up -d recreates the container when the embedder override is added or removed
  docker compose --project-directory $repo @composeFiles up -d
  if ($LASTEXITCODE -ne 0) { throw "docker compose up failed (exit $LASTEXITCODE)." }
} finally {
  Remove-Item Env:GUARD_TOKEN -ErrorAction SilentlyContinue
}

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ($true) {
  try {
    $ping = Invoke-RestMethod -Uri 'http://127.0.0.1:3001/api/ping' -TimeoutSec 5
    if ($ping.online) { break }
  } catch { }
  if ((Get-Date) -gt $deadline) { throw "AnythingLLM did not answer /api/ping within $TimeoutSeconds s." }
  Start-Sleep -Seconds 2
}
Write-Host "AnythingLLM is up at http://127.0.0.1:3001 (embedder: $Embedder)"

# First run: create a developer API key for scripts\anythingllm-client.py. This internal endpoint
# accepts unauthenticated calls only while the instance has no password (fresh local setup),
# and the app listens on loopback only.
$keyFile = Join-Path $repo '.run\anythingllm-api-key'
if (-not (Test-Path $keyFile)) {
  $res = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:3001/api/system/generate-api-key' `
    -ContentType 'application/json' -Body '{}'
  if (-not $res.apiKey.secret) { throw 'Could not create an AnythingLLM API key; create one in Settings > Developer API.' }
  [IO.File]::WriteAllText($keyFile, $res.apiKey.secret)
  Write-Host 'Created AnythingLLM API key (saved to .run\anythingllm-api-key)'
}
