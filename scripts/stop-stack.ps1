<#
.SYNOPSIS
  Stop the AnythingLLM container and the llm-api-guard process started for this project.
  App data in .\storage is kept so the next run does not re-embed the corpus.
#>
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
# compose interpolates GUARD_TOKEN even on "down"; a placeholder is enough to stop the container.
$env:GUARD_TOKEN = 'stopping'
try {
  docker compose --project-directory $repo -f (Join-Path $repo 'compose.yaml') down
} finally {
  Remove-Item Env:GUARD_TOKEN -ErrorAction SilentlyContinue
}
& (Join-Path $PSScriptRoot 'stop-guard.ps1')
