param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
)

$binDir = Join-Path $RepoRoot "node_modules\.bin"
$pkgDir = Join-Path $RepoRoot "node_modules\gitnexus"
$pkgBinDir = Join-Path $pkgDir "bin"

New-Item -ItemType Directory -Force -Path $binDir, $pkgBinDir | Out-Null

$cmdWrapper = @'
@echo off
setlocal
node "%~dp0..\..\tools\gitnexus-shim\gitnexus.mjs" %*
'@

$psWrapper = @'
& node "$PSScriptRoot\..\..\tools\gitnexus-shim\gitnexus.mjs" @args
'@

$packageJson = @'
{
  "name": "gitnexus",
  "version": "0.0.0-local-shim",
  "private": true,
  "bin": {
    "gitnexus": "bin/gitnexus.js"
  }
}
'@

$jsWrapper = @'
#!/usr/bin/env node
import "../../../tools/gitnexus-shim/gitnexus.mjs";
'@

Set-Content -LiteralPath (Join-Path $binDir "gitnexus.cmd") -Value $cmdWrapper -Encoding ASCII
Set-Content -LiteralPath (Join-Path $binDir "gitnexus.ps1") -Value $psWrapper -Encoding ASCII
Set-Content -LiteralPath (Join-Path $pkgDir "package.json") -Value $packageJson -Encoding ASCII
Set-Content -LiteralPath (Join-Path $pkgBinDir "gitnexus.js") -Value $jsWrapper -Encoding ASCII

Write-Host "Installed repo-local gitnexus shim under node_modules."
