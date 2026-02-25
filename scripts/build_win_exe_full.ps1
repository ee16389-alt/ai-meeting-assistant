param(
  [switch]$SkipPrepareAssets,
  [switch]$SkipBackend,
  [switch]$SkipNpmInstall
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Require-Command([string]$Name) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "$Name not found"
  }
}

Require-Command "python"
Require-Command "node"
Require-Command "npm"

Write-Host "[Info] Root: $root"
Write-Host "[Info] Python: $(python --version 2>&1)"
Write-Host "[Info] Node: $(node --version 2>$null)"
Write-Host "[Info] npm: $(npm --version)"

if (-not $SkipPrepareAssets) {
  Write-Host "[1/4] Preparing desktop model assets (GGUF + Sherpa-ONNX)..."
  python scripts\prepare_desktop_assets.py --include-sherpa
}

if (-not $SkipBackend) {
  Write-Host "[2/4] Building backend binary (PyInstaller onedir, external Sherpa pack)..."
  $env:PYINSTALLER_MODE = "onedir"
  $env:EMBED_SHERPA_ONNX = "0"
  python scripts\build_backend.py
}

Set-Location (Join-Path $root "desktop")

if (-not $SkipNpmInstall) {
  Write-Host "[3/4] Installing desktop dependencies..."
  npm ci
} else {
  Write-Host "[3/4] Skipped npm ci"
}

Write-Host "[4/4] Building Windows installer (bundled GGUF + Sherpa-ONNX)..."
$env:BUNDLE_GGUF = "1"
$env:BUNDLE_SHERPA_MODELS = "1"
npm run build:win

Write-Host "Done. Check desktop\dist\ for the generated Windows installer (.exe)."
