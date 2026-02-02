# Stock Trader App - Startup Script
# This script loads environment variables and starts the trading system

param(
    [switch]$Auto,        # Run automated trading (old behavior)
    [switch]$DryRun,      # Dry run mode for automated trading
    [switch]$Simple,      # Use simple_automation.py instead
    [switch]$Auth         # Run Schwab authentication only
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "=== Stock Trader App ===" -ForegroundColor Cyan
Write-Host ""

# Load .env file
$envFile = Join-Path $ScriptDir ".env"
if (Test-Path $envFile) {
    Write-Host "Loading environment variables from .env..." -ForegroundColor Green
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
            Write-Host "  Set: $name" -ForegroundColor DarkGray
        }
    }
} else {
    Write-Host "Warning: .env file not found at $envFile" -ForegroundColor Yellow
}

# Set Python path
$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "Virtual environment not found. Creating one..." -ForegroundColor Yellow
    python -m venv .venv
    & "$ScriptDir\.venv\Scripts\pip.exe" install numpy pandas yfinance matplotlib openai python-dotenv
}

# Quick check for openai package only (other deps should already be installed)
Write-Host ""
Write-Host "Checking dependencies..." -ForegroundColor Green
$hasOpenAI = & $PythonExe -c "import openai; print('ok')" 2>$null
if ($hasOpenAI -ne "ok") {
    Write-Host "Installing openai package..." -ForegroundColor Yellow
    & "$ScriptDir\.venv\Scripts\pip.exe" install -q openai python-dotenv
}

Write-Host ""

# Determine which mode to run
if ($Auth) {
    Write-Host "Running Schwab authentication..." -ForegroundColor Cyan
    & $PythonExe "$ScriptDir\schwab_auth.py"
}
elseif ($Simple) {
    Write-Host "Starting simple automation (no Schwab)..." -ForegroundColor Cyan
    & $PythonExe "$ScriptDir\simple_automation.py" --model rk-stockpicker @args
}
elseif ($Auto) {
    if ($DryRun) {
        Write-Host "Starting automated trading (DRY RUN)..." -ForegroundColor Yellow
        & $PythonExe "$ScriptDir\live_trading.py" --dry-run --model rk-stockpicker
    } else {
        Write-Host "Starting automated LIVE trading..." -ForegroundColor Red
        Write-Host "WARNING: Real orders will be placed!" -ForegroundColor Red
        & $PythonExe "$ScriptDir\live_trading.py" --live --model rk-stockpicker
    }
}
else {
    # Default: Interactive live trading mode
    Write-Host "Starting Interactive Trading Mode..." -ForegroundColor Cyan
    Write-Host "Schwab account is source of truth. Type 'help' for commands." -ForegroundColor DarkGray
    Write-Host ""
    & $PythonExe "$ScriptDir\interactive_trading.py" --model rk-stockpicker
}
