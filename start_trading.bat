@echo off
REM Stock Trader App - Quick Start (Windows Batch)
REM Double-click this file to start the trading system

cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0start_trading.ps1" %*
pause
