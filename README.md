# ChatGPT Micro-Cap Experiment
Welcome to the repo behind my 6-month live trading experiment where ChatGPT manages a real-money micro-cap portfolio.

## Overview on getting started: [Here](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Start%20Your%20Own/README.md)
   
## Repository Structure

- **`trading_script.py`** - Main trading engine with portfolio management and stop-loss automation
- **`Scripts and CSV Files/`** - My personal portfolio (updates every trading day)
- **`Start Your Own/`** - Template files and guide for starting your own experiment  
- **`Weekly Deep Research (MD|PDF)/`** - Research summaries and performance reports
- **`Experiment Details/`** - Documentation, methodology, prompts, and Q&A

# The Concept
Every day, I kept seeing the same ad about having some A.I. pick undervalued stocks. It was obvious it was trying to get me to subscribe to some garbage, so I just rolled my eyes.  
Then I started wondering, "How well would that actually work?"

So, starting with just $100, I wanted to answer a simple but powerful question:

**Can powerful large language models like ChatGPT actually generate alpha (or at least make smart trading decisions) using real-time data?**

## Each trading day:

- I provide it trading data on the stocks in its portfolio.  
- Strict stop-loss rules apply.  
- Every week I allow it to use deep research to reevaluate its account.  
- I track and publish performance data weekly on my blog: [Here](https://nathanbsmith729.substack.com)

## Research & Documentation

- [Research Index](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Experiment%20Details/Deep%20Research%20Index.md)  
- [Disclaimer](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Experiment%20Details/Disclaimer.md)  
- [Q&A](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Experiment%20Details/Q%26A.md)  
- [Prompts](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Experiment%20Details/Prompts.md)  
- [Starting Your Own](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Start%20Your%20Own/README.md)  
- [Research Summaries (MD)](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/tree/main/Weekly%20Deep%20Research%20(MD))  
- [Full Deep Research Reports (PDF)](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/tree/main/Weekly%20Deep%20Research%20(PDF))
- [Chats](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Experiment%20Details/Chats.md)
# Current Performance

<!-- To update performance chart: 
     1. Replace the image file with updated results
     2. Update the dates and description below
     3. Update the "Last Updated" date -->

**Current Portfolio Results**

![Latest Performance Results](Results.png)

**Current Status:** Portfolio is outperforming the S&P 500 benchmark

*Performance data is updated after each trading day. See the CSV files in `Scripts and CSV Files/` for detailed daily tracking.*

# Features of This Repo
- Live trading scripts — used to evaluate prices and update holdings daily  
- LLM-powered decision engine — ChatGPT picks the trades  
- Performance tracking — CSVs with daily PnL, total equity, and trade history  
- Visualization tools — Matplotlib graphs comparing ChatGPT vs. Index  
- Logs & trade data — auto-saved logs for transparency  

## Want to Contribute?

Contributions are very welcome! This project is community-oriented, and your help is invaluable.  

- **Issues:** If you notice a bug or have an idea for improvement, please.  
- **Pull Requests:** Feel free to submit a PR — I usually review within a few days.  
- **Collaboration:** High-value contributors may be invited as maintainers/admins to help shape the project’s future.  

Whether it’s fixing a typo, adding features, or discussing new ideas, all contributions are appreciated!


# Why This Matters
AI is being hyped across every industry, but can it really manage money without guidance?

This project is an attempt to find out — with transparency, data, and a real budget.

# Tech Stack & Features

## Core Technologies
- **Python** - Core scripting and automation
- **pandas + yFinance** - Market data fetching and analysis
- **Matplotlib** - Performance visualization and charting
- **ChatGPT-4** - AI-powered trading decision engine
- **Azure OpenAI** - LLM integration for automated trading decisions
- **Schwab API** - Live brokerage integration for real trading

## Key Features
- **Robust Data Sources** - Yahoo Finance primary, Stooq fallback for reliability
- **Automated Stop-Loss** - Automatic position management with configurable stop-losses
- **Interactive Trading** - Market-on-Open (MOO) and limit order support
- **Backtesting Support** - ASOF_DATE override for historical analysis
- **Performance Analytics** - CAPM analysis, Sharpe/Sortino ratios, drawdown metrics
- **Trade Logging** - Complete transparency with detailed execution logs
- **Live Schwab Trading** - Execute real trades directly through Schwab API

## System Requirements
- Python  3.11+
- Internet connection for market data
- ~10MB storage for CSV data files
- Schwab Developer Account (for live trading)
- Azure OpenAI or OpenAI API key (for automated recommendations)

# Quick Start - Automated Trading

## 1. Setup Environment

Create a `.env` file in the project root with your credentials:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-endpoint.cognitiveservices.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=your-deployment-name

# Schwab API Configuration (for live trading)
SCHWAB_APP_KEY=your-schwab-app-key
SCHWAB_APP_SECRET=your-schwab-app-secret
SCHWAB_CALLBACK_URL=https://127.0.0.1
```

## 2. Run the Trading Bot

**PowerShell (recommended):**
```powershell
# Interactive mode (default) - chat with your portfolio
.\start_trading.ps1

# Automated mode - runs trading cycle automatically
.\start_trading.ps1 -Auto

# Dry run - see recommendations without trading
.\start_trading.ps1 -DryRun

# Re-authenticate with Schwab
.\start_trading.ps1 -Auth

# Simple mode (no Schwab, just LLM recommendations)
.\start_trading.ps1 -Simple
```

**Or double-click:** `start_trading.bat`

## 3. Interactive Mode Features

The default interactive mode provides a conversational trading experience:

### Quick Commands
| Command | Description |
|---------|-------------|
| `status` or `s` | Show current portfolio status |
| `recommend` or `r` | Get LLM trading recommendations |
| `execute` or `e` | Execute a recommended trade |
| `orders` or `o` | Show open orders |
| `buy <TICKER> <SHARES> [STOP]` | Manual buy order with optional stop-loss |
| `sell <TICKER> <SHARES>` | Manual sell order |
| `sync` | Sync CSV with Schwab positions |
| `help` or `h` | Show help menu |
| `quit` or `q` | Exit |

### Natural Language Chat
You can also ask questions in plain English! The AI understands your portfolio context:

- *"What's my best performing stock?"*
- *"Should I sell CTXR?"*
- *"How much cash do I have?"*
- *"Give me a summary of my portfolio"*
- *"What's the risk on my current positions?"*

## 4. What It Does

1. ✅ Loads environment variables from `.env`
2. ✅ Connects to your Schwab account (auto-refreshes tokens)
3. ✅ Fetches real account data (positions, cash balance)
4. ✅ Schwab is the source of truth, compares with CSV
5. ✅ Calls Azure OpenAI for trading recommendations
6. ✅ Executes trades with automatic stop-loss orders
7. ✅ Auto-syncs CSV after each trade
8. ✅ Supports natural language queries about your portfolio
9. ✅ Logs all activity for review

## Scripts Overview

| Script | Description |
|--------|-------------|
| `start_trading.ps1` | Main startup script - loads env and runs trading |
| `start_trading.bat` | Double-click launcher for Windows |
| `interactive_trading.py` | Interactive trading with natural language chat |
| `live_trading.py` | Automated live trading with Schwab integration |
| `simple_automation.py` | LLM recommendations without Schwab |
| `trading_script.py` | Core trading engine and portfolio management |
| `schwab_auth.py` | Schwab OAuth authentication |
| `schwab_client.py` | Schwab API client for orders/positions |
| `check_orders.py` | Utility to view Schwab orders |

# Follow Along
The experiment runs from June 2025 to December 2025.  
Every trading day I will update the portfolio CSV file.  
If you feel inspired to do something similar, feel free to use this as a blueprint.

Updates are posted weekly on my blog, more coming soon!

Blog: [A.I Controls Stock Account](https://nathanbsmith729.substack.com)

Have feature requests or any advice?  

Please reach out here: **nathanbsmith.business@gmail.com**

