"""
Azure AI Foundry Chatbot for Stock Trading

This script connects your Azure AI ChatGPT 5.2 agent to the trading system,
automating the workflow of getting recommendations and executing trades.

Now with Schwab integration for real trading!

Setup:
1. Create a .env file with:
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_KEY=your-api-key
   AZURE_OPENAI_DEPLOYMENT=your-deployment-name
   SCHWAB_APP_KEY=your-schwab-app-key
   SCHWAB_APP_SECRET=your-schwab-secret

2. Run: python schwab_auth.py  (first time only, to authenticate)
3. Run: python azure_chatbot.py

"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Any

# Load .env file
from dotenv import load_dotenv
load_dotenv()

import pandas as pd

# Import from trading_script
from trading_script import (
    set_data_dir,
    load_latest_portfolio_state,
    log_manual_buy,
    log_manual_sell,
    daily_results,
    download_price_data,
    trading_day_window,
    check_weekend,
    last_trading_date,
    load_benchmarks,
    DATA_DIR,
    PORTFOLIO_CSV,
)

# Import Schwab client (optional - may not be authenticated)
try:
    from schwab_client import SchwabClient
    from schwab_auth import SchwabAuth
    SCHWAB_AVAILABLE = True
except ImportError:
    SCHWAB_AVAILABLE = False

# Import News client (optional - needs API key)
try:
    from news_client import NewsClient
    NEWS_AVAILABLE = True
except ImportError:
    NEWS_AVAILABLE = False

class AzureTradingChatbot:
    """Interactive chatbot that connects Azure AI to the trading system."""

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        deployment: str | None = None,
        data_dir: str = ".",
        enable_schwab: bool = True,
        live_trading: bool = False,  # Safety: default to paper trading
    ):
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-52-stockpicker")
        
        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Azure OpenAI credentials required. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
            )

        # Initialize Azure OpenAI client
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError("openai package required. Run: pip install openai")

        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version="2024-12-01-preview",
        )

        # Set up data directory
        self.data_path = Path(data_dir)
        set_data_dir(self.data_path)

        # Schwab integration
        self.schwab_client: SchwabClient | None = None
        self.live_trading = live_trading
        self.schwab_account_hash: str | None = None
        
        if enable_schwab and SCHWAB_AVAILABLE:
            self._init_schwab()

        # News integration
        self.news_client: NewsClient | None = None
        if NEWS_AVAILABLE:
            self._init_news()

        # Conversation history for context
        self.conversation_history: list[dict[str, str]] = []
        
        # System prompt for the trading agent
        self.system_prompt = self._build_system_prompt()

        # Load portfolio
        self.portfolio_df: pd.DataFrame
        self.cash: float
        self._load_portfolio()
    
    def _init_schwab(self) -> None:
        """Initialize Schwab client if authenticated."""
        try:
            auth = SchwabAuth()
            if auth.is_authenticated():
                self.schwab_client = SchwabClient(auth)
                
                # Try to get account hash (may fail if API is temporarily down)
                try:
                    accounts = self.schwab_client.get_account_numbers()
                    if accounts:
                        self.schwab_account_hash = accounts[0].get("hashValue")
                        # Save for future use
                        self.schwab_client.save_account_info(
                            accounts[0].get("accountNumber"),
                            self.schwab_account_hash
                        )
                        print(f"✅ Schwab connected: Account #{accounts[0].get('accountNumber')}")
                    else:
                        # Try cached account
                        self.schwab_account_hash = self.schwab_client.get_account_hash()
                        print(f"✅ Schwab connected (using cached account info)")
                except Exception as e:
                    # Try cached account hash
                    try:
                        self.schwab_account_hash = self.schwab_client.get_account_hash()
                        print(f"✅ Schwab connected (using cached account info)")
                    except:
                        print(f"⚠️ Schwab authenticated but account unavailable: {e}")
            else:
                print("ℹ️ Schwab not authenticated. Run 'python schwab_auth.py' to enable real trading.")
        except Exception as e:
            print(f"ℹ️ Schwab integration unavailable: {e}")
    
    def _init_news(self) -> None:
        """Initialize news client if API key available."""
        try:
            self.news_client = NewsClient()
            if self.news_client.api_key:
                print("✅ News/Sentiment: Finnhub connected")
            else:
                self.news_client = None
        except Exception as e:
            print(f"ℹ️ News integration unavailable: {e}")
            self.news_client = None

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the trading agent."""
        return """You are a professional-grade portfolio strategist managing a real-money micro-cap portfolio.

CORE RULES:
- Budget discipline: No new capital beyond what's shown. Track cash precisely.
- Execution limits: Full shares only. No options, shorting, leverage, margin, or derivatives. Long-only.
- Universe: Primarily U.S. micro-caps under $300M market cap. Consider liquidity (avg volume, spread).
- Risk control: Always set stop-losses on new positions.
- You have complete control over position sizing, risk management, and order types.

⚠️ CRITICAL - STOCK RESTRICTIONS (AVOID THESE):
Many micro-cap stocks have trading restrictions that will cause order rejections. AVOID recommending:
1. Stocks priced under $1.00 (penny stocks) - require fully cleared funds, often broker-assisted only
2. Foreign ADRs from restricted countries (China ADRs are usually OK, but check liquidity)
3. OTC/Pink Sheet stocks - these often require broker-assisted trades
4. Stocks with extremely low market cap (<$10M) - often have special restrictions
5. Recently IPO'd stocks (within 30 days) - may have settlement restrictions
6. Stocks flagged as "hard to borrow" or with high short interest restrictions

PREFER STOCKS THAT ARE:
- Listed on major exchanges (NYSE, NASDAQ) - not OTC
- Priced between $2.00 and $20.00 for micro-caps (sweet spot for tradability)
- Have average daily volume > 100,000 shares
- Are U.S. domiciled companies (not foreign ADRs unless major exchange listed)
- Have been trading for at least 6 months

If a stock has any red flags, find an alternative. There are thousands of tradeable micro-caps.

AVAILABLE TOOLS:
1. get_stock_prices - Retrieves real-time stock prices. ALWAYS use this before recommending trades.
2. get_stock_news - Retrieves news, sentiment, financials, and analyst ratings for a stock.
   Use this to research stocks before making recommendations.

RESEARCH PROCESS:
1. When asked for recommendations, first use get_stock_news to research potential picks
2. Consider news sentiment, analyst ratings, and financials in your analysis
3. Then use get_stock_prices to verify current prices
4. Verify the stock doesn't have trading restrictions (price > $1, major exchange, good volume)
5. Only then make your recommendation with full context

RESPONSE FORMAT FOR TRADES:
When recommending trades, include a JSON block like this:

```json
{
    "trades": [
        {
            "action": "buy",
            "ticker": "SYMBOL",
            "shares": 10,
            "order_type": "limit",
            "limit_price": 5.50,
            "stop_loss": 4.40,
            "reason": "Brief rationale including news/sentiment context"
        }
    ]
}
```

For order_type, use "limit" (preferred) or "market".
Always explain your reasoning before the JSON block, citing relevant news or sentiment.

If you have no trades to recommend, just say so - no JSON needed."""

    def _load_portfolio(self) -> None:
        """Load the current portfolio state."""
        try:
            result, self.cash = load_latest_portfolio_state()
            # Handle both DataFrame and list returns
            if isinstance(result, pd.DataFrame):
                self.portfolio_df = result
            elif isinstance(result, list):
                self.portfolio_df = pd.DataFrame(result) if result else pd.DataFrame(
                    columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"]
                )
            else:
                self.portfolio_df = pd.DataFrame(
                    columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"]
                )
        except Exception as e:
            print(f"Note: Could not load existing portfolio ({e}). Starting fresh.")
            self.portfolio_df = pd.DataFrame(
                columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"]
            )
            self.cash = 0.0
        
        # In live mode, sync cash from Schwab (source of truth)
        if self.live_trading and self.schwab_client:
            self._sync_from_schwab()
    
    def _sync_from_schwab(self) -> None:
        """Sync portfolio state from Schwab account (source of truth for live trading)."""
        if not self.schwab_client:
            return
        
        try:
            balance = self.schwab_client.get_balance()
            if "error" not in balance:
                schwab_cash = balance.get("cash_available", 0)
                if abs(schwab_cash - self.cash) > 0.01:  # Only update if different
                    print(f"💰 Syncing cash from Schwab: ${self.cash:.2f} → ${schwab_cash:.2f}")
                    self.cash = schwab_cash
                    self._update_csv_cash(schwab_cash)
            
            # Optionally sync positions too
            positions = self.schwab_client.get_positions()
            if positions:
                print(f"📊 Schwab has {len(positions)} position(s)")
        except Exception as e:
            print(f"⚠️ Could not sync from Schwab: {e}")
    
    def _update_csv_cash(self, new_cash: float) -> None:
        """Update the cash balance in the CSV file."""
        try:
            csv_path = self.data_path / "chatgpt_portfolio_update.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                if not df.empty:
                    # Update the most recent TOTAL row's cash balance
                    total_mask = df["Ticker"] == "TOTAL"
                    if total_mask.any():
                        # Get index of last TOTAL row
                        last_total_idx = df[total_mask].index[-1]
                        df.loc[last_total_idx, "Cash Balance"] = new_cash
                        df.to_csv(csv_path, index=False)
                        print(f"📝 Updated CSV cash balance to ${new_cash:.2f}")
        except Exception as e:
            print(f"⚠️ Could not update CSV: {e}")

    def get_portfolio_summary(self) -> str:
        """Generate a summary of current portfolio for the AI."""
        today = last_trading_date().date().isoformat()
        
        if self.portfolio_df.empty:
            holdings_text = "No current holdings"
            total_value = 0.0
        else:
            holdings_text = self.portfolio_df.to_string(index=False)
            total_value = self.portfolio_df["cost_basis"].sum() if "cost_basis" in self.portfolio_df.columns else 0.0

        total_equity = self.cash + total_value
        benchmarks = load_benchmarks()

        # Get benchmark prices
        benchmark_data = []
        s, e = trading_day_window()
        for ticker in benchmarks[:4]:  # Limit to 4 benchmarks
            try:
                fetch = download_price_data(ticker, start=s, end=e, progress=False)
                if not fetch.df.empty:
                    close = float(fetch.df["Close"].iloc[-1])
                    benchmark_data.append(f"{ticker}: ${close:.2f}")
            except Exception:
                pass

        benchmark_text = ", ".join(benchmark_data) if benchmark_data else "Unavailable"

        return f"""
=== Portfolio Status as of {today} ===

[ Holdings ]
{holdings_text}

[ Account Summary ]
Cash Balance: ${self.cash:,.2f}
Portfolio Value: ${total_value:,.2f}
Total Equity: ${total_equity:,.2f}

[ Benchmark Prices ]
{benchmark_text}

[ Available Actions ]
- You can recommend BUY or SELL trades
- Include stop-loss for all new positions
- Use limit orders when possible
{self.get_schwab_account_summary()}"""

    def get_schwab_account_summary(self) -> str:
        """Get real Schwab account info if connected."""
        if not self.schwab_client:
            return ""
        
        try:
            balance = self.schwab_client.get_balance()
            positions = self.schwab_client.get_positions()
            
            summary = f"""
[ SCHWAB REAL ACCOUNT ]
Cash Available: ${balance.get('cash_available', 0):,.2f}
Account Value: ${balance.get('account_value', 0):,.2f}
Buying Power: ${balance.get('buying_power', 0):,.2f}
"""
            if positions:
                summary += f"\nReal Positions ({len(positions)}):\n"
                for pos in positions:
                    symbol = pos.get("instrument", {}).get("symbol", "N/A")
                    qty = pos.get("longQuantity", 0) or pos.get("shortQuantity", 0)
                    avg_price = pos.get("averagePrice", 0)
                    summary += f"  {symbol}: {qty} shares @ ${avg_price:.2f}\n"
            else:
                summary += "\nNo positions in Schwab account.\n"
            
            return summary
        except Exception as e:
            return f"\n[ Schwab account error: {e} ]\n"

    def get_stock_price(self, ticker: str) -> dict[str, Any]:
        """Get current price data for a stock."""
        import numpy as np
        ticker = ticker.upper().strip()
        s, e = trading_day_window()
        
        try:
            fetch = download_price_data(ticker, start=s, end=e, progress=False)
            if fetch.df.empty:
                return {"ticker": ticker, "error": "No data available"}
            
            data = fetch.df
            result = {
                "ticker": ticker,
                "price": round(float(data["Close"].iloc[-1]), 2),
                "open": round(float(data["Open"].iloc[-1]), 2) if "Open" in data else None,
                "high": round(float(data["High"].iloc[-1]), 2),
                "low": round(float(data["Low"].iloc[-1]), 2),
                "volume": int(data["Volume"].iloc[-1]) if "Volume" in data else None,
                "source": fetch.source,
            }
            
            # Calculate change if we have 2+ days
            if len(data) >= 2:
                prev_close = float(data["Close"].iloc[-2])
                change = result["price"] - prev_close
                change_pct = (change / prev_close) * 100
                result["change"] = round(change, 2)
                result["change_pct"] = round(change_pct, 2)
            
            return result
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}

    def get_multiple_prices(self, tickers: list[str]) -> str:
        """Get prices for multiple tickers and format as text."""
        results = []
        for ticker in tickers:
            data = self.get_stock_price(ticker)
            if "error" in data:
                results.append(f"{data['ticker']}: {data['error']}")
            else:
                change_str = ""
                if "change_pct" in data:
                    sign = "+" if data["change"] >= 0 else ""
                    change_str = f" ({sign}{data['change_pct']:.2f}%)"
                results.append(
                    f"{data['ticker']}: ${data['price']:.2f}{change_str} "
                    f"| H: ${data['high']:.2f} L: ${data['low']:.2f} "
                    f"| Vol: {data.get('volume', 'N/A'):,}"
                )
        return "\n".join(results)
    
    def get_stock_news(self, ticker: str) -> str:
        """Get news, sentiment, and financials for a stock."""
        if not self.news_client:
            return f"News service not available. Set FINNHUB_API_KEY in .env"
        
        return self.news_client.get_stock_summary(ticker)

    # Define tools/functions for the AI to call
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_prices",
                "description": "Get current real-time stock prices, including open, high, low, close, volume, and percent change. Use this to look up any stock ticker before making trade recommendations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of stock ticker symbols to look up (e.g., ['AAPL', 'MSFT', 'TSLA'])"
                        }
                    },
                    "required": ["tickers"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_stock_news",
                "description": "Get comprehensive research data for a stock including: recent news headlines with sentiment, social media sentiment (Reddit/Twitter), key financial metrics (P/E, market cap, 52-week range), and analyst recommendations. Use this to research stocks before making recommendations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Stock ticker symbol to research (e.g., 'AAPL')"
                        }
                    },
                    "required": ["ticker"]
                }
            }
        }
    ]

    def chat(self, user_message: str) -> str:
        """Send a message to the AI and get a response."""
        # Add portfolio context to first message or when requested
        if not self.conversation_history or "portfolio" in user_message.lower() or "status" in user_message.lower():
            context = self.get_portfolio_summary()
            full_message = f"{context}\n\nUser: {user_message}"
        else:
            full_message = user_message

        self.conversation_history.append({"role": "user", "content": full_message})

        try:
            # First API call - may include tool calls
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    *self.conversation_history[-10:],
                ],
                tools=self.TOOLS,
                max_completion_tokens=2000,
            )

            assistant_message = response.choices[0].message

            # Check if the AI wants to call a function
            if assistant_message.tool_calls:
                # Process tool calls
                tool_results = []
                for tool_call in assistant_message.tool_calls:
                    if tool_call.function.name == "get_stock_prices":
                        # Parse the function arguments
                        args = json.loads(tool_call.function.arguments)
                        tickers = args.get("tickers", [])
                        
                        print(f"\n📈 AI is looking up prices for: {', '.join(tickers)}...")
                        
                        # Get the prices
                        price_data = []
                        for ticker in tickers:
                            data = self.get_stock_price(ticker)
                            price_data.append(data)
                        
                        # Format results for the AI
                        result_text = json.dumps(price_data, indent=2)
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "content": result_text,
                        })
                    
                    elif tool_call.function.name == "get_stock_news":
                        # Parse the function arguments
                        args = json.loads(tool_call.function.arguments)
                        ticker = args.get("ticker", "")
                        
                        print(f"\n📰 AI is researching news/sentiment for: {ticker}...")
                        
                        # Get news and sentiment
                        news_data = self.get_stock_news(ticker)
                        
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "content": news_data,
                        })
                
                # Add the assistant's tool call message and tool results to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })
                
                for result in tool_results:
                    self.conversation_history.append(result)

                # Second API call with the tool results
                response2 = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        *self.conversation_history[-12:],  # Include tool results
                    ],
                    max_completion_tokens=2000,
                )
                
                final_message = response2.choices[0].message.content or ""
                self.conversation_history.append({"role": "assistant", "content": final_message})
                return final_message
            
            else:
                # No tool calls, just return the response
                content = assistant_message.content or ""
                self.conversation_history.append({"role": "assistant", "content": content})
                return content

        except Exception as e:
            return f"Error communicating with Azure AI: {e}"

    def parse_trades_from_response(self, response: str) -> list[dict[str, Any]]:
        """Extract trade recommendations from AI response."""
        trades = []
        seen = set()  # Track duplicates by (ticker, action, shares)
        
        # Try to find JSON block in response
        json_patterns = [
            r'```json\s*(.*?)\s*```',  # ```json ... ``` (most specific)
            r'```\s*(\{.*?"trades".*?\})\s*```',  # ``` {...trades...} ```
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    if isinstance(match, str):
                        # Clean up the match
                        json_str = match.strip()
                        if not json_str.startswith("{"):
                            # Try to find the JSON object within
                            start = json_str.find("{")
                            if start >= 0:
                                json_str = json_str[start:]
                        data = json.loads(json_str)
                        if "trades" in data:
                            for trade in data["trades"]:
                                # Create a unique key for this trade
                                key = (
                                    trade.get("ticker", "").upper(),
                                    trade.get("action", "").lower(),
                                    trade.get("shares", 0)
                                )
                                if key not in seen:
                                    seen.add(key)
                                    trades.append(trade)
                            # Found trades, stop looking at more patterns
                            if trades:
                                return trades
                except json.JSONDecodeError:
                    continue

        return trades

    def execute_trade(self, trade: dict[str, Any], use_schwab: bool = False) -> str:
        """Execute a single trade and return result message."""
        action = trade.get("action", "").lower()
        ticker = trade.get("ticker", "").upper()
        shares = int(float(trade.get("shares", 0)))  # Schwab requires int
        order_type = trade.get("order_type", "limit").lower()
        limit_price = float(trade.get("limit_price", trade.get("price", 0)))
        stop_loss = float(trade.get("stop_loss", 0))
        reason = trade.get("reason", "AI recommendation")

        if not ticker or shares <= 0:
            return f"❌ Invalid trade: missing ticker or shares"

        # REAL SCHWAB TRADING
        if use_schwab and self.schwab_client and self.live_trading:
            return self._execute_schwab_trade(
                action=action,
                ticker=ticker,
                shares=shares,
                order_type=order_type,
                limit_price=limit_price,
                stop_loss=stop_loss,
                reason=reason,
            )

        # PAPER TRADING (local CSV tracking)
        if action == "buy":
            if limit_price <= 0:
                return f"❌ BUY {ticker}: No price specified"
            
            cost = shares * limit_price
            if cost > self.cash:
                return f"❌ BUY {ticker}: Insufficient cash (need ${cost:.2f}, have ${self.cash:.2f})"

            # Execute the buy using trading_script function
            try:
                self.cash, self.portfolio_df = log_manual_buy(
                    buy_price=limit_price,
                    shares=shares,
                    ticker=ticker,
                    stoploss=stop_loss,
                    cash=self.cash,
                    chatgpt_portfolio=self.portfolio_df,
                    interactive=False,  # Don't prompt for confirmation
                )
                return f"✅ [PAPER] BUY {int(shares)} {ticker} @ ${limit_price:.2f} (stop: ${stop_loss:.2f}) - {reason}"
            except Exception as e:
                return f"❌ BUY {ticker} failed: {e}"

        elif action == "sell":
            if limit_price <= 0:
                return f"❌ SELL {ticker}: No price specified"

            # Execute the sell
            try:
                self.cash, self.portfolio_df = log_manual_sell(
                    sell_price=limit_price,
                    shares_sold=shares,
                    ticker=ticker,
                    cash=self.cash,
                    chatgpt_portfolio=self.portfolio_df,
                    reason=reason,
                    interactive=False,
                )
                return f"✅ [PAPER] SELL {int(shares)} {ticker} @ ${limit_price:.2f} - {reason}"
            except Exception as e:
                return f"❌ SELL {ticker} failed: {e}"

        else:
            return f"❓ Unknown action: {action}"
    
    def _execute_schwab_trade(
        self,
        action: str,
        ticker: str,
        shares: int,
        order_type: str,
        limit_price: float,
        stop_loss: float,
        reason: str,
    ) -> str:
        """Execute a real trade via Schwab API and verify execution."""
        if not self.schwab_client:
            return "❌ Schwab client not connected"
        
        try:
            if action == "buy":
                if order_type == "market":
                    result = self.schwab_client.buy_market(ticker, shares, dry_run=False)
                else:
                    result = self.schwab_client.buy_limit(ticker, shares, limit_price, dry_run=False)
                
                if result.get("success"):
                    msg = f"✅ [SCHWAB] BUY {shares} {ticker}"
                    if order_type == "limit":
                        msg += f" @ ${limit_price:.2f}"
                    msg += f" - {reason}"
                    
                    # Verify the trade was actually placed
                    print("   🔍 Verifying trade...")
                    verification = self.schwab_client.verify_trade(ticker, "BUY", shares)
                    msg += f"\n   {verification['message']}"
                    
                    # Also set stop-loss if specified
                    if stop_loss > 0:
                        stop_result = self.schwab_client.set_stop_loss(
                            ticker, shares, stop_loss, dry_run=False
                        )
                        if stop_result.get("success"):
                            msg += f"\n   🛑 Stop-loss set @ ${stop_loss:.2f}"
                    
                    return msg
                else:
                    return f"❌ [SCHWAB] BUY {ticker} failed: {result.get('error', 'Unknown error')}"
            
            elif action == "sell":
                if order_type == "market":
                    result = self.schwab_client.sell_market(ticker, shares, dry_run=False)
                else:
                    result = self.schwab_client.sell_limit(ticker, shares, limit_price, dry_run=False)
                
                if result.get("success"):
                    msg = f"✅ [SCHWAB] SELL {shares} {ticker}"
                    if order_type == "limit":
                        msg += f" @ ${limit_price:.2f}"
                    msg += f" - {reason}"
                    
                    # Verify the trade was actually placed
                    print("   🔍 Verifying trade...")
                    verification = self.schwab_client.verify_trade(ticker, "SELL", shares)
                    msg += f"\n   {verification['message']}"
                    
                    return msg
                else:
                    return f"❌ [SCHWAB] SELL {ticker} failed: {result.get('error', 'Unknown error')}"
            
            else:
                return f"❓ Unknown action: {action}"
        
        except Exception as e:
            return f"❌ [SCHWAB] Trade failed: {e}"
        finally:
            # Always sync from Schwab after trade attempt to keep local state updated
            self._sync_from_schwab()

    def execute_trades_with_confirmation(self, trades: list[dict[str, Any]]) -> list[str]:
        """Execute trades with user confirmation."""
        if not trades:
            return ["No trades to execute."]

        results = []
        print("\n" + "=" * 50)
        print("📋 TRADE RECOMMENDATIONS")
        print("=" * 50)

        for i, trade in enumerate(trades, 1):
            action = trade.get("action", "?").upper()
            ticker = trade.get("ticker", "?")
            shares = trade.get("shares", 0)
            price = trade.get("limit_price", trade.get("price", 0))
            stop = trade.get("stop_loss", "N/A")
            reason = trade.get("reason", "")

            print(f"\n{i}. {action} {shares} {ticker} @ ${price}")
            if action == "BUY" and stop != "N/A":
                print(f"   Stop-loss: ${stop}")
            print(f"   Reason: {reason}")

        print("\n" + "-" * 50)
        
        # Show trading mode options - in LIVE mode, 'y' executes LIVE trades
        if self.schwab_client:
            mode_text = "🔴 LIVE" if self.live_trading else "📝 PAPER"
            print(f"Current mode: {mode_text}")
            print("\nOptions:")
            if self.live_trading:
                print("  y     - Execute ALL trades LIVE via Schwab ⚠️")
                print("  n     - Cancel")
                print("  1,2,3 - Execute specific trades LIVE via Schwab ⚠️")
                print("  paper - Execute ALL trades in paper mode only")
            else:
                print("  y     - Execute ALL trades (paper)")
                print("  n     - Cancel")
                print("  1,2,3 - Execute specific trades (paper)")
                print("  LIVE  - Execute ALL with REAL MONEY via Schwab ⚠️")
        else:
            print("Mode: 📝 PAPER (Schwab not connected)")
        
        confirm = input("\nYour choice: ").strip()

        # Determine if using Schwab for live trading
        use_schwab = False
        selected_indices = None
        
        # In LIVE mode, 'y' and number selections default to Schwab
        if self.live_trading and self.schwab_client:
            if confirm.lower() == "paper":
                # Explicit paper mode override
                use_schwab = False
                print("📝 Executing in PAPER mode...")
            elif confirm.lower() == "y":
                # In LIVE mode, 'y' means execute live
                use_schwab = True
                print("\n⚠️  EXECUTING REAL TRADES WITH SCHWAB ⚠️")
                double_confirm = input("Type 'CONFIRM' to proceed: ").strip()
                if double_confirm != "CONFIRM":
                    return ["❌ Live trading cancelled."]
            elif confirm.lower() == "n":
                return ["❌ Trades cancelled by user."]
            else:
                # Number selection in LIVE mode = live trades
                try:
                    selected_indices = [int(x.strip()) - 1 for x in confirm.split(",")]
                    use_schwab = True
                    print(f"\n⚠️  EXECUTING TRADES {[i+1 for i in selected_indices]} LIVE ⚠️")
                    double_confirm = input("Type 'CONFIRM' to proceed: ").strip()
                    if double_confirm != "CONFIRM":
                        return ["❌ Live trading cancelled."]
                except ValueError:
                    return ["❌ Invalid selection. Trades cancelled."]
        else:
            # PAPER mode or no Schwab - original logic
            if confirm.upper() == "LIVE" and self.schwab_client:
                use_schwab = True
                self.live_trading = True  # Switch to live mode
                print("\n⚠️  EXECUTING REAL TRADES WITH SCHWAB ⚠️")
                double_confirm = input("Type 'CONFIRM' to proceed: ").strip()
                if double_confirm != "CONFIRM":
                    return ["❌ Live trading cancelled."]
            elif confirm.lower() == "y":
                pass  # Paper trade all
            elif confirm.lower() == "n":
                return ["❌ Trades cancelled by user."]
            else:
                # Selective paper execution
                try:
                    selected_indices = [int(x.strip()) - 1 for x in confirm.split(",")]
                except ValueError:
                    return ["❌ Invalid selection. Trades cancelled."]

        # Execute trades
        trades_to_execute = trades if selected_indices is None else [trades[i] for i in selected_indices if 0 <= i < len(trades)]
        
        for trade in trades_to_execute:
            result = self.execute_trade(trade, use_schwab=use_schwab)
            results.append(result)
            print(result)

        return results

    def run_interactive(self) -> None:
        """Run the chatbot in interactive mode."""
        print("\n" + "=" * 60)
        print("🤖 AZURE AI STOCK TRADING CHATBOT")
        print("=" * 60)
        print(f"Connected to: {self.endpoint}")
        print(f"Deployment: {self.deployment}")
        
        # Show Schwab status
        if self.schwab_client:
            mode = "🔴 LIVE TRADING" if self.live_trading else "📝 PAPER TRADING"
            print(f"Schwab: ✅ Connected | Mode: {mode}")
        else:
            print("Schwab: ❌ Not connected (paper trading only)")
        
        print("\nCommands:")
        print("  /price TICKER [TICKER2 ...]  - Get current stock prices")
        print("  /news TICKER  - Get news, sentiment & financials for a stock")
        print("  /status  - Show current portfolio")
        print("  /daily   - Run daily portfolio update")
        print("  /execute - Execute last recommended trades")
        if self.schwab_client:
            print("  /schwab  - Show Schwab account details")
            print("  /orders  - Show Schwab orders")
            print("  /sync    - Sync cash/positions from Schwab")
            print("  /golive  - Enable LIVE trading mode ⚠️")
            print("  /paper   - Switch to paper trading mode")
        print("  /quit    - Exit chatbot")
        print("-" * 60)

        last_trades: list[dict[str, Any]] = []

        # Initial portfolio status
        print(self.get_portfolio_summary())

        while True:
            try:
                user_input = input("\n💬 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "/quit":
                print("Goodbye!")
                break

            elif user_input.lower() == "/status":
                self._load_portfolio()  # Refresh
                print(self.get_portfolio_summary())
                continue

            elif user_input.lower().startswith("/price"):
                parts = user_input.split()[1:]  # Get tickers after /price
                if not parts:
                    print("Usage: /price TICKER [TICKER2 ...]")
                    print("Example: /price AAPL MSFT TSLA")
                else:
                    print("\n📈 Fetching prices...")
                    print(self.get_multiple_prices(parts))
                continue
            
            elif user_input.lower().startswith("/news"):
                parts = user_input.split()[1:]  # Get ticker after /news
                if not parts:
                    print("Usage: /news TICKER")
                    print("Example: /news AAPL")
                elif not self.news_client:
                    print("❌ News service not available. Set FINNHUB_API_KEY in .env")
                    print("   Get free key at: https://finnhub.io/register")
                else:
                    ticker = parts[0].upper()
                    print(f"\n📰 Fetching news for {ticker}...")
                    print(self.get_stock_news(ticker))
                continue

            elif user_input.lower() == "/daily":
                print("\nRunning daily portfolio update...")
                try:
                    daily_results(self.portfolio_df, self.cash)
                except Exception as e:
                    print(f"Error running daily update: {e}")
                continue

            elif user_input.lower() == "/execute":
                if last_trades:
                    self.execute_trades_with_confirmation(last_trades)
                    self._load_portfolio()  # Refresh after trades
                else:
                    print("No pending trades. Ask the AI for recommendations first.")
                continue
            
            # Schwab-specific commands
            elif user_input.lower() == "/schwab":
                if self.schwab_client:
                    self.schwab_client.display_account_summary()
                else:
                    print("❌ Schwab not connected. Run 'python schwab_auth.py' first.")
                continue
            
            elif user_input.lower() == "/orders":
                if self.schwab_client:
                    self.schwab_client.display_orders()
                else:
                    print("❌ Schwab not connected.")
                continue
            
            elif user_input.lower() == "/golive":
                if not self.schwab_client:
                    print("❌ Schwab not connected. Run 'python schwab_auth.py' first.")
                    continue
                print("\n⚠️  WARNING: LIVE TRADING MODE ⚠️")
                print("This will execute REAL trades with REAL money!")
                confirm = input("Type 'I UNDERSTAND' to enable: ").strip()
                if confirm == "I UNDERSTAND":
                    self.live_trading = True
                    print("🔴 LIVE TRADING MODE ENABLED")
                    # Auto-sync from Schwab when going live
                    print("\ud83d\udd04 Syncing from Schwab...")
                    self._sync_from_schwab()
                else:
                    print("Live trading not enabled.")
                continue
            
            elif user_input.lower() == "/sync":
                if not self.schwab_client:
                    print("\u274c Schwab not connected.")
                    continue
                print("\ud83d\udd04 Syncing from Schwab...")
                self._sync_from_schwab()
                continue
            
            elif user_input.lower() == "/paper":
                self.live_trading = False
                print("📝 Paper trading mode enabled (safe)")
                continue

            # Send to AI
            print("\n🤔 Thinking...")
            response = self.chat(user_input)
            print(f"\n🤖 AI: {response}")

            # Check for trade recommendations
            trades = self.parse_trades_from_response(response)
            if trades:
                last_trades = trades
                print(f"\n📊 Found {len(trades)} trade recommendation(s).")
                execute_now = input("Execute now? (y/n): ").strip().lower()
                if execute_now == "y":
                    self.execute_trades_with_confirmation(trades)
                    self._load_portfolio()
                else:
                    print("Use /execute to execute these trades later.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Azure AI Stock Trading Chatbot")
    parser.add_argument("--endpoint", help="Azure OpenAI endpoint (or set AZURE_OPENAI_ENDPOINT)")
    parser.add_argument("--api-key", help="Azure OpenAI API key (or set AZURE_OPENAI_API_KEY)")
    parser.add_argument("--deployment", help="Azure deployment name (or set AZURE_OPENAI_DEPLOYMENT)")
    parser.add_argument("--data-dir", default=".", help="Data directory for CSV files")
    parser.add_argument("--live", action="store_true", help="Start in LIVE trading mode (real money!)")

    args = parser.parse_args()

    try:
        chatbot = AzureTradingChatbot(
            endpoint=args.endpoint,
            api_key=args.api_key,
            deployment=args.deployment,
            data_dir=args.data_dir,
            live_trading=args.live,
        )
        
        # Warn if starting in live mode
        if args.live:
            print("\n" + "⚠️" * 20)
            print("🔴 STARTING IN LIVE TRADING MODE - REAL MONEY!")
            print("⚠️" * 20 + "\n")
        
        chatbot.run_interactive()

    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nSet these environment variables:")
        print("  AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
        print("  AZURE_OPENAI_API_KEY=your-api-key")
        print("  AZURE_OPENAI_DEPLOYMENT=your-deployment-name")
        sys.exit(1)


if __name__ == "__main__":
    main()
