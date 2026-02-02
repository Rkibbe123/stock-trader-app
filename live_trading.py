"""
Live Trading System with Schwab Integration
Combines LLM-powered recommendations with real Schwab account trading.
"""

import os
import argparse
import json
import re
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime

import pandas as pd

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

from schwab_client import SchwabClient
from schwab_auth import SchwabAuth, interactive_authorization


def sync_csv_with_schwab(client: SchwabClient, csv_path: Path):
    """Sync CSV file with Schwab positions after trades"""
    positions = client.get_positions()
    balance = client.get_balance()
    cash = balance.get('cash_available', 0)
    total_equity = balance.get('account_value', 0)
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Build new CSV data
    rows = []
    for pos in positions:
        symbol = pos.get("instrument", {}).get("symbol", "N/A")
        qty = pos.get("longQuantity", 0) or pos.get("shortQuantity", 0)
        avg_price = pos.get("averagePrice", 0)
        market_value = pos.get("marketValue", 0)
        current_price = market_value / qty if qty > 0 else 0
        
        rows.append({
            'Date': today,
            'Ticker': symbol,
            'Action': 'HOLD',
            'Shares': qty,
            'Buy Price': avg_price,
            'Stop Loss': '',
            'Cost Basis': qty * avg_price,
            'Current Price': current_price,
            'Total Value': market_value,
            'PnL': market_value - (qty * avg_price),
            'Cash Balance': cash,
            'Total Equity': total_equity
        })
    
    # Add total row
    total_value = sum(pos.get("marketValue", 0) for pos in positions)
    rows.append({
        'Date': today,
        'Ticker': 'TOTAL',
        'Action': 'SYNC',
        'Shares': '',
        'Buy Price': '',
        'Stop Loss': '',
        'Cost Basis': '',
        'Current Price': '',
        'Total Value': total_value,
        'PnL': '',
        'Cash Balance': cash,
        'Total Equity': total_equity
    })
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV synced: {csv_path}")
    print(f"   Positions: {len(positions)}, Cash: ${cash:.2f}, Total: ${total_equity:.2f}")


def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse LLM response and extract trading decisions"""
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response: {e}")
        return {"error": "Failed to parse response", "raw_response": response}


def call_azure_openai(prompt: str, model: str = "rk-stockpicker") -> str:
    """Call Azure OpenAI API"""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    if not api_key:
        raise ValueError("No API key found. Set AZURE_OPENAI_API_KEY or OPENAI_API_KEY in .env")

    if endpoint:
        client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2024-02-15-preview"
        )
    else:
        client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional portfolio analyst. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=1500
    )
    return response.choices[0].message.content


def generate_trading_prompt(positions: List[dict], cash: float, total_value: float) -> str:
    """Generate a trading prompt with current Schwab portfolio data"""
    
    if not positions:
        holdings_text = "No current holdings"
    else:
        holdings_lines = []
        for pos in positions:
            symbol = pos.get("instrument", {}).get("symbol", "N/A")
            qty = pos.get("longQuantity", 0) or pos.get("shortQuantity", 0)
            avg_price = pos.get("averagePrice", 0)
            market_value = pos.get("marketValue", 0)
            holdings_lines.append(f"{symbol}: {qty} shares @ ${avg_price:.2f} (value: ${market_value:.2f})")
        holdings_text = "\n".join(holdings_lines)

    today = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""You are an aggressive professional portfolio analyst managing a LIVE trading account. 
Here is the current portfolio state as of {today}:

[ Current Holdings ]
{holdings_text}

[ Account Snapshot ]
Cash Available: ${cash:,.2f}
Total Account Value: ${total_value:,.2f}

IMPORTANT: This is a REAL account with REAL money. Be thoughtful but decisive.

Rules:
- You have ${cash:,.2f} in cash available for new positions
- Prefer U.S. micro-cap stocks (<$300M market cap) for aggressive growth
- Full shares only, no options or derivatives
- Use stop-losses for risk management (typically 15-25% below entry)
- Consider current positions when making new recommendations
- If you recommend selling, provide specific exit reasoning

Analyze the current market conditions and provide specific trading recommendations.

Respond with ONLY a JSON object in this exact format:
{{
    "analysis": "Brief market analysis and portfolio assessment",
    "trades": [
        {{
            "action": "buy",
            "ticker": "SYMBOL",
            "shares": 10,
            "price": 5.50,
            "stop_loss": 4.50,
            "reason": "Brief rationale"
        }},
        {{
            "action": "sell",
            "ticker": "EXISTING_SYMBOL",
            "shares": 5,
            "price": 8.00,
            "reason": "Exit rationale"
        }}
    ],
    "confidence": 0.75
}}

Be aggressive but not reckless. Recommend trades you are confident about."""
    return prompt


def execute_live_trades(client: SchwabClient, trades: List[Dict[str, Any]], dry_run: bool = True) -> List[Dict]:
    """Execute trades on Schwab account"""
    
    results = []
    
    for trade in trades:
        action = trade.get('action', '').lower()
        ticker = trade.get('ticker', '').upper()
        shares = int(trade.get('shares', 0))
        price = float(trade.get('price', 0))
        stop_loss = float(trade.get('stop_loss', 0))
        reason = trade.get('reason', 'LLM recommendation')
        
        if not ticker or shares <= 0:
            print(f"⚠️ Invalid trade: {trade}")
            continue
        
        result = {
            "ticker": ticker,
            "action": action,
            "shares": shares,
            "price": price,
            "status": "pending"
        }
        
        try:
            if action == 'buy':
                print(f"\n{'🔶 DRY RUN: ' if dry_run else '🛒 EXECUTING: '}BUY {shares} {ticker} @ ~${price:.2f}")
                print(f"   Reason: {reason}")
                
                if dry_run:
                    result["status"] = "simulated"
                    result["message"] = "Dry run - no order placed"
                else:
                    # Place market buy order
                    order_result = client.buy_market(ticker, shares, dry_run=False)
                    
                    if "error" in order_result:
                        result["status"] = "failed"
                        result["message"] = order_result["error"]
                        print(f"   ❌ Order failed: {order_result['error']}")
                    else:
                        result["status"] = "submitted"
                        result["order_id"] = order_result.get("location", "").split("/")[-1]
                        print(f"   ✅ Order submitted!")
                        
                        # Verify the trade
                        verification = client.verify_trade(ticker, "BUY", shares)
                        print(f"   {verification['message']}")
                        result["verification"] = verification
                        
                        # If buy successful and stop_loss specified, place stop-loss order
                        if stop_loss > 0 and verification.get("verified"):
                            print(f"   🛑 Setting stop-loss at ${stop_loss:.2f}...")
                            stop_result = client.set_stop_loss(ticker, shares, stop_loss, dry_run=False)
                            if "error" not in stop_result:
                                print(f"   ✅ Stop-loss order placed")
                            else:
                                print(f"   ⚠️ Stop-loss failed: {stop_result.get('error')}")
                
            elif action == 'sell':
                print(f"\n{'🔶 DRY RUN: ' if dry_run else '💰 EXECUTING: '}SELL {shares} {ticker} @ ~${price:.2f}")
                print(f"   Reason: {reason}")
                
                if dry_run:
                    result["status"] = "simulated"
                    result["message"] = "Dry run - no order placed"
                else:
                    # Place market sell order
                    order_result = client.sell_market(ticker, shares, dry_run=False)
                    
                    if "error" in order_result:
                        result["status"] = "failed"
                        result["message"] = order_result["error"]
                        print(f"   ❌ Order failed: {order_result['error']}")
                    else:
                        result["status"] = "submitted"
                        print(f"   ✅ Order submitted!")
                        
                        # Verify the trade
                        verification = client.verify_trade(ticker, "SELL", shares)
                        print(f"   {verification['message']}")
                        result["verification"] = verification
                        
            elif action == 'hold':
                print(f"\n📌 HOLD: {ticker} - {reason}")
                result["status"] = "hold"
                
        except Exception as e:
            result["status"] = "error"
            result["message"] = str(e)
            print(f"   ❌ Error: {e}")
        
        results.append(result)
    
    return results


def run_live_trading(model: str = "rk-stockpicker", dry_run: bool = True, auto_confirm: bool = False):
    """Main live trading function"""
    
    print("\n" + "="*60)
    print("🚀 LIVE TRADING SYSTEM" + (" [DRY RUN]" if dry_run else " [LIVE MODE]"))
    print("="*60)
    
    # Initialize Schwab client
    print("\n🔐 Connecting to Schwab...")
    
    try:
        auth = SchwabAuth()
        
        if not auth.is_authenticated():
            print("\n⚠️ Not authenticated with Schwab. Starting authorization flow...")
            auth = interactive_authorization()
            if not auth.is_authenticated():
                print("❌ Authentication failed. Exiting.")
                return
        
        client = SchwabClient(auth)
        print("✅ Connected to Schwab!")
        
    except Exception as e:
        print(f"❌ Failed to connect to Schwab: {e}")
        print("\nMake sure SCHWAB_APP_KEY and SCHWAB_APP_SECRET are set in .env")
        return
    
    # Get account info
    print("\n📊 Fetching account data...")
    
    try:
        balance = client.get_balance()
        if "error" in balance:
            print(f"❌ Could not fetch balance: {balance['error']}")
            return
        
        positions = client.get_positions()
        
        cash = balance.get("cash_available", 0)
        total_value = balance.get("account_value", 0)
        
        print(f"\n💵 Account Value: ${total_value:,.2f}")
        print(f"💰 Cash Available: ${cash:,.2f}")
        print(f"📈 Positions: {len(positions)}")
        
        if positions:
            print("\nCurrent Holdings:")
            for pos in positions:
                symbol = pos.get("instrument", {}).get("symbol", "N/A")
                qty = pos.get("longQuantity", 0) or pos.get("shortQuantity", 0)
                market_value = pos.get("marketValue", 0)
                print(f"   {symbol}: {qty} shares (${market_value:,.2f})")
        
    except Exception as e:
        print(f"❌ Error fetching account data: {e}")
        return
    
    # Generate prompt and call LLM
    print("\n🤖 Getting LLM recommendations...")
    
    try:
        prompt = generate_trading_prompt(positions, cash, total_value)
        response = call_azure_openai(prompt, model)
        parsed = parse_llm_response(response)
        
        if "error" in parsed:
            print(f"❌ LLM Error: {parsed['error']}")
            return
        
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return
    
    # Display recommendations
    analysis = parsed.get('analysis', 'No analysis provided')
    confidence = parsed.get('confidence', 0)
    trades = parsed.get('trades', [])
    
    print("\n" + "="*60)
    print("📋 LLM RECOMMENDATIONS")
    print("="*60)
    print(f"\n📊 Analysis: {analysis}")
    print(f"🎯 Confidence: {confidence:.0%}")
    print(f"📝 Recommended Trades: {len(trades)}")
    
    if trades:
        print("\nTrade Details:")
        for i, trade in enumerate(trades, 1):
            action = trade.get('action', 'unknown').upper()
            ticker = trade.get('ticker', 'N/A')
            shares = trade.get('shares', 0)
            price = trade.get('price', 0)
            stop_loss = trade.get('stop_loss', 0)
            reason = trade.get('reason', 'No reason')
            
            print(f"\n  {i}. {action} {shares} shares of {ticker}")
            print(f"     Price: ~${price:.2f}" + (f" | Stop: ${stop_loss:.2f}" if stop_loss else ""))
            print(f"     Reason: {reason}")
    
    if not trades:
        print("\n✅ No trades recommended at this time.")
        return
    
    # Confirm execution
    print("\n" + "="*60)
    
    if dry_run:
        print("🔶 DRY RUN MODE - No real orders will be placed")
        execute = True
    elif auto_confirm:
        print("⚠️ AUTO-CONFIRM ENABLED - Executing trades automatically!")
        execute = True
    else:
        print("⚠️ LIVE MODE - Real orders will be placed!")
        confirm = input("\nExecute these trades? (yes/no): ").strip().lower()
        execute = confirm == 'yes'
    
    if execute:
        print("\n" + "="*60)
        print("📈 EXECUTING TRADES" + (" [DRY RUN]" if dry_run else " [LIVE]"))
        print("="*60)
        
        results = execute_live_trades(client, trades, dry_run=dry_run)
        
        # Save results
        log_file = Path(__file__).parent / "live_trade_log.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "dry_run": dry_run,
                "analysis": analysis,
                "confidence": confidence,
                "trades": trades,
                "results": results
            }) + "\n")
        
        print(f"\n📝 Trade log saved to: {log_file}")
        
        # Auto-sync CSV with Schwab after live trades
        if not dry_run:
            print("\n🔄 Syncing CSV with Schwab...")
            sync_csv_with_schwab(client, Path(__file__).parent / "chatgpt_portfolio_update.csv")
            
            print("\n📊 Updated Account Status:")
            client.display_account_summary()
    else:
        print("\n❌ Trade execution cancelled.")
    
    print("\n" + "="*60)
    print("✅ Session complete!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Live Trading System with Schwab")
    parser.add_argument("--model", default="rk-stockpicker", help="Azure OpenAI deployment name")
    parser.add_argument("--dry-run", action="store_true", help="Simulate trades without executing")
    parser.add_argument("--live", action="store_true", help="Execute real trades (requires confirmation)")
    parser.add_argument("--auto-confirm", action="store_true", help="Skip confirmation prompt (DANGEROUS)")
    
    args = parser.parse_args()
    
    # Default to dry-run unless --live is specified
    dry_run = not args.live
    
    if not dry_run and not args.auto_confirm:
        print("\n⚠️  WARNING: You are about to run in LIVE mode!")
        print("   Real orders will be placed on your Schwab account.")
        print("   Make sure you understand the risks.\n")
    
    run_live_trading(
        model=args.model,
        dry_run=dry_run,
        auto_confirm=args.auto_confirm
    )


if __name__ == "__main__":
    main()
