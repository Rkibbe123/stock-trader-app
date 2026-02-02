"""
Interactive Live Trading System
- Connects to Schwab as source of truth
- Compares with local CSV files
- Waits for user commands before making recommendations
"""

import os
import json
import re
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from schwab_client import SchwabClient
from schwab_auth import SchwabAuth, interactive_authorization


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


def call_azure_openai(prompt: str, model: str = "rk-stockpicker", system_prompt: str = None) -> str:
    """Call Azure OpenAI API"""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    if not api_key:
        raise ValueError("No API key found. Set AZURE_OPENAI_API_KEY in .env")

    if endpoint:
        client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2024-02-15-preview"
        )
    else:
        client = openai.OpenAI(api_key=api_key)

    if system_prompt is None:
        system_prompt = "You are a professional portfolio analyst. Always respond with valid JSON."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=1500
    )
    return response.choices[0].message.content


def chat_with_llm(user_input: str, positions: List[Dict], balance: Dict, orders: List[Dict], model: str) -> str:
    """Handle natural language chat with portfolio context"""
    
    # Build portfolio context
    if positions:
        holdings_lines = []
        for pos in positions:
            holdings_lines.append(f"  {pos['ticker']}: {pos['shares']:.0f} shares @ ${pos['avg_price']:.4f} (value: ${pos['market_value']:.2f})")
        holdings_text = "\n".join(holdings_lines)
    else:
        holdings_text = "  No current holdings"
    
    # Build open orders context
    if orders:
        open_orders = [o for o in orders if o['status'] in ['WORKING', 'PENDING_ACTIVATION', 'QUEUED', 'ACCEPTED']]
        if open_orders:
            orders_lines = []
            for o in open_orders:
                price_str = f"@ ${o['price']:.2f}" if o['price'] else "MKT"
                orders_lines.append(f"  {o['order_type']} {o['action']} {o['shares']:.0f} {o['ticker']} {price_str}")
            orders_text = "\n".join(orders_lines)
        else:
            orders_text = "  No open orders"
    else:
        orders_text = "  No open orders"
    
    cash = balance.get('cash_available', 0)
    total_value = balance.get('account_value', 0)
    today = datetime.now().strftime("%Y-%m-%d")
    
    system_prompt = """You are an expert trading assistant helping manage a live stock portfolio. 
You have access to real-time portfolio data and can provide advice, analysis, and insights.
Be concise but helpful. If the user asks about executing trades, remind them to use the commands:
- 'buy TICKER SHARES [STOP]' for buying
- 'sell TICKER SHARES' for selling
- 'recommend' for AI trading recommendations
- 'execute' to execute recommendations

Be conversational and helpful. Answer questions about stocks, market conditions, portfolio strategy, etc."""

    prompt = f"""Current Date: {today}

PORTFOLIO CONTEXT:
=================
Account Value: ${total_value:,.2f}
Cash Available: ${cash:,.2f}

Current Holdings:
{holdings_text}

Open Orders:
{orders_text}

USER QUESTION:
{user_input}

Please respond helpfully to the user's question. If they're asking about trading, provide analysis and guidance."""

    try:
        response = call_azure_openai(prompt, model, system_prompt)
        return response
    except Exception as e:
        return f"Sorry, I encountered an error: {e}"


def load_csv_portfolio(csv_path: Path) -> pd.DataFrame:
    """Load portfolio from CSV file"""
    if not csv_path.exists():
        return pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])
    
    df = pd.read_csv(csv_path)
    # Filter to only position rows (not TOTAL/SYNC rows)
    if 'Ticker' in df.columns:
        df = df[~df['Ticker'].isin(['TOTAL', 'SYNC', ''])]
        df = df.rename(columns={'Ticker': 'ticker', 'Shares': 'shares', 'Stop Loss': 'stop_loss', 
                                 'Buy Price': 'buy_price', 'Cost Basis': 'cost_basis'})
    elif 'ticker' in df.columns:
        df = df[~df['ticker'].isin(['TOTAL', 'SYNC', ''])]
    
    return df


def get_schwab_positions(client: SchwabClient) -> List[Dict]:
    """Get positions from Schwab"""
    positions = client.get_positions()
    result = []
    for pos in positions:
        symbol = pos.get("instrument", {}).get("symbol", "N/A")
        qty = pos.get("longQuantity", 0) or pos.get("shortQuantity", 0)
        avg_price = pos.get("averagePrice", 0)
        market_value = pos.get("marketValue", 0)
        current_price = pos.get("currentDayProfitLossPercentage", 0)
        
        result.append({
            "ticker": symbol,
            "shares": qty,
            "avg_price": avg_price,
            "market_value": market_value,
            "current_price": market_value / qty if qty > 0 else 0
        })
    return result


def get_schwab_orders(client: SchwabClient) -> List[Dict]:
    """Get open orders from Schwab"""
    from datetime import timedelta
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%dT00:00:00.000Z')
    to_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%dT23:59:59.000Z')
    
    orders = client.get_orders(max_results=50, from_date=from_date, to_date=to_date)
    result = []
    
    for order in orders:
        status = order.get('status', 'N/A')
        order_type = order.get('orderType', 'N/A')
        stop_price = order.get('stopPrice')
        price = order.get('price')
        
        for leg in order.get('orderLegCollection', []):
            symbol = leg.get('instrument', {}).get('symbol', 'N/A')
            qty = leg.get('quantity', 0)
            instruction = leg.get('instruction', 'N/A')
            
            result.append({
                "ticker": symbol,
                "action": instruction,
                "shares": qty,
                "order_type": order_type,
                "price": stop_price or price,
                "status": status,
                "order_id": order.get('orderId')
            })
    
    return result


def compare_positions(schwab_positions: List[Dict], csv_df: pd.DataFrame) -> Dict:
    """Compare Schwab positions with CSV file"""
    differences = {
        "in_schwab_only": [],
        "in_csv_only": [],
        "quantity_mismatch": [],
        "matched": []
    }
    
    schwab_tickers = {p['ticker']: p for p in schwab_positions}
    csv_tickers = {}
    
    if not csv_df.empty and 'ticker' in csv_df.columns:
        for _, row in csv_df.iterrows():
            ticker = str(row.get('ticker', '')).upper()
            if ticker and ticker not in ['TOTAL', 'SYNC', 'NAN']:
                csv_tickers[ticker] = {
                    'ticker': ticker,
                    'shares': float(row.get('shares', 0)),
                    'buy_price': float(row.get('buy_price', 0)) if pd.notna(row.get('buy_price')) else 0
                }
    
    # Check Schwab positions
    for ticker, schwab_pos in schwab_tickers.items():
        if ticker in csv_tickers:
            csv_pos = csv_tickers[ticker]
            if abs(schwab_pos['shares'] - csv_pos['shares']) > 0.01:
                differences["quantity_mismatch"].append({
                    "ticker": ticker,
                    "schwab_shares": schwab_pos['shares'],
                    "csv_shares": csv_pos['shares']
                })
            else:
                differences["matched"].append(ticker)
        else:
            differences["in_schwab_only"].append(schwab_pos)
    
    # Check CSV positions not in Schwab
    for ticker, csv_pos in csv_tickers.items():
        if ticker not in schwab_tickers:
            differences["in_csv_only"].append(csv_pos)
    
    return differences


def display_portfolio(client: SchwabClient, csv_path: Path):
    """Display current portfolio status"""
    print("\n" + "="*70)
    print("📊 PORTFOLIO STATUS (Schwab = Source of Truth)")
    print("="*70)
    
    # Get Schwab data
    balance = client.get_balance()
    positions = get_schwab_positions(client)
    orders = get_schwab_orders(client)
    
    # Display balance
    print(f"\n💵 Account Value: ${balance.get('account_value', 0):,.2f}")
    print(f"💰 Cash Available: ${balance.get('cash_available', 0):,.2f}")
    print(f"💪 Buying Power: ${balance.get('buying_power', 0):,.2f}")
    
    # Display positions
    print(f"\n📈 POSITIONS ({len(positions)} total)")
    print("-"*70)
    if positions:
        print(f"{'Ticker':<10} {'Shares':>10} {'Avg Price':>12} {'Current':>12} {'Value':>12}")
        print("-"*70)
        for pos in positions:
            print(f"{pos['ticker']:<10} {pos['shares']:>10.0f} ${pos['avg_price']:>10.4f} "
                  f"${pos['current_price']:>10.4f} ${pos['market_value']:>10.2f}")
    else:
        print("  No positions")
    
    # Display open orders
    open_orders = [o for o in orders if o['status'] in ['WORKING', 'PENDING_ACTIVATION', 'QUEUED', 'ACCEPTED']]
    print(f"\n📋 OPEN ORDERS ({len(open_orders)} active)")
    print("-"*70)
    if open_orders:
        for order in open_orders:
            price_str = f"@ ${order['price']:.2f}" if order['price'] else "MKT"
            print(f"  [{order['status']}] {order['order_type']} {order['action']} "
                  f"{order['shares']:.0f} {order['ticker']} {price_str}")
    else:
        print("  No open orders")
    
    # Compare with CSV
    csv_df = load_csv_portfolio(csv_path)
    differences = compare_positions(positions, csv_df)
    
    print(f"\n🔄 CSV COMPARISON")
    print("-"*70)
    
    if differences["in_schwab_only"]:
        print("  ⚠️  In Schwab but NOT in CSV:")
        for pos in differences["in_schwab_only"]:
            print(f"      {pos['ticker']}: {pos['shares']:.0f} shares")
    
    if differences["in_csv_only"]:
        print("  ⚠️  In CSV but NOT in Schwab:")
        for pos in differences["in_csv_only"]:
            print(f"      {pos['ticker']}: {pos['shares']:.0f} shares")
    
    if differences["quantity_mismatch"]:
        print("  ⚠️  Quantity mismatches:")
        for diff in differences["quantity_mismatch"]:
            print(f"      {diff['ticker']}: Schwab={diff['schwab_shares']:.0f}, CSV={diff['csv_shares']:.0f}")
    
    if not any([differences["in_schwab_only"], differences["in_csv_only"], differences["quantity_mismatch"]]):
        print("  ✅ Schwab and CSV are in sync!")
    
    print("="*70)
    
    return positions, balance, orders


def get_recommendations(client: SchwabClient, positions: List[Dict], balance: Dict, model: str):
    """Get LLM trading recommendations"""
    print("\n🤖 Getting LLM recommendations...")
    
    cash = balance.get("cash_available", 0)
    total_value = balance.get("account_value", 0)
    
    # Build holdings text
    if not positions:
        holdings_text = "No current holdings"
    else:
        holdings_lines = []
        for pos in positions:
            holdings_lines.append(f"{pos['ticker']}: {pos['shares']:.0f} shares @ ${pos['avg_price']:.4f} "
                                  f"(value: ${pos['market_value']:.2f})")
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
        }}
    ],
    "confidence": 0.75
}}

Be aggressive but not reckless. Recommend trades you are confident about."""

    try:
        response = call_azure_openai(prompt, model)
        parsed = parse_llm_response(response)
        
        if "error" in parsed:
            print(f"❌ Error: {parsed['error']}")
            return None
        
        # Display recommendations
        analysis = parsed.get('analysis', 'No analysis provided')
        confidence = parsed.get('confidence', 0)
        trades = parsed.get('trades', [])
        
        print("\n" + "="*70)
        print("📋 LLM RECOMMENDATIONS")
        print("="*70)
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
        else:
            print("\n✅ No trades recommended at this time.")
        
        print("="*70)
        return parsed
        
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return None


def execute_trade(client: SchwabClient, trade: Dict, csv_path: Path = None):
    """Execute a single trade"""
    action = trade.get('action', '').lower()
    ticker = trade.get('ticker', '').upper()
    shares = int(trade.get('shares', 0))
    stop_loss = float(trade.get('stop_loss', 0))
    
    if not ticker or shares <= 0:
        print(f"⚠️ Invalid trade parameters")
        return False
    
    print(f"\n⚠️ EXECUTING: {action.upper()} {shares} {ticker}")
    confirm = input("Confirm? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print("❌ Trade cancelled")
        return False
    
    success = False
    try:
        if action == 'buy':
            result = client.buy_market(ticker, shares, dry_run=False)
            if "error" not in result:
                print(f"✅ Buy order submitted!")
                
                # Verify and place stop-loss
                verification = client.verify_trade(ticker, "BUY", shares)
                print(f"   {verification['message']}")
                success = verification.get("verified", False)
                
                if stop_loss > 0 and success:
                    print(f"🛑 Setting stop-loss at ${stop_loss:.2f}...")
                    client.set_stop_loss(ticker, shares, stop_loss, dry_run=False)
            else:
                print(f"❌ Order failed: {result['error']}")
                
        elif action == 'sell':
            result = client.sell_market(ticker, shares, dry_run=False)
            if "error" not in result:
                print(f"✅ Sell order submitted!")
                verification = client.verify_trade(ticker, "SELL", shares)
                print(f"   {verification['message']}")
                success = verification.get("verified", False)
            else:
                print(f"❌ Order failed: {result['error']}")
        
        # Auto-sync CSV after successful trade
        if success and csv_path:
            print("\n🔄 Auto-syncing CSV with Schwab...")
            sync_csv_with_schwab(client, csv_path)
                
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return success


def show_help():
    """Display help menu"""
    print("\n" + "="*70)
    print("📖 COMMANDS")
    print("="*70)
    print("""
  status, s      - Show current portfolio status
  recommend, r   - Get LLM trading recommendations
  execute, e     - Execute a recommended trade
  orders, o      - Show open orders
  buy <TICKER> <SHARES> [STOP]  - Manual buy order
  sell <TICKER> <SHARES>        - Manual sell order
  sync           - Sync CSV with Schwab positions
  help, h        - Show this help menu
  quit, q        - Exit
""")
    print("="*70)
    print("💡 TIP: You can also ask questions in natural language!")
    print("   Examples:")
    print("   • \"What's my best performing stock?\"")
    print("   • \"Should I sell CTXR?\"")
    print("   • \"How much cash do I have?\"")
    print("   • \"Give me a summary of my portfolio\"")
    print("="*70)


def sync_csv_with_schwab(client: SchwabClient, csv_path: Path):
    """Sync CSV file with Schwab positions"""
    print("\n🔄 Syncing CSV with Schwab...")
    
    positions = get_schwab_positions(client)
    balance = client.get_balance()
    cash = balance.get('cash_available', 0)
    total_equity = balance.get('account_value', 0)
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Build new CSV data
    rows = []
    for pos in positions:
        rows.append({
            'Date': today,
            'Ticker': pos['ticker'],
            'Action': 'HOLD',
            'Shares': pos['shares'],
            'Buy Price': pos['avg_price'],
            'Stop Loss': '',
            'Cost Basis': pos['shares'] * pos['avg_price'],
            'Current Price': pos['current_price'],
            'Total Value': pos['market_value'],
            'PnL': pos['market_value'] - (pos['shares'] * pos['avg_price']),
            'Cash Balance': cash,
            'Total Equity': total_equity
        })
    
    # Add total row
    total_value = sum(p['market_value'] for p in positions)
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


def run_interactive(model: str = "rk-stockpicker", csv_path: str = None):
    """Main interactive trading loop"""
    
    print("\n" + "="*70)
    print("🚀 INTERACTIVE LIVE TRADING SYSTEM")
    print("="*70)
    
    # Set CSV path
    if csv_path:
        portfolio_csv = Path(csv_path)
    else:
        portfolio_csv = Path(__file__).parent / "chatgpt_portfolio_update.csv"
    
    # Connect to Schwab
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
        return
    
    # Show initial status
    positions, balance, orders = display_portfolio(client, portfolio_csv)
    
    # Store last recommendations
    last_recommendations = None
    
    # Interactive loop
    show_help()
    
    while True:
        try:
            cmd = input("\n💬 How can I help you? >>> ").strip().lower()
            
            if not cmd:
                continue
            
            parts = cmd.split()
            command = parts[0]
            
            if command in ['quit', 'q', 'exit']:
                print("\n👋 Goodbye!")
                break
                
            elif command in ['help', 'h', '?']:
                show_help()
                
            elif command in ['status', 's']:
                positions, balance, orders = display_portfolio(client, portfolio_csv)
                
            elif command in ['recommend', 'r', 'rec']:
                last_recommendations = get_recommendations(client, positions, balance, model)
                
            elif command in ['orders', 'o']:
                orders = get_schwab_orders(client)
                print(f"\n📋 ALL ORDERS ({len(orders)} total)")
                print("-"*60)
                for order in orders:
                    price_str = f"@ ${order['price']:.2f}" if order['price'] else "MKT"
                    print(f"  [{order['status']}] {order['order_type']} {order['action']} "
                          f"{order['shares']:.0f} {order['ticker']} {price_str}")
                
            elif command in ['execute', 'e', 'exec']:
                if not last_recommendations or not last_recommendations.get('trades'):
                    print("⚠️ No recommendations available. Run 'recommend' first.")
                else:
                    trades = last_recommendations['trades']
                    print("\nWhich trade to execute?")
                    for i, t in enumerate(trades, 1):
                        print(f"  {i}. {t['action'].upper()} {t['shares']} {t['ticker']}")
                    
                    choice = input("Enter number (or 'all'): ").strip()
                    if choice == 'all':
                        for trade in trades:
                            execute_trade(client, trade, portfolio_csv)
                    elif choice.isdigit() and 1 <= int(choice) <= len(trades):
                        execute_trade(client, trades[int(choice) - 1], portfolio_csv)
                    else:
                        print("Invalid choice")
                    
                    # Refresh positions
                    positions, balance, orders = display_portfolio(client, portfolio_csv)
                
            elif command == 'buy' and len(parts) >= 3:
                ticker = parts[1].upper()
                shares = int(parts[2])
                stop_loss = float(parts[3]) if len(parts) > 3 else 0
                
                trade = {"action": "buy", "ticker": ticker, "shares": shares, "stop_loss": stop_loss}
                execute_trade(client, trade, portfolio_csv)
                positions, balance, orders = display_portfolio(client, portfolio_csv)
                
            elif command == 'sell' and len(parts) >= 3:
                ticker = parts[1].upper()
                shares = int(parts[2])
                
                trade = {"action": "sell", "ticker": ticker, "shares": shares}
                execute_trade(client, trade, portfolio_csv)
                positions, balance, orders = display_portfolio(client, portfolio_csv)
                
            elif command == 'sync':
                sync_csv_with_schwab(client, portfolio_csv)
                
            else:
                # Natural language chat - send any non-command input to LLM
                print("\n🤖 Thinking...")
                response = chat_with_llm(cmd, positions, balance, orders, model)
                print(f"\n{response}\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive Live Trading System")
    parser.add_argument("--model", default="rk-stockpicker", help="Azure OpenAI deployment name")
    parser.add_argument("--csv", help="Path to portfolio CSV file")
    
    args = parser.parse_args()
    
    run_interactive(model=args.model, csv_path=args.csv)


if __name__ == "__main__":
    main()
