"""Quick script to check Schwab orders"""
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timedelta
from schwab_client import SchwabClient

client = SchwabClient()
print("\n📋 Checking Schwab Orders...\n")

# Get all orders with proper date range
from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%dT00:00:00.000Z')
to_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%dT23:59:59.000Z')
orders = client.get_orders(max_results=20, from_date=from_date, to_date=to_date)
print(f"Found {len(orders)} orders:\n")

for order in orders:
    order_id = order.get('orderId', 'N/A')
    status = order.get('status', 'N/A')
    order_type = order.get('orderType', 'N/A')
    stop_price = order.get('stopPrice', None)
    price = order.get('price', None)
    entered = order.get('enteredTime', 'N/A')
    
    legs = order.get('orderLegCollection', [])
    for leg in legs:
        symbol = leg.get('instrument', {}).get('symbol', 'N/A')
        qty = leg.get('quantity', 0)
        instruction = leg.get('instruction', 'N/A')
        
        price_str = f"Stop @ ${stop_price}" if stop_price else (f"@ ${price}" if price else "MARKET")
        print(f"  [{status}] {order_type} {instruction} {qty} {symbol} {price_str}")
        print(f"           Order ID: {order_id}")
        print(f"           Entered: {entered}")
        print()

# Also show positions
print("\n📈 Current Positions:\n")
positions = client.get_positions()
for pos in positions:
    symbol = pos.get("instrument", {}).get("symbol", "N/A")
    qty = pos.get("longQuantity", 0) or pos.get("shortQuantity", 0)
    avg_price = pos.get("averagePrice", 0)
    market_value = pos.get("marketValue", 0)
    print(f"  {symbol}: {qty} shares @ ${avg_price:.4f} (value: ${market_value:.2f})")

# Show balance
print("\n💰 Account Balance:")
balance = client.get_balance()
print(f"  Cash Available: ${balance.get('cash_available', 0):.2f}")
print(f"  Account Value: ${balance.get('account_value', 0):.2f}")
