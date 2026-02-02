"""
Schwab Trader API Client
Provides methods to interact with Schwab's trading API
"""

import os
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
import requests
from schwab_auth import SchwabAuth

# Schwab API base URL
SCHWAB_API_BASE = "https://api.schwabapi.com/trader/v1"


class SchwabClient:
    """
    Client for Schwab Trader API.
    Handles account info, positions, orders, and quotes.
    """
    
    def __init__(self, auth: Optional[SchwabAuth] = None):
        """Initialize with optional auth object"""
        self.auth = auth or SchwabAuth()
        self._account_hash = None  # Cache for account hash
        
        # Try to load cached account hash immediately
        self._load_cached_account()
    
    def _load_cached_account(self):
        """Load cached account info if available"""
        try:
            import json
            from pathlib import Path
            token_file = Path(__file__).parent / ".schwab_account.json"
            if token_file.exists():
                with open(token_file) as f:
                    data = json.load(f)
                    self._account_hash = data.get("account_hash")
        except:
            pass
    
    def _get_headers(self, include_content_type: bool = False) -> dict:
        """Get headers with valid access token"""
        token = self.auth.get_valid_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
        # Only include Content-Type for POST/PUT requests (Schwab API may reject GET with it)
        if include_content_type:
            headers["Content-Type"] = "application/json"
        return headers
    
    def _make_request(self, method: str, endpoint: str, data: dict = None, params: dict = None, retries: int = 3) -> dict:
        """Make an authenticated request to Schwab API with retry logic"""
        import time
        
        url = f"{SCHWAB_API_BASE}{endpoint}"
        # Only include Content-Type for requests with body
        include_content_type = method.upper() in ["POST", "PUT"]
        headers = self._get_headers(include_content_type=include_content_type)
        
        for attempt in range(retries):
            try:
                if method.upper() == "GET":
                    response = requests.get(url, headers=headers, params=params)
                elif method.upper() == "POST":
                    response = requests.post(url, headers=headers, json=data)
                elif method.upper() == "PUT":
                    response = requests.put(url, headers=headers, json=data)
                elif method.upper() == "DELETE":
                    response = requests.delete(url, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Handle different response codes
                if response.status_code == 200:
                    return response.json() if response.text else {}
                elif response.status_code == 201:
                    # Order created successfully
                    return {"success": True, "location": response.headers.get("Location")}
                elif response.status_code == 204:
                    # No content (successful delete)
                    return {"success": True}
                elif response.status_code == 500 or (response.status_code == 400 and "500" in response.text):
                    # Internal server error - retry
                    if attempt < retries - 1:
                        wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
                        time.sleep(wait_time)
                        continue
                    else:
                        return {"error": "Schwab API temporarily unavailable", "status_code": 500}
                else:
                    error_msg = f"API Error {response.status_code}: {response.text}"
                    return {"error": error_msg, "status_code": response.status_code}
                    
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}
    
    # ============ Account Methods ============
    
    def get_accounts(self) -> List[dict]:
        """Get all linked accounts"""
        result = self._make_request("GET", "/accounts")
        if isinstance(result, list):
            return result
        # Handle case where result might be wrapped
        if isinstance(result, dict) and "securitiesAccount" in result:
            return [result]
        return result.get("accounts", []) if isinstance(result, dict) else []
    
    def get_account_numbers(self) -> List[dict]:
        """Get account numbers and hashes"""
        result = self._make_request("GET", "/accounts/accountNumbers")
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "error" in result:
            # Try to use cached account info
            print("⚠️ Could not fetch accounts, trying cached data...")
            return []
        return []
    
    def get_account_hash(self) -> str:
        """Get the primary account hash (cached)"""
        if self._account_hash:
            return self._account_hash
        
        accounts = self.get_account_numbers()
        if accounts:
            self._account_hash = accounts[0].get("hashValue")
            return self._account_hash
        
        # Fallback: try to read from token file
        try:
            import json
            from pathlib import Path
            token_file = Path(__file__).parent / ".schwab_account.json"
            if token_file.exists():
                with open(token_file) as f:
                    data = json.load(f)
                    self._account_hash = data.get("account_hash")
                    if self._account_hash:
                        print(f"📂 Loaded cached account hash")
                        return self._account_hash
        except:
            pass
        
        raise Exception("No accounts found. Schwab API may be temporarily unavailable.")
    
    def save_account_info(self, account_number: str, account_hash: str):
        """Save account info for later use"""
        import json
        from pathlib import Path
        token_file = Path(__file__).parent / ".schwab_account.json"
        with open(token_file, "w") as f:
            json.dump({"account_number": account_number, "account_hash": account_hash}, f)
    
    def get_account_details(self, account_hash: str = None, include_positions: bool = True) -> dict:
        """Get detailed account information including positions"""
        try:
            account_hash = account_hash or self.get_account_hash()
        except Exception as e:
            return {"error": str(e)}
        
        params = {"fields": "positions"} if include_positions else {}
        return self._make_request("GET", f"/accounts/{account_hash}", params=params)
    
    def get_positions(self, account_hash: str = None) -> List[dict]:
        """Get current positions in the account"""
        account = self.get_account_details(account_hash, include_positions=True)
        if "error" in account:
            return []
        
        positions = account.get("securitiesAccount", {}).get("positions", [])
        return positions
    
    def get_balance(self, account_hash: str = None) -> dict:
        """Get account balance information"""
        try:
            account_hash = account_hash or self.get_account_hash()
        except Exception as e:
            return {"error": str(e)}
        
        account = self.get_account_details(account_hash, include_positions=False)
        if "error" in account:
            return account
        
        balances = account.get("securitiesAccount", {}).get("currentBalances", {})
        return {
            "cash_available": balances.get("cashAvailableForTrading", 0),
            "cash_balance": balances.get("cashBalance", 0),
            "account_value": balances.get("liquidationValue", 0),
            "buying_power": balances.get("buyingPower", balances.get("cashAvailableForTrading", 0)),
        }
    
    # ============ Order Methods ============
    
    def get_orders(self, account_hash: str = None, max_results: int = 100, 
                   from_date: str = None, to_date: str = None, status: str = None) -> List[dict]:
        """Get orders for an account"""
        account_hash = account_hash or self.get_account_hash()
        
        params = {"maxResults": max_results}
        if from_date:
            params["fromEnteredTime"] = from_date
        if to_date:
            params["toEnteredTime"] = to_date
        if status:
            params["status"] = status
        
        result = self._make_request("GET", f"/accounts/{account_hash}/orders", params=params)
        return result if isinstance(result, list) else []
    
    def place_order(self, order: dict, account_hash: str = None) -> dict:
        """Place an order"""
        account_hash = account_hash or self.get_account_hash()
        return self._make_request("POST", f"/accounts/{account_hash}/orders", data=order)
    
    def cancel_order(self, order_id: str, account_hash: str = None) -> dict:
        """Cancel an order"""
        account_hash = account_hash or self.get_account_hash()
        return self._make_request("DELETE", f"/accounts/{account_hash}/orders/{order_id}")
    
    def get_order(self, order_id: str, account_hash: str = None) -> dict:
        """Get a specific order"""
        account_hash = account_hash or self.get_account_hash()
        return self._make_request("GET", f"/accounts/{account_hash}/orders/{order_id}")
    
    # ============ Order Builder Methods ============
    
    def build_market_order(self, symbol: str, quantity: int, instruction: str = "BUY") -> dict:
        """
        Build a market order.
        instruction: BUY, SELL, BUY_TO_COVER, SELL_SHORT
        """
        return {
            "orderType": "MARKET",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [{
                "instruction": instruction,
                "quantity": quantity,
                "instrument": {
                    "symbol": symbol.upper(),
                    "assetType": "EQUITY"
                }
            }]
        }
    
    def build_limit_order(self, symbol: str, quantity: int, limit_price: float, 
                          instruction: str = "BUY", duration: str = "DAY") -> dict:
        """
        Build a limit order.
        instruction: BUY, SELL, BUY_TO_COVER, SELL_SHORT
        duration: DAY, GOOD_TILL_CANCEL
        """
        return {
            "orderType": "LIMIT",
            "session": "NORMAL",
            "price": str(limit_price),
            "duration": duration,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [{
                "instruction": instruction,
                "quantity": quantity,
                "instrument": {
                    "symbol": symbol.upper(),
                    "assetType": "EQUITY"
                }
            }]
        }
    
    def build_stop_order(self, symbol: str, quantity: int, stop_price: float,
                         instruction: str = "SELL") -> dict:
        """Build a stop order (typically for stop-loss)"""
        return {
            "orderType": "STOP",
            "session": "NORMAL",
            "stopPrice": str(stop_price),
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [{
                "instruction": instruction,
                "quantity": quantity,
                "instrument": {
                    "symbol": symbol.upper(),
                    "assetType": "EQUITY"
                }
            }]
        }
    
    def build_stop_limit_order(self, symbol: str, quantity: int, stop_price: float,
                               limit_price: float, instruction: str = "SELL") -> dict:
        """Build a stop-limit order"""
        return {
            "orderType": "STOP_LIMIT",
            "session": "NORMAL",
            "price": str(limit_price),
            "stopPrice": str(stop_price),
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [{
                "instruction": instruction,
                "quantity": quantity,
                "instrument": {
                    "symbol": symbol.upper(),
                    "assetType": "EQUITY"
                }
            }]
        }
    
    # ============ Convenience Trading Methods ============
    
    def buy_market(self, symbol: str, quantity: int, dry_run: bool = True) -> dict:
        """Buy shares at market price"""
        order = self.build_market_order(symbol, quantity, "BUY")
        
        if dry_run:
            print(f"🔶 DRY RUN: Would buy {quantity} shares of {symbol} at market")
            return {"dry_run": True, "order": order}
        
        print(f"🛒 Placing BUY order: {quantity} shares of {symbol} at market")
        return self.place_order(order)
    
    def sell_market(self, symbol: str, quantity: int, dry_run: bool = True) -> dict:
        """Sell shares at market price"""
        order = self.build_market_order(symbol, quantity, "SELL")
        
        if dry_run:
            print(f"🔶 DRY RUN: Would sell {quantity} shares of {symbol} at market")
            return {"dry_run": True, "order": order}
        
        print(f"💰 Placing SELL order: {quantity} shares of {symbol} at market")
        return self.place_order(order)
    
    def buy_limit(self, symbol: str, quantity: int, limit_price: float, dry_run: bool = True) -> dict:
        """Buy shares at limit price"""
        order = self.build_limit_order(symbol, quantity, limit_price, "BUY")
        
        if dry_run:
            print(f"🔶 DRY RUN: Would buy {quantity} shares of {symbol} at ${limit_price:.2f}")
            return {"dry_run": True, "order": order}
        
        print(f"🛒 Placing LIMIT BUY: {quantity} shares of {symbol} at ${limit_price:.2f}")
        return self.place_order(order)
    
    def sell_limit(self, symbol: str, quantity: int, limit_price: float, dry_run: bool = True) -> dict:
        """Sell shares at limit price"""
        order = self.build_limit_order(symbol, quantity, limit_price, "SELL")
        
        if dry_run:
            print(f"🔶 DRY RUN: Would sell {quantity} shares of {symbol} at ${limit_price:.2f}")
            return {"dry_run": True, "order": order}
        
        print(f"💰 Placing LIMIT SELL: {quantity} shares of {symbol} at ${limit_price:.2f}")
        return self.place_order(order)
    
    def set_stop_loss(self, symbol: str, quantity: int, stop_price: float, dry_run: bool = True) -> dict:
        """Set a stop-loss order"""
        order = self.build_stop_order(symbol, quantity, stop_price, "SELL")
        
        if dry_run:
            print(f"🔶 DRY RUN: Would set stop-loss for {quantity} shares of {symbol} at ${stop_price:.2f}")
            return {"dry_run": True, "order": order}
        
        print(f"🛑 Setting STOP LOSS: {quantity} shares of {symbol} at ${stop_price:.2f}")
        return self.place_order(order)
    
    # ============ Display Methods ============
    
    def display_account_summary(self):
        """Display a summary of the account"""
        print("\n" + "="*60)
        print("📊 SCHWAB ACCOUNT SUMMARY")
        print("="*60)
        
        # Get balance
        balance = self.get_balance()
        if "error" not in balance:
            print(f"\n💵 Account Value: ${balance['account_value']:,.2f}")
            print(f"💰 Cash Available: ${balance['cash_available']:,.2f}")
            print(f"🏦 Cash Balance: ${balance['cash_balance']:,.2f}")
            print(f"💪 Buying Power: ${balance['buying_power']:,.2f}")
        else:
            print(f"❌ Could not fetch balance: {balance['error']}")
        
        # Get positions
        positions = self.get_positions()
        if positions:
            print(f"\n📈 POSITIONS ({len(positions)} total)")
            print("-"*50)
            for pos in positions:
                symbol = pos.get("instrument", {}).get("symbol", "N/A")
                qty = pos.get("longQuantity", 0) or pos.get("shortQuantity", 0)
                avg_price = pos.get("averagePrice", 0)
                market_value = pos.get("marketValue", 0)
                
                print(f"  {symbol}: {qty} shares @ ${avg_price:.2f} = ${market_value:,.2f}")
        else:
            print("\n📈 No positions")
        
        print("="*60)
    
    def display_orders(self, status: str = None):
        """Display recent orders"""
        orders = self.get_orders(status=status)
        
        print("\n" + "="*60)
        print(f"📋 ORDERS" + (f" ({status})" if status else ""))
        print("="*60)
        
        if not orders:
            print("No orders found")
            return
        
        for order in orders[:10]:  # Show last 10
            order_id = order.get("orderId", "N/A")
            status = order.get("status", "N/A")
            order_type = order.get("orderType", "N/A")
            
            legs = order.get("orderLegCollection", [])
            if legs:
                leg = legs[0]
                symbol = leg.get("instrument", {}).get("symbol", "N/A")
                qty = leg.get("quantity", 0)
                instruction = leg.get("instruction", "N/A")
                
                price = order.get("price", order.get("stopPrice", "MKT"))
                print(f"  [{status}] {instruction} {qty} {symbol} @ {price} (ID: {order_id})")
        
        print("="*60)
    
    def verify_trade(self, ticker: str, expected_action: str, expected_shares: int, 
                     timeout_seconds: int = 15) -> dict:
        """
        Verify a trade was executed by checking orders and positions.
        Retries up to timeout_seconds to allow order processing.
        Returns verification status and details.
        """
        import time
        
        result = {
            "verified": False,
            "order_status": None,
            "position_found": False,
            "position_shares": 0,
            "cash_balance": 0,
            "message": ""
        }
        
        ticker = ticker.upper()
        expected_action = expected_action.upper()
        
        # Retry loop - check every 3 seconds up to timeout
        max_attempts = timeout_seconds // 3
        for attempt in range(max_attempts):
            if attempt > 0:
                print(f"   ⏳ Waiting for order to process... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(3)
            
            try:
                # Check recent orders for this ticker - need date range for Schwab API
                from datetime import datetime, timedelta
                from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%dT00:00:00.000Z')
                to_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%dT23:59:59.000Z')
                orders = self.get_orders(max_results=20, from_date=from_date, to_date=to_date)
                matching_orders = []
                
                for order in orders:
                    legs = order.get("orderLegCollection", [])
                    for leg in legs:
                        symbol = leg.get("instrument", {}).get("symbol", "").upper()
                        instruction = leg.get("instruction", "").upper()
                        qty = leg.get("quantity", 0)
                        
                        # Check if this matches our trade
                        if symbol == ticker:
                            order_status = order.get("status", "UNKNOWN")
                            matching_orders.append({
                                "order_id": order.get("orderId"),
                                "status": order_status,
                                "instruction": instruction,
                                "quantity": qty,
                                "price": order.get("price", order.get("stopPrice")),
                                "filled_qty": order.get("filledQuantity", 0),
                                "status_description": order.get("statusDescription", "")
                            })
                
                if matching_orders:
                    latest = matching_orders[0]  # Most recent
                    result["order_status"] = latest["status"]
                    
                    # FILLED is what we want - order is complete
                    if latest["status"] == "FILLED":
                        result["verified"] = True
                        break  # Success! Stop retrying
                    elif latest["status"] in ["WORKING", "PENDING_ACTIVATION", "QUEUED", "ACCEPTED", "AWAITING_PARENT_ORDER"]:
                        # Order is in progress - might fill soon
                        result["verified"] = True
                        if attempt < max_attempts - 1:
                            continue  # Keep checking for FILLED
                        break
                    elif latest["status"] in ["CANCELED", "REJECTED", "EXPIRED"]:
                        result["verified"] = False
                        # Include the rejection reason
                        reason = latest.get("status_description", "")
                        if reason:
                            # Truncate long reasons
                            reason = reason[:150] + "..." if len(reason) > 150 else reason
                            result["message"] = f"❌ Order {latest['status']}: {reason}"
                        else:
                            result["message"] = f"❌ Order {latest['status']}"
                        break
                else:
                    # No matching order found yet - keep trying
                    if attempt < max_attempts - 1:
                        continue
                        
            except Exception as e:
                result["message"] = f"⚠️ Verification error: {e}"
                break
        
        try:
            # Check positions
            positions = self.get_positions()
            for pos in positions:
                symbol = pos.get("instrument", {}).get("symbol", "").upper()
                if symbol == ticker:
                    result["position_found"] = True
                    result["position_shares"] = pos.get("longQuantity", 0) or pos.get("shortQuantity", 0)
                    break
            
            # Check balance
            balance = self.get_balance()
            result["cash_balance"] = balance.get("cash_available", 0)
            
            # Build message
            if result["verified"]:
                if result["order_status"] == "FILLED":
                    result["message"] = f"✅ VERIFIED: Order FILLED"
                    if result["position_found"]:
                        result["message"] += f" | Position: {result['position_shares']} shares"
                else:
                    result["message"] = f"⏳ Order {result['order_status']} (may be pending fill)"
            else:
                if result.get("order_status"):
                    result["message"] = f"⚠️ Order found with status: {result['order_status']}"
                else:
                    result["message"] = f"❌ No matching order found for {ticker}"
            
            result["message"] += f" | Cash: ${result['cash_balance']:.2f}"
            
        except Exception as e:
            if not result["message"]:
                result["message"] = f"⚠️ Verification error: {e}"
        
        return result


# ============ Quick Test ============

if __name__ == "__main__":
    print("Testing Schwab Client...")
    
    try:
        client = SchwabClient()
        
        if not client.auth.is_authenticated():
            print("\n⚠️ Not authenticated. Please run schwab_auth.py first.")
        else:
            print("\n✅ Authenticated! Fetching account info...")
            client.display_account_summary()
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nRun 'python schwab_auth.py' to authenticate first.")
