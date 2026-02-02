"""
Schwab OAuth 2.0 Authentication Module
Handles the three-legged OAuth flow for Schwab Trader API
"""

import os
import json
import base64
import webbrowser
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()

# Schwab OAuth endpoints
SCHWAB_AUTH_URL = "https://api.schwabapi.com/v1/oauth/authorize"
SCHWAB_TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"

# Token storage file
TOKEN_FILE = Path(__file__).parent / ".schwab_tokens.json"


class SchwabAuth:
    """Handles Schwab OAuth authentication and token management"""
    
    def __init__(self):
        self.app_key = os.getenv("SCHWAB_APP_KEY")
        self.app_secret = os.getenv("SCHWAB_APP_SECRET")
        self.callback_url = os.getenv("SCHWAB_CALLBACK_URL", "https://127.0.0.1")
        
        if not self.app_key or not self.app_secret:
            raise ValueError("SCHWAB_APP_KEY and SCHWAB_APP_SECRET must be set in .env")
        
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        self.refresh_expires_at = None
        
        # Try to load existing tokens
        self._load_tokens()
    
    def _get_basic_auth_header(self) -> str:
        """Generate Base64 encoded Basic Auth header"""
        credentials = f"{self.app_key}:{self.app_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"
    
    def _save_tokens(self):
        """Save tokens to file for persistence"""
        token_data = {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_expires_at": self.token_expires_at.isoformat() if self.token_expires_at else None,
            "refresh_expires_at": self.refresh_expires_at.isoformat() if self.refresh_expires_at else None,
        }
        with open(TOKEN_FILE, "w") as f:
            json.dump(token_data, f, indent=2)
        print("💾 Tokens saved to .schwab_tokens.json")
    
    def _load_tokens(self):
        """Load tokens from file if they exist"""
        if TOKEN_FILE.exists():
            try:
                with open(TOKEN_FILE, "r") as f:
                    token_data = json.load(f)
                
                self.access_token = token_data.get("access_token")
                self.refresh_token = token_data.get("refresh_token")
                
                if token_data.get("token_expires_at"):
                    self.token_expires_at = datetime.fromisoformat(token_data["token_expires_at"])
                if token_data.get("refresh_expires_at"):
                    self.refresh_expires_at = datetime.fromisoformat(token_data["refresh_expires_at"])
                
                print("📂 Loaded existing tokens from .schwab_tokens.json")
                return True
            except Exception as e:
                print(f"⚠️ Could not load tokens: {e}")
        return False
    
    def get_authorization_url(self) -> str:
        """Generate the authorization URL for user login"""
        params = {
            "client_id": self.app_key,
            "redirect_uri": self.callback_url
        }
        url = f"{SCHWAB_AUTH_URL}?{urllib.parse.urlencode(params)}"
        return url
    
    def start_authorization(self):
        """
        Start the OAuth authorization flow.
        Opens browser for user to login and authorize.
        """
        auth_url = self.get_authorization_url()
        
        print("\n" + "="*60)
        print("🔐 SCHWAB AUTHORIZATION REQUIRED")
        print("="*60)
        print("\nOpening browser for Schwab login...")
        print("\nIf browser doesn't open, manually visit:")
        print(f"\n{auth_url}\n")
        print("="*60)
        print("\nAfter login, you'll be redirected to a page that may show 404.")
        print("That's expected! Copy the ENTIRE URL from your browser's address bar.")
        print("="*60)
        
        # Try to open browser
        try:
            webbrowser.open(auth_url)
        except Exception as e:
            print(f"Could not open browser: {e}")
        
        return auth_url
    
    def extract_auth_code(self, redirect_url: str) -> str:
        """
        Extract the authorization code from the redirect URL.
        The URL will look like: https://127.0.0.1/?code=AUTH_CODE&session=SESSION_ID
        """
        parsed = urllib.parse.urlparse(redirect_url)
        params = urllib.parse.parse_qs(parsed.query)
        
        if "code" not in params:
            raise ValueError("No authorization code found in URL. Make sure you copied the entire URL.")
        
        # URL decode the code (replace %40 with @, etc.)
        code = urllib.parse.unquote(params["code"][0])
        return code
    
    def exchange_code_for_tokens(self, auth_code: str) -> dict:
        """
        Exchange authorization code for access and refresh tokens.
        Step 2 of OAuth flow.
        """
        headers = {
            "Authorization": self._get_basic_auth_header(),
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": self.callback_url
        }
        
        print("\n🔄 Exchanging authorization code for tokens...")
        
        response = requests.post(SCHWAB_TOKEN_URL, headers=headers, data=data)
        
        if response.status_code != 200:
            print(f"❌ Token exchange failed: {response.status_code}")
            print(f"Response: {response.text}")
            raise Exception(f"Token exchange failed: {response.text}")
        
        token_data = response.json()
        
        # Store tokens
        self.access_token = token_data["access_token"]
        self.refresh_token = token_data["refresh_token"]
        
        # Calculate expiration times
        # Access token: 30 minutes, Refresh token: 7 days
        self.token_expires_at = datetime.now() + timedelta(seconds=token_data.get("expires_in", 1800))
        self.refresh_expires_at = datetime.now() + timedelta(days=7)
        
        # Save tokens to file
        self._save_tokens()
        
        print("✅ Successfully obtained tokens!")
        print(f"   Access token expires: {self.token_expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Refresh token expires: {self.refresh_expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return token_data
    
    def refresh_access_token(self) -> dict:
        """
        Refresh the access token using the refresh token.
        Step 4 of OAuth flow.
        """
        if not self.refresh_token:
            raise Exception("No refresh token available. Please re-authorize.")
        
        headers = {
            "Authorization": self._get_basic_auth_header(),
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token
        }
        
        print("\n🔄 Refreshing access token...")
        
        response = requests.post(SCHWAB_TOKEN_URL, headers=headers, data=data)
        
        if response.status_code != 200:
            print(f"❌ Token refresh failed: {response.status_code}")
            print(f"Response: {response.text}")
            # If refresh fails, tokens are likely expired
            self.access_token = None
            self.refresh_token = None
            raise Exception("Refresh token expired. Please re-authorize.")
        
        token_data = response.json()
        
        # Update tokens
        self.access_token = token_data["access_token"]
        self.refresh_token = token_data["refresh_token"]
        
        # Update expiration times
        self.token_expires_at = datetime.now() + timedelta(seconds=token_data.get("expires_in", 1800))
        self.refresh_expires_at = datetime.now() + timedelta(days=7)
        
        # Save updated tokens
        self._save_tokens()
        
        print("✅ Access token refreshed!")
        print(f"   New expiration: {self.token_expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return token_data
    
    def get_valid_access_token(self) -> str:
        """
        Get a valid access token, refreshing if necessary.
        This is the main method to call before making API requests.
        """
        # Check if we have tokens
        if not self.access_token:
            raise Exception("Not authenticated. Please run authorization flow first.")
        
        # Check if access token is expired (with 5 minute buffer)
        if self.token_expires_at and datetime.now() >= (self.token_expires_at - timedelta(minutes=5)):
            print("⚠️ Access token expired or expiring soon...")
            
            # Check if refresh token is still valid
            if self.refresh_expires_at and datetime.now() >= self.refresh_expires_at:
                raise Exception("Refresh token expired. Please re-authorize.")
            
            # Refresh the token
            self.refresh_access_token()
        
        return self.access_token
    
    def is_authenticated(self) -> bool:
        """Check if we have valid authentication"""
        if not self.access_token or not self.refresh_token:
            return False
        
        # Check if refresh token is expired
        if self.refresh_expires_at and datetime.now() >= self.refresh_expires_at:
            return False
        
        return True
    
    def get_auth_status(self) -> dict:
        """Get current authentication status"""
        return {
            "authenticated": self.is_authenticated(),
            "has_access_token": bool(self.access_token),
            "has_refresh_token": bool(self.refresh_token),
            "access_token_expires": self.token_expires_at.strftime('%Y-%m-%d %H:%M:%S') if self.token_expires_at else None,
            "refresh_token_expires": self.refresh_expires_at.strftime('%Y-%m-%d %H:%M:%S') if self.refresh_expires_at else None,
        }


def interactive_authorization():
    """
    Interactive authorization flow for command line use.
    """
    auth = SchwabAuth()
    
    # Check if already authenticated
    if auth.is_authenticated():
        print("\n✅ Already authenticated with Schwab!")
        status = auth.get_auth_status()
        print(f"   Access token expires: {status['access_token_expires']}")
        print(f"   Refresh token expires: {status['refresh_token_expires']}")
        
        reauth = input("\nDo you want to re-authorize anyway? (y/n): ").strip().lower()
        if reauth != 'y':
            return auth
    
    # Start authorization
    auth.start_authorization()
    
    # Wait for user to paste the redirect URL
    print("\n")
    redirect_url = input("Paste the redirect URL here: ").strip()
    
    # Extract code and exchange for tokens
    try:
        auth_code = auth.extract_auth_code(redirect_url)
        print(f"\n📝 Authorization code extracted: {auth_code[:20]}...")
        
        auth.exchange_code_for_tokens(auth_code)
        print("\n🎉 Schwab authorization complete!")
        
    except Exception as e:
        print(f"\n❌ Authorization failed: {e}")
        raise
    
    return auth


if __name__ == "__main__":
    # Run interactive authorization when script is run directly
    auth = interactive_authorization()
    
    if auth.is_authenticated():
        print("\n" + "="*60)
        print("✅ You're now authenticated with Schwab!")
        print("You can now use the trading bot with real account access.")
        print("="*60)
