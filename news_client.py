"""
News and Sentiment Client for Stock Trading

Fetches market news and sentiment data from Finnhub API.
Free tier: 60 calls/minute

Setup:
1. Get free API key at https://finnhub.io/register
2. Add to .env: FINNHUB_API_KEY=your-key-here
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Any
from dotenv import load_dotenv

load_dotenv()


class NewsClient:
    """Fetch stock news and sentiment from Finnhub."""
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            print("⚠️ FINNHUB_API_KEY not set. News features disabled.")
            print("   Get free key at: https://finnhub.io/register")
        
    def _make_request(self, endpoint: str, params: dict = None) -> dict | list:
        """Make API request to Finnhub."""
        if not self.api_key:
            return {"error": "API key not configured"}
        
        params = params or {}
        params["token"] = self.api_key
        
        try:
            response = requests.get(
                f"{self.BASE_URL}/{endpoint}",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def get_company_news(self, symbol: str, days_back: int = 7) -> list[dict]:
        """
        Get recent news for a specific company.
        
        Args:
            symbol: Stock ticker (e.g., "AAPL")
            days_back: How many days of news to fetch
            
        Returns:
            List of news articles with headline, summary, source, url, datetime
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        result = self._make_request("company-news", {
            "symbol": symbol.upper(),
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d")
        })
        
        if isinstance(result, dict) and "error" in result:
            return []
        
        # Format and limit results
        news = []
        for article in result[:10]:  # Limit to 10 articles
            news.append({
                "headline": article.get("headline", ""),
                "summary": article.get("summary", "")[:300],  # Truncate
                "source": article.get("source", ""),
                "url": article.get("url", ""),
                "datetime": datetime.fromtimestamp(article.get("datetime", 0)).strftime("%Y-%m-%d %H:%M"),
                "sentiment": self._simple_sentiment(article.get("headline", "") + " " + article.get("summary", ""))
            })
        
        return news
    
    def get_market_news(self, category: str = "general") -> list[dict]:
        """
        Get general market news.
        
        Args:
            category: "general", "forex", "crypto", "merger"
            
        Returns:
            List of market news articles
        """
        result = self._make_request("news", {"category": category})
        
        if isinstance(result, dict) and "error" in result:
            return []
        
        news = []
        for article in result[:10]:
            news.append({
                "headline": article.get("headline", ""),
                "summary": article.get("summary", "")[:300],
                "source": article.get("source", ""),
                "url": article.get("url", ""),
                "datetime": datetime.fromtimestamp(article.get("datetime", 0)).strftime("%Y-%m-%d %H:%M"),
                "category": article.get("category", ""),
            })
        
        return news
    
    def get_sentiment(self, symbol: str) -> dict:
        """
        Get social sentiment for a stock (Reddit, Twitter mentions).
        
        Returns:
            Dict with buzz score, sentiment scores, etc.
        """
        result = self._make_request("stock/social-sentiment", {"symbol": symbol.upper()})
        
        if isinstance(result, dict) and "error" in result:
            return {"symbol": symbol, "error": result["error"]}
        
        # Aggregate Reddit + Twitter data
        reddit = result.get("reddit", [])
        twitter = result.get("twitter", [])
        
        sentiment = {
            "symbol": symbol.upper(),
            "reddit_mentions": sum(r.get("mention", 0) for r in reddit[-7:]),  # Last 7 days
            "reddit_positive": sum(r.get("positiveScore", 0) for r in reddit[-7:]),
            "reddit_negative": sum(r.get("negativeScore", 0) for r in reddit[-7:]),
            "twitter_mentions": sum(t.get("mention", 0) for t in twitter[-7:]),
            "twitter_positive": sum(t.get("positiveScore", 0) for t in twitter[-7:]),
            "twitter_negative": sum(t.get("negativeScore", 0) for t in twitter[-7:]),
        }
        
        # Calculate overall sentiment
        total_positive = sentiment["reddit_positive"] + sentiment["twitter_positive"]
        total_negative = sentiment["reddit_negative"] + sentiment["twitter_negative"]
        total = total_positive + total_negative
        
        if total > 0:
            sentiment["overall_score"] = (total_positive - total_negative) / total
            sentiment["overall_label"] = "Bullish" if sentiment["overall_score"] > 0.1 else "Bearish" if sentiment["overall_score"] < -0.1 else "Neutral"
        else:
            sentiment["overall_score"] = 0
            sentiment["overall_label"] = "No data"
        
        return sentiment
    
    def get_basic_financials(self, symbol: str) -> dict:
        """
        Get basic financial metrics for a stock.
        
        Returns:
            Dict with P/E, market cap, 52-week high/low, etc.
        """
        result = self._make_request("stock/metric", {
            "symbol": symbol.upper(),
            "metric": "all"
        })
        
        if isinstance(result, dict) and "error" in result:
            return {"symbol": symbol, "error": result["error"]}
        
        metrics = result.get("metric", {})
        
        return {
            "symbol": symbol.upper(),
            "market_cap": metrics.get("marketCapitalization"),
            "pe_ratio": metrics.get("peBasicExclExtraTTM"),
            "pb_ratio": metrics.get("pbAnnual"),
            "52_week_high": metrics.get("52WeekHigh"),
            "52_week_low": metrics.get("52WeekLow"),
            "52_week_return": metrics.get("52WeekPriceReturnDaily"),
            "beta": metrics.get("beta"),
            "dividend_yield": metrics.get("dividendYieldIndicatedAnnual"),
            "eps_ttm": metrics.get("epsBasicExclExtraItemsTTM"),
            "revenue_growth": metrics.get("revenueGrowthTTMYoy"),
        }
    
    def get_recommendation_trends(self, symbol: str) -> dict:
        """
        Get analyst recommendation trends.
        
        Returns:
            Dict with buy/hold/sell counts from analysts
        """
        result = self._make_request("stock/recommendation", {"symbol": symbol.upper()})
        
        if isinstance(result, list) and len(result) > 0:
            latest = result[0]
            return {
                "symbol": symbol.upper(),
                "period": latest.get("period"),
                "strong_buy": latest.get("strongBuy", 0),
                "buy": latest.get("buy", 0),
                "hold": latest.get("hold", 0),
                "sell": latest.get("sell", 0),
                "strong_sell": latest.get("strongSell", 0),
            }
        
        return {"symbol": symbol, "error": "No recommendations found"}
    
    def _simple_sentiment(self, text: str) -> str:
        """Simple keyword-based sentiment (fallback when API sentiment unavailable)."""
        text = text.lower()
        
        positive = ["surge", "jump", "gain", "rise", "bullish", "growth", "profit", 
                   "beat", "exceed", "strong", "rally", "soar", "upgrade", "buy"]
        negative = ["fall", "drop", "decline", "bearish", "loss", "miss", "weak",
                   "crash", "plunge", "downgrade", "sell", "cut", "warning", "risk"]
        
        pos_count = sum(1 for word in positive if word in text)
        neg_count = sum(1 for word in negative if word in text)
        
        if pos_count > neg_count:
            return "Positive"
        elif neg_count > pos_count:
            return "Negative"
        return "Neutral"
    
    def get_stock_summary(self, symbol: str, current_price: float = None) -> str:
        """
        Get a comprehensive summary for AI context.
        Combines news, sentiment, financials, and tradability into a formatted string.
        """
        symbol = symbol.upper()
        summary_parts = [f"\n📰 NEWS & SENTIMENT FOR {symbol}"]
        summary_parts.append("=" * 40)
        
        # Check tradability first
        tradability = self.check_tradability(symbol, current_price)
        if tradability.get("warnings"):
            summary_parts.append("\n⚠️ TRADABILITY WARNINGS:")
            for warning in tradability["warnings"]:
                summary_parts.append(f"   {warning}")
            if not tradability.get("tradeable"):
                summary_parts.append("   🚫 THIS STOCK MAY NOT BE TRADEABLE VIA API - CONSIDER ALTERNATIVES")
        
        # Add exchange info
        if tradability.get("profile"):
            profile = tradability["profile"]
            summary_parts.append(f"\n📋 Company Info:")
            summary_parts.append(f"   Exchange: {profile.get('exchange', 'Unknown')}")
            summary_parts.append(f"   Country: {profile.get('country', 'Unknown')}")
            if profile.get('industry'):
                summary_parts.append(f"   Industry: {profile['industry']}")
        
        # Get news
        news = self.get_company_news(symbol, days_back=3)
        if news:
            summary_parts.append(f"\n📰 Recent Headlines ({len(news)} articles):")
            for i, article in enumerate(news[:5], 1):
                sentiment_emoji = "📈" if article["sentiment"] == "Positive" else "📉" if article["sentiment"] == "Negative" else "➖"
                summary_parts.append(f"  {i}. {sentiment_emoji} {article['headline'][:80]}")
                summary_parts.append(f"     Source: {article['source']} | {article['datetime']}")
        else:
            summary_parts.append("\n📰 No recent news found")
        
        # Get sentiment
        sentiment = self.get_sentiment(symbol)
        if "error" not in sentiment:
            summary_parts.append(f"\n💬 Social Sentiment: {sentiment['overall_label']} (score: {sentiment['overall_score']:.2f})")
            summary_parts.append(f"   Reddit: {sentiment['reddit_mentions']} mentions | Twitter: {sentiment['twitter_mentions']} mentions")
        
        # Get financials
        financials = self.get_basic_financials(symbol)
        if "error" not in financials and financials.get("market_cap"):
            summary_parts.append(f"\n📊 Key Metrics:")
            if financials.get("market_cap"):
                mc = financials["market_cap"]
                mc_str = f"${mc/1000:.1f}B" if mc > 1000 else f"${mc:.0f}M"
                summary_parts.append(f"   Market Cap: {mc_str}")
            if financials.get("pe_ratio"):
                summary_parts.append(f"   P/E Ratio: {financials['pe_ratio']:.1f}")
            if financials.get("52_week_high") and financials.get("52_week_low"):
                summary_parts.append(f"   52-Week Range: ${financials['52_week_low']:.2f} - ${financials['52_week_high']:.2f}")
            if financials.get("beta"):
                summary_parts.append(f"   Beta: {financials['beta']:.2f}")
        
        # Get analyst recommendations
        recs = self.get_recommendation_trends(symbol)
        if "error" not in recs:
            total = recs["strong_buy"] + recs["buy"] + recs["hold"] + recs["sell"] + recs["strong_sell"]
            if total > 0:
                summary_parts.append(f"\n🎯 Analyst Ratings ({total} analysts):")
                summary_parts.append(f"   Strong Buy: {recs['strong_buy']} | Buy: {recs['buy']} | Hold: {recs['hold']} | Sell: {recs['sell']} | Strong Sell: {recs['strong_sell']}")
        
        return "\n".join(summary_parts)
    
    def get_market_overview(self) -> str:
        """Get general market news for context."""
        summary_parts = ["\n🌍 MARKET NEWS OVERVIEW"]
        summary_parts.append("=" * 40)
        
        news = self.get_market_news("general")
        if news:
            for i, article in enumerate(news[:5], 1):
                summary_parts.append(f"\n{i}. {article['headline'][:100]}")
                summary_parts.append(f"   Source: {article['source']} | {article['datetime']}")
        else:
            summary_parts.append("No market news available")
        
        return "\n".join(summary_parts)

    def get_company_profile(self, symbol: str) -> dict:
        """
        Get company profile including exchange, industry, and IPO date.
        Useful for checking if a stock is tradeable.
        """
        result = self._make_request("stock/profile2", {"symbol": symbol.upper()})
        
        if isinstance(result, dict) and "error" in result:
            return {"symbol": symbol, "error": result["error"]}
        
        if not result or not result.get("name"):
            return {"symbol": symbol, "error": "Company not found"}
        
        return {
            "symbol": symbol.upper(),
            "name": result.get("name"),
            "exchange": result.get("exchange"),
            "industry": result.get("finnhubIndustry"),
            "country": result.get("country"),
            "ipo_date": result.get("ipo"),
            "market_cap": result.get("marketCapitalization"),
            "shares_outstanding": result.get("shareOutstanding"),
            "website": result.get("weburl"),
        }
    
    def check_tradability(self, symbol: str, current_price: float = None) -> dict:
        """
        Check if a stock is likely tradeable via API (not broker-assisted).
        
        Returns:
            Dict with tradeable status and any warnings
        """
        warnings = []
        is_tradeable = True
        
        # Get company profile for exchange info
        profile = self.get_company_profile(symbol)
        
        if "error" in profile:
            warnings.append(f"Could not verify company profile: {profile['error']}")
        else:
            # Check exchange - prefer major US exchanges
            exchange = profile.get("exchange", "").upper()
            major_exchanges = ["NASDAQ", "NYSE", "NEW YORK STOCK EXCHANGE", "AMEX", "ARCA", "AMERICAN"]
            otc_exchanges = ["OTC", "PINK", "GREY"]
            
            is_major = any(ex in exchange for ex in major_exchanges)
            is_otc = any(otc in exchange for otc in otc_exchanges)
            
            if is_otc:
                warnings.append(f"⚠️ OTC/Pink Sheet stock ({exchange}) - may require broker-assisted trade")
                is_tradeable = False
            elif not is_major:
                warnings.append(f"⚠️ Non-major exchange ({exchange}) - verify tradability")
            
            # Check country - foreign stocks may have restrictions
            country = profile.get("country", "").upper()
            if country and country not in ["US", "USA", "UNITED STATES"]:
                if country in ["CN", "CHINA"]:
                    warnings.append(f"⚠️ Chinese company - may have ADR restrictions")
                else:
                    warnings.append(f"ℹ️ Foreign company ({country}) - check for ADR restrictions")
            
            # Check IPO date - recent IPOs may have restrictions
            ipo_date = profile.get("ipo_date")
            if ipo_date:
                try:
                    from datetime import datetime
                    ipo = datetime.strptime(ipo_date, "%Y-%m-%d")
                    days_since_ipo = (datetime.now() - ipo).days
                    if days_since_ipo < 30:
                        warnings.append(f"⚠️ Recent IPO ({days_since_ipo} days ago) - may have settlement restrictions")
                except:
                    pass
            
            # Check market cap - very small caps often have restrictions
            market_cap = profile.get("market_cap")
            if market_cap and market_cap < 10:  # Under $10M
                warnings.append(f"⚠️ Very low market cap (${market_cap}M) - may require cleared funds")
        
        # Check price if provided
        if current_price is not None:
            if current_price < 1.0:
                warnings.append(f"⚠️ Penny stock (${current_price:.2f}) - requires fully cleared funds, may need broker")
                is_tradeable = False
            elif current_price < 2.0:
                warnings.append(f"ℹ️ Low-priced stock (${current_price:.2f}) - may have extra restrictions")
        
        return {
            "symbol": symbol.upper(),
            "tradeable": is_tradeable and len([w for w in warnings if "⚠️" in w]) == 0,
            "warnings": warnings,
            "profile": profile if "error" not in profile else None
        }


# ============ Quick Test ============

if __name__ == "__main__":
    print("Testing News Client...")
    
    client = NewsClient()
    
    if not client.api_key:
        print("\n⚠️ Set FINNHUB_API_KEY in .env to test")
        print("   Get free key at: https://finnhub.io/register")
    else:
        # Test with a stock
        print("\n" + "=" * 60)
        print(client.get_stock_summary("AAPL"))
        
        print("\n" + "=" * 60)
        print(client.get_market_overview())
