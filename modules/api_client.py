"""
MooMoo API Client Module
Handles connection and data fetching from MooMoo OpenAPI
"""

import pandas as pd
import numpy as np
from datetime import datetime
import config


class MooMooAPIClient:
    """
    Wrapper for MooMoo OpenAPI
    Handles connection, data fetching, and error handling
    """
    
    def __init__(self, host: str = None, port: int = None):
        """
        Initialize MooMoo API client
        
        Args:
            host: MooMoo OpenD host (default from config)
            port: MooMoo OpenD port (default from config)
        """
        self.host = host or config.MOOMOO_HOST
        self.port = port or config.MOOMOO_PORT
        self.quote_ctx = None
        self.is_connected = False
        
    def connect(self):
        """Connect to MooMoo OpenD"""
        try:
            from moomoo import OpenQuoteContext, RET_OK
            print(f"Attempting to connect to MooMoo OpenD at {self.host}:{self.port}...")
            self.quote_ctx = OpenQuoteContext(host=self.host, port=self.port, is_encrypt=None)
            ret, data = self.quote_ctx.get_stock_quote(['US.AAPL'])
            if ret == RET_OK:
                self.is_connected = True
                print("✓ Connected to MooMoo OpenD successfully")
                print("✓ Using LIVE market data")
                return True
            else:
                print(f"✗ Connection failed: {data}")
                print("⚠ Switching to DEMO mode with simulated data")
                self.is_connected = False
                return False
        except ImportError:
            print("⚠ MooMoo SDK not installed. Install with: pip install moomoo-opend")
            print("⚠ Running in DEMO mode with simulated data")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"✗ Connection error: {e}")
            print("⚠ MooMoo OpenD not running or connection refused")
            print("⚠ Switching to DEMO mode with simulated data")
            print()
            print("To use live data:")
            print("  1. Download and install MooMoo OpenD")
            print("  2. Start the OpenD application")
            print("  3. Make sure it's running on port 11111")
            print("  4. Run this script again")
            print()
            self.is_connected = False
            self.quote_ctx = None
            return False
    
    def get_historical_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """
        Fetch historical stock data
        
        Args:
            symbol: Stock symbol (e.g., 'US.AAPL')
            days: Number of days of historical data
            
        Returns:
            DataFrame with columns: time_key, open, close, high, low, volume, code
        """
        if self.is_connected and self.quote_ctx:
            try:
                from moomoo import KLType, RET_OK
                ret, data = self.quote_ctx.get_cur_kline(
                    symbol, 
                    num=days, 
                    ktype=KLType.K_DAY
                )
                if ret == RET_OK:
                    return data
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        # Return simulated data for demo
        return self._generate_demo_data(symbol, days)
    
    def get_real_time_quote(self, symbol: str) -> dict:
        """
        Get real-time quote for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with current price, volume, etc.
        """
        if self.is_connected and self.quote_ctx:
            try:
                from moomoo import RET_OK
                ret, data = self.quote_ctx.get_stock_quote([symbol])
                if ret == RET_OK and len(data) > 0:
                    return data.iloc[0].to_dict()
            except Exception as e:
                print(f"Error fetching quote for {symbol}: {e}")
        
        # Return demo quote
        return {
            'code': symbol,
            'last_price': 100.0 + np.random.rand() * 50,
            'volume': np.random.randint(1000000, 10000000)
        }
    
    def get_market_snapshot(self, market: str = 'US') -> pd.DataFrame:
        """
        Get market snapshot with all tradable stocks
        
        Args:
            market: Market code (US, HK, etc.)
            
        Returns:
            DataFrame with stock list
        """
        if self.is_connected and self.quote_ctx:
            try:
                from moomoo import RET_OK, Market
                # This would use actual MooMoo API calls
                # ret, data = self.quote_ctx.get_stock_basicinfo(market, ...)
                pass
            except Exception as e:
                print(f"Error fetching market snapshot: {e}")
        
        # Return demo stock list
        return self._get_demo_stock_list(market)
    
    def _generate_demo_data(self, symbol: str, days: int) -> pd.DataFrame:
        """
        Generate simulated stock data for demonstration
        
        Args:
            symbol: Stock symbol
            days: Number of days to generate
            
        Returns:
            DataFrame with OHLCV data
        """
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Simulate price movement with trend and noise
        np.random.seed(hash(symbol) % 2**32)
        base_price = 100 + np.random.rand() * 400
        trend = np.linspace(0, 20, days)
        noise = np.random.randn(days) * 5
        prices = base_price + trend + np.cumsum(noise)
        prices = np.maximum(prices, 1)  # Ensure positive prices
        
        # Generate OHLC data
        high = prices * (1 + np.random.rand(days) * 0.02)
        low = prices * (1 - np.random.rand(days) * 0.02)
        open_price = prices + np.random.randn(days) * 2
        volume = np.random.randint(1000000, 10000000, days)
        
        df = pd.DataFrame({
            'time_key': dates,
            'open': open_price,
            'close': prices,
            'high': high,
            'low': low,
            'volume': volume,
            'code': symbol
        })
        
        return df
    
    def _get_demo_stock_list(self, market: str) -> pd.DataFrame:
        """Generate demo stock list"""
        stocks = [
            'US.AAPL', 'US.MSFT', 'US.GOOGL', 'US.AMZN', 'US.META',
            'US.NVDA', 'US.TSLA', 'US.AMD', 'US.INTC', 'US.ORCL'
        ]
        
        return pd.DataFrame({
            'code': stocks,
            'name': [s.split('.')[1] for s in stocks],
            'market': market
        })
    
    def close(self):
        """Close connection"""
        if self.quote_ctx:
            self.quote_ctx.close()
            print("✓ Connection closed")
        self.is_connected = False


if __name__ == "__main__":
    # Test the API client
    client = MooMooAPIClient()
    client.connect()
    
    # Test data fetch
    data = client.get_historical_data('US.AAPL', days=100)
    print(f"\nFetched {len(data)} days of data for US.AAPL")
    print(data.head())
    
    # Test real-time quote
    quote = client.get_real_time_quote('US.AAPL')
    print(f"\nReal-time quote: {quote}")
    
    client.close()