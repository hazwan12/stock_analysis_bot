"""
Technical Analysis Module
Implements various technical indicators and calculations
"""

import pandas as pd
import numpy as np
from typing import Tuple


class TechnicalAnalyzer:
    """
    Performs technical analysis calculations
    Includes: Moving Averages, RSI, MACD, Bollinger Bands, Volume analysis
    """
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average
        
        Args:
            data: Price series
            period: Number of periods
            
        Returns:
            SMA series
        """
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            data: Price series
            period: Number of periods
            
        Returns:
            EMA series
        """
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            data: Price series
            period: RSI period (default 14)
            
        Returns:
            RSI series (0-100)
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.Series, 
                      fast: int = 12, 
                      slow: int = 26, 
                      signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD, Signal line, and Histogram
        
        Args:
            data: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Tuple of (MACD, Signal, Histogram)
        """
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, 
                                 period: int = 20, 
                                 std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            data: Price series
            period: Moving average period
            std_dev: Number of standard deviations
            
        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band)
        """
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_volume_profile(volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate average volume
        
        Args:
            volume: Volume series
            period: Averaging period
            
        Returns:
            Average volume series
        """
        return volume.rolling(window=period).mean()
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (volatility indicator)
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period
            
        Returns:
            ATR series
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, 
                            low: pd.Series, 
                            close: pd.Series, 
                            period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator (%K and %D)
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Lookback period
            
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        stoch_d = stoch_k.rolling(window=3).mean()
        
        return stoch_k, stoch_d
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume
        
        Args:
            close: Close price series
            volume: Volume series
            
        Returns:
            OBV series
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def calculate_momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate price momentum
        
        Args:
            data: Price series
            period: Lookback period
            
        Returns:
            Momentum series
        """
        return data.diff(period)
    
    @staticmethod
    def calculate_roc(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate Rate of Change
        
        Args:
            data: Price series
            period: Lookback period
            
        Returns:
            ROC series (percentage)
        """
        return ((data - data.shift(period)) / data.shift(period)) * 100


if __name__ == "__main__":
    # Test technical indicators
    from modules.api_client import MooMooAPIClient
    
    # Fetch data
    client = MooMooAPIClient()
    client.connect()
    data = client.get_historical_data('US.AAPL', days=100)
    
    # Calculate indicators
    ta = TechnicalAnalyzer()
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    
    # Test each indicator
    print("Testing Technical Indicators:\n")
    
    sma_20 = ta.calculate_sma(close, 20)
    print(f"SMA(20): {sma_20.iloc[-1]:.2f}")
    
    ema_12 = ta.calculate_ema(close, 12)
    print(f"EMA(12): {ema_12.iloc[-1]:.2f}")
    
    rsi = ta.calculate_rsi(close)
    print(f"RSI(14): {rsi.iloc[-1]:.2f}")
    
    macd, signal, hist = ta.calculate_macd(close)
    print(f"MACD: {macd.iloc[-1]:.4f}")
    
    upper, middle, lower = ta.calculate_bollinger_bands(close)
    print(f"BB Upper: {upper.iloc[-1]:.2f}")
    print(f"BB Middle: {middle.iloc[-1]:.2f}")
    print(f"BB Lower: {lower.iloc[-1]:.2f}")
    
    atr = ta.calculate_atr(high, low, close)
    print(f"ATR(14): {atr.iloc[-1]:.2f}")
    
    stoch_k, stoch_d = ta.calculate_stochastic(high, low, close)
    print(f"Stochastic %K: {stoch_k.iloc[-1]:.2f}")
    print(f"Stochastic %D: {stoch_d.iloc[-1]:.2f}")
    
    client.close()