"""
Trend Analyzer Module
Analyzes market trends and generates buy/sell signals
"""

import pandas as pd
import numpy as np
from typing import Dict
from modules.technical_analysis import TechnicalAnalyzer


class TrendAnalyzer:
    """
    Analyzes market trends using technical indicators
    Generates actionable buy/sell signals with scoring
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize TrendAnalyzer
        
        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data.copy()
        self.signals = {}
        self.ta = TechnicalAnalyzer()
        
    def analyze(self) -> Dict:
        """
        Perform complete technical analysis
        
        Returns:
            Dictionary with signals, scores, and recommendations
        """
        close_prices = self.data['close']
        volume = self.data['volume']
        high = self.data['high']
        low = self.data['low']
        
        # Calculate all indicators
        self.data['SMA_20'] = self.ta.calculate_sma(close_prices, 20)
        self.data['SMA_50'] = self.ta.calculate_sma(close_prices, 50)
        self.data['EMA_12'] = self.ta.calculate_ema(close_prices, 12)
        self.data['RSI'] = self.ta.calculate_rsi(close_prices)
        
        macd, signal, hist = self.ta.calculate_macd(close_prices)
        self.data['MACD'] = macd
        self.data['MACD_Signal'] = signal
        self.data['MACD_Hist'] = hist
        
        upper_bb, middle_bb, lower_bb = self.ta.calculate_bollinger_bands(close_prices)
        self.data['BB_Upper'] = upper_bb
        self.data['BB_Middle'] = middle_bb
        self.data['BB_Lower'] = lower_bb
        
        self.data['Volume_MA'] = self.ta.calculate_volume_profile(volume)
        self.data['ATR'] = self.ta.calculate_atr(high, low, close_prices)
        
        # Generate signals
        self._generate_signals()
        
        return self.signals
    
    def _generate_signals(self):
        """Generate trading signals based on technical indicators"""
        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2]
        
        signals = []
        score = 0
        
        # 1. Moving Average Crossover (Golden/Death Cross)
        if pd.notna(latest['SMA_20']) and pd.notna(latest['SMA_50']):
            if latest['SMA_20'] > latest['SMA_50'] and prev['SMA_20'] <= prev['SMA_50']:
                signals.append("🟢 Golden Cross: SMA20 crossed above SMA50 (Bullish)")
                score += 2
            elif latest['SMA_20'] < latest['SMA_50'] and prev['SMA_20'] >= prev['SMA_50']:
                signals.append("🔴 Death Cross: SMA20 crossed below SMA50 (Bearish)")
                score -= 2
        
        # 2. RSI Analysis (Overbought/Oversold)
        if pd.notna(latest['RSI']):
            if latest['RSI'] < 30:
                signals.append(f"🟢 RSI Oversold: {latest['RSI']:.1f} (Potential buy)")
                score += 1
            elif latest['RSI'] > 70:
                signals.append(f"🔴 RSI Overbought: {latest['RSI']:.1f} (Potential sell)")
                score -= 1
            elif 30 <= latest['RSI'] <= 40:
                signals.append(f"🟡 RSI Near Oversold: {latest['RSI']:.1f}")
            elif 60 <= latest['RSI'] <= 70:
                signals.append(f"🟠 RSI Near Overbought: {latest['RSI']:.1f}")
        
        # 3. MACD Crossover
        if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
            if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
                signals.append("🟢 MACD Bullish Crossover")
                score += 1
            elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
                signals.append("🔴 MACD Bearish Crossover")
                score -= 1
        
        # 4. Bollinger Bands
        if pd.notna(latest['BB_Lower']) and pd.notna(latest['BB_Upper']):
            if latest['close'] < latest['BB_Lower']:
                signals.append("🟢 Price below lower Bollinger Band (Oversold)")
                score += 1
            elif latest['close'] > latest['BB_Upper']:
                signals.append("🔴 Price above upper Bollinger Band (Overbought)")
                score -= 1
            
            # Check for squeeze (low volatility)
            bb_width = (latest['BB_Upper'] - latest['BB_Lower']) / latest['BB_Middle']
            if bb_width < 0.05:
                signals.append("📊 Bollinger Band Squeeze (Potential breakout)")
        
        # 5. Volume Analysis
        if pd.notna(latest['Volume_MA']):
            volume_ratio = latest['volume'] / latest['Volume_MA']
            if volume_ratio > 1.5:
                signals.append(f"📊 High volume: {volume_ratio:.2f}x average (Strong momentum)")
                if score > 0:
                    score += 0.5  # Boost existing bullish signals
                elif score < 0:
                    score -= 0.5  # Boost existing bearish signals
            elif volume_ratio < 0.5:
                signals.append(f"📊 Low volume: {volume_ratio:.2f}x average (Weak momentum)")
        
        # 6. Price Trend
        price_change = ((latest['close'] - prev['close']) / prev['close']) * 100
        if abs(price_change) > 2:
            if price_change > 0:
                signals.append(f"📈 Strong upward move: {price_change:.2f}% today")
            else:
                signals.append(f"📉 Strong downward move: {price_change:.2f}% today")
        else:
            if price_change > 0:
                signals.append(f"📈 Price up {price_change:.2f}% today")
            else:
                signals.append(f"📉 Price down {abs(price_change):.2f}% today")
        
        # 7. Volatility Check
        if pd.notna(latest['ATR']):
            atr_pct = (latest['ATR'] / latest['close']) * 100
            if atr_pct > 3:
                signals.append(f"⚠️  High volatility: {atr_pct:.2f}% ATR")
        
        # Generate recommendation based on total score
        if score >= 3:
            recommendation = "🟢 STRONG BUY"
        elif score >= 1:
            recommendation = "🟡 BUY"
        elif score <= -3:
            recommendation = "🔴 STRONG SELL"
        elif score <= -1:
            recommendation = "🟠 SELL"
        else:
            recommendation = "⚪ HOLD"
        
        # Compile results
        self.signals = {
            'recommendation': recommendation,
            'score': int(score),
            'signals': signals,
            'current_price': latest['close'],
            'rsi': latest['RSI'],
            'macd': latest['MACD'],
            'volume_ratio': latest['volume'] / latest['Volume_MA'] if pd.notna(latest['Volume_MA']) and latest['Volume_MA'] > 0 else 0,
            'volatility': (latest['ATR'] / latest['close']) * 100 if pd.notna(latest['ATR']) else 0,
            'price_change': price_change
        }
    
    def get_signal_strength(self) -> str:
        """
        Get signal strength classification
        
        Returns:
            Signal strength (Very Strong, Strong, Moderate, Weak)
        """
        if not self.signals:
            self.analyze()
        
        score = abs(self.signals['score'])
        if score >= 4:
            return "Very Strong"
        elif score >= 3:
            return "Strong"
        elif score >= 2:
            return "Moderate"
        else:
            return "Weak"
    
    def get_risk_level(self) -> str:
        """
        Assess risk level based on volatility and indicators
        
        Returns:
            Risk level (Low, Medium, High)
        """
        if not self.signals:
            self.analyze()
        
        volatility = self.signals['volatility']
        rsi = self.signals['rsi']
        
        # High risk if high volatility or extreme RSI
        if volatility > 4 or rsi < 20 or rsi > 80:
            return "High"
        elif volatility > 2.5 or rsi < 30 or rsi > 70:
            return "Medium"
        else:
            return "Low"


if __name__ == "__main__":
    # Test trend analyzer
    import sys
    sys.path.append('..')
    from modules.api_client import MooMooAPIClient
    
    client = MooMooAPIClient()
    client.connect()
    
    # Fetch and analyze
    data = client.get_historical_data('US.AAPL', days=100)
    
    analyzer = TrendAnalyzer(data)
    results = analyzer.analyze()
    
    # Display results
    print(f"\nAnalysis for US.AAPL")
    print(f"Current Price: ${results['current_price']:.2f}")
    print(f"Recommendation: {results['recommendation']}")
    print(f"Score: {results['score']}")
    print(f"Signal Strength: {analyzer.get_signal_strength()}")
    print(f"Risk Level: {analyzer.get_risk_level()}")
    print(f"\nSignals:")
    for signal in results['signals']:
        print(f"  {signal}")
    
    client.close()