"""
Backtesting Module
Tests trading strategies on historical data and provides performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# Import from other modules
from modules.technical_analysis import TechnicalAnalyzer
from modules.trend_analyzer import TrendAnalyzer
import config


@dataclass
class Trade:
    """Represents a single trade"""
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    shares: int = 0
    trade_type: str = 'LONG'
    signal_score: int = 0
    
    @property
    def is_open(self) -> bool:
        return self.exit_date is None
    
    @property
    def pnl(self) -> float:
        """Calculate profit/loss"""
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.shares
    
    @property
    def pnl_percent(self) -> float:
        """Calculate percentage return"""
        if self.exit_price is None:
            return 0.0
        return ((self.exit_price - self.entry_price) / self.entry_price) * 100
    
    @property
    def holding_days(self) -> int:
        """Calculate holding period"""
        if self.exit_date is None:
            return 0
        return (self.exit_date - self.entry_date).days


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, initial_capital: float = None, commission: float = None):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting capital for backtesting
            commission: Commission rate (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.commission = commission or config.BACKTEST_COMMISSION
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.dates: List[datetime] = []
        self.current_position: Optional[Trade] = None
        self.cash = self.initial_capital
        self.results = {}
        
    def run_backtest(self, symbol: str, data: pd.DataFrame, 
                     buy_threshold: int = None,
                     sell_threshold: int = None,
                     stop_loss_pct: float = None,
                     take_profit_pct: float = None) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            symbol: Stock symbol
            data: Historical price data
            buy_threshold: Score threshold for buy signals
            sell_threshold: Score threshold for sell signals
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        # Use config defaults if not specified
        buy_threshold = buy_threshold or config.DEFAULT_BUY_THRESHOLD
        sell_threshold = sell_threshold or config.DEFAULT_SELL_THRESHOLD
        stop_loss_pct = stop_loss_pct or config.DEFAULT_STOP_LOSS
        take_profit_pct = take_profit_pct or config.DEFAULT_TAKE_PROFIT
        
        print(f"\n{'='*70}")
        print(f"BACKTESTING {symbol}")
        print(f"{'='*70}")
        print(f"Period: {data['time_key'].iloc[0]} to {data['time_key'].iloc[-1]}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Commission: {self.commission*100}%")
        print(f"Buy Threshold: {buy_threshold}, Sell Threshold: {sell_threshold}")
        print(f"Stop Loss: {stop_loss_pct*100}%, Take Profit: {take_profit_pct*100}%")
        print(f"{'='*70}\n")
        
        # Ensure data is sorted by date
        data = data.sort_values('time_key').reset_index(drop=True)
        
        # Calculate indicators for entire dataset
        analyzer = TrendAnalyzer(data)
        analyzed_data = analyzer.data.copy()
        
        # Need at least 50 days for indicators
        start_idx = 50
        
        # Simulate trading day by day
        for i in range(start_idx, len(analyzed_data)):
            current_date = analyzed_data.iloc[i]['time_key']
            current_price = analyzed_data.iloc[i]['close']
            
            # Calculate signals for current day
            daily_signal = self._calculate_daily_signal(analyzed_data.iloc[:i+1])
            
            # Check if we need to close current position
            if self.current_position and not self.current_position.is_open:
                self.current_position = None
                
            if self.current_position:
                # Check stop loss
                if current_price <= self.current_position.entry_price * (1 - stop_loss_pct):
                    self._close_position(current_date, current_price, "STOP_LOSS")
                # Check take profit
                elif current_price >= self.current_position.entry_price * (1 + take_profit_pct):
                    self._close_position(current_date, current_price, "TAKE_PROFIT")
            
            # Generate trading signals
            if daily_signal['score'] >= buy_threshold and self.current_position is None:
                self._open_position(current_date, current_price, daily_signal['score'])
                
            elif daily_signal['score'] <= sell_threshold and self.current_position:
                self._close_position(current_date, current_price, "SIGNAL")
            
            # Calculate portfolio value
            portfolio_value = self.cash
            if self.current_position and not self.current_position.is_open:
                portfolio_value += self.current_position.shares * current_price
            elif self.current_position:
                portfolio_value += self.current_position.shares * current_price
                
            self.equity_curve.append(portfolio_value)
            self.dates.append(current_date)
        
        # Close any remaining open positions
        if self.current_position and self.current_position.is_open:
            last_price = analyzed_data.iloc[-1]['close']
            last_date = analyzed_data.iloc[-1]['time_key']
            self._close_position(last_date, last_price, "END_OF_BACKTEST")
        
        # Calculate performance metrics
        self.results = self._calculate_metrics(data, symbol)
        self._print_results()
        
        return self.results
    
    def _calculate_daily_signal(self, historical_data: pd.DataFrame) -> Dict:
        """Calculate trading signal for a specific day"""
        if len(historical_data) < 50:
            return {'score': 0, 'signals': []}
        
        latest = historical_data.iloc[-1]
        prev = historical_data.iloc[-2]
        
        score = 0
        
        # Moving Average Crossover
        if pd.notna(latest['SMA_20']) and pd.notna(latest['SMA_50']):
            if latest['SMA_20'] > latest['SMA_50'] and prev['SMA_20'] <= prev['SMA_50']:
                score += 2
            elif latest['SMA_20'] < latest['SMA_50'] and prev['SMA_20'] >= prev['SMA_50']:
                score -= 2
        
        # RSI
        if pd.notna(latest['RSI']):
            if latest['RSI'] < 30:
                score += 1
            elif latest['RSI'] > 70:
                score -= 1
        
        # MACD
        if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
            if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
                score += 1
            elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
                score -= 1
        
        # Bollinger Bands
        if pd.notna(latest['BB_Lower']) and pd.notna(latest['BB_Upper']):
            if latest['close'] < latest['BB_Lower']:
                score += 1
            elif latest['close'] > latest['BB_Upper']:
                score -= 1
        
        return {'score': score, 'signals': []}
    
    def _open_position(self, date: datetime, price: float, score: int):
        """Open a new position"""
        available_cash = self.cash * 0.95
        commission_cost = available_cash * self.commission
        shares = int((available_cash - commission_cost) / price)
        
        if shares > 0:
            cost = shares * price
            total_cost = cost + (cost * self.commission)
            
            self.current_position = Trade(
                entry_date=date,
                entry_price=price,
                shares=shares,
                signal_score=score
            )
            
            self.cash -= total_cost
            print(f"📈 BUY  | {date.date()} | Price: ${price:.2f} | Shares: {shares} | Cost: ${total_cost:,.2f} | Score: {score}")
    
    def _close_position(self, date: datetime, price: float, reason: str):
        """Close current position"""
        if not self.current_position:
            return
        
        proceeds = self.current_position.shares * price
        commission_cost = proceeds * self.commission
        net_proceeds = proceeds - commission_cost
        
        self.current_position.exit_date = date
        self.current_position.exit_price = price
        
        self.cash += net_proceeds
        self.trades.append(self.current_position)
        
        pnl = self.current_position.pnl - (self.current_position.entry_price * self.current_position.shares * self.commission * 2)
        pnl_pct = self.current_position.pnl_percent
        
        emoji = "✅" if pnl > 0 else "❌"
        print(f"{emoji} SELL | {date.date()} | Price: ${price:.2f} | P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%) | Reason: {reason} | Days: {self.current_position.holding_days}")
        
        self.current_position = None
    
    def _calculate_metrics(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Calculate performance metrics"""
        if len(self.trades) == 0:
            return {
                'error': 'No trades executed during backtest period',
                'total_trades': 0
            }
        
        final_value = self.equity_curve[-1]
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = (len(winning_trades) / len(self.trades)) * 100 if self.trades else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum([t.pnl for t in winning_trades]) / sum([t.pnl for t in losing_trades])) if losing_trades else 0
        
        # Calculate max drawdown
        peak = self.initial_capital
        max_drawdown = 0
        for value in self.equity_curve:
            if value > peak:
                peak = value
            drawdown = ((peak - value) / peak) * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        else:
            sharpe_ratio = 0
        
        # Buy and hold comparison
        start_price = data.iloc[50]['close']
        end_price = data.iloc[-1]['close']
        buy_hold_return = ((end_price - start_price) / start_price) * 100
        
        avg_holding_days = np.mean([t.holding_days for t in self.trades])
        
        return {
            'symbol': symbol,
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'buy_hold_return': buy_hold_return,
            'alpha': total_return - buy_hold_return,
            'avg_holding_days': avg_holding_days,
            'best_trade': max([t.pnl_percent for t in self.trades]) if self.trades else 0,
            'worst_trade': min([t.pnl_percent for t in self.trades]) if self.trades else 0,
        }
    
    def _print_results(self):
        """Print backtest results"""
        r = self.results
        
        if 'error' in r:
            print(f"\n⚠️  {r['error']}")
            return
        
        print(f"\n{'='*70}")
        print(f"BACKTEST RESULTS - {r['symbol']}")
        print(f"{'='*70}\n")
        
        print("💰 PERFORMANCE SUMMARY:")
        print(f"  Initial Capital:    ${r['initial_capital']:,.2f}")
        print(f"  Final Value:        ${r['final_value']:,.2f}")
        print(f"  Total Return:       {r['total_return']:+.2f}%")
        print(f"  Buy & Hold Return:  {r['buy_hold_return']:+.2f}%")
        print(f"  Alpha (vs B&H):     {r['alpha']:+.2f}%")
        print()
        
        print("📊 TRADE STATISTICS:")
        print(f"  Total Trades:       {r['total_trades']}")
        print(f"  Winning Trades:     {r['winning_trades']} ({r['win_rate']:.1f}%)")
        print(f"  Losing Trades:      {r['losing_trades']}")
        print(f"  Average Win:        ${r['avg_win']:,.2f}")
        print(f"  Average Loss:       ${r['avg_loss']:,.2f}")
        print(f"  Profit Factor:      {r['profit_factor']:.2f}")
        print(f"  Best Trade:         {r['best_trade']:+.2f}%")
        print(f"  Worst Trade:        {r['worst_trade']:+.2f}%")
        print(f"  Avg Holding Period: {r['avg_holding_days']:.1f} days")
        print()
        
        print("📈 RISK METRICS:")
        print(f"  Max Drawdown:       {r['max_drawdown']:.2f}%")
        print(f"  Sharpe Ratio:       {r['sharpe_ratio']:.2f}")
        print()
        
        if r['total_return'] > r['buy_hold_return'] and r['win_rate'] > 55:
            rating = "🌟 EXCELLENT - Strategy outperforms buy & hold"
        elif r['total_return'] > 0 and r['win_rate'] > 50:
            rating = "✅ GOOD - Profitable strategy"
        elif r['total_return'] > 0:
            rating = "⚠️  FAIR - Profitable but needs improvement"
        else:
            rating = "❌ POOR - Strategy needs significant adjustments"
        
        print(f"OVERALL RATING: {rating}")
        print(f"{'='*70}\n")
    
    def plot_results(self, symbol: str):
        """Plot backtest results"""
        if len(self.equity_curve) == 0:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'Backtest Results - {symbol}', fontsize=16, fontweight='bold')
        
        # 1. Equity Curve
        ax1 = axes[0, 0]
        ax1.plot(self.dates, self.equity_curve, label='Strategy', linewidth=2, color='blue')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial Capital')
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Drawdown
        ax2 = axes[0, 1]
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = ((equity_series - running_max) / running_max) * 100
        ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        ax2.plot(drawdown, color='red', linewidth=1)
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Days')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Trade P&L Distribution
        ax3 = axes[1, 0]
        trade_pnls = [t.pnl_percent for t in self.trades]
        if trade_pnls:
            ax3.hist(trade_pnls, bins=30, edgecolor='black', alpha=0.7, color='green')
            ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax3.set_title('Trade P&L Distribution')
            ax3.set_xlabel('Return (%)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # 4. Win/Loss Ratio
        ax4 = axes[1, 1]
        wins = len([t for t in self.trades if t.pnl > 0])
        losses = len([t for t in self.trades if t.pnl <= 0])
        ax4.pie([wins, losses], labels=['Wins', 'Losses'], autopct='%1.1f%%',
                colors=['green', 'red'], startangle=90)
        ax4.set_title(f'Win Rate: {(wins/(wins+losses)*100):.1f}%' if (wins+losses) > 0 else 'Win Rate: N/A')
        
        # 5. Monthly Returns
        ax5 = axes[2, 0]
        if len(self.dates) > 0:
            equity_df = pd.DataFrame({'date': self.dates, 'value': self.equity_curve})
            equity_df['date'] = pd.to_datetime(equity_df['date'])
            equity_df.set_index('date', inplace=True)
            monthly_returns = equity_df['value'].resample('M').last().pct_change() * 100
            
            if len(monthly_returns) > 1:
                monthly_returns_pivot = monthly_returns.to_frame()
                monthly_returns_pivot['year'] = monthly_returns_pivot.index.year
                monthly_returns_pivot['month'] = monthly_returns_pivot.index.month
                pivot = monthly_returns_pivot.pivot(index='year', columns='month', values='value')
                
                sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                           cbar_kws={'label': 'Return (%)'}, ax=ax5)
                ax5.set_title('Monthly Returns (%)')
                ax5.set_xlabel('Month')
                ax5.set_ylabel('Year')
        
        # 6. Cumulative Returns
        ax6 = axes[2, 1]
        returns = pd.Series(self.equity_curve).pct_change().fillna(0)
        cumulative_returns = (1 + returns).cumprod() - 1
        ax6.plot(cumulative_returns * 100, linewidth=2, color='purple')
        ax6.set_title('Cumulative Returns')
        ax6.set_xlabel('Days')
        ax6.set_ylabel('Cumulative Return (%)')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='gray', linestyle='--')
        
        plt.tight_layout()
        
        # Save to charts directory
        import os
        os.makedirs(config.CHART_PATH, exist_ok=True)
        filename = f'{config.CHART_PATH}/backtest_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Chart saved as: {filename}")
        plt.show()


class ParameterOptimizer:
    """Optimize strategy parameters"""
    
    def __init__(self, symbol: str, data: pd.DataFrame, initial_capital: float = None):
        self.symbol = symbol
        self.data = data
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.results = []
    
    def optimize(self, param_grid: Dict) -> pd.DataFrame:
        """Test multiple parameter combinations"""
        from itertools import product
        
        print(f"\n{'='*70}")
        print("PARAMETER OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Testing {np.prod([len(v) for v in param_grid.values()])} combinations...")
        print(f"{'='*70}\n")
        
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        for i, params in enumerate(combinations, 1):
            print(f"\nTesting combination {i}/{len(combinations)}: {params}")
            
            engine = BacktestEngine(initial_capital=self.initial_capital)
            result = engine.run_backtest(
                self.symbol,
                self.data,
                buy_threshold=params.get('buy_threshold', 2),
                sell_threshold=params.get('sell_threshold', -2),
                stop_loss_pct=params.get('stop_loss_pct', 0.05),
                take_profit_pct=params.get('take_profit_pct', 0.15)
            )
            
            if 'error' not in result:
                result.update(params)
                self.results.append(result)
        
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('total_return', ascending=False)
        
        print(f"\n{'='*70}")
        print("OPTIMIZATION RESULTS - TOP 10")
        print(f"{'='*70}\n")
        print(results_df.head(10).to_string(index=False))
        
        return results_df


if __name__ == "__main__":
    from modules.api_client import MooMooAPIClient
    
    api_client = MooMooAPIClient()
    api_client.connect()
    
    symbol = 'US.AAPL'
    data = api_client.get_historical_data(symbol, days=500)
    
    engine = BacktestEngine()
    results = engine.run_backtest(symbol, data)
    engine.plot_results(symbol)
    
    api_client.close()