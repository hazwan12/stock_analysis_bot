"""
Paper Trading Simulator
Simulates day-by-day trading using historical data
Tests your strategy as if you were trading in real-time
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass

from modules.api_client import MooMooAPIClient
from modules.portfolio_manager import PortfolioManager
from modules.portfolio_recommender import PortfolioRecommender
import config


@dataclass
class PaperTradingDay:
    """Records one day of paper trading"""
    date: datetime
    starting_cash: float
    starting_value: float
    recommendations: List[Dict]
    executed_trades: List[Dict]
    ending_cash: float
    ending_value: float
    daily_return_pct: float
    total_return_pct: float


class PaperTradingSimulator:
    """
    Simulates real-time trading using historical data
    Day-by-day portfolio management with bot recommendations
    """
    
    def __init__(self, initial_capital: float = 10000):
        """
        Initialize paper trading simulator
        
        Args:
            initial_capital: Starting cash amount
        """
        self.initial_capital = initial_capital
        self.api_client = MooMooAPIClient()
        self.trading_days: List[PaperTradingDay] = []
        self.holdings = {}  # {symbol: {'shares': float, 'avg_price': float}}
        self.cash = initial_capital
        self.total_value = initial_capital
        
    def run_simulation(self, 
                      start_date: str,
                      num_days: int,
                      auto_execute: bool = True,
                      buy_threshold: int = 2,
                      sell_threshold: int = -2) -> Dict:
        """
        Run paper trading simulation
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            num_days: Number of trading days to simulate
            auto_execute: Automatically follow all recommendations
            buy_threshold: Signal score needed to buy
            sell_threshold: Signal score needed to sell
            
        Returns:
            Simulation results and statistics
        """
        print(f"\n{'='*80}")
        print(f"PAPER TRADING SIMULATION")
        print(f"{'='*80}")
        print(f"Start Date: {start_date}")
        print(f"Duration: {num_days} days")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Auto-Execute: {auto_execute}")
        print(f"Buy Threshold: {buy_threshold}, Sell Threshold: {sell_threshold}")
        print(f"{'='*80}\n")
        
        self.api_client.connect()
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        current_date = start_dt
        
        # Simulate day by day
        for day_num in range(1, num_days + 1):
            print(f"\n{'='*80}")
            print(f"DAY {day_num}: {current_date.strftime('%Y-%m-%d')}")
            print(f"{'='*80}")
            
            # Get market data as of this date
            day_result = self._simulate_day(
                current_date, 
                buy_threshold, 
                sell_threshold,
                auto_execute
            )
            
            self.trading_days.append(day_result)
            
            # Display day summary
            self._print_day_summary(day_result, day_num)
            
            # Move to next day
            current_date += timedelta(days=1)
            
            # Optional: pause between days for user review
            if not auto_execute and day_num < num_days:
                input("\nPress Enter to continue to next day...")
        
        # Calculate final results
        results = self._calculate_results()
        self._print_final_results(results)
        
        # Generate visualization
        self._plot_results()
        
        self.api_client.close()
        return results
    
    def _simulate_day(self, 
                     current_date: datetime,
                     buy_threshold: int,
                     sell_threshold: int,
                     auto_execute: bool) -> PaperTradingDay:
        """Simulate one day of trading"""
        
        starting_cash = self.cash
        starting_value = self._calculate_portfolio_value(current_date)
        
        # Get recommendations based on current portfolio
        recommendations = self._generate_recommendations(
            current_date,
            buy_threshold,
            sell_threshold
        )
        
        # Execute trades
        executed_trades = []
        if auto_execute:
            executed_trades = self._execute_recommendations(
                recommendations,
                current_date
            )
        
        # Calculate ending values
        ending_cash = self.cash
        ending_value = self._calculate_portfolio_value(current_date)
        
        # Calculate returns
        daily_return = ((ending_value - starting_value) / starting_value) * 100 if starting_value > 0 else 0
        total_return = ((ending_value - self.initial_capital) / self.initial_capital) * 100
        
        return PaperTradingDay(
            date=current_date,
            starting_cash=starting_cash,
            starting_value=starting_value,
            recommendations=recommendations,
            executed_trades=executed_trades,
            ending_cash=ending_cash,
            ending_value=ending_value,
            daily_return_pct=daily_return,
            total_return_pct=total_return
        )
    
    def _generate_recommendations(self,
                                 current_date: datetime,
                                 buy_threshold: int,
                                 sell_threshold: int) -> List[Dict]:
        """Generate recommendations for current date"""
        
        from modules.trend_analyzer import TrendAnalyzer
        from modules.universe_scanner import StockUniverseScanner
        
        recommendations = []
        
        # Analyze existing holdings
        for symbol, holding in self.holdings.items():
            # Get data up to current date
            all_data = self.api_client.get_historical_data(symbol, days=100)
            
            # Filter to only data before current date
            data = all_data[all_data['time_key'] <= current_date].copy()
            
            if len(data) < 50:
                continue
            
            # Analyze
            analyzer = TrendAnalyzer(data)
            signals = analyzer.analyze()
            
            current_price = data['close'].iloc[-1]
            gain_loss = ((current_price - holding['avg_price']) / holding['avg_price']) * 100
            
            # Sell conditions
            if signals['score'] <= sell_threshold or gain_loss < -15 or gain_loss > 30:
                recommendations.append({
                    'action': 'SELL',
                    'symbol': symbol,
                    'shares': holding['shares'],
                    'price': current_price,
                    'reason': f"Score: {signals['score']}, P&L: {gain_loss:+.1f}%"
                })
            else:
                recommendations.append({
                    'action': 'HOLD',
                    'symbol': symbol,
                    'shares': holding['shares'],
                    'price': current_price,
                    'reason': f"Score: {signals['score']}, P&L: {gain_loss:+.1f}%"
                })
        
        # Find new opportunities (limited to top 3 to avoid over-trading)
        if self.cash > 100:
            scanner = StockUniverseScanner(max_workers=1)
            scanner.api_client = self.api_client
            
            # Get universe but filter to current date
            symbols = scanner.get_stock_universe()[:10]  # Scan top 10 only
            
            buy_opportunities = []
            for symbol in symbols:
                if symbol in self.holdings:
                    continue
                
                all_data = self.api_client.get_historical_data(symbol, days=100)
                data = all_data[all_data['time_key'] <= current_date].copy()
                
                if len(data) < 50:
                    continue
                
                analyzer = TrendAnalyzer(data)
                signals = analyzer.analyze()
                
                if signals['score'] >= buy_threshold:
                    current_price = data['close'].iloc[-1]
                    max_shares = int((self.cash * 0.3) / current_price)
                    
                    if max_shares >= 1:
                        buy_opportunities.append({
                            'action': 'BUY',
                            'symbol': symbol,
                            'shares': max_shares,
                            'price': current_price,
                            'reason': f"Score: {signals['score']}",
                            'score': signals['score']
                        })
            
            # Sort by score and take top 3
            buy_opportunities.sort(key=lambda x: x['score'], reverse=True)
            recommendations.extend(buy_opportunities[:3])
        
        return recommendations
    
    def _execute_recommendations(self,
                                recommendations: List[Dict],
                                current_date: datetime) -> List[Dict]:
        """Execute trading recommendations"""
        
        executed = []
        
        # Execute sells first
        for rec in recommendations:
            if rec['action'] == 'SELL':
                symbol = rec['symbol']
                shares = rec['shares']
                price = rec['price']
                
                proceeds = shares * price * (1 - config.COMMISSION_RATE)
                self.cash += proceeds
                
                # Remove from holdings
                if symbol in self.holdings:
                    del self.holdings[symbol]
                
                executed.append({
                    'date': current_date,
                    'action': 'SELL',
                    'symbol': symbol,
                    'shares': shares,
                    'price': price,
                    'amount': proceeds
                })
                
                print(f"  🔴 SELL: {symbol} - {shares} shares @ ${price:.2f} = ${proceeds:,.2f}")
        
        # Execute buys
        for rec in recommendations:
            if rec['action'] == 'BUY':
                symbol = rec['symbol']
                shares = rec['shares']
                price = rec['price']
                
                cost = shares * price * (1 + config.COMMISSION_RATE)
                
                if cost <= self.cash:
                    self.cash -= cost
                    
                    # Add to holdings
                    if symbol in self.holdings:
                        old_shares = self.holdings[symbol]['shares']
                        old_avg = self.holdings[symbol]['avg_price']
                        new_shares = old_shares + shares
                        new_avg = ((old_shares * old_avg) + (shares * price)) / new_shares
                        self.holdings[symbol] = {'shares': new_shares, 'avg_price': new_avg}
                    else:
                        self.holdings[symbol] = {'shares': shares, 'avg_price': price}
                    
                    executed.append({
                        'date': current_date,
                        'action': 'BUY',
                        'symbol': symbol,
                        'shares': shares,
                        'price': price,
                        'amount': cost
                    })
                    
                    print(f"  🟢 BUY: {symbol} - {shares} shares @ ${price:.2f} = ${cost:,.2f}")
        
        return executed
    
    def _calculate_portfolio_value(self, current_date: datetime) -> float:
        """Calculate total portfolio value"""
        
        holdings_value = 0
        for symbol, holding in self.holdings.items():
            # Get price as of current date
            all_data = self.api_client.get_historical_data(symbol, days=10)
            data = all_data[all_data['time_key'] <= current_date]
            
            if len(data) > 0:
                current_price = data['close'].iloc[-1]
                holdings_value += holding['shares'] * current_price
        
        return self.cash + holdings_value
    
    def _print_day_summary(self, day: PaperTradingDay, day_num: int):
        """Print daily summary"""
        
        print(f"\n📊 Portfolio Status:")
        print(f"   Cash: ${day.ending_cash:,.2f}")
        print(f"   Holdings Value: ${day.ending_value - day.ending_cash:,.2f}")
        print(f"   Total Value: ${day.ending_value:,.2f}")
        print(f"   Daily Return: {day.daily_return_pct:+.2f}%")
        print(f"   Total Return: {day.total_return_pct:+.2f}%")
        
        if self.holdings:
            print(f"\n💼 Current Holdings:")
            for symbol, holding in self.holdings.items():
                print(f"   {symbol}: {holding['shares']:.0f} shares @ ${holding['avg_price']:.2f} avg")
    
    def _calculate_results(self) -> Dict:
        """Calculate final simulation results"""
        
        if not self.trading_days:
            return {}
        
        total_trades = sum(len(day.executed_trades) for day in self.trading_days)
        buy_trades = sum(1 for day in self.trading_days for trade in day.executed_trades if trade['action'] == 'BUY')
        sell_trades = sum(1 for day in self.trading_days for trade in day.executed_trades if trade['action'] == 'SELL')
        
        daily_returns = [day.daily_return_pct for day in self.trading_days]
        positive_days = len([r for r in daily_returns if r > 0])
        
        final_value = self.trading_days[-1].ending_value
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate buy and hold comparison
        # Assume buying SPY on day 1 and holding
        first_day_cash = self.initial_capital
        buy_hold_return = 0  # Simplified - would need SPY data
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'total_days': len(self.trading_days),
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'positive_days': positive_days,
            'win_rate': (positive_days / len(self.trading_days)) * 100,
            'avg_daily_return': np.mean(daily_returns),
            'best_day': max(daily_returns),
            'worst_day': min(daily_returns),
            'max_value': max(day.ending_value for day in self.trading_days),
            'min_value': min(day.ending_value for day in self.trading_days),
        }
    
    def _print_final_results(self, results: Dict):
        """Print final simulation results"""
        
        print(f"\n{'='*80}")
        print(f"PAPER TRADING SIMULATION - FINAL RESULTS")
        print(f"{'='*80}\n")
        
        print("💰 PERFORMANCE:")
        print(f"   Initial Capital:  ${results['initial_capital']:,.2f}")
        print(f"   Final Value:      ${results['final_value']:,.2f}")
        print(f"   Total Return:     {results['total_return_pct']:+.2f}%")
        print(f"   Profit/Loss:      ${results['final_value'] - results['initial_capital']:+,.2f}")
        print()
        
        print("📊 TRADING ACTIVITY:")
        print(f"   Total Days:       {results['total_days']}")
        print(f"   Total Trades:     {results['total_trades']}")
        print(f"   Buy Trades:       {results['buy_trades']}")
        print(f"   Sell Trades:      {results['sell_trades']}")
        print(f"   Positive Days:    {results['positive_days']} ({results['win_rate']:.1f}%)")
        print()
        
        print("📈 DAILY RETURNS:")
        print(f"   Average:          {results['avg_daily_return']:+.2f}%")
        print(f"   Best Day:         {results['best_day']:+.2f}%")
        print(f"   Worst Day:        {results['worst_day']:+.2f}%")
        print()
        
        print("💼 PORTFOLIO RANGE:")
        print(f"   Maximum Value:    ${results['max_value']:,.2f}")
        print(f"   Minimum Value:    ${results['min_value']:,.2f}")
        print()
        
        # Rating
        if results['total_return_pct'] > 10 and results['win_rate'] > 60:
            rating = "🌟 EXCELLENT - Strategy performed very well!"
        elif results['total_return_pct'] > 5 and results['win_rate'] > 50:
            rating = "✅ GOOD - Profitable strategy"
        elif results['total_return_pct'] > 0:
            rating = "⚠️  FAIR - Marginally profitable"
        else:
            rating = "❌ POOR - Strategy lost money"
        
        print(f"OVERALL RATING: {rating}")
        print(f"{'='*80}\n")
    
    def _plot_results(self):
        """Plot simulation results"""
        
        if not self.trading_days:
            return
        
        dates = [day.date for day in self.trading_days]
        values = [day.ending_value for day in self.trading_days]
        returns = [day.total_return_pct for day in self.trading_days]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Portfolio value over time
        ax1.plot(dates, values, linewidth=2, color='blue', label='Portfolio Value')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial Capital')
        ax1.set_title('Paper Trading: Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Cumulative return
        ax2.plot(dates, returns, linewidth=2, color='green', label='Total Return')
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.set_title('Paper Trading: Cumulative Return', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Mark buy/sell trades
        for day in self.trading_days:
            for trade in day.executed_trades:
                if trade['action'] == 'BUY':
                    ax1.plot(day.date, day.ending_value, 'g^', markersize=8, alpha=0.6)
                elif trade['action'] == 'SELL':
                    ax1.plot(day.date, day.ending_value, 'rv', markersize=8, alpha=0.6)
        
        plt.tight_layout()
        
        import os
        os.makedirs(config.CHART_PATH, exist_ok=True)
        filename = f'{config.CHART_PATH}/paper_trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Chart saved: {filename}")
        plt.show()


if __name__ == "__main__":
    # Test paper trading simulator
    simulator = PaperTradingSimulator(initial_capital=10000)
    
    results = simulator.run_simulation(
        start_date='2024-01-01',
        num_days=30,
        auto_execute=True,
        buy_threshold=2,
        sell_threshold=-2
    )
    
    print("\n✅ Paper trading simulation complete!")
