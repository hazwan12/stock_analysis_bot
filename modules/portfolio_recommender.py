"""
Portfolio Recommendation Engine
Provides intelligent buy/sell/hold recommendations based on portfolio holdings
"""

import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime

from modules.portfolio_manager import PortfolioManager, Portfolio
from modules.api_client import MooMooAPIClient
from modules.trend_analyzer import TrendAnalyzer
from modules.universe_scanner import StockUniverseScanner
import config


class PortfolioRecommender:
    """Generate intelligent recommendations for portfolio management"""
    
    def __init__(self, portfolio_manager: PortfolioManager):
        self.pm = portfolio_manager
        self.api_client = MooMooAPIClient()
    
    def generate_recommendations(self, portfolio_id: int, 
                                available_cash: float = None) -> List[Dict]:
        """
        Generate comprehensive recommendations for a portfolio
        
        Args:
            portfolio_id: Portfolio ID
            available_cash: Cash available for investment (None = use from portfolio)
        
        Returns:
            List of recommendations with actions and reasoning
        """
        print(f"\n{'='*80}")
        print(f"GENERATING PORTFOLIO RECOMMENDATIONS")
        print(f"{'='*80}\n")
        
        portfolio = self.pm.get_portfolio(portfolio_id)
        if not portfolio:
            print(f"❌ Portfolio {portfolio_id} not found")
            return []
        
        # Get current holdings
        holdings = self.pm.get_holdings(portfolio_id)
        
        # Get current prices
        current_prices = self._get_current_prices(holdings)
        
        # Analyze existing holdings
        holding_analysis = self._analyze_holdings(holdings, current_prices)
        
        # Scan for new opportunities
        new_opportunities = self._find_new_opportunities(holdings)
        
        # Calculate available cash
        if available_cash is None:
            summary = self.pm.calculate_portfolio_value(portfolio_id, current_prices)
            available_cash = summary['cash_available'] + portfolio.recurring_amount
        
        # Generate recommendations
        recommendations = []
        
        # 1. Sell recommendations (from existing holdings)
        for symbol, analysis in holding_analysis.items():
            if analysis['action'] == 'SELL':
                recommendations.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'reason': analysis['reason'],
                    'target_shares': analysis['shares'],
                    'current_price': analysis['current_price'],
                    'expected_proceeds': analysis['shares'] * analysis['current_price']
                })
        
        # 2. Hold recommendations (keep existing)
        for symbol, analysis in holding_analysis.items():
            if analysis['action'] == 'HOLD':
                recommendations.append({
                    'symbol': symbol,
                    'action': 'HOLD',
                    'reason': analysis['reason'],
                    'target_shares': analysis['shares'],
                    'current_price': analysis['current_price']
                })
        
        # 3. Buy more recommendations (existing holdings)
        for symbol, analysis in holding_analysis.items():
            if analysis['action'] == 'BUY_MORE':
                recommendations.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'reason': analysis['reason'],
                    'target_shares': analysis['suggested_shares'],
                    'current_price': analysis['current_price'],
                    'estimated_cost': analysis['suggested_shares'] * analysis['current_price']
                })
        
        # 4. New buy recommendations
        for opp in new_opportunities:
            if opp['action'] == 'BUY':
                recommendations.append({
                    'symbol': opp['symbol'],
                    'action': 'BUY',
                    'reason': opp['reason'],
                    'target_shares': opp['suggested_shares'],
                    'current_price': opp['current_price'],
                    'estimated_cost': opp['suggested_shares'] * opp['current_price']
                })
        
        # Optimize allocation
        recommendations = self._optimize_allocation(recommendations, available_cash)
        
        # Save recommendations
        self.pm.save_recommendations(portfolio_id, recommendations)
        
        # Display recommendations
        self._display_recommendations(recommendations, available_cash)
        
        return recommendations
    
    def _get_current_prices(self, holdings: List) -> Dict[str, float]:
        """Get current prices for all holdings"""
        self.api_client.connect()
        
        prices = {}
        for holding in holdings:
            data = self.api_client.get_historical_data(holding.symbol, days=1)
            if data is not None and len(data) > 0:
                prices[holding.symbol] = data['close'].iloc[-1]
        
        self.api_client.close()
        return prices
    
    def _analyze_holdings(self, holdings: List, current_prices: Dict[str, float]) -> Dict:
        """Analyze each holding and determine action"""
        self.api_client.connect()
        
        analysis = {}
        
        for holding in holdings:
            symbol = holding.symbol
            current_price = current_prices.get(symbol, holding.average_price)
            
            # Fetch recent data
            data = self.api_client.get_historical_data(symbol, days=100)
            
            if data is None or len(data) < 50:
                analysis[symbol] = {
                    'action': 'HOLD',
                    'reason': 'Insufficient data for analysis',
                    'shares': holding.shares,
                    'current_price': current_price
                }
                continue
            
            # Analyze current signals
            analyzer = TrendAnalyzer(data)
            signals = analyzer.analyze()
            
            # Calculate performance
            gain_loss_pct = ((current_price - holding.average_price) / holding.average_price) * 100
            
            # Decision logic
            action = 'HOLD'
            reason = ''
            suggested_shares = 0
            
            # Sell conditions
            if signals['score'] <= -2:
                action = 'SELL'
                reason = f"Strong sell signal (Score: {signals['score']}). {signals['recommendation']}"
            elif gain_loss_pct < -15:
                action = 'SELL'
                reason = f"Stop loss triggered: {gain_loss_pct:.1f}% loss"
            elif gain_loss_pct > 30:
                action = 'SELL'
                reason = f"Take profit: {gain_loss_pct:.1f}% gain achieved"
            
            # Buy more conditions
            elif signals['score'] >= 3:
                action = 'BUY_MORE'
                reason = f"Strong buy signal (Score: {signals['score']}). Good opportunity to add."
                suggested_shares = max(1, int(holding.shares * 0.2))  # Suggest 20% more
            
            # Hold conditions
            else:
                action = 'HOLD'
                reason = f"Neutral signals (Score: {signals['score']}). Current P&L: {gain_loss_pct:+.1f}%"
            
            analysis[symbol] = {
                'action': action,
                'reason': reason,
                'shares': holding.shares,
                'current_price': current_price,
                'gain_loss_pct': gain_loss_pct,
                'signal_score': signals['score'],
                'suggested_shares': suggested_shares
            }
        
        self.api_client.close()
        return analysis
    
    def _find_new_opportunities(self, existing_holdings: List) -> List[Dict]:
        """Find new stocks to buy (not currently held)"""
        print("\n🔍 Scanning for new opportunities...")
        
        # Get list of currently held symbols
        held_symbols = [h.symbol for h in existing_holdings]
        
        # Quick scan of universe
        scanner = StockUniverseScanner(max_workers=3)
        scanner.connect()
        
        results = scanner.scan_universe(mode='quick', top_n=10, parallel=True)
        
        scanner.close()
        
        # Filter out stocks already held
        opportunities = []
        
        for _, row in results.iterrows():
            if row['status'] == 'OK' and row['symbol'] not in held_symbols:
                if row['current_recommendation'] in ['🟢 STRONG BUY', '🟡 BUY']:
                    opportunities.append({
                        'symbol': row['symbol'],
                        'action': 'BUY',
                        'reason': f"New opportunity: {row['current_recommendation']} (Score: {row['current_signal_score']})",
                        'current_price': row['current_price'],
                        'signal_score': row['current_signal_score'],
                        'suggested_shares': 0  # Will be calculated in allocation
                    })
        
        return opportunities[:5]  # Top 5 new opportunities
    
    def _optimize_allocation(self, recommendations: List[Dict], 
                           available_cash: float) -> List[Dict]:
        """Optimize share allocations based on available cash"""
        # Separate buy and non-buy recommendations
        buy_recs = [r for r in recommendations if r['action'] == 'BUY']
        other_recs = [r for r in recommendations if r['action'] != 'BUY']
        
        if not buy_recs or available_cash <= 0:
            return recommendations
        
        # Calculate total buy cost if we follow all recommendations
        total_cost = sum(r.get('estimated_cost', 0) for r in buy_recs)
        
        if total_cost <= available_cash:
            # We have enough cash for all recommendations
            return recommendations
        
        # Need to scale down or prioritize
        # Allocate based on signal strength and diversification
        
        # Sort by signal score (highest first)
        buy_recs.sort(key=lambda x: x.get('signal_score', 0), reverse=True)
        
        # Allocate cash proportionally
        cash_per_stock = available_cash / len(buy_recs)
        
        for rec in buy_recs:
            max_shares = int(cash_per_stock / rec['current_price'])
            rec['target_shares'] = max(1, max_shares)
            rec['estimated_cost'] = rec['target_shares'] * rec['current_price']
        
        return other_recs + buy_recs
    
    def _display_recommendations(self, recommendations: List[Dict], available_cash: float):
        """Display recommendations in a clear format"""
        print(f"\n{'='*80}")
        print(f"PORTFOLIO RECOMMENDATIONS")
        print(f"{'='*80}\n")
        
        print(f"💵 Available Cash: ${available_cash:,.2f}\n")
        
        # Group by action
        sells = [r for r in recommendations if r['action'] == 'SELL']
        buys = [r for r in recommendations if r['action'] == 'BUY']
        holds = [r for r in recommendations if r['action'] == 'HOLD']
        
        if sells:
            print(f"🔴 SELL RECOMMENDATIONS ({len(sells)}):")
            print("-" * 80)
            for i, rec in enumerate(sells, 1):
                print(f"{i}. {rec['symbol']}")
                print(f"   Shares: {rec['target_shares']:.0f} @ ${rec['current_price']:.2f}")
                print(f"   Expected Proceeds: ${rec.get('expected_proceeds', 0):,.2f}")
                print(f"   Reason: {rec['reason']}")
                print()
        
        if buys:
            print(f"🟢 BUY RECOMMENDATIONS ({len(buys)}):")
            print("-" * 80)
            total_buy_cost = 0
            for i, rec in enumerate(buys, 1):
                cost = rec.get('estimated_cost', 0)
                total_buy_cost += cost
                print(f"{i}. {rec['symbol']}")
                print(f"   Buy: {rec['target_shares']:.0f} shares @ ${rec['current_price']:.2f}")
                print(f"   Estimated Cost: ${cost:,.2f}")
                print(f"   Reason: {rec['reason']}")
                print()
            
            print(f"Total Buy Cost: ${total_buy_cost:,.2f}")
            print(f"Remaining Cash: ${available_cash - total_buy_cost:,.2f}\n")
        
        if holds:
            print(f"⚪ HOLD POSITIONS ({len(holds)}):")
            print("-" * 80)
            for rec in holds:
                print(f"• {rec['symbol']}: {rec['target_shares']:.0f} shares @ ${rec['current_price']:.2f}")
                print(f"  {rec['reason']}")
            print()
        
        print(f"{'='*80}\n")
    
    def execute_recommendations_interactive(self, portfolio_id: int):
        """Interactive mode to execute recommendations"""
        recommendations = self.pm.get_recommendations(portfolio_id)
        
        if not recommendations:
            print("📭 No pending recommendations")
            return
        
        print(f"\n{'='*80}")
        print("EXECUTE RECOMMENDATIONS")
        print(f"{'='*80}\n")
        
        executed = []
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n[{i}/{len(recommendations)}] {rec['action']}: {rec['symbol']}")
            print(f"Shares: {rec['target_shares']:.0f}")
            print(f"Reason: {rec['reason']}")
            
            response = input("\nDid you execute this? (y/n/skip): ").strip().lower()
            
            if response == 'y':
                # Record the transaction
                if rec['action'] == 'BUY':
                    price = float(input(f"Enter executed price (suggested: ${rec.get('current_price', 0):.2f}): $"))
                    shares = float(input(f"Enter shares bought (suggested: {rec['target_shares']:.0f}): "))
                    self.pm.add_holding(portfolio_id, rec['symbol'], shares, price)
                    executed.append(rec['symbol'])
                
                elif rec['action'] == 'SELL':
                    price = float(input(f"Enter executed price: $"))
                    shares = float(input(f"Enter shares sold: "))
                    self.pm.sell_holding(portfolio_id, rec['symbol'], shares, price)
                    executed.append(rec['symbol'])
            
            elif response == 'skip':
                break
        
        if executed:
            print(f"\n✅ Executed {len(executed)} transactions: {', '.join(executed)}")
        else:
            print("\n📝 No transactions executed")


if __name__ == "__main__":
    # Test recommender
    pm = PortfolioManager()
    recommender = PortfolioRecommender(pm)
    
    # Create test portfolio
    portfolio = pm.create_portfolio("Test Portfolio", 10000, 500)
    
    if portfolio:
        # Add some holdings
        pm.add_holding(portfolio.portfolio_id, 'US.AAPL', 10, 150.00)
        pm.add_holding(portfolio.portfolio_id, 'US.MSFT', 15, 300.00)
        
        # Generate recommendations
        recommendations = recommender.generate_recommendations(portfolio.portfolio_id)