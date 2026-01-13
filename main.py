"""
Stock Analysis Bot - Main Execution File
Portfolio-first approach with improved user experience
"""

import sys
import os
from datetime import datetime
import pandas as pd

# Add modules directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.api_client import MooMooAPIClient
from modules.trend_analyzer import TrendAnalyzer
from modules.backtesting import BacktestEngine
from modules.walk_forward import WalkForwardAnalyzer
from modules.universe_scanner import StockUniverseScanner
from modules.portfolio_manager import PortfolioManager
from modules.portfolio_recommender import PortfolioRecommender
from modules.paper_trading import PaperTradingSimulator
from utils.logger import setup_logger
import config

# Setup logging
logger = setup_logger()

# Global portfolio manager
pm = PortfolioManager()


def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"{title.center(80)}")
    print(f"{'='*80}\n")


def print_welcome():
    """Print welcome screen"""
    print_header("STOCK ANALYSIS BOT")
    print("Developed for systematic trading with MooMoo")
    print(f"Version: 1.0")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def portfolio_selection_menu():
    """Main portfolio selection menu"""
    while True:
        print_header("SELECT OR CREATE PORTFOLIO")
        
        # List existing portfolios
        portfolios = pm.list_portfolios()
        
        if portfolios:
            print("YOUR PORTFOLIOS:")
            print(f"{'ID':<5} {'Name':<30} {'Total Invested':<18} {'Created':<20}")
            print("-" * 80)
            for p in portfolios:
                print(f"{p.portfolio_id:<5} {p.name:<30} ${p.total_invested:<17,.2f} {p.created_date:<20}")
            print()
        else:
            print("📭 No portfolios found. Create your first portfolio to get started!\n")
        
        print("OPTIONS:")
        print("  N  - Create New Portfolio")
        if portfolios:
            print("  [ID] - Select portfolio by ID (e.g., 1, 2, 3)")
        print("  Q  - Quit")
        print()
        
        choice = input("Your choice: ").strip().upper()
        
        if choice == 'Q':
            print("\n👋 Goodbye!")
            return None
        
        elif choice == 'N':
            portfolio = create_new_portfolio()
            if portfolio:
                return portfolio.portfolio_id
        
        else:
            # Try to parse as portfolio ID
            try:
                portfolio_id = int(choice)
                portfolio = pm.get_portfolio(portfolio_id)
                if portfolio:
                    return portfolio_id
                else:
                    print(f"\n❌ Portfolio {portfolio_id} not found")
            except ValueError:
                print("\n❌ Invalid choice")
        
        input("\nPress Enter to continue...")


def create_new_portfolio():
    """Create a new portfolio"""
    print_header("CREATE NEW PORTFOLIO")
    
    print("Set up your investment portfolio\n")
    
    name = input("Portfolio Name (e.g., 'Growth Portfolio', 'Retirement Fund'): ").strip()
    if not name:
        print("❌ Portfolio name cannot be empty")
        return None
    
    try:
        print("\n💰 INVESTMENT AMOUNTS:")
        initial_investment = float(input("Initial Investment Amount: $").strip().replace(',', ''))
        recurring_amount = float(input("Monthly Recurring Investment: $").strip().replace(',', ''))
        
        print("\n📝 NOTES (Optional):")
        notes = input("Strategy/Notes (press Enter to skip): ").strip()
        
        portfolio = pm.create_portfolio(name, initial_investment, recurring_amount, notes)
        
        if portfolio:
            print(f"\n✅ Portfolio '{name}' created successfully!")
            print(f"\n💡 Next steps:")
            print(f"   1. Add your initial holdings")
            print(f"   2. Generate recommendations")
            print(f"   3. Start scanning for opportunities")
            input("\nPress Enter to continue...")
            return portfolio
        
    except ValueError:
        print("❌ Invalid amount entered")
        return None


def portfolio_management_menu(portfolio_id: int):
    """Portfolio management and scanning menu"""
    while True:
        portfolio = pm.get_portfolio(portfolio_id)
        if not portfolio:
            print("❌ Portfolio not found")
            return
        
        # Get current portfolio status
        holdings = pm.get_holdings(portfolio_id)
        
        # Get current prices for status
        if holdings:
            api_client = MooMooAPIClient()
            api_client.connect()
            current_prices = {}
            for h in holdings:
                data = api_client.get_historical_data(h.symbol, days=1)
                if data is not None and len(data) > 0:
                    current_prices[h.symbol] = data['close'].iloc[-1]
            api_client.close()
            
            summary = pm.calculate_portfolio_value(portfolio_id, current_prices)
        else:
            summary = None
        
        # Display portfolio header
        print_header(f"PORTFOLIO: {portfolio.name}")
        
        if summary:
            total_return_color = "🟢" if summary['total_return_pct'] >= 0 else "🔴"
            print(f"💰 Total Value: ${summary['total_value']:,.2f} | "
                  f"Invested: ${summary['total_invested']:,.2f} | "
                  f"{total_return_color} Return: {summary['total_return_pct']:+.2f}%")
            print(f"📊 Holdings: {len(holdings)} positions | Cash: ${summary['cash_available']:,.2f}")
        else:
            print(f"💰 Total Invested: ${portfolio.total_invested:,.2f}")
            print(f"📊 No holdings yet - Add your first stocks!")
        
        print()
        print("="*80)
        
        print("\nPORTFOLIO MANAGEMENT:")
        print("  1. View Portfolio Details")
        print("  2. Add Holdings (Buy Stocks)")
        print("  3. Sell Holdings")
        print("  4. View Transaction History")
        
        print("\nRECOMMENDATIONS & SCANNING:")
        print("  5. Generate Portfolio Recommendations")
        print("  6. Execute Recommendations")
        print("  7. Quick Universe Scan")
        print("  8. Three-Tier Deep Scan")
        
        print("\nANALYSIS TOOLS:")
        print("  9. Analyze Single Stock")
        print("  10. Backtest Stock Strategy")
        print("  11. Run Paper Trading Simulation")
        
        print("\n  0. Back to Portfolio Selection")
        print()
        
        choice = input("Your choice: ").strip()
        
        if choice == '0':
            return
        
        elif choice == '1':
            view_portfolio_details(portfolio_id)
        
        elif choice == '2':
            add_holdings_to_portfolio(portfolio_id)
        
        elif choice == '3':
            sell_holdings_from_portfolio(portfolio_id)
        
        elif choice == '4':
            view_transaction_history(portfolio_id)
        
        elif choice == '5':
            generate_portfolio_recommendations(portfolio_id)
        
        elif choice == '6':
            execute_portfolio_recommendations(portfolio_id)
        
        elif choice == '7':
            quick_scan_for_portfolio(portfolio_id)
        
        elif choice == '8':
            deep_scan_for_portfolio(portfolio_id)
        
        elif choice == '9':
            analyze_stock_for_portfolio(portfolio_id)
        
        elif choice == '10':
            backtest_stock_for_portfolio(portfolio_id)

        elif choice == '11':
            run_paper_trading_simulation(portfolio_id)
        
        else:
            print("❌ Invalid choice")
        
        input("\nPress Enter to continue...")


def view_portfolio_details(portfolio_id: int):
    """View detailed portfolio information"""
    print_header("PORTFOLIO DETAILS")
    
    portfolio = pm.get_portfolio(portfolio_id)
    holdings = pm.get_holdings(portfolio_id)
    
    if not holdings:
        print("📭 No holdings in this portfolio yet")
        print("\n💡 Use option 2 to add your first stocks")
        return
    
    # Get current prices
    api_client = MooMooAPIClient()
    api_client.connect()
    
    current_prices = {}
    print("Fetching current prices...")
    for h in holdings:
        data = api_client.get_historical_data(h.symbol, days=1)
        if data is not None and len(data) > 0:
            current_prices[h.symbol] = data['close'].iloc[-1]
    
    api_client.close()
    
    # Display summary
    pm.display_portfolio_summary(portfolio_id, current_prices)


def add_holdings_to_portfolio(portfolio_id: int):
    """Add holdings to portfolio"""
    print_header("ADD HOLDINGS")
    
    print("Add stocks to your portfolio\n")
    
    while True:
        symbol = input("Stock Symbol (e.g., US.AAPL) or 'done' to finish: ").strip().upper()
        
        if symbol.lower() == 'done':
            break
        
        try:
            shares = float(input(f"Number of shares: ").strip())
            price = float(input(f"Purchase price per share: $").strip().replace(',', ''))
            
            success = pm.add_holding(portfolio_id, symbol, shares, price)
            
            if success:
                print(f"✅ Added to portfolio")
            
            another = input("\nAdd another stock? (y/n): ").strip().lower()
            if another != 'y':
                break
                
        except ValueError:
            print("❌ Invalid input")


def sell_holdings_from_portfolio(portfolio_id: int):
    """Sell holdings from portfolio"""
    print_header("SELL HOLDINGS")
    
    holdings = pm.get_holdings(portfolio_id)
    
    if not holdings:
        print("📭 No holdings to sell")
        return
    
    print("Your current holdings:")
    for i, h in enumerate(holdings, 1):
        print(f"  {i}. {h.symbol}: {h.shares:.2f} shares @ ${h.average_price:.2f} avg")
    print()
    
    try:
        choice = int(input("Select holding to sell (number): ").strip())
        if 1 <= choice <= len(holdings):
            holding = holdings[choice - 1]
            
            print(f"\nSelling {holding.symbol} ({holding.shares:.2f} shares available)")
            shares = float(input("Number of shares to sell: ").strip())
            price = float(input(f"Sale price per share: $").strip().replace(',', ''))
            
            success = pm.sell_holding(portfolio_id, holding.symbol, shares, price)
            
            if success:
                proceeds = shares * price
                print(f"\n✅ Sold {shares} shares for ${proceeds:,.2f}")
        else:
            print("❌ Invalid selection")
    
    except ValueError:
        print("❌ Invalid input")


def view_transaction_history(portfolio_id: int):
    """View transaction history"""
    print_header("TRANSACTION HISTORY")
    
    transactions = pm.get_transactions(portfolio_id, limit=20)
    
    if not transactions:
        print("📭 No transactions yet")
        return
    
    print(f"Last {len(transactions)} transactions:\n")
    print(f"{'Date':<20} {'Type':<6} {'Symbol':<10} {'Shares':<10} {'Price':<12} {'Total':<15}")
    print("-" * 80)
    
    for t in transactions:
        type_emoji = "🟢" if t.transaction_type == "BUY" else "🔴"
        print(f"{t.transaction_date:<20} {type_emoji}{t.transaction_type:<5} {t.symbol:<10} "
              f"{t.shares:<10.2f} ${t.price:<11.2f} ${t.total_amount:<14,.2f}")


def generate_portfolio_recommendations(portfolio_id: int):
    """Generate recommendations for portfolio"""
    print_header("GENERATE RECOMMENDATIONS")
    
    portfolio = pm.get_portfolio(portfolio_id)
    
    # Check if portfolio has holdings
    holdings = pm.get_holdings(portfolio_id)
    if not holdings:
        print("⚠️  No holdings found. Recommendations will focus on new opportunities.")
        print()
    
    # Ask about additional cash
    print(f"Current recurring amount: ${portfolio.recurring_amount:,.2f}")
    add_more = input("\nAdd additional cash for this investment cycle? (y/n): ").strip().lower()
    
    available_cash = None
    if add_more == 'y':
        try:
            additional = float(input("Enter additional amount: $").strip().replace(',', ''))
            available_cash = portfolio.recurring_amount + additional
            print(f"\n💵 Total available for investment: ${available_cash:,.2f}")
        except ValueError:
            print("❌ Invalid amount, using recurring amount only")
    
    # Generate recommendations
    print("\n🔍 Analyzing portfolio and scanning opportunities...")
    print("This may take a few minutes...\n")
    
    recommender = PortfolioRecommender(pm)
    recommendations = recommender.generate_recommendations(portfolio_id, available_cash)
    
    if recommendations:
        print(f"\n✅ Generated {len(recommendations)} recommendations")
        print("💡 Use option 6 to execute these recommendations")


def execute_portfolio_recommendations(portfolio_id: int):
    """Execute portfolio recommendations interactively"""
    print_header("EXECUTE RECOMMENDATIONS")
    
    recommendations = pm.get_recommendations(portfolio_id)
    
    if not recommendations:
        print("📭 No pending recommendations")
        print("\n💡 Use option 5 to generate recommendations first")
        return
    
    print(f"You have {len(recommendations)} pending recommendations\n")
    
    recommender = PortfolioRecommender(pm)
    recommender.execute_recommendations_interactive(portfolio_id)


def quick_scan_for_portfolio(portfolio_id: int):
    """Quick universe scan optimized for portfolio"""
    print_header("QUICK UNIVERSE SCAN")
    
    print("🔍 Scanning stock universe for opportunities...")
    print("Mode: Quick (5-10 minutes)\n")
    
    scanner = StockUniverseScanner()
    scanner.connect()
    
    # Get current holdings to exclude
    holdings = pm.get_holdings(portfolio_id)
    held_symbols = [h.symbol for h in holdings]
    
    results = scanner.scan_universe(mode='quick', top_n=20, parallel=True)
    
    # Filter and show results
    print(f"\n{'='*80}")
    print("TOP OPPORTUNITIES (excluding your current holdings)")
    print(f"{'='*80}\n")
    
    new_opportunities = results[~results['symbol'].isin(held_symbols)]
    
    if len(new_opportunities) > 0:
        display_cols = ['symbol', 'current_recommendation', 'current_signal_score', 
                       'current_price', 'overall_score']
        print(new_opportunities.head(10)[display_cols].to_string(index=False))
    else:
        print("No new opportunities found")
    
    # Export
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{config.SCAN_PATH}/portfolio_{portfolio_id}_scan_{timestamp}.csv"
    scanner.export_results(results, filename)
    
    scanner.close()


def deep_scan_for_portfolio(portfolio_id: int):
    """Three-tier deep scan for portfolio"""
    print_header("THREE-TIER DEEP SCAN")
    
    print("🔍 Performing comprehensive analysis...")
    print("This will take 30-60 minutes but provides high-confidence recommendations\n")
    
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        return
    
    scanner = StockUniverseScanner()
    scanner.connect()
    
    # Tier 1: Quick Scan
    print("\n" + "="*80)
    print("TIER 1: QUICK SCAN")
    print("="*80)
    quick_results = scanner.scan_universe(mode='quick', top_n=30, parallel=True)
    
    # Tier 2: Backtest top candidates
    print("\n" + "="*80)
    print("TIER 2: BACKTEST VALIDATION")
    print("="*80)
    tier1_symbols = quick_results.nlargest(15, 'overall_score')['symbol'].tolist()
    backtest_results = scanner.scan_universe(symbols=tier1_symbols, mode='backtest', 
                                            top_n=10, parallel=False)
    
    # Tier 3: Walk-forward on best
    print("\n" + "="*80)
    print("TIER 3: WALK-FORWARD VALIDATION")
    print("="*80)
    tier2_symbols = backtest_results.nlargest(8, 'overall_score')['symbol'].tolist()
    final_results = scanner.scan_universe(symbols=tier2_symbols, mode='full', 
                                         top_n=5, parallel=False)
    
    # Display final recommendations
    print_header("DEEP SCAN RESULTS")
    
    print("🎯 HIGH-CONFIDENCE OPPORTUNITIES:\n")
    
    if len(final_results) > 0:
        for i, row in final_results.head(5).iterrows():
            print(f"{row['rank']}. {row['symbol']} - {row['current_recommendation']}")
            print(f"   WF Avg Return: {row['wf_avg_return']:+.2f}%")
            print(f"   Consistency: {row['wf_consistency']:.1f}/100")
            print(f"   Price: ${row['current_price']:.2f}")
            print()
    
    scanner.close()


def analyze_stock_for_portfolio(portfolio_id: int):
    """Analyze a specific stock"""
    print_header("ANALYZE SINGLE STOCK")
    
    symbol = input("Enter stock symbol (e.g., US.AAPL): ").strip().upper()
    
    api_client = MooMooAPIClient()
    api_client.connect()
    
    print(f"\nFetching data for {symbol}...")
    data = api_client.get_historical_data(symbol, days=100)
    
    if data is None or len(data) < 50:
        print(f"❌ Unable to fetch data for {symbol}")
        api_client.close()
        return
    
    # Analyze
    analyzer = TrendAnalyzer(data)
    results = analyzer.analyze()
    
    # Display
    print(f"\n{'='*80}")
    print(f"ANALYSIS: {symbol}")
    print(f"{'='*80}\n")
    
    print(f"Current Price: ${results['current_price']:.2f}")
    print(f"Recommendation: {results['recommendation']}")
    print(f"Signal Score: {results['score']}")
    print(f"Signal Strength: {analyzer.get_signal_strength()}")
    print(f"Risk Level: {analyzer.get_risk_level()}")
    
    print(f"\n📊 Technical Indicators:")
    print(f"  RSI: {results['rsi']:.2f}")
    print(f"  MACD: {results['macd']:.4f}")
    print(f"  Volume Ratio: {results['volume_ratio']:.2f}x average")
    print(f"  Volatility: {results['volatility']:.2f}%")
    
    print(f"\n📈 Signals Detected:")
    for signal in results['signals']:
        print(f"  {signal}")
    
    api_client.close()


def backtest_stock_for_portfolio(portfolio_id: int):
    """Backtest a stock strategy"""
    print_header("BACKTEST STOCK STRATEGY")
    
    symbol = input("Enter stock symbol (e.g., US.AAPL): ").strip().upper()
    
    api_client = MooMooAPIClient()
    api_client.connect()
    
    print(f"\nFetching historical data for {symbol}...")
    data = api_client.get_historical_data(symbol, days=500)
    
    if data is None or len(data) < 100:
        print(f"❌ Insufficient data for {symbol}")
        api_client.close()
        return
    
    # Run backtest
    engine = BacktestEngine()
    results = engine.run_backtest(symbol, data)
    
    # Plot results
    if 'error' not in results:
        plot = input("\nGenerate charts? (y/n): ").strip().lower()
        if plot == 'y':
            engine.plot_results(symbol)
    
    api_client.close()


def run_paper_trading_simulation(portfolio_id: int):
    """Run paper trading simulation to test strategy"""
    print_header("PAPER TRADING SIMULATION")
    
    print("📝 Day-by-day trading simulation using historical data")
    print("   Unlike backtesting, this shows you each day's decisions in sequence")
    print("   Perfect for understanding the 'lived experience' of your strategy
")
    
    # Quick start vs custom configuration
    print("SIMULATION MODE:")
    print("-" * 80)
    print("  Q. Quick Start    - 30 days, default settings (recommended for first try)")
    print("  C. Custom Setup   - Full control over all parameters")
    print()
    
    mode = input("Your choice (Q/C, default: Q): ").strip().upper() or "Q"
    
    try:
        if mode == 'Q':
            # Quick start with sensible defaults
            from datetime import timedelta
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            num_days = 30
            initial_capital = 10000.0
            buy_threshold = config.DEFAULT_BUY_THRESHOLD
            sell_threshold = config.DEFAULT_SELL_THRESHOLD
            auto_execute = True
            
            print("
🚀 QUICK START CONFIGURATION:")
            print("=" * 80)
            print(f"Start Date:        {start_date} (~60 days ago)")
            print(f"Duration:          {num_days} trading days")
            print(f"Initial Capital:   ${initial_capital:,.2f}")
            print(f"Buy Threshold:     {buy_threshold} (Buy on scores >= {buy_threshold})")
            print(f"Sell Threshold:    {sell_threshold} (Sell on scores <= {sell_threshold})")
            print(f"Execution Mode:    Auto-Execute")
            print(f"Estimated Time:    ~{num_days * 2} seconds ({num_days} days × 2 sec/day)")
            print("=" * 80)
            
            confirm = input("
Start quick simulation? (y/n): ").strip().lower()
            if confirm != 'y':
                print("❌ Simulation cancelled")
                return
        
        else:
            # Custom configuration with better UX
            print("
📋 CUSTOM CONFIGURATION:")
            print("=" * 80)
            
            # Date selection with presets
            print("
START DATE:")
            print("  1. 30 days ago")
            print("  2. 60 days ago (recommended)")
            print("  3. 90 days ago")
            print("  4. Custom date (YYYY-MM-DD)")
            
            date_choice = input("
Your choice (1-4, default: 2): ").strip() or "2"
            
            from datetime import timedelta
            if date_choice == "1":
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            elif date_choice == "2":
                start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            elif date_choice == "3":
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            else:
                while True:
                    start_date = input("Enter date (YYYY-MM-DD, e.g., 2024-01-01): ").strip()
                    try:
                        input_date = datetime.strptime(start_date, '%Y-%m-%d')
                        if input_date >= datetime.now():
                            print("⚠️  Date must be in the past")
                        elif input_date < datetime.now() - timedelta(days=730):
                            confirm = input("⚠️  Date is >2 years old. Data may be limited. Continue? (y/n): ").lower()
                            if confirm == 'y':
                                break
                        else:
                            break
                    except ValueError:
                        print("❌ Invalid format. Use YYYY-MM-DD")
            
            print(f"✓ Start date: {start_date}")
            
            # Number of days with validation
            while True:
                try:
                    days_input = input("
Number of days to simulate (1-365, default: 30): ").strip()
                    if not days_input:
                        num_days = 30
                        break
                    num_days = int(days_input)
                    if 1 <= num_days <= 365:
                        break
                    else:
                        print("⚠️  Please enter a number between 1 and 365")
                except ValueError:
                    print("⚠️  Please enter a valid number")
            
            print(f"✓ Duration: {num_days} days")
            
            # Capital with better validation
            while True:
                try:
                    capital_input = input("
Initial capital (e.g., 10000 or 10,000, default: $10,000): ").strip()
                    if not capital_input:
                        initial_capital = 10000.0
                        break
                    # Remove formatting
                    capital_input = capital_input.replace('$', '').replace(',', '')
                    initial_capital = float(capital_input)
                    
                    if initial_capital <= 0:
                        print("⚠️  Capital must be positive")
                    elif initial_capital < 1000:
                        confirm = input(f"⚠️  ${initial_capital:,.2f} is quite low. Continue? (y/n): ").lower()
                        if confirm == 'y':
                            break
                    elif initial_capital > 1000000:
                        confirm = input(f"💰 ${initial_capital:,.2f} is large. Confirm? (y/n): ").lower()
                        if confirm == 'y':
                            break
                    else:
                        break
                except ValueError:
                    print("⚠️  Please enter a valid dollar amount")
            
            print(f"✓ Capital: ${initial_capital:,.2f}")
            
            # Strategy settings
            print("
STRATEGY SETTINGS:")
            while True:
                try:
                    buy_input = input(f"Buy signal threshold (default: {config.DEFAULT_BUY_THRESHOLD}): ").strip()
                    buy_threshold = int(buy_input) if buy_input else config.DEFAULT_BUY_THRESHOLD
                    if -5 <= buy_threshold <= 5:
                        break
                    print("⚠️  Threshold must be between -5 and 5")
                except ValueError:
                    print("⚠️  Please enter a valid number")
            
            while True:
                try:
                    sell_input = input(f"Sell signal threshold (default: {config.DEFAULT_SELL_THRESHOLD}): ").strip()
                    sell_threshold = int(sell_input) if sell_input else config.DEFAULT_SELL_THRESHOLD
                    if -5 <= sell_threshold <= 5:
                        break
                    print("⚠️  Threshold must be between -5 and 5")
                except ValueError:
                    print("⚠️  Please enter a valid number")
            
            print(f"✓ Thresholds: Buy={buy_threshold}, Sell={sell_threshold}")
            
            # Execution mode
            print("
EXECUTION MODE:")
            print("  1. Auto-Execute (Automatically follow all recommendations)")
            print("  2. Manual Review (Review each day, more educational)")
            mode_choice = input("
Your choice (1-2, default: 1): ").strip() or "1"
            auto_execute = (mode_choice == "1")
            
            print(f"✓ Mode: {'Auto-Execute' if auto_execute else 'Manual Review'}")
            
            # Final confirmation
            print("
" + "=" * 80)
            print("SIMULATION SUMMARY:")
            print("=" * 80)
            print(f"Start Date:        {start_date}")
            print(f"Duration:          {num_days} trading days")
            print(f"Initial Capital:   ${initial_capital:,.2f}")
            print(f"Buy Threshold:     {buy_threshold}")
            print(f"Sell Threshold:    {sell_threshold}")
            print(f"Execution Mode:    {'Auto-Execute' if auto_execute else 'Manual Review'}")
            print(f"Estimated Time:    ~{num_days * 2} seconds")
            print("=" * 80)
            
            confirm = input("
Start simulation? (y/n): ").strip().lower()
            if confirm != 'y':
                print("❌ Simulation cancelled")
                return
        
        # Run simulation
        print("
🚀 Starting simulation...")
        print("Progress will be shown below.
")
        
        simulator = PaperTradingSimulator(initial_capital=initial_capital)
        
        results = simulator.run_simulation(
            start_date=start_date,
            num_days=num_days,
            auto_execute=auto_execute,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold
        )
        
        # Post-simulation options
        if results:
            print("
" + "=" * 80)
            print("SIMULATION COMPLETE!")
            print("=" * 80)
            print(f"✓ Chart saved to: data/charts/paper_trading_*.png")
            print()
            
            save = input("Save detailed results to CSV? (y/n): ").strip().lower()
            
            if save == 'y':
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{config.BACKTEST_PATH}/paper_trading_sim_{timestamp}.csv"
                
                results_df = pd.DataFrame([results])
                results_df.to_csv(filename, index=False)
                
                print(f"✅ Results saved: {filename}")
                print("💡 TIP: Compare multiple runs by checking the data/backtests/ folder")
            
            # Option to run again
            print()
            run_again = input("Run another simulation with different parameters? (y/n): ").strip().lower()
            if run_again == 'y':
                run_paper_trading_simulation(portfolio_id)
        
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
        print("💡 TIP: Try Quick Start mode (Q) for easier setup")
    except KeyboardInterrupt:
        print("

⚠️  Simulation interrupted by user")
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        print("💡 TIP: Check that MooMoo API is connected and data is available")


def main():
    """Main entry point"""
    # Create necessary directories
    os.makedirs(config.DATA_PATH, exist_ok=True)
    os.makedirs(config.SCAN_PATH, exist_ok=True)
    os.makedirs(config.BACKTEST_PATH, exist_ok=True)
    os.makedirs(config.CHART_PATH, exist_ok=True)
    
    # Print welcome
    print_welcome()
    
    # Main loop
    while True:
        # Portfolio selection
        portfolio_id = portfolio_selection_menu()
        
        if portfolio_id is None:
            break
        
        # Portfolio management menu
        portfolio_management_menu(portfolio_id)
    
    print("\n✅ Session ended. Happy trading! 📈")


if __name__ == "__main__":
    main()