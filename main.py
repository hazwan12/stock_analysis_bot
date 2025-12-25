"""
Stock Analysis Bot - Main Execution File

This is the primary entry point for running the stock analysis bot.
Choose from various analysis modes and features.
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
from utils.logger import setup_logger
import config

# Setup logging
logger = setup_logger()


def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"{title.center(80)}")
    print(f"{'='*80}\n")


def analyze_single_stock(symbol: str):
    """Analyze a single stock with current signals"""
    print_header(f"ANALYZING {symbol}")
    
    api_client = MooMooAPIClient()
    api_client.connect()
    
    # Fetch data
    data = api_client.get_historical_data(symbol, days=100)
    
    # Analyze
    analyzer = TrendAnalyzer(data)
    results = analyzer.analyze()
    
    # Display results
    print(f"Symbol: {symbol}")
    print(f"Current Price: ${results['current_price']:.2f}")
    print(f"\nRECOMMENDATION: {results['recommendation']}")
    print(f"Signal Score: {results['score']}")
    print(f"\nTechnical Indicators:")
    print(f"  RSI: {results['rsi']:.2f}")
    print(f"  MACD: {results['macd']:.4f}")
    print(f"  Volume Ratio: {results['volume_ratio']:.2f}x")
    print(f"\nSignals:")
    for signal in results['signals']:
        print(f"  {signal}")
    
    api_client.close()
    return results


def backtest_stock(symbol: str):
    """Run backtest on a single stock"""
    print_header(f"BACKTESTING {symbol}")
    
    api_client = MooMooAPIClient()
    api_client.connect()
    
    # Fetch data
    data = api_client.get_historical_data(symbol, days=config.BACKTEST_DAYS)
    
    # Run backtest
    engine = BacktestEngine(
        initial_capital=config.INITIAL_CAPITAL,
        commission=config.BACKTEST_COMMISSION
    )
    
    results = engine.run_backtest(
        symbol=symbol,
        data=data,
        buy_threshold=config.DEFAULT_BUY_THRESHOLD,
        sell_threshold=config.DEFAULT_SELL_THRESHOLD,
        stop_loss_pct=config.DEFAULT_STOP_LOSS,
        take_profit_pct=config.DEFAULT_TAKE_PROFIT
    )
    
    # Plot results
    engine.plot_results(symbol)
    
    api_client.close()
    return results


def walk_forward_test(symbol: str):
    """Run walk-forward analysis on a stock"""
    print_header(f"WALK-FORWARD TESTING {symbol}")
    
    api_client = MooMooAPIClient()
    api_client.connect()
    
    # Fetch data
    data = api_client.get_historical_data(symbol, days=750)
    
    # Create analyzer
    wf_analyzer = WalkForwardAnalyzer(
        symbol=symbol,
        data=data,
        initial_capital=config.INITIAL_CAPITAL,
        train_window_days=config.WF_TRAIN_WINDOW,
        test_window_days=config.WF_TEST_WINDOW,
        step_size=config.WF_STEP_SIZE
    )
    
    # Parameter grid
    param_grid = {
        'buy_threshold': [1, 2, 3],
        'sell_threshold': [-1, -2, -3],
        'stop_loss_pct': [0.03, 0.05, 0.07],
        'take_profit_pct': [0.10, 0.15, 0.20]
    }
    
    # Run analysis
    results = wf_analyzer.run_walk_forward_test(param_grid)
    
    # Visualize and export
    wf_analyzer.plot_walk_forward_results()
    wf_analyzer.export_results()
    
    api_client.close()
    return results


def scan_universe(mode: str = 'quick'):
    """Scan stock universe"""
    print_header(f"UNIVERSE SCAN - {mode.upper()} MODE")
    
    scanner = StockUniverseScanner(
        initial_capital=config.INITIAL_CAPITAL,
        min_price=config.SCANNER_MIN_PRICE,
        max_price=config.SCANNER_MAX_PRICE,
        min_avg_volume=config.SCANNER_MIN_VOLUME,
        max_workers=config.SCANNER_MAX_WORKERS
    )
    
    scanner.connect()
    
    # Run scan
    results = scanner.scan_universe(
        mode=mode,
        top_n=20,
        parallel=True
    )
    
    # Export results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{config.SCAN_PATH}/scan_{mode}_{timestamp}.csv"
    scanner.export_results(results, filename)
    
    scanner.close()
    return results


def full_pipeline(symbol: str):
    """Run complete analysis pipeline on a stock"""
    print_header(f"FULL ANALYSIS PIPELINE - {symbol}")
    
    print("Stage 1/3: Current Signal Analysis...")
    signal_results = analyze_single_stock(symbol)
    
    print("\nStage 2/3: Backtesting...")
    backtest_results = backtest_stock(symbol)
    
    print("\nStage 3/3: Walk-Forward Validation...")
    wf_results = walk_forward_test(symbol)
    
    # Summary
    print_header("PIPELINE SUMMARY")
    print(f"Symbol: {symbol}")
    print(f"\nCurrent Analysis:")
    print(f"  Recommendation: {signal_results['recommendation']}")
    print(f"  Signal Score: {signal_results['score']}")
    
    if 'error' not in backtest_results:
        print(f"\nBacktest Results:")
        print(f"  Total Return: {backtest_results['total_return']:+.2f}%")
        print(f"  Win Rate: {backtest_results['win_rate']:.1f}%")
        print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    
    if 'error' not in wf_results:
        print(f"\nWalk-Forward Results:")
        print(f"  Avg Return: {wf_results['avg_test_return']:+.2f}%")
        print(f"  Consistency: {wf_results['consistency_score']:.1f}/100")
        print(f"  Win Windows: {wf_results['win_rate_windows']:.1f}%")
        print(f"  Degradation: {wf_results['degradation_pct']:+.1f}%")
    
    return {
        'signals': signal_results,
        'backtest': backtest_results,
        'walk_forward': wf_results
    }


def interactive_menu():
    """Interactive menu for selecting operations"""
    while True:
        print_header("STOCK ANALYSIS BOT - MAIN MENU")
        print("1. Analyze Single Stock (Current Signals)")
        print("2. Backtest Single Stock")
        print("3. Walk-Forward Test Single Stock")
        print("4. Full Pipeline (All 3 analyses)")
        print("5. Quick Universe Scan")
        print("6. Backtest Universe Scan")
        print("7. Full Universe Scan (Walk-Forward)")
        print("8. Three-Tier Universe Scan (Quick → Backtest → Full)")
        print("9. Exit")
        print()
        
        choice = input("Enter your choice (1-9): ").strip()
        
        if choice == '1':
            symbol = input("Enter stock symbol (e.g., US.AAPL): ").strip()
            analyze_single_stock(symbol)
            
        elif choice == '2':
            symbol = input("Enter stock symbol (e.g., US.AAPL): ").strip()
            backtest_stock(symbol)
            
        elif choice == '3':
            symbol = input("Enter stock symbol (e.g., US.AAPL): ").strip()
            walk_forward_test(symbol)
            
        elif choice == '4':
            symbol = input("Enter stock symbol (e.g., US.AAPL): ").strip()
            full_pipeline(symbol)
            
        elif choice == '5':
            scan_universe(mode='quick')
            
        elif choice == '6':
            scan_universe(mode='backtest')
            
        elif choice == '7':
            scan_universe(mode='full')
            
        elif choice == '8':
            three_tier_scan()
            
        elif choice == '9':
            print("\nExiting... Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")


def three_tier_scan():
    """Run three-tier progressive universe scan"""
    print_header("THREE-TIER UNIVERSE SCAN")
    
    scanner = StockUniverseScanner(
        initial_capital=config.INITIAL_CAPITAL,
        min_price=config.SCANNER_MIN_PRICE,
        max_price=config.SCANNER_MAX_PRICE,
        min_avg_volume=config.SCANNER_MIN_VOLUME,
        max_workers=config.SCANNER_MAX_WORKERS
    )
    
    scanner.connect()
    
    # Tier 1: Quick Scan
    print("\n" + "="*80)
    print("TIER 1: QUICK SCAN")
    print("="*80)
    quick_results = scanner.scan_universe(mode='quick', top_n=30)
    
    # Create watchlist for tier 2
    tier1_watchlist = scanner.create_watchlist(quick_results, {
        'min_score': 20,
        'recommendations': ['🟢 STRONG BUY', '🟡 BUY']
    })
    
    # Tier 2: Backtest Scan
    print("\n" + "="*80)
    print("TIER 2: BACKTEST SCAN")
    print("="*80)
    backtest_results = scanner.scan_universe(
        symbols=tier1_watchlist,
        mode='backtest',
        top_n=15
    )
    
    # Create watchlist for tier 3
    tier2_watchlist = scanner.create_watchlist(backtest_results, {
        'min_return': 5.0,
        'min_score': 50
    })
    
    # Tier 3: Walk-Forward Scan
    print("\n" + "="*80)
    print("TIER 3: WALK-FORWARD SCAN")
    print("="*80)
    full_results = scanner.scan_universe(
        symbols=tier2_watchlist,
        mode='full',
        top_n=10
    )
    
    # Final watchlist
    final_watchlist = scanner.create_watchlist(full_results, {
        'min_return': 3.0,
        'max_degradation': 40.0,
        'min_consistency': 60.0
    })
    
    # Export all results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    scanner.export_results(quick_results, 
                          f"{config.SCAN_PATH}/tier1_quick_{timestamp}.csv")
    scanner.export_results(backtest_results, 
                          f"{config.SCAN_PATH}/tier2_backtest_{timestamp}.csv")
    scanner.export_results(full_results, 
                          f"{config.SCAN_PATH}/tier3_full_{timestamp}.csv")
    
    # Summary
    print_header("THREE-TIER SCAN COMPLETE")
    print(f"Tier 1 (Quick): {len(quick_results)} stocks scanned → {len(tier1_watchlist)} qualified")
    print(f"Tier 2 (Backtest): {len(tier1_watchlist)} stocks analyzed → {len(tier2_watchlist)} qualified")
    print(f"Tier 3 (Walk-Forward): {len(tier2_watchlist)} stocks validated → {len(final_watchlist)} qualified")
    print(f"\n🎯 FINAL HIGH-CONFIDENCE WATCHLIST:")
    print(f"   {', '.join(final_watchlist)}")
    
    scanner.close()
    return final_watchlist


def main():
    """Main entry point"""
    print_header("STOCK ANALYSIS BOT")
    print("Developed for systematic trading with MooMoo")
    print(f"Version: 1.0")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create necessary directories
    os.makedirs(config.DATA_PATH, exist_ok=True)
    os.makedirs(config.SCAN_PATH, exist_ok=True)
    os.makedirs(config.BACKTEST_PATH, exist_ok=True)
    os.makedirs(config.CHART_PATH, exist_ok=True)
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'analyze' and len(sys.argv) > 2:
            symbol = sys.argv[2]
            analyze_single_stock(symbol)
            
        elif command == 'backtest' and len(sys.argv) > 2:
            symbol = sys.argv[2]
            backtest_stock(symbol)
            
        elif command == 'walkforward' and len(sys.argv) > 2:
            symbol = sys.argv[2]
            walk_forward_test(symbol)
            
        elif command == 'pipeline' and len(sys.argv) > 2:
            symbol = sys.argv[2]
            full_pipeline(symbol)
            
        elif command == 'scan':
            mode = sys.argv[2] if len(sys.argv) > 2 else 'quick'
            scan_universe(mode)
            
        elif command == 'threescan':
            three_tier_scan()
            
        else:
            print("Invalid command. Use:")
            print("  python main.py analyze <SYMBOL>")
            print("  python main.py backtest <SYMBOL>")
            print("  python main.py walkforward <SYMBOL>")
            print("  python main.py pipeline <SYMBOL>")
            print("  python main.py scan [quick|backtest|full]")
            print("  python main.py threescan")
    else:
        # Run interactive menu
        interactive_menu()


if __name__ == "__main__":
    main()