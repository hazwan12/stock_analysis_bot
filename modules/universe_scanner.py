"""
Stock Universe Scanner Module
Scans entire stock universe to find best opportunities
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

from modules.api_client import MooMooAPIClient
from modules.trend_analyzer import TrendAnalyzer
from modules.backtesting import BacktestEngine
from modules.walk_forward import WalkForwardAnalyzer
import config


@dataclass
class StockScore:
    """Comprehensive stock scoring"""
    symbol: str
    current_signal_score: int = 0
    current_price: float = 0.0
    current_recommendation: str = ""
    backtest_return: float = 0.0
    backtest_win_rate: float = 0.0
    backtest_sharpe: float = 0.0
    backtest_alpha: float = 0.0
    backtest_trades: int = 0
    wf_avg_return: float = 0.0
    wf_consistency: float = 0.0
    wf_win_windows: float = 0.0
    wf_degradation: float = 0.0
    overall_score: float = 0.0
    rank: int = 0
    avg_volume: float = 0.0
    volatility: float = 0.0
    sector: str = "Unknown"
    status: str = "OK"
    error_msg: str = ""
    
    def to_dict(self):
        return asdict(self)


class StockUniverseScanner:
    """Scans stock universe to find best opportunities"""
    
    def __init__(self, 
                 initial_capital: float = None,
                 min_price: float = None,
                 max_price: float = None,
                 min_avg_volume: int = None,
                 max_workers: int = None):
        """Initialize Universe Scanner"""
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.min_price = min_price or config.SCANNER_MIN_PRICE
        self.max_price = max_price or config.SCANNER_MAX_PRICE
        self.min_avg_volume = min_avg_volume or config.SCANNER_MIN_VOLUME
        self.max_workers = max_workers or config.SCANNER_MAX_WORKERS
        self.api_client = MooMooAPIClient()
        self.results: List[StockScore] = []
        
    def get_stock_universe(self, market: str = "US") -> List[str]:
        """Get list of stocks to scan"""
        us_stocks = [
            # Tech
            'US.AAPL', 'US.MSFT', 'US.GOOGL', 'US.AMZN', 'US.META',
            'US.NVDA', 'US.TSLA', 'US.AMD', 'US.INTC', 'US.ORCL',
            # Finance
            'US.JPM', 'US.BAC', 'US.WFC', 'US.GS', 'US.MS',
            'US.C', 'US.BLK', 'US.SCHW', 'US.AXP', 'US.V',
            # Healthcare
            'US.JNJ', 'US.UNH', 'US.PFE', 'US.ABBV', 'US.TMO',
            'US.MRK', 'US.ABT', 'US.DHR', 'US.LLY', 'US.BMY',
            # Consumer
            'US.WMT', 'US.HD', 'US.PG', 'US.KO', 'US.PEP',
            'US.COST', 'US.NKE', 'US.MCD', 'US.SBUX', 'US.DIS',
            # Industrial
            'US.BA', 'US.CAT', 'US.GE', 'US.MMM', 'US.HON',
            'US.UPS', 'US.LMT', 'US.RTX', 'US.DE', 'US.UNP',
            # Energy
            'US.XOM', 'US.CVX', 'US.COP', 'US.SLB', 'US.EOG',
            # Communication
            'US.T', 'US.VZ', 'US.NFLX', 'US.CMCSA', 'US.TMUS',
            # Others
            'US.BRK.B', 'US.BABA', 'US.CRM', 'US.PYPL', 'US.SQ',
        ]
        
        print(f"\n{'='*80}")
        print(f"STOCK UNIVERSE: {len(us_stocks)} stocks")
        print(f"{'='*80}\n")
        
        return us_stocks
    
    def analyze_single_stock(self, symbol: str, mode: str = 'quick') -> StockScore:
        """Analyze a single stock"""
        score = StockScore(symbol=symbol)
        
        try:
            days = 500 if mode in ['backtest', 'full'] else 100
            data = self.api_client.get_historical_data(symbol, days)
            
            if data is None or len(data) < 50:
                score.status = "INSUFFICIENT_DATA"
                score.error_msg = f"Not enough data"
                return score
            
            # Current signals
            analyzer = TrendAnalyzer(data)
            signals = analyzer.analyze()
            
            score.current_signal_score = signals.get('score', 0)
            score.current_price = signals.get('current_price', 0.0)
            score.current_recommendation = signals.get('recommendation', 'N/A')
            score.avg_volume = data['volume'].tail(20).mean()
            score.volatility = data['close'].pct_change().std() * 100
            
            if mode == 'quick':
                score.overall_score = self._calculate_quick_score(score)
                score.status = "OK"
                return score
            
            # Backtest
            if mode in ['backtest', 'full']:
                engine = BacktestEngine(initial_capital=self.initial_capital)
                bt_results = engine.run_backtest(symbol, data)
                
                if 'error' not in bt_results:
                    score.backtest_return = bt_results['total_return']
                    score.backtest_win_rate = bt_results['win_rate']
                    score.backtest_sharpe = bt_results['sharpe_ratio']
                    score.backtest_alpha = bt_results['alpha']
                    score.backtest_trades = bt_results['total_trades']
                
                if mode == 'backtest':
                    score.overall_score = self._calculate_backtest_score(score)
                    score.status = "OK"
                    return score
            
            # Walk-forward
            if mode == 'full':
                if len(data) < 400:
                    score.status = "INSUFFICIENT_DATA_WF"
                    score.error_msg = "Need 400+ days for walk-forward"
                    score.overall_score = self._calculate_backtest_score(score)
                    return score
                
                wf = WalkForwardAnalyzer(
                    symbol=symbol,
                    data=data,
                    initial_capital=self.initial_capital,
                    train_window_days=200,
                    test_window_days=40,
                    step_size=40
                )
                
                param_grid = {
                    'buy_threshold': [2, 3],
                    'sell_threshold': [-2, -3],
                    'stop_loss_pct': [0.05],
                    'take_profit_pct': [0.15]
                }
                
                wf_results = wf.run_walk_forward_test(param_grid)
                
                if 'error' not in wf_results:
                    score.wf_avg_return = wf_results['avg_test_return']
                    score.wf_consistency = wf_results['consistency_score']
                    score.wf_win_windows = wf_results['win_rate_windows']
                    score.wf_degradation = wf_results['degradation_pct']
                
                score.overall_score = self._calculate_full_score(score)
                score.status = "OK"
            
        except Exception as e:
            score.status = "ERROR"
            score.error_msg = str(e)
            print(f"✗ Error analyzing {symbol}: {e}")
        
        return score
    
    def _calculate_quick_score(self, score: StockScore) -> float:
        """Calculate score for quick mode"""
        return score.current_signal_score * 10
    
    def _calculate_backtest_score(self, score: StockScore) -> float:
        """Calculate score for backtest mode"""
        s = 0.0
        s += score.current_signal_score * 3
        s += max(0, score.backtest_return) * 2
        if score.backtest_win_rate > 50:
            s += (score.backtest_win_rate - 50) * 0.3
        s += max(0, score.backtest_sharpe) * 10
        return s
    
    def _calculate_full_score(self, score: StockScore) -> float:
        """Calculate score for full mode"""
        s = 0.0
        s += score.current_signal_score * 2
        s += max(0, score.wf_avg_return) * 3
        s += score.wf_consistency * 0.25
        s += score.wf_win_windows * 0.1
        
        if score.wf_degradation < 30:
            s += 10
        elif score.wf_degradation < 50:
            s += 5
        else:
            s -= 5
        
        return max(0, s)
    
    def scan_universe(self, 
                     symbols: Optional[List[str]] = None,
                     mode: str = 'quick',
                     top_n: int = 20,
                     parallel: bool = True) -> pd.DataFrame:
        """Scan stock universe"""
        print(f"\n{'='*80}")
        print(f"UNIVERSE SCAN - MODE: {mode.upper()}")
        print(f"{'='*80}\n")
        
        if symbols is None:
            symbols = self.get_stock_universe()
        
        start_time = time.time()
        
        if parallel and len(symbols) > 5:
            print(f"⚡ Parallel processing ({self.max_workers} workers)")
            results = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_symbol = {
                    executor.submit(self.analyze_single_stock, symbol, mode): symbol 
                    for symbol in symbols
                }
                
                completed = 0
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        results.append(result)
                        completed += 1
                        
                        if completed % 10 == 0:
                            elapsed = time.time() - start_time
                            rate = completed / elapsed
                            remaining = len(symbols) - completed
                            eta = remaining / rate if rate > 0 else 0
                            print(f"Progress: {completed}/{len(symbols)} ({completed/len(symbols)*100:.1f}%) | ETA: {eta/60:.1f} min")
                    except Exception as e:
                        print(f"✗ Failed {symbol}: {e}")
        else:
            print(f"📊 Sequential processing")
            results = []
            for i, symbol in enumerate(symbols, 1):
                print(f"\nAnalyzing {symbol} ({i}/{len(symbols)})...")
                result = self.analyze_single_stock(symbol, mode)
                results.append(result)
                
                if i % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed
                    remaining = len(symbols) - i
                    eta = remaining / rate if rate > 0 else 0
                    print(f"Progress: {i}/{len(symbols)} | ETA: {eta/60:.1f} min")
                
                time.sleep(0.5)
        
        self.results = results
        elapsed = time.time() - start_time
        
        self.results.sort(key=lambda x: x.overall_score, reverse=True)
        
        for i, result in enumerate(self.results, 1):
            result.rank = i
        
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        print(f"\n{'='*80}")
        print(f"SCAN COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Stocks analyzed: {len(results)}")
        print(f"Avg time/stock: {elapsed/len(results):.1f} seconds")
        print(f"{'='*80}\n")
        
        self._display_top_stocks(df, top_n, mode)
        
        return df
    
    def _display_top_stocks(self, df: pd.DataFrame, top_n: int, mode: str):
        """Display top stocks"""
        print(f"\n{'='*80}")
        print(f"TOP {top_n} OPPORTUNITIES")
        print(f"{'='*80}\n")
        
        valid_df = df[df['status'] == 'OK'].copy()
        
        if len(valid_df) == 0:
            print("No valid results")
            return
        
        top_stocks = valid_df.head(top_n)
        
        if mode == 'quick':
            display_cols = ['rank', 'symbol', 'current_recommendation', 
                          'current_signal_score', 'current_price', 
                          'volatility', 'overall_score']
            print(top_stocks[display_cols].to_string(index=False))
            
        elif mode == 'backtest':
            display_cols = ['rank', 'symbol', 'current_recommendation',
                          'backtest_return', 'backtest_win_rate', 
                          'backtest_sharpe', 'backtest_alpha', 
                          'overall_score']
            print(top_stocks[display_cols].to_string(index=False))
            
        elif mode == 'full':
            display_cols = ['rank', 'symbol', 'current_recommendation',
                          'wf_avg_return', 'wf_consistency', 
                          'wf_win_windows', 'wf_degradation',
                          'overall_score']
            
            display = top_stocks[display_cols].copy()
            display['wf_avg_return'] = display['wf_avg_return'].apply(lambda x: f"{x:+.2f}%")
            display['wf_consistency'] = display['wf_consistency'].apply(lambda x: f"{x:.1f}")
            display['wf_win_windows'] = display['wf_win_windows'].apply(lambda x: f"{x:.1f}%")
            display['wf_degradation'] = display['wf_degradation'].apply(lambda x: f"{x:+.1f}%")
            display['overall_score'] = display['overall_score'].apply(lambda x: f"{x:.1f}")
            
            print(display.to_string(index=False))
        
        print()
    
    def create_watchlist(self, df: pd.DataFrame, criteria: Dict) -> List[str]:
        """Create filtered watchlist"""
        filtered = df[df['status'] == 'OK'].copy()
        
        if 'min_score' in criteria:
            filtered = filtered[filtered['overall_score'] >= criteria['min_score']]
        
        if 'min_return' in criteria:
            if 'wf_avg_return' in filtered.columns:
                filtered = filtered[filtered['wf_avg_return'] >= criteria['min_return']]
            elif 'backtest_return' in filtered.columns:
                filtered = filtered[filtered['backtest_return'] >= criteria['min_return']]
        
        if 'max_degradation' in criteria and 'wf_degradation' in filtered.columns:
            filtered = filtered[filtered['wf_degradation'] <= criteria['max_degradation']]
        
        if 'min_consistency' in criteria and 'wf_consistency' in filtered.columns:
            filtered = filtered[filtered['wf_consistency'] >= criteria['min_consistency']]
        
        if 'recommendations' in criteria:
            filtered = filtered[filtered['current_recommendation'].isin(criteria['recommendations'])]
        
        watchlist = filtered['symbol'].tolist()
        
        print(f"\n{'='*80}")
        print(f"WATCHLIST: {len(watchlist)} stocks")
        print(f"{'='*80}")
        print(f"Criteria: {criteria}")
        print(f"\nStocks: {', '.join(watchlist)}")
        print(f"{'='*80}\n")
        
        return watchlist
    
    def export_results(self, df: pd.DataFrame, filename: Optional[str] = None):
        """Export results to CSV"""
        import os
        
        if filename is None:
            os.makedirs(config.SCAN_PATH, exist_ok=True)
            filename = f'{config.SCAN_PATH}/scan_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        df.to_csv(filename, index=False)
        print(f"✅ Results exported: {filename}")
        
    def connect(self):
        """Connect to API"""
        return self.api_client.connect()
    
    def close(self):
        """Close connections"""
        self.api_client.close()


if __name__ == "__main__":
    scanner = StockUniverseScanner()
    scanner.connect()
    
    # Quick scan
    results = scanner.scan_universe(mode='quick', top_n=20, parallel=True)
    scanner.export_results(results)
    
    # Create watchlist
    watchlist = scanner.create_watchlist(results, {
        'min_score': 20,
        'recommendations': ['🟢 STRONG BUY', '🟡 BUY']
    })
    
    scanner.close()