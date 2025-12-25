"""
Walk-Forward Testing Module
Validates strategy robustness by testing on rolling time windows
Prevents overfitting and look-ahead bias
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from modules.backtesting import BacktestEngine, ParameterOptimizer, Trade
import config


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window"""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    optimized_params: Dict = None
    train_results: Dict = None
    test_results: Dict = None


class WalkForwardAnalyzer:
    """
    Performs walk-forward analysis to validate strategy robustness
    """
    
    def __init__(self, symbol: str, data: pd.DataFrame, 
                 initial_capital: float = None,
                 train_window_days: int = None,
                 test_window_days: int = None,
                 step_size: int = None):
        """
        Initialize Walk-Forward Analyzer
        
        Args:
            symbol: Stock symbol
            data: Historical price data
            initial_capital: Starting capital
            train_window_days: Days for training/optimization
            test_window_days: Days for out-of-sample testing
            step_size: Days to move forward between windows
        """
        self.symbol = symbol
        self.data = data.sort_values('time_key').reset_index(drop=True)
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.train_window_days = train_window_days or config.WF_TRAIN_WINDOW
        self.test_window_days = test_window_days or config.WF_TEST_WINDOW
        self.step_size = step_size or config.WF_STEP_SIZE
        self.windows: List[WalkForwardWindow] = []
        self.aggregate_results = {}
        
    def create_windows(self) -> List[WalkForwardWindow]:
        """Create train/test window splits"""
        print(f"\n{'='*80}")
        print(f"CREATING WALK-FORWARD WINDOWS")
        print(f"{'='*80}")
        print(f"Total data points: {len(self.data)}")
        print(f"Training window: {self.train_window_days} days")
        print(f"Testing window: {self.test_window_days} days")
        print(f"Step size: {self.step_size} days")
        print(f"{'='*80}\n")
        
        windows = []
        window_id = 1
        start_idx = self.train_window_days
        
        while start_idx + self.test_window_days <= len(self.data):
            train_start_idx = start_idx - self.train_window_days
            train_end_idx = start_idx
            test_start_idx = start_idx
            test_end_idx = start_idx + self.test_window_days
            
            train_data = self.data.iloc[train_start_idx:train_end_idx].copy()
            test_data = self.data.iloc[test_start_idx:test_end_idx].copy()
            
            window = WalkForwardWindow(
                window_id=window_id,
                train_start=train_data['time_key'].iloc[0],
                train_end=train_data['time_key'].iloc[-1],
                test_start=test_data['time_key'].iloc[0],
                test_end=test_data['time_key'].iloc[-1],
                train_data=train_data,
                test_data=test_data
            )
            
            windows.append(window)
            
            print(f"Window {window_id}:")
            print(f"  Train: {window.train_start.date()} to {window.train_end.date()} ({len(train_data)} days)")
            print(f"  Test:  {window.test_start.date()} to {window.test_end.date()} ({len(test_data)} days)")
            
            window_id += 1
            start_idx += self.step_size
        
        self.windows = windows
        print(f"\nCreated {len(windows)} walk-forward windows")
        return windows
    
    def run_walk_forward_test(self, param_grid: Dict, 
                              optimization_metric: str = 'total_return') -> Dict:
        """
        Run complete walk-forward analysis
        
        Args:
            param_grid: Parameters to optimize
            optimization_metric: Metric to optimize
        """
        if not self.windows:
            self.create_windows()
        
        print(f"\n{'='*80}")
        print(f"WALK-FORWARD ANALYSIS - {self.symbol}")
        print(f"{'='*80}")
        print(f"Optimizing for: {optimization_metric}")
        print(f"Parameter grid: {param_grid}")
        print(f"{'='*80}\n")
        
        all_test_results = []
        all_train_results = []
        
        for i, window in enumerate(self.windows, 1):
            print(f"\n{'='*80}")
            print(f"WINDOW {window.window_id} of {len(self.windows)}")
            print(f"{'='*80}")
            
            # Optimize on training data
            print(f"\n📚 TRAINING PHASE")
            print(f"Period: {window.train_start.date()} to {window.train_end.date()}")
            
            optimizer = ParameterOptimizer(
                symbol=self.symbol,
                data=window.train_data,
                initial_capital=self.initial_capital
            )
            
            train_results_df = optimizer.optimize(param_grid)
            
            if len(train_results_df) == 0:
                print("⚠️  No valid results in training, skipping")
                continue
            
            best_train_result = train_results_df.iloc[0]
            window.optimized_params = {
                'buy_threshold': int(best_train_result['buy_threshold']),
                'sell_threshold': int(best_train_result['sell_threshold']),
                'stop_loss_pct': float(best_train_result.get('stop_loss_pct', 0.05)),
                'take_profit_pct': float(best_train_result.get('take_profit_pct', 0.15))
            }
            window.train_results = best_train_result.to_dict()
            
            print(f"\n✅ Best training parameters:")
            for key, value in window.optimized_params.items():
                print(f"  {key}: {value}")
            
            # Test on out-of-sample data
            print(f"\n🧪 TESTING PHASE")
            print(f"Period: {window.test_start.date()} to {window.test_end.date()}")
            
            test_engine = BacktestEngine(initial_capital=self.initial_capital)
            test_result = test_engine.run_backtest(
                symbol=self.symbol,
                data=window.test_data,
                **window.optimized_params
            )
            
            if 'error' not in test_result:
                window.test_results = test_result
                all_test_results.append(test_result)
                all_train_results.append(window.train_results)
        
        # Aggregate results
        self.aggregate_results = self._calculate_aggregate_metrics(
            all_train_results, all_test_results
        )
        
        self._print_aggregate_results()
        
        return self.aggregate_results
    
    def _calculate_aggregate_metrics(self, train_results: List[Dict], 
                                    test_results: List[Dict]) -> Dict:
        """Calculate aggregate performance across all windows"""
        if not test_results:
            return {'error': 'No valid test results'}
        
        test_returns = [r['total_return'] for r in test_results]
        test_win_rates = [r['win_rate'] for r in test_results]
        test_sharpes = [r['sharpe_ratio'] for r in test_results]
        test_trades = [r['total_trades'] for r in test_results]
        test_alphas = [r['alpha'] for r in test_results]
        
        train_returns = [r['total_return'] for r in train_results]
        
        degradation = [train_r['total_return'] - test_r['total_return'] 
                      for train_r, test_r in zip(train_results, test_results)]
        
        profitable_windows = len([r for r in test_returns if r > 0])
        
        return {
            'symbol': self.symbol,
            'total_windows': len(test_results),
            'profitable_windows': profitable_windows,
            'win_rate_windows': (profitable_windows / len(test_results)) * 100,
            'avg_test_return': np.mean(test_returns),
            'median_test_return': np.median(test_returns),
            'std_test_return': np.std(test_returns),
            'min_test_return': np.min(test_returns),
            'max_test_return': np.max(test_returns),
            'avg_test_win_rate': np.mean(test_win_rates),
            'avg_test_sharpe': np.mean(test_sharpes),
            'total_test_trades': sum(test_trades),
            'avg_test_alpha': np.mean(test_alphas),
            'avg_train_return': np.mean(train_returns),
            'avg_degradation': np.mean(degradation),
            'degradation_pct': (np.mean(degradation) / np.mean(train_returns)) * 100 if np.mean(train_returns) != 0 else 0,
            'consistency_score': self._calculate_consistency_score(test_returns),
            'return_stability': 1 / (1 + np.std(test_returns)),
            'worst_window_loss': min(test_returns),
            'best_window_gain': max(test_returns),
        }
    
    def _calculate_consistency_score(self, returns: List[float]) -> float:
        """Calculate consistency score (0-100)"""
        if len(returns) < 2:
            return 0
        
        positive_ratio = len([r for r in returns if r > 0]) / len(returns)
        volatility_penalty = 1 / (1 + np.std(returns) / 10)
        mean_return_bonus = min(np.mean(returns) / 10, 1.0) if np.mean(returns) > 0 else 0
        
        score = (positive_ratio * 0.5 + volatility_penalty * 0.3 + mean_return_bonus * 0.2) * 100
        return max(0, min(100, score))
    
    def _print_aggregate_results(self):
        """Print aggregate results"""
        r = self.aggregate_results
        
        if 'error' in r:
            print(f"\n⚠️  {r['error']}")
            return
        
        print(f"\n{'='*80}")
        print(f"WALK-FORWARD AGGREGATE RESULTS - {r['symbol']}")
        print(f"{'='*80}\n")
        
        print("🎯 OVERALL PERFORMANCE:")
        print(f"  Total Windows: {r['total_windows']}")
        print(f"  Profitable Windows: {r['profitable_windows']} ({r['win_rate_windows']:.1f}%)")
        print(f"  Consistency Score: {r['consistency_score']:.1f}/100")
        print()
        
        print("📊 OUT-OF-SAMPLE STATISTICS:")
        print(f"  Average Return: {r['avg_test_return']:+.2f}%")
        print(f"  Median Return: {r['median_test_return']:+.2f}%")
        print(f"  Std Deviation: {r['std_test_return']:.2f}%")
        print(f"  Best Window: {r['max_test_return']:+.2f}%")
        print(f"  Worst Window: {r['min_test_return']:+.2f}%")
        print(f"  Average Win Rate: {r['avg_test_win_rate']:.1f}%")
        print(f"  Average Sharpe: {r['avg_test_sharpe']:.2f}")
        print(f"  Average Alpha: {r['avg_test_alpha']:+.2f}%")
        print()
        
        print("🔬 ROBUSTNESS ANALYSIS:")
        print(f"  Avg Training Return: {r['avg_train_return']:+.2f}%")
        print(f"  Performance Degradation: {r['avg_degradation']:+.2f}% ({r['degradation_pct']:+.2f}%)")
        print(f"  Return Stability: {r['return_stability']:.3f}")
        print()
        
        if r['avg_test_return'] > 5 and r['win_rate_windows'] > 60 and r['consistency_score'] > 60:
            rating = "🌟 EXCELLENT - Strategy is robust"
        elif r['avg_test_return'] > 2 and r['win_rate_windows'] > 50:
            rating = "✅ GOOD - Strategy shows promise"
        elif r['avg_test_return'] > 0:
            rating = "⚠️  FAIR - Marginally profitable"
        else:
            rating = "❌ POOR - Fails validation"
        
        print(f"ROBUSTNESS RATING: {rating}")
        
        if r['degradation_pct'] > 50:
            print(f"\n⚠️  WARNING: High degradation suggests overfitting!")
        
        print(f"{'='*80}\n")
    
    def plot_walk_forward_results(self):
        """Visualize walk-forward results"""
        if not self.windows or not self.aggregate_results:
            print("No results to plot")
            return
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        window_ids = [w.window_id for w in self.windows if w.test_results]
        train_returns = [w.train_results['total_return'] for w in self.windows if w.test_results]
        test_returns = [w.test_results['total_return'] for w in self.windows if w.test_results]
        test_win_rates = [w.test_results['win_rate'] for w in self.windows if w.test_results]
        test_sharpes = [w.test_results['sharpe_ratio'] for w in self.windows if w.test_results]
        test_trades = [w.test_results['total_trades'] for w in self.windows if w.test_results]
        test_dates = [w.test_start for w in self.windows if w.test_results]
        
        # 1. Train vs Test Returns
        ax1 = fig.add_subplot(gs[0, :2])
        x = np.arange(len(window_ids))
        width = 0.35
        ax1.bar(x - width/2, train_returns, width, label='Training', alpha=0.8, color='blue')
        ax1.bar(x + width/2, test_returns, width, label='Out-of-Sample', alpha=0.8, color='orange')
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax1.set_xlabel('Window')
        ax1.set_ylabel('Return (%)')
        ax1.set_title('Training vs Out-of-Sample Returns')
        ax1.set_xticks(x)
        ax1.set_xticklabels(window_ids)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Return Distribution
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(test_returns, bins=15, edgecolor='black', alpha=0.7, color='green')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.axvline(x=np.mean(test_returns), color='blue', linestyle='--', linewidth=2, label='Mean')
        ax2.set_xlabel('Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Return Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative Performance
        ax3 = fig.add_subplot(gs[1, :2])
        cumulative_returns = np.cumprod([1 + r/100 for r in test_returns]) - 1
        ax3.plot(window_ids, [r * 100 for r in cumulative_returns], 
                marker='o', linewidth=2, markersize=6, color='purple')
        ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax3.set_xlabel('Window')
        ax3.set_ylabel('Cumulative Return (%)')
        ax3.set_title('Cumulative Performance')
        ax3.grid(True, alpha=0.3)
        ax3.fill_between(window_ids, [r * 100 for r in cumulative_returns], 0, alpha=0.3, color='purple')
        
        # 4. Win Rate by Window
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.bar(window_ids, test_win_rates, alpha=0.7, color='teal')
        ax4.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50%')
        ax4.axhline(y=np.mean(test_win_rates), color='blue', linestyle='--', linewidth=2, label='Avg')
        ax4.set_xlabel('Window')
        ax4.set_ylabel('Win Rate (%)')
        ax4.set_title('Win Rate by Window')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Sharpe Ratio
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(window_ids, test_sharpes, marker='s', linewidth=2, markersize=6, color='darkgreen')
        ax5.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, label='1.0')
        ax5.axhline(y=np.mean(test_sharpes), color='blue', linestyle='--', linewidth=2, label='Avg')
        ax5.set_xlabel('Window')
        ax5.set_ylabel('Sharpe Ratio')
        ax5.set_title('Risk-Adjusted Returns')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Trade Count
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.bar(window_ids, test_trades, alpha=0.7, color='coral')
        ax6.set_xlabel('Window')
        ax6.set_ylabel('Trades')
        ax6.set_title('Trading Activity')
        ax6.grid(True, alpha=0.3)
        
        # 7. Performance Over Time
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.scatter(test_dates, test_returns, s=100, alpha=0.6, c=test_returns, 
                   cmap='RdYlGn', edgecolors='black', linewidth=1)
        ax7.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax7.set_xlabel('Date')
        ax7.set_ylabel('Return (%)')
        ax7.set_title('Performance Across Time')
        ax7.grid(True, alpha=0.3)
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.suptitle(f'Walk-Forward Analysis - {self.symbol}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        import os
        os.makedirs(config.CHART_PATH, exist_ok=True)
        filename = f'{config.CHART_PATH}/walk_forward_{self.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Chart saved: {filename}")
        plt.show()
    
    def export_results(self):
        """Export results to CSV"""
        window_data = []
        for w in self.windows:
            if w.test_results:
                window_data.append({
                    'Window': w.window_id,
                    'Train Start': w.train_start.date(),
                    'Train End': w.train_end.date(),
                    'Test Start': w.test_start.date(),
                    'Test End': w.test_end.date(),
                    'Buy Threshold': w.optimized_params['buy_threshold'],
                    'Sell Threshold': w.optimized_params['sell_threshold'],
                    'Stop Loss %': w.optimized_params['stop_loss_pct'],
                    'Take Profit %': w.optimized_params['take_profit_pct'],
                    'Train Return %': w.train_results['total_return'],
                    'Test Return %': w.test_results['total_return'],
                    'Test Win Rate %': w.test_results['win_rate'],
                    'Test Sharpe': w.test_results['sharpe_ratio'],
                    'Test Trades': w.test_results['total_trades'],
                    'Test Alpha %': w.test_results['alpha'],
                })
        
        df = pd.DataFrame(window_data)
        
        import os
        os.makedirs(config.BACKTEST_PATH, exist_ok=True)
        filename = f'{config.BACKTEST_PATH}/walk_forward_{self.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(filename, index=False)
        print(f"✅ Results exported: {filename}")
        
        return df


if __name__ == "__main__":
    from modules.api_client import MooMooAPIClient
    
    api_client = MooMooAPIClient()
    api_client.connect()
    
    symbol = 'US.AAPL'
    data = api_client.get_historical_data(symbol, days=750)
    
    wf_analyzer = WalkForwardAnalyzer(symbol, data)
    
    param_grid = {
        'buy_threshold': [1, 2, 3],
        'sell_threshold': [-1, -2, -3],
        'stop_loss_pct': [0.03, 0.05, 0.07],
        'take_profit_pct': [0.10, 0.15, 0.20]
    }
    
    results = wf_analyzer.run_walk_forward_test(param_grid)
    wf_analyzer.plot_walk_forward_results()
    wf_analyzer.export_results()
    
    api_client.close()