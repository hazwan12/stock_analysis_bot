"""
Configuration settings for Stock Analysis Bot
"""

# API Configuration
MOOMOO_HOST = '127.0.0.1'
MOOMOO_PORT = 11111

# Trading Parameters
INITIAL_CAPITAL = 10000
COMMISSION_RATE = 0.001  # 0.1%

# Signal Thresholds
DEFAULT_BUY_THRESHOLD = 2
DEFAULT_SELL_THRESHOLD = -2
DEFAULT_STOP_LOSS = 0.05  # 5%
DEFAULT_TAKE_PROFIT = 0.15  # 15%

# Backtesting Parameters
BACKTEST_DAYS = 500
BACKTEST_COMMISSION = 0.001

# Walk-Forward Parameters
WF_TRAIN_WINDOW = 250
WF_TEST_WINDOW = 50
WF_STEP_SIZE = 50

# Scanner Parameters
SCANNER_MIN_PRICE = 10.0
SCANNER_MAX_PRICE = 500.0
SCANNER_MIN_VOLUME = 2000000
SCANNER_MAX_WORKERS = 5

# Data Paths
DATA_PATH = './data'
SCAN_PATH = './data/scans'
BACKTEST_PATH = './data/backtests'
CHART_PATH = './data/charts'

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = './stock_bot.log'