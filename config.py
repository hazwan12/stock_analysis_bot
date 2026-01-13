"""
Configuration Settings for Stock Analysis Bot
All configurable parameters in one place
"""

# ============================================================================
# API Configuration
# ============================================================================

# MooMoo OpenD Connection
MOOMOO_HOST = '127.0.0.1'
MOOMOO_PORT = 11111

# API Rate Limiting (MooMoo API Compliance)
API_RATE_LIMIT = 45  # Requests per minute (safe margin below 60 limit)
API_CACHE_DURATION = 5  # Minutes to cache data


# ============================================================================
# Trading Parameters
# ============================================================================

# Portfolio Defaults
INITIAL_CAPITAL = 10000  # Default starting capital for new portfolios
COMMISSION_RATE = 0.001  # Trading commission (0.1%)

# Signal Thresholds
DEFAULT_BUY_THRESHOLD = 2  # Signal score needed to trigger buy (range: -5 to +5)
DEFAULT_SELL_THRESHOLD = -2  # Signal score needed to trigger sell

# Risk Management
DEFAULT_STOP_LOSS = 0.05  # Stop loss at -5% (0.05 = 5%)
DEFAULT_TAKE_PROFIT = 0.15  # Take profit at +15% (0.15 = 15%)


# ============================================================================
# Backtesting Parameters
# ============================================================================

BACKTEST_DAYS = 500  # Number of days to use for backtesting
BACKTEST_COMMISSION = 0.001  # Commission rate for backtesting (0.1%)


# ============================================================================
# Walk-Forward Testing Parameters
# ============================================================================

WF_TRAIN_WINDOW = 250  # Training window size in days (~1 year)
WF_TEST_WINDOW = 50  # Testing window size in days (~2 months)
WF_STEP_SIZE = 50  # Days to step forward between windows


# ============================================================================
# Scanner Parameters
# ============================================================================

# Stock Filtering
SCANNER_MIN_PRICE = 10.0  # Minimum stock price to consider ($10)
SCANNER_MAX_PRICE = 500.0  # Maximum stock price to consider ($500)
SCANNER_MIN_VOLUME = 2000000  # Minimum average daily volume (2M shares)

# Scanning Behavior
SCANNER_MAX_WORKERS = 1  # Number of parallel workers (1 = sequential, API compliant)
SCANNER_UNIVERSE_SIZE = 30  # Number of stocks in universe (reduced from 70 for speed)
SCANNER_ENABLE_PARALLEL = False  # Enable/disable parallel processing (False for API compliance)
SCANNER_SHOW_PROGRESS = True  # Show progress indicators during scans


# ============================================================================
# Data Storage Paths
# ============================================================================

DATA_PATH = './data'  # Main data directory
SCAN_PATH = './data/scans'  # Universe scan results
BACKTEST_PATH = './data/backtests'  # Backtest results and reports
CHART_PATH = './data/charts'  # Generated charts and visualizations


# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = 'INFO'  # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = './stock_bot.log'  # Log file location


# ============================================================================
# Advanced Settings (Optional)
# ============================================================================

# Technical Indicator Periods (for customization)
TECHNICAL_RSI_PERIOD = 14  # RSI calculation period
TECHNICAL_MACD_FAST = 12  # MACD fast EMA period
TECHNICAL_MACD_SLOW = 26  # MACD slow EMA period
TECHNICAL_MACD_SIGNAL = 9  # MACD signal line period
TECHNICAL_BB_PERIOD = 20  # Bollinger Bands period
TECHNICAL_BB_STD_DEV = 2  # Bollinger Bands standard deviation multiplier

# Portfolio Allocation
PORTFOLIO_MAX_POSITION_SIZE = 0.35  # Max 35% of cash per stock (for diversification)
PORTFOLIO_CASH_RESERVE = 0.05  # Keep 5% cash reserve (don't invest 100%)

# Scanning Time Estimates (for display purposes)
ESTIMATE_QUICK_SCAN_MIN = 7  # Minimum minutes for quick scan
ESTIMATE_QUICK_SCAN_MAX = 10  # Maximum minutes for quick scan
ESTIMATE_BACKTEST_SCAN_MIN = 15  # Minimum minutes for backtest scan
ESTIMATE_BACKTEST_SCAN_MAX = 20  # Maximum minutes for backtest scan
ESTIMATE_FULL_SCAN_MIN = 25  # Minimum minutes for full walk-forward scan
ESTIMATE_FULL_SCAN_MAX = 35  # Maximum minutes for full walk-forward scan


# ============================================================================
# Feature Flags (Enable/Disable Features)
# ============================================================================

ENABLE_CACHING = True  # Enable data caching (recommended)
ENABLE_RATE_LIMITING = True  # Enable API rate limiting (required for MooMoo)
ENABLE_DEMO_MODE_FALLBACK = True  # Fallback to demo mode if API fails
ENABLE_PROGRESS_TRACKING = True  # Show progress bars and ETAs


# ============================================================================
# Notification Settings (Future Feature)
# ============================================================================

# Email Notifications (not yet implemented)
ENABLE_EMAIL_NOTIFICATIONS = False
EMAIL_SMTP_SERVER = 'smtp.gmail.com'
EMAIL_SMTP_PORT = 587
EMAIL_FROM = ''  # Your email
EMAIL_TO = ''  # Recipient email
EMAIL_PASSWORD = ''  # App password

# SMS Notifications (not yet implemented)
ENABLE_SMS_NOTIFICATIONS = False


# ============================================================================
# Development/Debug Settings
# ============================================================================

DEBUG_MODE = False  # Enable debug mode (more verbose logging)
DEMO_MODE_FORCED = False  # Force demo mode even if API is available
SAVE_RAW_DATA = False  # Save raw API responses for debugging