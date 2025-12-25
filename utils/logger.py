"""
Logging Utility Module
Configures logging for the entire application
"""

import logging
import sys
from datetime import datetime
import config


def setup_logger(name: str = 'StockBot', level: str = None) -> logging.Logger:
    """
    Setup and configure logger
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    # Get log level from config if not specified
    if level is None:
        level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    else:
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (DEBUG and above)
    try:
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create log file: {e}")
    
    return logger


def log_trade(symbol: str, action: str, price: float, shares: int, reason: str = ""):
    """
    Log a trade action
    
    Args:
        symbol: Stock symbol
        action: BUY or SELL
        price: Trade price
        shares: Number of shares
        reason: Reason for trade
    """
    logger = logging.getLogger('StockBot')
    logger.info(f"TRADE | {action} | {symbol} | {shares} shares @ ${price:.2f} | {reason}")


def log_signal(symbol: str, signal: str, score: int):
    """
    Log a trading signal
    
    Args:
        symbol: Stock symbol
        signal: Signal description
        score: Signal score
    """
    logger = logging.getLogger('StockBot')
    logger.info(f"SIGNAL | {symbol} | {signal} | Score: {score}")


def log_performance(symbol: str, returns: float, win_rate: float):
    """
    Log performance metrics
    
    Args:
        symbol: Stock symbol
        returns: Total returns
        win_rate: Win rate percentage
    """
    logger = logging.getLogger('StockBot')
    logger.info(f"PERFORMANCE | {symbol} | Returns: {returns:.2f}% | Win Rate: {win_rate:.1f}%")


if __name__ == "__main__":
    # Test logger
    logger = setup_logger()
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test trade logging
    log_trade("US.AAPL", "BUY", 150.25, 10, "Strong buy signal")
    log_signal("US.AAPL", "🟢 STRONG BUY", 3)
    log_performance("US.AAPL", 15.5, 62.3)