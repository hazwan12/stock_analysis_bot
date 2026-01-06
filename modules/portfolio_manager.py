"""
Portfolio Manager Module
Manages user investment portfolios with holdings tracking and recommendations
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import os

import config


@dataclass
class Portfolio:
    """Represents an investment portfolio"""
    portfolio_id: int
    name: str
    initial_investment: float
    recurring_amount: float
    created_date: str
    total_invested: float = 0.0
    current_value: float = 0.0
    total_return_pct: float = 0.0
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Holding:
    """Represents a stock holding in a portfolio"""
    holding_id: int
    portfolio_id: int
    symbol: str
    shares: float
    average_price: float
    purchase_date: str
    current_price: float = 0.0
    current_value: float = 0.0
    gain_loss: float = 0.0
    gain_loss_pct: float = 0.0
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Transaction:
    """Represents a buy/sell transaction"""
    transaction_id: int
    portfolio_id: int
    symbol: str
    transaction_type: str  # BUY or SELL
    shares: float
    price: float
    total_amount: float
    transaction_date: str
    notes: str = ""
    
    def to_dict(self):
        return asdict(self)


class PortfolioManager:
    """Manages multiple investment portfolios with holdings tracking"""
    
    def __init__(self, db_path: str = None):
        """Initialize Portfolio Manager with database"""
        if db_path is None:
            db_path = os.path.join(config.DATA_PATH, 'portfolios.db')
        
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Portfolios table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolios (
                portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                initial_investment REAL NOT NULL,
                recurring_amount REAL NOT NULL,
                created_date TEXT NOT NULL,
                total_invested REAL DEFAULT 0,
                notes TEXT
            )
        ''')
        
        # Holdings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS holdings (
                holding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                shares REAL NOT NULL,
                average_price REAL NOT NULL,
                purchase_date TEXT NOT NULL,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id),
                UNIQUE(portfolio_id, symbol)
            )
        ''')
        
        # Transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                transaction_type TEXT NOT NULL,
                shares REAL NOT NULL,
                price REAL NOT NULL,
                total_amount REAL NOT NULL,
                transaction_date TEXT NOT NULL,
                notes TEXT,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
            )
        ''')
        
        # Recommendations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                reason TEXT,
                target_shares REAL,
                priority INTEGER,
                recommendation_date TEXT NOT NULL,
                executed INTEGER DEFAULT 0,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_portfolio(self, name: str, initial_investment: float, 
                        recurring_amount: float, notes: str = "") -> Portfolio:
        """Create a new portfolio"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        created_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            cursor.execute('''
                INSERT INTO portfolios (name, initial_investment, recurring_amount, 
                                      created_date, total_invested, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (name, initial_investment, recurring_amount, created_date, 
                  initial_investment, notes))
            
            conn.commit()
            portfolio_id = cursor.lastrowid
            
            portfolio = Portfolio(
                portfolio_id=portfolio_id,
                name=name,
                initial_investment=initial_investment,
                recurring_amount=recurring_amount,
                created_date=created_date,
                total_invested=initial_investment
            )
            
            print(f"✅ Portfolio '{name}' created successfully!")
            print(f"   ID: {portfolio_id}")
            print(f"   Initial Investment: ${initial_investment:,.2f}")
            print(f"   Recurring Amount: ${recurring_amount:,.2f}")
            
            return portfolio
            
        except sqlite3.IntegrityError:
            print(f"❌ Portfolio '{name}' already exists!")
            return None
        finally:
            conn.close()
    
    def list_portfolios(self) -> List[Portfolio]:
        """List all portfolios"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM portfolios')
        rows = cursor.fetchall()
        
        portfolios = []
        for row in rows:
            portfolio = Portfolio(
                portfolio_id=row[0],
                name=row[1],
                initial_investment=row[2],
                recurring_amount=row[3],
                created_date=row[4],
                total_invested=row[5]
            )
            portfolios.append(portfolio)
        
        conn.close()
        return portfolios
    
    def get_portfolio(self, portfolio_id: int) -> Optional[Portfolio]:
        """Get a specific portfolio"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM portfolios WHERE portfolio_id = ?', (portfolio_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return Portfolio(
                portfolio_id=row[0],
                name=row[1],
                initial_investment=row[2],
                recurring_amount=row[3],
                created_date=row[4],
                total_invested=row[5]
            )
        return None
    
    def add_holding(self, portfolio_id: int, symbol: str, shares: float, 
                   price: float, transaction_date: str = None) -> bool:
        """Add or update a holding in portfolio"""
        if transaction_date is None:
            transaction_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if holding already exists
            cursor.execute('''
                SELECT shares, average_price FROM holdings 
                WHERE portfolio_id = ? AND symbol = ?
            ''', (portfolio_id, symbol))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing holding (average down/up)
                old_shares, old_avg_price = existing
                new_shares = old_shares + shares
                new_avg_price = ((old_shares * old_avg_price) + (shares * price)) / new_shares
                
                cursor.execute('''
                    UPDATE holdings 
                    SET shares = ?, average_price = ?, purchase_date = ?
                    WHERE portfolio_id = ? AND symbol = ?
                ''', (new_shares, new_avg_price, transaction_date, portfolio_id, symbol))
                
                print(f"✅ Updated {symbol}: {old_shares} → {new_shares} shares @ ${new_avg_price:.2f} avg")
            else:
                # Insert new holding
                cursor.execute('''
                    INSERT INTO holdings (portfolio_id, symbol, shares, average_price, purchase_date)
                    VALUES (?, ?, ?, ?, ?)
                ''', (portfolio_id, symbol, shares, price, transaction_date))
                
                print(f"✅ Added {shares} shares of {symbol} @ ${price:.2f}")
            
            # Record transaction
            total_amount = shares * price
            cursor.execute('''
                INSERT INTO transactions (portfolio_id, symbol, transaction_type, 
                                        shares, price, total_amount, transaction_date)
                VALUES (?, ?, 'BUY', ?, ?, ?, ?)
            ''', (portfolio_id, symbol, shares, price, total_amount, transaction_date))
            
            # Update total invested
            cursor.execute('''
                UPDATE portfolios 
                SET total_invested = total_invested + ?
                WHERE portfolio_id = ?
            ''', (total_amount, portfolio_id))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error adding holding: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_holdings(self, portfolio_id: int) -> List[Holding]:
        """Get all holdings for a portfolio"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM holdings WHERE portfolio_id = ?
        ''', (portfolio_id,))
        
        rows = cursor.fetchall()
        holdings = []
        
        for row in rows:
            holding = Holding(
                holding_id=row[0],
                portfolio_id=row[1],
                symbol=row[2],
                shares=row[3],
                average_price=row[4],
                purchase_date=row[5]
            )
            holdings.append(holding)
        
        conn.close()
        return holdings
    
    def update_holding_prices(self, portfolio_id: int, price_data: Dict[str, float]):
        """Update current prices for all holdings"""
        holdings = self.get_holdings(portfolio_id)
        
        for holding in holdings:
            if holding.symbol in price_data:
                holding.current_price = price_data[holding.symbol]
                holding.current_value = holding.shares * holding.current_price
                holding.gain_loss = holding.current_value - (holding.shares * holding.average_price)
                holding.gain_loss_pct = (holding.gain_loss / (holding.shares * holding.average_price)) * 100
        
        return holdings
    
    def sell_holding(self, portfolio_id: int, symbol: str, shares: float, 
                    price: float, transaction_date: str = None) -> bool:
        """Sell shares from a holding"""
        if transaction_date is None:
            transaction_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT shares FROM holdings 
                WHERE portfolio_id = ? AND symbol = ?
            ''', (portfolio_id, symbol))
            
            result = cursor.fetchone()
            if not result:
                print(f"❌ No holding found for {symbol}")
                return False
            
            current_shares = result[0]
            if shares > current_shares:
                print(f"❌ Cannot sell {shares} shares. Only {current_shares} available.")
                return False
            
            new_shares = current_shares - shares
            
            if new_shares == 0:
                # Remove holding completely
                cursor.execute('''
                    DELETE FROM holdings 
                    WHERE portfolio_id = ? AND symbol = ?
                ''', (portfolio_id, symbol))
                print(f"✅ Sold all {shares} shares of {symbol} @ ${price:.2f}")
            else:
                # Update shares
                cursor.execute('''
                    UPDATE holdings 
                    SET shares = ?
                    WHERE portfolio_id = ? AND symbol = ?
                ''', (new_shares, portfolio_id, symbol))
                print(f"✅ Sold {shares} shares of {symbol} @ ${price:.2f} ({new_shares} remaining)")
            
            # Record transaction
            total_amount = shares * price
            cursor.execute('''
                INSERT INTO transactions (portfolio_id, symbol, transaction_type, 
                                        shares, price, total_amount, transaction_date)
                VALUES (?, ?, 'SELL', ?, ?, ?, ?)
            ''', (portfolio_id, symbol, shares, price, total_amount, transaction_date))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error selling holding: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_transactions(self, portfolio_id: int, limit: int = 50) -> List[Transaction]:
        """Get transaction history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM transactions 
            WHERE portfolio_id = ?
            ORDER BY transaction_date DESC
            LIMIT ?
        ''', (portfolio_id, limit))
        
        rows = cursor.fetchall()
        transactions = []
        
        for row in rows:
            transaction = Transaction(
                transaction_id=row[0],
                portfolio_id=row[1],
                symbol=row[2],
                transaction_type=row[3],
                shares=row[4],
                price=row[5],
                total_amount=row[6],
                transaction_date=row[7],
                notes=row[8] or ""
            )
            transactions.append(transaction)
        
        conn.close()
        return transactions
    
    def save_recommendations(self, portfolio_id: int, recommendations: List[Dict]):
        """Save recommendations for a portfolio"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        recommendation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Clear old recommendations
        cursor.execute('''
            DELETE FROM recommendations 
            WHERE portfolio_id = ? AND executed = 0
        ''', (portfolio_id,))
        
        # Insert new recommendations
        for i, rec in enumerate(recommendations, 1):
            cursor.execute('''
                INSERT INTO recommendations 
                (portfolio_id, symbol, action, reason, target_shares, priority, recommendation_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (portfolio_id, rec['symbol'], rec['action'], rec['reason'], 
                  rec.get('target_shares', 0), i, recommendation_date))
        
        conn.commit()
        conn.close()
    
    def get_recommendations(self, portfolio_id: int) -> List[Dict]:
        """Get current recommendations for a portfolio"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol, action, reason, target_shares, priority, recommendation_date
            FROM recommendations
            WHERE portfolio_id = ? AND executed = 0
            ORDER BY priority
        ''', (portfolio_id,))
        
        rows = cursor.fetchall()
        recommendations = []
        
        for row in rows:
            recommendations.append({
                'symbol': row[0],
                'action': row[1],
                'reason': row[2],
                'target_shares': row[3],
                'priority': row[4],
                'date': row[5]
            })
        
        conn.close()
        return recommendations
    
    def calculate_portfolio_value(self, portfolio_id: int, current_prices: Dict[str, float]) -> Dict:
        """Calculate current portfolio value and performance"""
        portfolio = self.get_portfolio(portfolio_id)
        holdings = self.update_holding_prices(portfolio_id, current_prices)
        
        total_value = sum(h.current_value for h in holdings)
        total_cost = sum(h.shares * h.average_price for h in holdings)
        total_gain_loss = total_value - total_cost
        
        if total_cost > 0:
            total_return_pct = (total_gain_loss / total_cost) * 100
        else:
            total_return_pct = 0
        
        cash_available = portfolio.total_invested - total_cost
        
        return {
            'portfolio_id': portfolio_id,
            'portfolio_name': portfolio.name,
            'total_invested': portfolio.total_invested,
            'cash_available': cash_available,
            'holdings_value': total_value,
            'total_value': total_value + cash_available,
            'total_gain_loss': total_gain_loss,
            'total_return_pct': total_return_pct,
            'holdings': [h.to_dict() for h in holdings]
        }
    
    def display_portfolio_summary(self, portfolio_id: int, current_prices: Dict[str, float]):
        """Display detailed portfolio summary"""
        summary = self.calculate_portfolio_value(portfolio_id, current_prices)
        
        print(f"\n{'='*80}")
        print(f"PORTFOLIO SUMMARY: {summary['portfolio_name']}")
        print(f"{'='*80}\n")
        
        print(f"💰 PORTFOLIO VALUE:")
        print(f"   Total Invested:     ${summary['total_invested']:,.2f}")
        print(f"   Cash Available:     ${summary['cash_available']:,.2f}")
        print(f"   Holdings Value:     ${summary['holdings_value']:,.2f}")
        print(f"   Total Value:        ${summary['total_value']:,.2f}")
        print(f"   Gain/Loss:          ${summary['total_gain_loss']:,.2f} ({summary['total_return_pct']:+.2f}%)")
        print()
        
        if summary['holdings']:
            print(f"📊 HOLDINGS ({len(summary['holdings'])} positions):")
            print(f"{'Symbol':<10} {'Shares':<10} {'Avg Cost':<12} {'Current':<12} {'Value':<14} {'Gain/Loss':<15}")
            print("-" * 80)
            
            for h in summary['holdings']:
                gain_loss_str = f"${h['gain_loss']:,.2f} ({h['gain_loss_pct']:+.2f}%)"
                print(f"{h['symbol']:<10} {h['shares']:<10.2f} "
                      f"${h['average_price']:<11.2f} ${h['current_price']:<11.2f} "
                      f"${h['current_value']:<13,.2f} {gain_loss_str:<15}")
        else:
            print("📊 No holdings yet")
        
        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Test portfolio manager
    pm = PortfolioManager()
    
    # Create a portfolio
    portfolio = pm.create_portfolio(
        name="My Growth Portfolio",
        initial_investment=10000,
        recurring_amount=500
    )
    
    if portfolio:
        # Add some holdings
        pm.add_holding(portfolio.portfolio_id, 'US.AAPL', 10, 150.00)
        pm.add_holding(portfolio.portfolio_id, 'US.MSFT', 15, 300.00)
        
        # Display summary
        current_prices = {'US.AAPL': 185.00, 'US.MSFT': 350.00}
        pm.display_portfolio_summary(portfolio.portfolio_id, current_prices)