"""
Rate Limiter and Caching Utilities
Handles API rate limiting and data caching for MooMoo API
"""

import time
from datetime import datetime, timedelta
from typing import Any, Optional
import threading


class RateLimiter:
    """
    Rate limiter for API calls
    Ensures we don't exceed MooMoo's API limits
    """
    
    def __init__(self, max_calls_per_minute: int = 45):
        """
        Initialize rate limiter
        
        Args:
            max_calls_per_minute: Maximum API calls allowed per minute
                                 Default: 45 (safe margin below 60 limit)
        """
        self.max_calls = max_calls_per_minute
        self.calls = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if we've hit the rate limit"""
        with self.lock:
            now = datetime.now()
            
            # Remove calls older than 1 minute
            cutoff = now - timedelta(minutes=1)
            self.calls = [call_time for call_time in self.calls if call_time > cutoff]
            
            # Check if we're at the limit
            if len(self.calls) >= self.max_calls:
                # Calculate wait time
                oldest_call = self.calls[0]
                wait_seconds = 61 - (now - oldest_call).total_seconds()
                
                if wait_seconds > 0:
                    print(f"⏳ Rate limit reached. Waiting {wait_seconds:.1f}s...")
                    time.sleep(wait_seconds)
                    # Clear old calls after waiting
                    self.calls = []
            
            # Record this call
            self.calls.append(now)
    
    def get_stats(self) -> dict:
        """Get current rate limiter statistics"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        recent_calls = [c for c in self.calls if c > cutoff]
        
        return {
            'calls_last_minute': len(recent_calls),
            'max_calls': self.max_calls,
            'remaining_capacity': self.max_calls - len(recent_calls)
        }


class DataCache:
    """
    Simple data cache with time-based expiration
    Reduces redundant API calls
    """
    
    def __init__(self, cache_duration_minutes: int = 5):
        """
        Initialize cache
        
        Args:
            cache_duration_minutes: How long to keep cached data
        """
        self.cache = {}
        self.duration = timedelta(minutes=cache_duration_minutes)
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cached data if still valid
        
        Args:
            key: Cache key (e.g., stock symbol)
            
        Returns:
            Cached data or None if expired/missing
        """
        with self.lock:
            if key in self.cache:
                data, timestamp = self.cache[key]
                
                # Check if expired
                if datetime.now() - timestamp < self.duration:
                    return data
                else:
                    # Remove expired entry
                    del self.cache[key]
        
        return None
    
    def set(self, key: str, data: Any):
        """
        Store data in cache
        
        Args:
            key: Cache key
            data: Data to cache
        """
        with self.lock:
            self.cache[key] = (data, datetime.now())
    
    def clear(self):
        """Clear all cached data"""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        with self.lock:
            now = datetime.now()
            valid_entries = 0
            expired_entries = 0
            
            for key, (data, timestamp) in self.cache.items():
                if now - timestamp < self.duration:
                    valid_entries += 1
                else:
                    expired_entries += 1
            
            return {
                'total_entries': len(self.cache),
                'valid_entries': valid_entries,
                'expired_entries': expired_entries
            }
    
    def cleanup_expired(self):
        """Remove expired entries from cache"""
        with self.lock:
            now = datetime.now()
            expired_keys = [
                key for key, (data, timestamp) in self.cache.items()
                if now - timestamp >= self.duration
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            return len(expired_keys)


class ProgressTracker:
    """
    Track and display progress for long-running operations
    """
    
    def __init__(self, total: int, operation_name: str = "Processing"):
        """
        Initialize progress tracker
        
        Args:
            total: Total number of items to process
            operation_name: Name of the operation
        """
        self.total = total
        self.current = 0
        self.operation_name = operation_name
        self.start_time = datetime.now()
        self.lock = threading.Lock()
    
    def update(self, increment: int = 1, item_name: str = ""):
        """
        Update progress
        
        Args:
            increment: Number of items completed
            item_name: Name of current item being processed
        """
        with self.lock:
            self.current += increment
            
            # Calculate metrics
            elapsed = (datetime.now() - self.start_time).total_seconds()
            progress_pct = (self.current / self.total) * 100
            
            if self.current > 0:
                avg_time_per_item = elapsed / self.current
                remaining_items = self.total - self.current
                eta_seconds = avg_time_per_item * remaining_items
                eta_minutes = eta_seconds / 60
            else:
                eta_minutes = 0
            
            # Display progress
            item_info = f" - {item_name}" if item_name else ""
            print(f"[{self.current}/{self.total}] {progress_pct:.1f}% complete{item_info} | ETA: {eta_minutes:.1f} min")
    
    def finish(self):
        """Mark operation as complete"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"\n✅ {self.operation_name} complete!")
        print(f"   Processed: {self.total} items in {elapsed/60:.1f} minutes")
        print(f"   Average: {elapsed/self.total:.1f} seconds per item")


if __name__ == "__main__":
    # Test rate limiter
    print("Testing Rate Limiter...")
    limiter = RateLimiter(max_calls_per_minute=5)  # Low limit for testing
    
    for i in range(10):
        limiter.wait_if_needed()
        print(f"Call {i+1}: {limiter.get_stats()}")
        time.sleep(0.5)
    
    print("\nTesting Cache...")
    cache = DataCache(cache_duration_minutes=1)
    
    # Store data
    cache.set('AAPL', {'price': 150})
    print(f"Cached: {cache.get('AAPL')}")
    
    # Wait and check expiration
    time.sleep(2)
    print(f"After 2s: {cache.get('AAPL')}")
    
    print(f"\nCache stats: {cache.get_stats()}")
    
    print("\nTesting Progress Tracker...")
    tracker = ProgressTracker(10, "Test Operation")
    for i in range(10):
        time.sleep(0.3)
        tracker.update(1, f"Item {i+1}")
    tracker.finish()