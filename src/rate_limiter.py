"""
Rate limiter for API requests
"""

import time
import logging
from typing import Dict, List, Tuple, Optional
import threading

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple in-memory rate limiter to restrict API access

    Features:
    - Tracks requests by IP address/key
    - Sliding window implementation for accurate rate limiting
    - Thread-safe using a lock
    """

    def __init__(self, window_size: int = 3600):
        """
        Initialize rate limiter

        Args:
            window_size: Time window in seconds (default: 1 hour)
        """
        # Map of IP address -> list of timestamps
        self.requests: Dict[str, List[float]] = {}
        # Lock for thread safety
        self.lock = threading.RLock()
        # Window size in seconds
        self.window_size = window_size

    def _clean_old_requests(self, key: str):
        """Remove requests older than the window size"""
        if key not in self.requests:
            return

        current_time = time.time()
        cutoff = current_time - self.window_size

        # Filter out old requests
        self.requests[key] = [
            timestamp for timestamp in self.requests[key] if timestamp > cutoff
        ]

    def add_request(self, key: str) -> None:
        """
        Record a new request for the given key

        Args:
            key: Identifier for the client (typically IP address)
        """
        with self.lock:
            if key not in self.requests:
                self.requests[key] = []

            self._clean_old_requests(key)
            self.requests[key].append(time.time())

    def get_request_count(self, key: str) -> int:
        """
        Get the number of requests made by the key in the current window

        Args:
            key: Identifier for the client

        Returns:
            Number of requests in the current time window
        """
        with self.lock:
            if key not in self.requests:
                return 0

            self._clean_old_requests(key)
            return len(self.requests[key])

    def can_make_request(self, key: str, limit: int) -> bool:
        """
        Check if the key can make a new request based on the limit

        Args:
            key: Identifier for the client
            limit: Maximum number of requests allowed in the window

        Returns:
            True if the request can be made, False otherwise
        """
        return self.get_request_count(key) < limit

    def get_remaining_requests(self, key: str, limit: int) -> int:
        """
        Get the number of remaining requests the key can make

        Args:
            key: Identifier for the client
            limit: Maximum number of requests allowed in the window

        Returns:
            Number of remaining requests
        """
        count = self.get_request_count(key)
        return max(0, limit - count)

    def get_reset_time(self, key: str) -> Optional[float]:
        """
        Get the time when the oldest request will expire

        Args:
            key: Identifier for the client

        Returns:
            Timestamp when the rate limit will reset, or None if no requests
        """
        with self.lock:
            if key not in self.requests or not self.requests[key]:
                return None

            oldest_request = min(self.requests[key])
            return oldest_request + self.window_size

    def get_wait_time(self, key: str) -> Optional[float]:
        """
        Get the time to wait before a new request can be made

        Args:
            key: Identifier for the client

        Returns:
            Time in seconds to wait, or None if no wait is needed
        """
        reset_time = self.get_reset_time(key)
        if reset_time is None:
            return None

        wait_time = reset_time - time.time()
        return max(0, wait_time)


# Global rate limiter instance
transcript_limiter = RateLimiter(window_size=360)  # 1 hour window. add bcak 3600
