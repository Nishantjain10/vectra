"""
Concurrency utilities for thread-safe operations.

This module provides thread-safe wrappers and utilities to prevent data races
in the in-memory vector database storage.
"""
import threading
from contextlib import contextmanager
from typing import Any, Generator
from functools import wraps


class ReadWriteLock:
    """
    A readers-writer lock implementation.
    
    Multiple readers can acquire the lock simultaneously, but writers
    have exclusive access. This improves performance for read-heavy workloads
    which is common in vector databases.
    
    Time Complexity: O(1) for acquiring/releasing locks
    Space Complexity: O(1) 
    """
    
    def __init__(self):
        self._read_ready = threading.Condition(threading.RLock())
        self._readers = 0
        
    def acquire_read(self) -> None:
        """Acquire a read lock. Multiple readers can hold the lock simultaneously."""
        with self._read_ready:
            self._readers += 1
            
    def release_read(self) -> None:
        """Release a read lock."""
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notifyAll()
                
    def acquire_write(self) -> None:
        """Acquire a write lock. Exclusive access - no other readers or writers."""
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()
            
    def release_write(self) -> None:
        """Release a write lock."""
        self._read_ready.release()
        
    @contextmanager
    def read_lock(self) -> Generator[None, None, None]:
        """Context manager for read operations."""
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()
            
    @contextmanager
    def write_lock(self) -> Generator[None, None, None]:
        """Context manager for write operations."""
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()


def thread_safe_read(func):
    """
    Decorator to make a method thread-safe for read operations.
    
    The decorated method must be part of a class that has a '_lock' attribute
    which is a ReadWriteLock instance.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_lock'):
            raise AttributeError(f"{self.__class__.__name__} must have a '_lock' attribute")
        
        with self._lock.read_lock():
            return func(self, *args, **kwargs)
    return wrapper


def thread_safe_write(func):
    """
    Decorator to make a method thread-safe for write operations.
    
    The decorated method must be part of a class that has a '_lock' attribute
    which is a ReadWriteLock instance.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_lock'):
            raise AttributeError(f"{self.__class__.__name__} must have a '_lock' attribute")
        
        with self._lock.write_lock():
            return func(self, *args, **kwargs)
    return wrapper


class ThreadSafeCounter:
    """
    A thread-safe counter for generating sequential IDs or tracking statistics.
    
    Uses a simple lock for thread safety. Could be optimized with atomic
    operations for high-performance scenarios.
    """
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
        
    def increment(self) -> int:
        """Increment the counter and return the new value."""
        with self._lock:
            self._value += 1
            return self._value
            
    def decrement(self) -> int:
        """Decrement the counter and return the new value."""
        with self._lock:
            self._value -= 1
            return self._value
            
    @property
    def value(self) -> int:
        """Get the current value thread-safely."""
        with self._lock:
            return self._value


class ThreadSafeSingleton:
    """
    A thread-safe singleton metaclass.
    
    Ensures that only one instance of a class exists, even in multi-threaded
    environments. Useful for storage managers and configuration objects.
    """
    
    _instances = {}
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                # Double-checked locking pattern
                if cls not in cls._instances:
                    cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]
