import time
import os
import sys
from typing import Dict, Any, Optional
from enum import Enum
from pathlib import Path
import threading
from collections import defaultdict

class LoggerColors(Enum):
    """
    Enum for defining colors used in logging.
    """
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

class Logger():
    def __init__(self):
        self._last_log_times = defaultdict(float)
        self._lock = threading.Lock()
    
    def log(self, name: str, color: LoggerColors, message: str, delay_ms: float, force: bool = False) -> None:
        """
        Logs a message with rate limiting based on the name.
        
        Args:
            name: The category/name for the log message
            color: The color to apply to the message
            message: The content to log
            delay_ms: Minimum delay in milliseconds between logs of the same name
            force: If True, bypasses rate limiting and always logs the message
        """
        current_time = time.time()
        
        with self._lock:
            # Check if enough time has passed since the last log with this name
            last_log_time = self._last_log_times[name]
            time_elapsed_ms = (current_time - last_log_time) * 1000
            
            if force or time_elapsed_ms >= delay_ms or last_log_time == 0:
                # Update the last log time for this name
                self._last_log_times[name] = current_time
                
                timestamp = time.strftime("[%H:%M:%S]", time.localtime(current_time))
                
                colored_message = f"{color.value}{message}{LoggerColors.RESET.value}"
                print(f"{timestamp} [{name}] : {colored_message}", flush=True)