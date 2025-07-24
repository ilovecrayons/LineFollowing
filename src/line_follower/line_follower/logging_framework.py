#!/usr/bin/env python3
"""
Custom logging framework for Line Follower system.
Provides rate-limited, configurable logging with component-based control.
"""

import time
import os
import sys
from typing import Dict, Any, Optional
from enum import Enum
from pathlib import Path
import threading
from collections import defaultdict

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class LogLevel(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"
    SYSTEM = "system"


class LoggerFramework:
    """
    Centralized logging framework with rate limiting and configuration-based control.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure one logging instance across the system."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    # Force stdout to be unbuffered for real-time output
                    # Try multiple approaches to ensure unbuffered output
                    try:
                        # Method 1: Python 3.7+ reconfigure
                        sys.stdout.reconfigure(line_buffering=True)
                    except (AttributeError, OSError):
                        pass
                    
                    # Method 2: Set stdout to unbuffered mode
                    try:
                        if hasattr(sys.stdout, 'buffer'):
                            sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
                    except (AttributeError, OSError):
                        pass
                    
                    # Set stderr to unbuffered too
                    try:
                        if hasattr(sys.stderr, 'buffer'):
                            sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)
                    except (AttributeError, OSError):
                        pass
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.config = {}
        self.rate_limiters = defaultdict(lambda: defaultdict(float))
        self.load_config()
        
    def load_config(self):
        """Load logging configuration from YAML file."""
        config_path = Path(__file__).parent / "logging_config.yaml"
        
        # Fallback configuration 
        default_config = {
            'global': {
                'enable_all': True, 
                'console_output': True, 
                'file_output': False,
                'unbuffered_output': True  # Force immediate output, no buffering
            },
            'rate_limits': {'high_frequency': 0.5, 'medium_frequency': 1.0, 'low_frequency': 5.0},
            'components': {
                'tracker': {
                    'enabled': True, 'rate_limit': 'medium_frequency',
                    'levels': {
                        'system_events': True, 'line_detection': True, 
                        'coordinate_transform': False, 'velocity_commands': False, 'control_loops': False
                    }
                },
                'linreg': {
                    'enabled': True, 'rate_limit': 'high_frequency',
                    'levels': {
                        'detection_events': True, 'processing_details': False, 
                        'method_attempts': False, 'pixel_analysis': False, 'visualization_info': False
                    }
                },
                'detector': {
                    'enabled': True, 'rate_limit': 'medium_frequency',
                    'levels': {'image_info': True, 'curve_detection': True, 'processing_steps': False}
                },
                'system': {
                    'enabled': True, 'rate_limit': 'low_frequency',
                    'levels': {'startup': True, 'errors': True, 'warnings': True, 'debug': False}
                }
            },
            'formatting': {'timestamp': True, 'component_prefix': True, 'colorized': True},
            'colors': {
                'error': '\033[91m', 'warning': '\033[93m', 'info': '\033[92m',
                'debug': '\033[94m', 'system': '\033[95m', 'reset': '\033[0m'
            }
        }
        
        if YAML_AVAILABLE and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                return
            except Exception as e:
                print(f"Warning: Could not load YAML config: {e}, using defaults")
                
        self.config = default_config
    
    def _write_direct(self, message: str):
        """
        Use direct OS-level write to bypass Python's buffering.
        This ensures immediate output even when regular print buffering occurs.
        """
        try:
            os.write(sys.stdout.fileno(), (message + '\n').encode())
        except (IOError, OSError):
            # Fall back to regular print if direct write fails
            print(message, flush=True)
    
    def reload_config(self):
        """Reload configuration from file (useful for runtime config changes)."""
        self.load_config()
        
    def _should_log(self, component: str, category: str) -> bool:
        """Check if a log message should be processed based on configuration."""
        # Check global enable
        if not self.config.get('global', {}).get('enable_all', True):
            return False
            
        # Check component enable
        comp_config = self.config.get('components', {}).get(component, {})
        if not comp_config.get('enabled', True):
            return False
            
        # Check specific category enable  
        levels = comp_config.get('levels', {})
        if category in levels:
            return levels[category]
            
        # Default to enabled if not specified
        return True
        
    def _check_rate_limit(self, component: str, category: str) -> bool:
        """Check if message passes rate limiting."""
        current_time = time.time()
        
        # Get rate limit for this component
        comp_config = self.config.get('components', {}).get(component, {})
        rate_limit_key = comp_config.get('rate_limit', 'medium_frequency')
        rate_limit_seconds = self.config.get('rate_limits', {}).get(rate_limit_key, 1.0)
        
        # Check last log time for this component+category
        last_log_time = self.rate_limiters[component][category]
        
        if current_time - last_log_time >= rate_limit_seconds:
            self.rate_limiters[component][category] = current_time
            return True
            
        return False
        
    def _format_message(self, component: str, level: LogLevel, message: str) -> str:
        """Format log message according to configuration."""
        parts = []
        
        # Timestamp
        if self.config.get('formatting', {}).get('timestamp', True):
            parts.append(f"[{time.strftime('%H:%M:%S')}]")
            
        # Component prefix  
        if self.config.get('formatting', {}).get('component_prefix', True):
            parts.append(f"[{component.upper()}]")
            
        # Log level
        if self.config.get('formatting', {}).get('log_level', True):
            parts.append(f"[{level.value.upper()}]")
            
        # Message
        parts.append(message)
        
        formatted = " ".join(parts)
        
        # Colorize if enabled
        if self.config.get('formatting', {}).get('colorized', True):
            colors = self.config.get('colors', {})
            color = colors.get(level.value, '')
            reset = colors.get('reset', '')
            formatted = f"{color}{formatted}{reset}"
            
        return formatted
        
    def log(self, component: str, category: str, level: LogLevel, message: str, force: bool = False):
        """
        Main logging function.
        
        Args:
            component: Component name (e.g., 'tracker', 'linreg', 'detector')
            category: Log category (e.g., 'line_detection', 'system_events')  
            level: Log level (ERROR, WARNING, INFO, DEBUG, SYSTEM)
            message: Message to log
            force: If True, bypass rate limiting and configuration checks
        """
        # Force logging for errors regardless of config
        if level == LogLevel.ERROR:
            force = True
            
        if not force:
            # Check if we should log this message
            if not self._should_log(component, category):
                return
                
            # Check rate limiting (except for errors)
            if level != LogLevel.ERROR and not self._check_rate_limit(component, category):
                return
                
        # Format and output message
        formatted_message = self._format_message(component, level, message)
        
        # Console output with immediate flush to prevent buffering
        if self.config.get('global', {}).get('console_output', True):
            if self.config.get('global', {}).get('unbuffered_output', True):
                # Use direct OS-level write for truly unbuffered output
                self._write_direct(formatted_message)
            else:
                print(formatted_message, flush=True)
                sys.stdout.flush()
            
        # File output (if enabled)
        if self.config.get('global', {}).get('file_output', False):
            self._write_to_file(formatted_message)
            
    def _write_to_file(self, message: str):
        """Write message to log file with rotation."""
        # TODO: Implement file logging with rotation
        pass
        
    # Convenience methods for different log levels
    def error(self, component: str, category: str, message: str):
        self.log(component, category, LogLevel.ERROR, message, force=True)
        
    def warning(self, component: str, category: str, message: str):
        self.log(component, category, LogLevel.WARNING, message)
        
    def info(self, component: str, category: str, message: str):
        self.log(component, category, LogLevel.INFO, message)
        
    def debug(self, component: str, category: str, message: str):
        self.log(component, category, LogLevel.DEBUG, message)
        
    def system(self, component: str, category: str, message: str):
        self.log(component, category, LogLevel.SYSTEM, message)


class ComponentLogger:
    """
    Component-specific logger wrapper that automatically includes component name.
    """
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.framework = LoggerFramework()
        
    def error(self, category: str, message: str):
        self.framework.error(self.component_name, category, message)
        
    def warning(self, category: str, message: str):
        self.framework.warning(self.component_name, category, message)
        
    def info(self, category: str, message: str):
        self.framework.info(self.component_name, category, message)
        
    def debug(self, category: str, message: str):
        self.framework.debug(self.component_name, category, message)
        
    def system(self, category: str, message: str):
        self.framework.system(self.component_name, category, message)


# Global convenience functions
def get_logger(component_name: str) -> ComponentLogger:
    """Get a component-specific logger."""
    return ComponentLogger(component_name)


def reload_logging_config():
    """Reload logging configuration from file."""
    LoggerFramework().reload_config()


# Example usage:
if __name__ == "__main__":
    # Example of how to use the logging framework
    logger = get_logger("test_component")
    
    # These will be rate-limited based on configuration
    for i in range(10):
        logger.info("detection_events", f"Test message {i}")
        logger.debug("processing_details", f"Debug message {i}")
        time.sleep(0.1)
        
    # Errors are always shown
    logger.error("system_errors", "This error will always be shown")
