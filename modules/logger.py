# modules/logger.py

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

# Configuration from environment variables
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.getenv("LOG_DIR", "logs")
MAX_LOG_SIZE = int(os.getenv("MAX_LOG_SIZE", "5000000"))  # 5MB default
BACKUP_COUNT = int(os.getenv("BACKUP_COUNT", "5"))
ENABLE_CONSOLE_LOGS = os.getenv("ENABLE_CONSOLE_LOGS", "true").lower() == "true"

# Log file names
APP_LOG_FILE = "app.log"
USER_ACTIVITY_LOG_FILE = "user_activity.log"
ERROR_LOG_FILE = "errors.log"

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

def get_log_level(level_str: str) -> int:
    """Convert string log level to logging constant."""
    level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    return level_mapping.get(level_str.upper(), logging.INFO)

def create_rotating_file_handler(
    filename: str, 
    level: int = None,
    max_bytes: int = MAX_LOG_SIZE,
    backup_count: int = BACKUP_COUNT
) -> RotatingFileHandler:
    """Create a rotating file handler with consistent formatting."""
    log_path = os.path.join(LOG_DIR, filename)
    handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    
    if level is not None:
        handler.setLevel(level)
    
    # Detailed formatter for file logs
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    return handler

def create_console_handler(level: int = None) -> logging.StreamHandler:
    """Create console handler with simpler formatting."""
    handler = logging.StreamHandler(sys.stdout)
    
    if level is not None:
        handler.setLevel(level)
    
    # Simpler formatter for console
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    return handler

def setup_logger(
    name: str,
    level: Optional[int] = None,
    file_handler: Optional[logging.Handler] = None,
    console_enabled: bool = ENABLE_CONSOLE_LOGS
) -> logging.Logger:
    """
    Set up a logger with file and optionally console handlers.
    Prevents duplicate handlers in Streamlit hot reload scenarios.
    """
    logger = logging.getLogger(name)
    
    # Set level
    if level is None:
        level = get_log_level(LOG_LEVEL)
    logger.setLevel(level)
    
    # Check if handlers already exist to prevent duplicates
    has_file_handler = any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
    has_console_handler = any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout 
                            for h in logger.handlers)
    
    # Add file handler if not present
    if not has_file_handler and file_handler:
        logger.addHandler(file_handler)
    
    # Add console handler if not present and enabled
    if not has_console_handler and console_enabled:
        console_handler = create_console_handler(level)
        logger.addHandler(console_handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger

# Main application logger
app_file_handler = create_rotating_file_handler(APP_LOG_FILE)
logger = setup_logger("MeetingSummarizer", file_handler=app_file_handler)

# User activity logger - for tracking user interactions
user_activity_handler = create_rotating_file_handler(
    USER_ACTIVITY_LOG_FILE, 
    level=logging.INFO  # Only INFO and above for user activities
)
user_logger = setup_logger(
    "UserActivity", 
    level=logging.INFO,
    file_handler=user_activity_handler,
    console_enabled=False  # Don't spam console with user activities
)

# Error logger - for system errors and exceptions
error_handler = create_rotating_file_handler(
    ERROR_LOG_FILE,
    level=logging.ERROR  # Only ERROR and CRITICAL
)
error_logger = setup_logger(
    "SystemErrors",
    level=logging.ERROR, 
    file_handler=error_handler,
    console_enabled=True  # Always show errors on console
)

def log_user_activity(action: str, details: Optional[str] = None, user_id: Optional[str] = None):
    """
    Log user activity with structured format.
    
    Args:
        action: Action performed (e.g., "file_upload", "transcription_started")
        details: Additional details about the action
        user_id: Optional user identifier
    """
    message_parts = [f"ACTION: {action}"]
    
    if user_id:
        message_parts.append(f"USER: {user_id}")
    
    if details:
        message_parts.append(f"DETAILS: {details}")
    
    user_logger.info(" | ".join(message_parts))

def log_error(error: Exception, context: Optional[str] = None, extra_data: Optional[dict] = None):
    """
    Log system errors with structured format and context.
    
    Args:
        error: Exception that occurred
        context: Additional context about where/when the error occurred
        extra_data: Additional structured data to log
    """
    message_parts = [f"ERROR: {type(error).__name__}: {str(error)}"]
    
    if context:
        message_parts.append(f"CONTEXT: {context}")
    
    if extra_data:
        extra_str = " | ".join([f"{k}={v}" for k, v in extra_data.items()])
        message_parts.append(f"DATA: {extra_str}")
    
    error_logger.error(" | ".join(message_parts), exc_info=True)

def log_performance(operation: str, duration: float, success: bool = True, **kwargs):
    """
    Log performance metrics for operations.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        success: Whether operation succeeded
        **kwargs: Additional metrics to log
    """
    status = "SUCCESS" if success else "FAILED"
    message_parts = [f"PERF: {operation}", f"DURATION: {duration:.3f}s", f"STATUS: {status}"]
    
    for key, value in kwargs.items():
        message_parts.append(f"{key.upper()}: {value}")
    
    logger.info(" | ".join(message_parts))

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module/component.
    Uses the same configuration as the main logger.
    """
    module_logger = logging.getLogger(f"MeetingSummarizer.{name}")
    
    # Inherit from parent logger configuration
    if not module_logger.handlers:
        module_logger.setLevel(get_log_level(LOG_LEVEL))
        module_logger.propagate = True  # Let parent handle the logging
    
    return module_logger

def configure_third_party_loggers():
    """Configure logging levels for third-party libraries to reduce noise."""
    # Reduce noise from common libraries
    noisy_loggers = [
        'urllib3.connectionpool',
        'requests.packages.urllib3.connectionpool', 
        'PIL.PngImagePlugin',
        'matplotlib.font_manager',
        'transformers.tokenization_utils_base',
        'transformers.configuration_utils',
        'openai._base_client'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

def setup_streamlit_logging():
    """Special configuration for Streamlit applications."""
    # Streamlit can be noisy, reduce its logging
    st_logger = logging.getLogger('streamlit')
    st_logger.setLevel(logging.WARNING)
    
    # Watchdog can be very noisy in development
    watchdog_logger = logging.getLogger('watchdog')
    watchdog_logger.setLevel(logging.WARNING)

# Initialize third-party logger configuration
configure_third_party_loggers()

# Context manager for performance logging
class LogPerformance:
    """Context manager to automatically log operation performance."""
    
    def __init__(self, operation: str, logger_instance: logging.Logger = None, **kwargs):
        self.operation = operation
        self.logger = logger_instance or logger
        self.extra_data = kwargs
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        success = exc_type is None
        
        if not success:
            log_error(exc_val, f"During operation: {self.operation}")
        
        log_performance(self.operation, duration, success, **self.extra_data)

# Export main loggers and utilities
__all__ = [
    'logger',           # Main application logger
    'user_logger',      # User activity logger  
    'error_logger',     # Error logger
    'log_user_activity', # Helper for user activity logging
    'log_error',        # Helper for error logging
    'log_performance',  # Helper for performance logging
    'get_logger',       # Get module-specific logger
    'LogPerformance',   # Context manager for performance logging
    'setup_streamlit_logging'  # Streamlit-specific setup
]

# Log that the logging system has been initialized
logger.info(f"🚀 Logging system initialized - Level: {LOG_LEVEL}, Dir: {LOG_DIR}")