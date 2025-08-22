# modules/logger.py

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_FILE = "app.log"
LOG_LEVEL = logging.DEBUG  # You can change to INFO in production

# Create log directory if needed
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", LOG_FILE)

# Set up file handler with rotation (max 1MB per file, keep 3 backups)
file_handler = RotatingFileHandler(
    log_path,
    maxBytes=1_000_000,
    backupCount=3,
    encoding='utf-8'
)
file_handler.setLevel(LOG_LEVEL)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)

# Formatter (timestamp + level + module + message)
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Set up logger
logger = logging.getLogger("MeetingSummarizer")
logger.setLevel(LOG_LEVEL)

# Avoid adding handlers multiple times in dev mode
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
