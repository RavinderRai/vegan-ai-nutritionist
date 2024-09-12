import logging
import os
from typing import Optional
from logging.handlers import RotatingFileHandler

def setup_logger(name: Optional[str] = 'default_logger', log_file: str = "pipeline.log", level: int = logging.INFO) -> logging.Logger:
    """
    Function to set up a logger for machine learning pipelines.
    
    Parameters:
    - name (optional): str, name of the logger (can be the name of the pipeline/module)
    - log_file: str, name of the log file
    - level: int, logging level, default is INFO
    
    Returns:
    - logger: logging.Logger instance
    """
    
    # Create a logger with the provided name and set the level(INFO, DEBUG, etc.)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent multiple log handlers in case of re-importing the module
    if not logger.hasHandlers():
        # Create log directory if it doesn't exist
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        
        # File handler with rotating logs to avoid large log files
        file_handler = RotatingFileHandler(os.path.join(log_dir, log_file), maxBytes=5*1024*1024, backupCount=5)
        file_handler.setLevel(level)
        
        # Console handler to also print logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Define a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Apply the format to both handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger



