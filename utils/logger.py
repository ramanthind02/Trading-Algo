import os
import sys
import logging
import warnings
import pandas as pd
from datetime import datetime


# Custom filter to suppress FeatureExtractor warnings
class FeatureExtractorFilter(logging.Filter):
    def filter(self, record):
        # Filter out log messages containing "class FeatureExtractor"
        if hasattr(record, 'getMessage'):
            message = record.getMessage()
            if "class FeatureExtractor" in message or "FeatureExtractor:" in message:
                return False
        return True


def setup_logger(log_level=logging.INFO, log_to_file=True):
    # Suppress FeatureExtractor warnings at the global level
    warnings.filterwarnings("ignore", message=".*class FeatureExtractor.*")
    warnings.filterwarnings("ignore", message=".*FeatureExtractor.*")
    
    # Suppress pandas performance warnings
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    warnings.filterwarnings("ignore", message=".*DataFrame is highly fragmented.*")
    warnings.filterwarnings("ignore", message=".*frame.insert many times.*")
    warnings.filterwarnings("ignore", message=".*Consider joining all columns at once.*")
    
    if log_to_file:
        # Create logs directory if it doesn't exist
        log_dir = "C:/Users/Administrator/Desktop/logs"
        os.makedirs(log_dir, exist_ok=True)
    
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(log_dir, f"log_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear any existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create log formatting
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set formatting of loggers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    if log_to_file:
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
    
    # Add the FeatureExtractor filter to both handlers
    feature_filter = FeatureExtractorFilter()

    # Add handlers to logger
    console_handler.addFilter(feature_filter)
    logger.addHandler(console_handler)
    if log_to_file:
        file_handler.addFilter(feature_filter)
        logger.addHandler(file_handler)
        
    # Apply the filter to the root logger and common loggers
    logger.addFilter(feature_filter)
    
    # Also apply to common library loggers that might be causing this
    for logger_name in ['polars', 'functime', 'machine_learning', '__main__']:
        lib_logger = logging.getLogger(logger_name)
        lib_logger.addFilter(feature_filter)
    
    # Reduce APScheduler logging verbosity
    logging.getLogger('apscheduler').setLevel(logging.ERROR)
        
    return logger

def get_logger(name):
    # Apply the filter to any new logger created
    logger = logging.getLogger(name)
    logger.addFilter(FeatureExtractorFilter())
    return logger
