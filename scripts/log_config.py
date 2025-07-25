from pathlib import Path
import logging
import sys
from datetime import datetime

# Set project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
LOG_DIR = PROJECT_ROOT / "logs"


def setup_logging(
    session_id: str = datetime.now().strftime('%Y%m%d_%H%M%S'),
    user_id: str = "default",
    task_type: str = "Chat"
    ) -> None:
    """
    Configures logging to output to console and a unique file per session.
    """
    # Create logs directory if it doesn't exist
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Include task type in log directory path
    log_path = LOG_DIR / task_type
    log_path.mkdir(parents=True, exist_ok=True)

    # Create log filename
    log_filename = log_path / f"{user_id}_{session_id}_{task_type}.log"

    # Define log format
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
    )
    log_level = logging.INFO

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers if any
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging configured. Log file: {log_filename}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module
    
    Args:
        name: Module name or __name__ from calling module
        
    Returns:
        Logger instance with standardized configuration
    """
    return logging.getLogger(name)