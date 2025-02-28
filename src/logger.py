import logging

def setup_logger() -> logging.Logger:
    """
    This function sets up a logger with a file handler to log messages at the INFO level.
    If the logger already has handlers, this function does nothing.

    Parameters:
    None

    Returns:
    logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create a file handler
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Add filehandlers if they are not added 
    if not logger.hasHandlers():
        logger.addHandler(file_handler)

    return logger
