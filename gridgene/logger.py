import logging


def get_logger(name: str = None, level=logging.INFO):
    """
    Get a logger instance with a standard format and no duplicated handlers.

    Parameters
    ----------
    name : str, optional
        Name of the logger. Defaults to None which uses root logger.
    level : int, optional
        Logging level, defaults to logging.INFO.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
