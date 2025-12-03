import logging
from colorlog import ColoredFormatter

def setup_logger(level=logging.INFO, log_file=None):
    """
    Configures the root logger with colored console output and optional file logging.

    :param level: Logging level (e.g., logging.DEBUG)
    :param log_file: Path to a log file. If provided, logs will also be written there.
    """
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler with colorlog
    console_handler = logging.StreamHandler()
    log_format = "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s %(blue)s%(message)s"
    console_formatter = ColoredFormatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_format = "%(asctime)s [%(levelname)s] %(message)s"
        file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.setLevel(level)
    return logger


def color_metric(metric_name, value):
    """
    Adds ANSI color formatting to metric names for console output.
    """
    COLORS = {
        "mF1": "\033[96m",
        "mUAR": "\033[91m",       # light cyan
        "ACC": "\033[32m",        # green
        "CCC": "\033[33m",        # yellow
        "mean_emo": "\033[1;34m", # bold blue
        "mean_pkl": "\033[1;35m", # bold magenta
    }
    END = "\033[0m"
    color = COLORS.get(metric_name, "")
    return f"{color}{metric_name}:{value:.4f}{END}"


def color_split(name: str) -> str:
    """
    Adds ANSI color formatting to dataset split names for console output.
    """
    SPLIT_COLORS = {
        "TRAIN": "\033[1;33m",  # bright yellow
        "Dev": "\033[1;31m",    # bright red
        "Test": "\033[1;35m",   # bright magenta
    }
    END = "\033[0m"
    return f"{SPLIT_COLORS.get(name, '')}{name}{END}"
