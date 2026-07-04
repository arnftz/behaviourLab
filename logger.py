import logging
from pathlib import Path

Path("logs").mkdir(exist_ok=True)

logger = logging.getLogger("behaviourLab")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

# Everything goes to the log file
file_handler = logging.FileHandler(
    "logs/behaviourlab.log",
    encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Terminal only gets INFO, ERROR and CRITICAL
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

class ConsoleFilter(logging.Filter):
    def filter(self, record):
        return record.levelno in (
            logging.INFO,
            logging.ERROR,
            logging.CRITICAL,
        )

console_handler.addFilter(ConsoleFilter())

logger.addHandler(file_handler)
logger.addHandler(console_handler)