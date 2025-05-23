import os
import logging
from logging.config import dictConfig

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] %(message)s",
        },
        "simple": {
            "format": "[%(levelname)s] %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": os.path.join(".obz", "obz.log"),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 5,
            "formatter": "default",
        },
    },
    "root": {
        "level": os.getenv("LOG_LEVEL", "INFO"),
        "handlers": ["console", "file"],
    },
    "loggers": {
        "obz_client": {
            "level": os.getenv("OBZ_CLIENT_LOG_LEVEL", "INFO"),
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "data_inspector": {
            "level": os.getenv("DATA_INSPECTOR_LOG_LEVEL", "INFO"),
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "xai": {
            "level": os.getenv("XAI_LOG_LEVEL", "INFO"),
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
}

def setup_logging():
    try:
        os.makedirs(".obz", exist_ok=True)
        dictConfig(LOGGING_CONFIG)
    except Exception as e:
        print(f"Failed to configure logging: {e}")
        logging.basicConfig(level=logging.ERROR)