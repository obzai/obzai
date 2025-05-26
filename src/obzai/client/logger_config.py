# Obz AI - Copyright (C) 2025 Alethia XAI Sp. z o.o.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

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