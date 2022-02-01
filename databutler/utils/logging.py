import os
import sys

from loguru import logger

#  Remove any existing handlers
from databutler.utils import paths

logger.remove()

#  Add in sys.stderr with our own configuration
logger.add(sys.stderr, level="INFO")

#  Add in a file-log
logger.add(os.path.join(paths.get_logging_dir_path(), "databutler_{time}.log"),
           retention=50,  # Max. 50 logs
           level="TRACE")
