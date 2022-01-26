import os
import sys

from loguru import logger as _logger

logger = _logger

logger.remove()

#  Add default handler
logger.add(sys.stdout, diagnose=False)
