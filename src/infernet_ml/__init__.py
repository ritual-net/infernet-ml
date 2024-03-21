"""
specify module level settings here
"""

import logging
import os
from sys import stdout
from typing import Union

from dotenv import load_dotenv

# load dotenv files defined in module
load_dotenv()

LOGLEVEL: Union[str, int] = os.getenv("LOGLEVEL", default=logging.INFO)

# define the default logging config for the infernet_ml module here
logging.basicConfig(
    level=LOGLEVEL,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
    stream=stdout,
)
