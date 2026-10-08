"""
Parse dates and times from HiST file names.

HiST HDF5 files are expected to have names containing a ``YYYY-MM-DD``
date, e.g. ``2013-03-27-CamSer7196.h5``.
"""

from datetime import datetime, timezone
from pathlib import Path
import re





